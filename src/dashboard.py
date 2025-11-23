"""
Web Dashboard for Real-time Frame Visualization

This module provides a web server that receives compressed frames via UDP
and streams them to connected browsers via WebSocket. This allows non-blocking
visualization of training/gameplay frames.

Usage:
    1. Start the dashboard server: python -m src.dashboard
    2. Open browser to http://localhost:8080
    3. Send frames via UDP to localhost:9999
"""

import asyncio
import base64
import cv2
import io
import json
import numpy as np
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from aiohttp import web
import aiohttp

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def query_gpu_status():
    """Collect GPU availability and utilization information."""
    gpus = []
    source = None
    timestamp = datetime.utcnow().isoformat() + 'Z'

    nvidia_query = [
        'nvidia-smi',
        '--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'
    ]

    try:
        result = subprocess.run(
            nvidia_query,
            capture_output=True,
            text=True,
            check=True
        )
        source = 'nvidia-smi'
        for line in result.stdout.strip().splitlines():
            parts = [item.strip() for item in line.split(',')]
            if len(parts) >= 5:
                index, name, mem_total, mem_used, util = parts[:5]
                try:
                    gpus.append({
                        'index': int(index),
                        'name': name,
                        'memory_total_mb': int(mem_total),
                        'memory_used_mb': int(mem_used),
                        'utilization_gpu': int(util)
                    })
                except ValueError:
                    continue
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    if not gpus:
        try:
            import torch
            if torch.cuda.is_available():
                source = 'torch'
                for index in range(torch.cuda.device_count()):
                    gpus.append({
                        'index': index,
                        'name': torch.cuda.get_device_name(index),
                        'memory_total_mb': None,
                        'memory_used_mb': None,
                        'utilization_gpu': None
                    })
        except Exception:
            pass

    return {
        'gpus': gpus,
        'source': source,
        'timestamp': timestamp
    }


class FrameReceiver:
    """Receives compressed JPEG frames via UDP."""
    
    def __init__(self, host='0.0.0.0', port=9999, max_packet_size=65507):
        self.host = host
        self.port = port
        self.max_packet_size = max_packet_size
        self.socket = None
        self.latest_frames = {}  # {(trainer_name, channel_name): compressed_jpeg_bytes}
        self.last_frame_time = {}
        self.removed_trainers = set()
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
    def start(self):
        """Start the UDP receiver in a background thread."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # 1MB buffer
        self.socket.bind((self.host, self.port))
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        print(f"UDP Frame Receiver started on {self.host}:{self.port}")
        
    def stop(self):
        """Stop the UDP receiver."""
        self.running = False
        if self.socket:
            self.socket.close()
        if self.thread:
            self.thread.join(timeout=2)
            
    def _receive_loop(self):
        """Main loop to receive UDP packets."""
        while self.running:
            try:
                # Receive packet
                data, addr = self.socket.recvfrom(self.max_packet_size)
                
                # Parse packet: [trainer_name_len(2)][trainer_name][channel_name_len(2)][channel_name][compressed_jpeg]
                if len(data) < 4:  # Need at least 2 bytes for trainer name len + 2 for channel name len
                    continue
                
                offset = 0
                
                # Parse trainer name
                trainer_name_len = struct.unpack('!H', data[offset:offset+2])[0]
                offset += 2
                if len(data) < offset + trainer_name_len:
                    continue
                trainer_name = data[offset:offset + trainer_name_len].decode('utf-8')
                offset += trainer_name_len
                
                # Parse channel name
                if len(data) < offset + 2:
                    continue
                channel_name_len = struct.unpack('!H', data[offset:offset+2])[0]
                offset += 2
                if len(data) < offset + channel_name_len:
                    continue
                channel_name = data[offset:offset + channel_name_len].decode('utf-8')
                offset += channel_name_len
                
                # Rest is compressed JPEG data
                compressed_frame = data[offset:]
                
                # Store latest frame for this trainer/channel combination
                with self.lock:
                    self.latest_frames[(trainer_name, channel_name)] = compressed_frame
                    self.last_frame_time[trainer_name] = time.time()
                    
            except Exception as e:
                if self.running:
                    print(f"Error receiving frame: {e}")
                    import traceback
                    traceback.print_exc()
                    
    def get_latest_frames(self):
        """Get the latest frames for all channels."""
        with self.lock:
            return dict(self.latest_frames)

    def clear_trainer(self, trainer_name):
        """Remove all cached frames for a trainer."""
        removed = False
        with self.lock:
            keys_to_remove = [key for key in self.latest_frames.keys() if key[0] == trainer_name]
            for key in keys_to_remove:
                self.latest_frames.pop(key, None)
                removed = True
            self.last_frame_time.pop(trainer_name, None)
            if removed:
                self.removed_trainers.add(trainer_name)
        return removed

    def consume_removed_trainers(self):
        """Return trainers cleared since last check and reset the list."""
        with self.lock:
            removed = list(self.removed_trainers)
            self.removed_trainers.clear()
            return removed

    def list_trainers(self):
        """Return metadata about cached trainers and their channels."""
        summary = {}
        with self.lock:
            for (trainer_name, channel_name) in self.latest_frames.keys():
                data = summary.setdefault(trainer_name, {
                    'trainer': trainer_name,
                    'channels': set(),
                    'last_frame_ts': self.last_frame_time.get(trainer_name)
                })
                data['channels'].add(channel_name)

        trainers = []
        for trainer_name, data in summary.items():
            trainers.append({
                'trainer': trainer_name,
                'channels': sorted(data['channels']),
                'last_frame_ts': data.get('last_frame_ts')
            })

        trainers.sort(key=lambda item: item.get('last_frame_ts') or 0, reverse=True)
        return trainers


class TrainerProcessManager:
    """Launch and track training processes requested from the dashboard."""

    def __init__(self, project_root):
        self.project_root = Path(project_root).resolve()
        self.agents_dir = (self.project_root / 'agents').resolve()
        self.checkpoint_dir = (self.project_root / 'checkpoint').resolve()
        self.config_output_dir = self.project_root / 'runs' / 'dashboard_configs'
        self.log_dir = self.project_root / 'runs' / 'dashboard_logs'
        self.config_output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.python_executable = sys.executable
        self.processes = {}
        self._id_counter = 0
        self._lock = asyncio.Lock()
        self.allowed_config_roots = [self.agents_dir]
        if self.checkpoint_dir.exists():
            self.allowed_config_roots.append(self.checkpoint_dir)

    def list_configs(self):
        """List available configuration files in agents/ and checkpoint/ directories."""
        configs = set()
        configs.update(self._collect_configs(self.agents_dir, pattern='*.y*ml'))
        configs.update(self._collect_checkpoint_configs())
        return sorted(configs)

    def _collect_configs(self, base_dir, pattern='*.y*ml'):
        if not base_dir.exists():
            return set()
        entries = set()
        for path in base_dir.rglob(pattern):
            if path.is_file():
                try:
                    entries.add(str(path.resolve().relative_to(self.project_root)))
                except ValueError:
                    continue
        return entries

    def _collect_checkpoint_configs(self):
        """Return config.yaml files under checkpoint directory."""
        entries = set()
        if not self.checkpoint_dir.exists():
            return entries
        for config_file in self.checkpoint_dir.rglob('config.yaml'):
            if config_file.is_file():
                try:
                    entries.add(str(config_file.resolve().relative_to(self.project_root)))
                except ValueError:
                    continue
        return entries

    def read_config(self, relative_path):
        """Read the contents of a configuration file under agents/."""
        resolved = self._resolve_config_path(relative_path)
        return resolved.read_text()

    def _resolve_config_path(self, path_value):
        if not path_value:
            raise ValueError('config_path is required')
        candidate = Path(path_value)
        if not candidate.is_absolute():
            candidate = (self.project_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        candidate_str = str(candidate)
        if not any(candidate_str.startswith(str(root)) for root in self.allowed_config_roots):
            raise ValueError('Configuration must live inside the agents/ or checkpoint/ directories')
        if not candidate.exists():
            raise ValueError(f'Configuration not found: {candidate}')
        return candidate

    def _to_relative(self, path_value):
        try:
            return str(Path(path_value).resolve().relative_to(self.project_root))
        except Exception:
            return str(path_value)

    async def spawn_trainer(self,
                            config_path,
                            *,
                            config_text=None,
                            restart=False,
                            use_dashboard=True,
                            jpeg_quality=85,
                            cuda_visible_devices=None,
                            suppress_tqdm=False,
                            discard_logs=False):
        resolved = self._resolve_config_path(config_path)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        final_config = resolved
        snapshot_path = None
        config_text_clean = config_text if config_text and config_text.strip() else None

        if config_text_clean is not None:
            snapshot_name = f"dashboard_{timestamp}_{resolved.stem}.yaml"
            snapshot_path = self.config_output_dir / snapshot_name
            snapshot_path.write_text(config_text_clean)
            final_config = snapshot_path

        log_file_path = None
        if not discard_logs:
            log_file_path = self.log_dir / f"trainer_{timestamp}_{resolved.stem}.log"
            log_handle = open(log_file_path, 'w')
        else:
            log_handle = open(os.devnull, 'w')

        cmd = [
            self.python_executable,
            str(self.project_root / 'main.py'),
            '--config',
            str(final_config)
        ]

        if restart:
            cmd.append('--restart')
        if use_dashboard:
            cmd.append('--use_dashboard')
            cmd.extend(['--jpeg_quality', str(jpeg_quality)])

        env = os.environ.copy()
        if cuda_visible_devices is not None and str(cuda_visible_devices).strip():
            env['CUDA_VISIBLE_DEVICES'] = str(cuda_visible_devices).strip()
        if suppress_tqdm:
            env['TQDM_DISABLE'] = '1'

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=log_handle,
                stderr=log_handle,
                cwd=str(self.project_root),
                env=env
            )
        except Exception:
            log_handle.close()
            if snapshot_path and snapshot_path.exists():
                snapshot_path.unlink(missing_ok=True)
            raise

        async with self._lock:
            self._id_counter += 1
            entry_id = self._id_counter
            entry = {
                'id': entry_id,
                'pid': process.pid,
                'status': 'running',
                'returncode': None,
                'base_config': self._to_relative(resolved),
                'config_path': self._to_relative(final_config),
                'uses_snapshot': snapshot_path is not None,
                'log_path': self._to_relative(log_file_path) if log_file_path else None,
                'log_mode': 'discard' if discard_logs else 'file',
                'started_at': time.time(),
                'finished_at': None,
                'process': process,
                'log_handle': log_handle,
                'cuda_visible_devices': str(cuda_visible_devices).strip() if cuda_visible_devices else None,
                'suppress_tqdm': bool(suppress_tqdm),
            }
            self.processes[entry_id] = entry

        asyncio.create_task(self._monitor_process(entry_id))
        return self._public_entry(entry)

    async def _monitor_process(self, entry_id):
        async with self._lock:
            entry = self.processes.get(entry_id)
        if not entry:
            return
        process = entry['process']
        try:
            returncode = await process.wait()
        finally:
            entry['log_handle'].close()

        async with self._lock:
            entry = self.processes.get(entry_id)
            if not entry:
                return
            entry['returncode'] = returncode
            entry['status'] = 'finished' if returncode == 0 else 'failed'
            entry['finished_at'] = time.time()

    async def get_processes_snapshot(self):
        async with self._lock:
            entries = [self._public_entry(entry) for entry in self.processes.values()]
        entries.sort(key=lambda item: item['id'], reverse=True)
        return entries

    def _public_entry(self, entry):
        return {
            'id': entry['id'],
            'pid': entry['pid'],
            'status': entry['status'],
            'returncode': entry['returncode'],
            'base_config': entry['base_config'],
            'config_path': entry['config_path'],
            'uses_snapshot': entry['uses_snapshot'],
            'log_path': entry['log_path'],
             'log_mode': entry.get('log_mode'),
            'started_at': entry['started_at'],
            'finished_at': entry['finished_at'],
            'cuda_visible_devices': entry.get('cuda_visible_devices'),
            'suppress_tqdm': entry.get('suppress_tqdm', False),
        }


class DashboardServer:
    """Web server that serves frames to browser clients."""
    
    def __init__(self, web_port=8080, udp_port=9999):
        self.web_port = web_port
        self.receiver = FrameReceiver(port=udp_port)
        self.app = web.Application()
        self.websockets = set()
        self.trainer_manager = TrainerProcessManager(PROJECT_ROOT)
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup HTTP and WebSocket routes."""
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/ws', self.handle_websocket)
        self.app.router.add_get('/api/configs', self.handle_get_configs)
        self.app.router.add_get('/api/configs/content', self.handle_get_config_content)
        self.app.router.add_get('/api/trainers', self.handle_get_trainers)
        self.app.router.add_get('/api/gpus', self.handle_get_gpus)
        self.app.router.add_post('/api/trainers/clear', self.handle_clear_trainer)
        self.app.router.add_post('/api/trainers/spawn', self.handle_spawn_trainer)
        
    async def handle_index(self, request):
        """Serve the HTML dashboard page."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Training Dashboard</title>
    <style>
        body {
            background-color: #1e1e1e;
            color: #d4d4d4;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #4ec9b0;
        }
        #controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .panel {
            background-color: #252526;
            border: 2px solid #3e3e42;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            min-width: 0;
        }
        .panel-header {
            padding: 12px 16px;
            border-bottom: 1px solid #3e3e42;
            font-weight: bold;
            color: #4ec9b0;
        }
        .panel-body {
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            min-width: 0;
        }
        .panel-help {
            font-size: 12px;
            color: #9e9e9e;
        }
        .panel-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
            min-height: 40px;
        }
        .trainer-row {
            display: flex;
            justify-content: space-between;
            gap: 10px;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #3e3e42;
            flex-wrap: wrap;
        }
        .trainer-row:last-child {
            border-bottom: none;
        }
        .trainer-meta {
            font-size: 12px;
            color: #9e9e9e;
            word-break: break-word;
        }
        .trainer-row strong {
            overflow-wrap: anywhere;
        }
        .trainer-row div {
            min-width: 0;
        }
        select, textarea, input[type="number"] {
            background-color: #1e1e1e;
            color: #d4d4d4;
            border: 1px solid #3e3e42;
            border-radius: 4px;
            padding: 6px;
            max-width: 100%;
        }
        textarea {
            min-height: 180px;
            font-family: 'Fira Code', 'Courier New', monospace;
            resize: vertical;
        }
        button {
            border: none;
            border-radius: 4px;
            padding: 8px 12px;
            cursor: pointer;
            font-weight: bold;
        }
        .primary-button {
            background-color: #0e639c;
            color: #ffffff;
        }
        .control-button {
            background-color: #3a3d41;
            color: #d4d4d4;
        }
        .danger-button {
            background-color: #c23c2a;
            color: #ffffff;
        }
        .inline-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
        }
        .status {
            text-align: center;
            padding: 10px;
            margin-top: 10px;
            border-radius: 5px;
        }
        .status.connected {
            background-color: #0e7d0e;
        }
        .status.disconnected {
            background-color: #a31515;
        }
        .stats {
            text-align: center;
            margin-top: 10px;
            font-size: 12px;
            color: #858585;
        }
        #frames-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            margin-top: 30px;
        }
        .trainer-section {
            width: 100%;
            margin-bottom: 30px;
        }
        .trainer-header {
            background-color: #2d2d30;
            padding: 10px 20px;
            border-radius: 8px 8px 0 0;
            border: 2px solid #007acc;
            font-size: 18px;
            font-weight: bold;
            color: #007acc;
        }
        .trainer-frames {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            padding: 20px;
            background-color: #252526;
            border: 2px solid #007acc;
            border-top: none;
            border-radius: 0 0 8px 8px;
        }
        .frame-box {
            background-color: #252526;
            border: 2px solid #3e3e42;
            border-radius: 8px;
            padding: 15px;
            min-width: 300px;
        }
        .frame-title {
            color: #4ec9b0;
            font-weight: bold;
            margin-bottom: 10px;
            text-align: center;
        }
        .frame-image {
            max-width: 100%;
            image-rendering: pixelated;
            border: 1px solid #3e3e42;
            background-color: #1e1e1e;
        }
        .status-text {
            font-size: 12px;
            min-height: 16px;
        }
        .text-muted {
            color: #9e9e9e;
            font-size: 12px;
        }
        .gpu-status-grid {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .gpu-card {
            border: 1px solid #3e3e42;
            border-radius: 6px;
            padding: 8px;
            background-color: #1e1e1e;
        }
        .badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            background-color: #3a3d41;
            font-size: 11px;
            margin-left: 6px;
        }
    </style>
</head>
<body>
    <h1>🎮 Training Dashboard</h1>
    <div id="status" class="status disconnected">Connecting...</div>
    <div id="stats" class="stats">FPS: 0 | Frames: 0</div>
    <div id="controls">
        <div class="panel">
            <div class="panel-header">Active Sessions</div>
            <div class="panel-body">
                <p class="panel-help">Hide finished runs from the grid once training stops sending frames.</p>
                <div id="trainer-list" class="panel-list">No trainers detected yet.</div>
            </div>
        </div>
        <div class="panel">
            <div class="panel-header">Launch Trainer</div>
            <div class="panel-body">
                <label for="config-select">Agent Config</label>
                <div class="inline-row">
                    <select id="config-select"></select>
                    <button id="load-config" class="control-button">Load</button>
                </div>
                <label for="gpu-select">CUDA Device Visibility</label>
                <div class="inline-row">
                    <select id="gpu-select">
                        <option value="">All GPUs (default)</option>
                    </select>
                    <button id="refresh-gpus" class="control-button" type="button">Refresh GPUs</button>
                </div>
                <div id="gpu-select-help" class="text-muted">Populate GPUs using the Refresh button.</div>
                <textarea id="config-editor" spellcheck="false" placeholder="YAML config will appear here..."></textarea>
                <div class="inline-row">
                    <label>JPEG Quality
                        <input type="number" id="jpeg-quality" value="85" min="10" max="100">
                    </label>
                    <label>
                        <input type="checkbox" id="restart-training">
                        Restart checkpoints
                    </label>
                    <label>
                        <input type="checkbox" id="suppress-tqdm">
                        Disable tqdm output
                    </label>
                    <label>
                        <input type="checkbox" id="discard-logs">
                        Discard logs
                    </label>
                </div>
                <button id="launch-trainer" class="primary-button">Launch Trainer</button>
                <div id="launch-status" class="status-text"></div>
            </div>
        </div>
        <div class="panel">
            <div class="panel-header">Trainer Processes</div>
            <div class="panel-body">
                <div id="process-list" class="panel-list">No processes launched from the dashboard yet.</div>
            </div>
        </div>
        <div class="panel">
            <div class="panel-header">GPU Status</div>
            <div class="panel-body">
                <div id="gpu-status" class="gpu-status-grid">Loading GPU info...</div>
                <div class="text-muted" id="gpu-updated">&nbsp;</div>
            </div>
        </div>
    </div>
    <div id="frames-container"></div>
    
    <script>
        let ws = null;
        let frameCount = 0;
        let lastFrameTime = Date.now();
        let fps = 0;
        let trainerRefreshTimer = null;
        let gpuRefreshTimer = null;
        
        document.addEventListener('DOMContentLoaded', () => {
            loadConfigList();
            loadGpuList();
            refreshTrainerList();
            trainerRefreshTimer = setInterval(refreshTrainerList, 5000);
            gpuRefreshTimer = setInterval(loadGpuList, 7000);
            document.getElementById('load-config').addEventListener('click', loadSelectedConfig);
            document.getElementById('config-select').addEventListener('change', loadSelectedConfig);
            document.getElementById('launch-trainer').addEventListener('click', launchTrainerFromEditor);
            document.getElementById('refresh-gpus').addEventListener('click', loadGpuList);
        });

        function safeId(value) {
            return value.replace(/[^a-zA-Z0-9_-]/g, '_');
        }
        
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(protocol + '//' + window.location.host + '/ws');
            
            ws.onopen = function() {
                document.getElementById('status').textContent = 'Connected';
                document.getElementById('status').className = 'status connected';
                document.getElementById('frames-container').innerHTML = '';
            };
            
            ws.onclose = function() {
                document.getElementById('status').textContent = 'Disconnected - Reconnecting...';
                document.getElementById('status').className = 'status disconnected';
                setTimeout(connect, 2000);
            };
            
            ws.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateFrames(data.frames || {}, data.removed_trainers || []);
                updateStats();
            };
        }
        
        async function loadConfigList() {
            const select = document.getElementById('config-select');
            select.innerHTML = '';
            try {
                const response = await fetch('/api/configs');
                const data = await response.json();
                if (!data.configs || data.configs.length === 0) {
                    const option = document.createElement('option');
                    option.value = '';
                    option.textContent = 'No configs found in agents/';
                    select.appendChild(option);
                    return;
                }
                data.configs.forEach((cfg) => {
                    const option = document.createElement('option');
                    option.value = cfg;
                    option.textContent = cfg;
                    select.appendChild(option);
                });
                await loadSelectedConfig();
            } catch (error) {
                console.error('Failed to list configs', error);
            }
        }

        async function loadSelectedConfig() {
            const select = document.getElementById('config-select');
            const editor = document.getElementById('config-editor');
            const path = select.value;
            if (!path) {
                editor.value = '';
                return;
            }
            try {
                const response = await fetch(`/api/configs/content?path=${encodeURIComponent(path)}`);
                const data = await response.json();
                if (data.error) {
                    editor.value = data.error;
                } else {
                    editor.value = data.content;
                }
            } catch (error) {
                console.error('Failed to load config', error);
                editor.value = '# Unable to load config';
            }
        }

        async function loadGpuList() {
            const select = document.getElementById('gpu-select');
            const help = document.getElementById('gpu-select-help');
            try {
                const response = await fetch('/api/gpus');
                const data = await response.json();
                populateGpuSelect(select, data.gpus || []);
                renderGpuStatus(data);
                if (help) {
                    help.textContent = data.gpus && data.gpus.length
                        ? `Fetched ${data.gpus.length} GPU${data.gpus.length === 1 ? '' : 's'} via ${data.source || 'unknown'}.`
                        : 'No GPUs detected. Using default visibility.';
                }
            } catch (error) {
                console.error('Failed to load GPU info', error);
                if (help) {
                    help.textContent = 'Unable to query GPUs. Using default visibility.';
                }
                renderGpuStatus({ gpus: [], source: null });
            }
        }

        function populateGpuSelect(select, gpus) {
            if (!select) {
                return;
            }
            const currentValue = select.value;
            const options = [
                { value: '', label: 'All GPUs (default)' }
            ];
            gpus.forEach((gpu) => {
                options.push({
                    value: String(gpu.index),
                    label: `GPU ${gpu.index} • ${gpu.name}`
                });
            });
            select.innerHTML = '';
            options.forEach((opt) => {
                const option = document.createElement('option');
                option.value = opt.value;
                option.textContent = opt.label;
                select.appendChild(option);
            });
            if (options.some((opt) => opt.value === currentValue)) {
                select.value = currentValue;
            }
        }

        function renderGpuStatus(data) {
            const container = document.getElementById('gpu-status');
            const updated = document.getElementById('gpu-updated');
            if (!container) {
                return;
            }
            container.innerHTML = '';
            const gpus = data && data.gpus ? data.gpus : [];
            if (!gpus.length) {
                container.textContent = 'No GPU information available.';
            } else {
                gpus.forEach((gpu) => {
                    const card = document.createElement('div');
                    card.className = 'gpu-card';
                    const title = document.createElement('div');
                    title.innerHTML = `<strong>GPU ${gpu.index}</strong> <span class="badge">${gpu.name}</span>`;
                    const stats = document.createElement('div');
                    stats.className = 'trainer-meta';
                    const util = gpu.utilization_gpu !== null && gpu.utilization_gpu !== undefined
                        ? `${gpu.utilization_gpu}%`
                        : 'n/a';
                    const memTotal = gpu.memory_total_mb ? `${gpu.memory_total_mb} MB` : 'n/a';
                    const memUsed = gpu.memory_used_mb ? `${gpu.memory_used_mb} MB` : 'n/a';
                    stats.textContent = `Utilization: ${util} | Memory: ${memUsed} / ${memTotal}`;
                    card.appendChild(title);
                    card.appendChild(stats);
                    container.appendChild(card);
                });
            }
            if (updated) {
                const source = data && data.source ? data.source : 'unknown';
                updated.textContent = `Updated ${new Date().toLocaleTimeString()} via ${source}.`;
            }
        }

        function setLaunchStatus(message, isError = false) {
            const statusEl = document.getElementById('launch-status');
            statusEl.textContent = message || '';
            statusEl.style.color = isError ? '#f48771' : '#4ec9b0';
        }

        async function launchTrainerFromEditor() {
            const path = document.getElementById('config-select').value;
            const editorText = document.getElementById('config-editor').value;
            const restart = document.getElementById('restart-training').checked;
            const jpegQuality = parseInt(document.getElementById('jpeg-quality').value, 10) || 85;
            const cudaVisible = document.getElementById('gpu-select').value;
            const suppressTqdm = document.getElementById('suppress-tqdm').checked;
            const discardLogs = document.getElementById('discard-logs').checked;
            if (!path) {
                setLaunchStatus('Pick a configuration file first.', true);
                return;
            }
            setLaunchStatus('Launching trainer...');
            try {
                const response = await fetch('/api/trainers/spawn', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        config_path: path,
                        config_text: editorText && editorText.trim() ? editorText : null,
                        restart: restart,
                        use_dashboard: true,
                        jpeg_quality: jpegQuality,
                        cuda_visible_devices: cudaVisible || null,
                        suppress_tqdm: suppressTqdm,
                        discard_logs: discardLogs
                    })
                });
                const data = await response.json();
                if (!response.ok || data.error) {
                    setLaunchStatus(data.error || 'Failed to launch trainer', true);
                } else {
                    setLaunchStatus(`Started trainer (pid ${data.process.pid}).`);
                    refreshTrainerList();
                }
            } catch (error) {
                console.error('Failed to launch trainer', error);
                setLaunchStatus('Failed to launch trainer', true);
            }
        }

        async function refreshTrainerList() {
            try {
                const response = await fetch('/api/trainers');
                const data = await response.json();
                renderTrainerList(data.sessions || []);
                renderProcessList(data.processes || []);
            } catch (error) {
                console.error('Failed to pull trainer metadata', error);
            }
        }

        function renderTrainerList(sessions) {
            const container = document.getElementById('trainer-list');
            container.innerHTML = '';
            if (!sessions.length) {
                container.textContent = 'No cached frames yet.';
                return;
            }
            sessions.forEach((session) => {
                const row = document.createElement('div');
                row.className = 'trainer-row';
                const info = document.createElement('div');
                const meta = document.createElement('div');
                const lastSeen = session.last_frame_iso ? new Date(session.last_frame_iso).toLocaleString() : 'unknown';
                const channels = (session.channels && session.channels.length) ? session.channels.join(', ') : '—';
                info.innerHTML = `<strong>${session.trainer}</strong>`;
                meta.className = 'trainer-meta';
                meta.textContent = `Channels: ${channels} | Last frame: ${lastSeen}`;
                info.appendChild(meta);
                const button = document.createElement('button');
                button.className = 'danger-button';
                button.textContent = 'Remove';
                button.addEventListener('click', () => clearTrainer(session.trainer));
                row.appendChild(info);
                row.appendChild(button);
                container.appendChild(row);
            });
        }

        function renderProcessList(processes) {
            const container = document.getElementById('process-list');
            container.innerHTML = '';
            if (!processes.length) {
                container.textContent = 'No processes launched from the dashboard yet.';
                return;
            }
            processes.forEach((proc) => {
                const row = document.createElement('div');
                row.className = 'trainer-row';
                const info = document.createElement('div');
                const details = document.createElement('div');
                const statusLine = `Status: ${proc.status}${proc.returncode !== null ? ` (code ${proc.returncode})` : ''}`;
                const started = proc.started_at_iso ? new Date(proc.started_at_iso).toLocaleString() : 'unknown';
                const finished = proc.finished_at_iso ? new Date(proc.finished_at_iso).toLocaleString() : '—';
                const logPath = proc.log_path || '—';
                const gpu = proc.cuda_visible_devices ? proc.cuda_visible_devices : 'All';
                const tqdmState = proc.suppress_tqdm ? 'disabled' : 'enabled';
                const logMode = proc.log_mode === 'discard' ? 'discarded' : 'file';
                info.innerHTML = `<strong>#${proc.id} • ${proc.base_config}</strong>`;
                details.className = 'trainer-meta';
                details.innerHTML = `${statusLine} | PID: ${proc.pid} | Log: ${logPath} (${logMode})<br>GPU: ${gpu} | tqdm: ${tqdmState}<br>Started: ${started} | Finished: ${finished}`;
                info.appendChild(details);
                row.appendChild(info);
                container.appendChild(row);
            });
        }

        async function clearTrainer(trainerName) {
            try {
                await fetch('/api/trainers/clear', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ trainer_name: trainerName })
                });
                refreshTrainerList();
            } catch (error) {
                console.error('Failed to clear trainer', error);
            }
        }

        function removeTrainerSection(trainer) {
            const section = document.getElementById('trainer-' + safeId(trainer));
            if (section) {
                section.remove();
            }
        }

        function updateFrames(frames, removedTrainers) {
            const container = document.getElementById('frames-container');

            (removedTrainers || []).forEach(removeTrainerSection);
            
            const trainerGroups = {};
            for (const [key, imageData] of Object.entries(frames)) {
                const [trainer, channel] = key.split('::');
                if (!trainerGroups[trainer]) {
                    trainerGroups[trainer] = {};
                }
                trainerGroups[trainer][channel] = imageData;
            }
            
            for (const [trainer, channels] of Object.entries(trainerGroups)) {
                const trainerId = 'trainer-' + safeId(trainer);
                const framesId = 'frames-' + safeId(trainer);
                let trainerSection = document.getElementById(trainerId);
                
                if (!trainerSection) {
                    trainerSection = document.createElement('div');
                    trainerSection.className = 'trainer-section';
                    trainerSection.id = trainerId;
                    trainerSection.dataset.trainer = trainer;
                    
                    const header = document.createElement('div');
                    header.className = 'trainer-header';
                    header.textContent = '🎮 ' + trainer;
                    
                    const framesDiv = document.createElement('div');
                    framesDiv.className = 'trainer-frames';
                    framesDiv.id = framesId;
                    framesDiv.dataset.trainer = trainer;
                    
                    trainerSection.appendChild(header);
                    trainerSection.appendChild(framesDiv);
                    container.appendChild(trainerSection);
                }
                
                const framesDiv = document.getElementById(framesId);
                
                for (const [channel, imageData] of Object.entries(channels)) {
                    const frameId = `frame-${safeId(trainer)}-${safeId(channel)}`;
                    const imgId = `img-${safeId(trainer)}-${safeId(channel)}`;
                    let frameBox = document.getElementById(frameId);
                    
                    if (!frameBox) {
                        frameBox = document.createElement('div');
                        frameBox.className = 'frame-box';
                        frameBox.id = frameId;
                        
                        const title = document.createElement('div');
                        title.className = 'frame-title';
                        title.textContent = channel;
                        
                        const img = document.createElement('img');
                        img.className = 'frame-image';
                        img.id = imgId;
                        
                        frameBox.appendChild(title);
                        frameBox.appendChild(img);
                        framesDiv.appendChild(frameBox);
                    }
                    
                    const img = document.getElementById(imgId);
                    img.src = 'data:image/jpeg;base64,' + imageData;
                }
            }
            
            frameCount++;
        }
        
        function updateStats() {
            const now = Date.now();
            const elapsed = (now - lastFrameTime) / 1000;
            
            if (elapsed >= 1.0) {
                fps = (frameCount / elapsed).toFixed(1);
                document.getElementById('stats').textContent = 
                    `FPS: ${fps} | Frames: ${frameCount}`;
                lastFrameTime = now;
                frameCount = 0;
            }
        }
        
        connect();
    </script>
</body>
</html>
        """
        return web.Response(text=html, content_type='text/html')
    
    async def handle_get_configs(self, request):
        """Return a list of available agent configuration files."""
        configs = self.trainer_manager.list_configs()
        return web.json_response({'configs': configs})

    async def handle_get_config_content(self, request):
        """Return the contents of a configuration file."""
        config_path = request.query.get('path')
        if not config_path:
            return web.json_response({'error': 'path query parameter is required'}, status=400)
        try:
            content = self.trainer_manager.read_config(config_path)
        except ValueError as exc:
            return web.json_response({'error': str(exc)}, status=400)
        return web.json_response({'path': config_path, 'content': content})

    async def handle_get_trainers(self, request):
        """Return metadata for active frame sessions and launched trainers."""
        sessions = self.receiver.list_trainers()
        for session in sessions:
            ts = session.get('last_frame_ts')
            session['last_frame_iso'] = datetime.utcfromtimestamp(ts).isoformat() + 'Z' if ts else None
        processes = await self.trainer_manager.get_processes_snapshot()
        for process in processes:
            started = process.get('started_at')
            finished = process.get('finished_at')
            process['started_at_iso'] = datetime.utcfromtimestamp(started).isoformat() + 'Z' if started else None
            process['finished_at_iso'] = datetime.utcfromtimestamp(finished).isoformat() + 'Z' if finished else None
        return web.json_response({'sessions': sessions, 'processes': processes})

    async def handle_get_gpus(self, request):
        """Return GPU metadata and utilization information."""
        return web.json_response(query_gpu_status())

    async def handle_clear_trainer(self, request):
        """Remove cached frames for a trainer so it disappears from the UI."""
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.json_response({'error': 'Invalid JSON payload'}, status=400)
        trainer_name = payload.get('trainer_name') if isinstance(payload, dict) else None
        if not trainer_name:
            return web.json_response({'error': 'trainer_name is required'}, status=400)
        removed = self.receiver.clear_trainer(trainer_name)
        return web.json_response({'removed': removed})

    async def handle_spawn_trainer(self, request):
        """Launch a new training process from the browser."""
        try:
            payload = await request.json()
        except json.JSONDecodeError:
            return web.json_response({'error': 'Invalid JSON payload'}, status=400)

        if not isinstance(payload, dict):
            return web.json_response({'error': 'Invalid payload'}, status=400)

        config_path = payload.get('config_path')
        config_text = payload.get('config_text')
        restart = bool(payload.get('restart', False))
        use_dashboard = bool(payload.get('use_dashboard', True))
        jpeg_quality = payload.get('jpeg_quality', 85)
        cuda_visible_devices = payload.get('cuda_visible_devices')
        suppress_tqdm = bool(payload.get('suppress_tqdm', False))
        discard_logs = bool(payload.get('discard_logs', False))

        if not config_path:
            return web.json_response({'error': 'config_path is required'}, status=400)

        try:
            jpeg_quality = int(jpeg_quality)
        except (TypeError, ValueError):
            return web.json_response({'error': 'jpeg_quality must be an integer'}, status=400)

        try:
            process_info = await self.trainer_manager.spawn_trainer(
                config_path=config_path,
                config_text=config_text,
                restart=restart,
                use_dashboard=use_dashboard,
                jpeg_quality=jpeg_quality,
                cuda_visible_devices=cuda_visible_devices,
                suppress_tqdm=suppress_tqdm,
                discard_logs=discard_logs
            )
        except ValueError as exc:
            return web.json_response({'error': str(exc)}, status=400)
        except Exception as exc:
            return web.json_response({'error': str(exc)}, status=500)

        return web.json_response({'process': process_info})
        
    async def handle_websocket(self, request):
        """Handle WebSocket connections from clients."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.websockets.add(ws)
        
        try:
            # Keep connection alive and send frames
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.ERROR:
                    print(f'WebSocket connection closed with exception {ws.exception()}')
        finally:
            self.websockets.discard(ws)
            
        return ws
        
    async def broadcast_frames(self):
        """Periodically broadcast frames to all connected WebSocket clients."""
        while True:
            try:
                await asyncio.sleep(0.033)  # ~30 FPS
                
                if not self.websockets:
                    continue
                    
                # Get latest frames from UDP receiver (already compressed as JPEG)
                frames_data = self.receiver.get_latest_frames()
                removed_trainers = self.receiver.consume_removed_trainers()
                
                if not frames_data and not removed_trainers:
                    continue
                    
                # Convert compressed frames to base64 for web transmission
                frames_base64 = {}
                for (trainer_name, channel_name), compressed_bytes in frames_data.items():
                    try:
                        # Convert to base64 for web transmission
                        frame_base64 = base64.b64encode(compressed_bytes).decode('utf-8')
                        
                        # Use trainer::channel as key for organization
                        key = f"{trainer_name}::{channel_name}"
                        frames_base64[key] = frame_base64
                    except Exception as e:
                        print(f"Error encoding frame for {trainer_name}/{channel_name}: {e}")
                    
                # Broadcast to all connected clients
                message = json.dumps({'frames': frames_base64, 'removed_trainers': removed_trainers})
                
                # Remove closed websockets
                closed = set()
                for ws in self.websockets:
                    if ws.closed:
                        closed.add(ws)
                    else:
                        try:
                            await ws.send_str(message)
                        except Exception as e:
                            print(f"Error sending to websocket: {e}")
                            closed.add(ws)
                            
                self.websockets -= closed
                
            except Exception as e:
                print(f"Error in broadcast loop: {e}")
                
    async def start_async(self):
        """Start the async components."""
        # Start frame broadcaster task
        asyncio.create_task(self.broadcast_frames())
        
    def start(self):
        """Start the dashboard server."""
        # Start UDP receiver
        self.receiver.start()
        
        # Setup async startup
        self.app.on_startup.append(lambda app: self.start_async())
        
        # Start web server
        print(f"Starting web dashboard on http://0.0.0.0:{self.web_port}")
        print(f"  → Local access: http://localhost:{self.web_port}")
        print(f"  → Network access: http://<your-ip>:{self.web_port}")
        print(f"Listening for UDP frames on port {self.receiver.port}")
        print()
        print("If you can't connect from another machine:")
        print("  1. Check firewall: sudo ufw allow {}/tcp".format(self.web_port))
        print("  2. Verify IP address: ip addr show")
        print()
        web.run_app(self.app, host='0.0.0.0', port=self.web_port)
        
    def stop(self):
        """Stop the dashboard server."""
        self.receiver.stop()


class FrameSender:
    """Helper class to send frames to the dashboard via UDP."""
    
    def __init__(self, trainer_name, host='localhost', port=9999, quality=85):
        """
        Initialize frame sender.
        
        Args:
            trainer_name: Name of the trainer (e.g., config['agent']['name'])
            host: Dashboard server address
            port: UDP port
            quality: JPEG compression quality (1-100, default 85)
        """
        self.trainer_name = trainer_name
        self.host = host
        self.port = port
        self.quality = quality
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
    def send_frame(self, frame, channel='default'):
        """
        Send a compressed frame to the dashboard via UDP.
        Frame is JPEG-compressed before sending to fit in UDP packet.
        
        Args:
            frame: numpy array (BGR or grayscale image)
            channel: string identifier for this frame stream
        """
        try:
            # Convert grayscale to BGR if needed
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # Ensure frame is uint8
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            # Compress frame as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            _, compressed = cv2.imencode('.jpg', frame, encode_param)
            compressed_bytes = compressed.tobytes()
            
            # Create packet: [trainer_name_len(2)][trainer_name][channel_name_len(2)][channel_name][compressed_jpeg]
            trainer_bytes = self.trainer_name.encode('utf-8')
            trainer_len = struct.pack('!H', len(trainer_bytes))
            
            channel_bytes = channel.encode('utf-8')
            channel_len = struct.pack('!H', len(channel_bytes))
            
            # Assemble packet
            packet = trainer_len + trainer_bytes + channel_len + channel_bytes + compressed_bytes
            
            # Check if packet fits in UDP limit
            if len(packet) > 65507:
                print(f"Warning: Packet too large ({len(packet)} bytes). Consider reducing quality or frame size.")
                return
            
            # Send via UDP
            self.socket.sendto(packet, (self.host, self.port))
            
        except Exception as e:
            print(f"Error sending frame: {e}")
            
    def close(self):
        """Close the UDP socket."""
        self.socket.close()


def main():
    """Run the dashboard server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Training Dashboard Server')
    parser.add_argument('--web-port', type=int, default=8080,
                       help='Web server port (default: 8080)')
    parser.add_argument('--udp-port', type=int, default=9999,
                       help='UDP receiver port (default: 9999)')
    
    args = parser.parse_args()
    
    server = DashboardServer(
        web_port=args.web_port, 
        udp_port=args.udp_port
    )
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop()


if __name__ == '__main__':
    main()
