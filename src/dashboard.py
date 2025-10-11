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
import socket
import struct
import threading
from aiohttp import web
import aiohttp


class FrameReceiver:
    """Receives compressed JPEG frames via UDP."""
    
    def __init__(self, host='0.0.0.0', port=9999, max_packet_size=65507):
        self.host = host
        self.port = port
        self.max_packet_size = max_packet_size
        self.socket = None
        self.latest_frames = {}  # {(trainer_name, channel_name): compressed_jpeg_bytes}
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
                    
            except Exception as e:
                if self.running:
                    print(f"Error receiving frame: {e}")
                    import traceback
                    traceback.print_exc()
                    
    def get_latest_frames(self):
        """Get the latest frames for all channels."""
        with self.lock:
            return dict(self.latest_frames)


class DashboardServer:
    """Web server that serves frames to browser clients."""
    
    def __init__(self, web_port=8080, udp_port=9999):
        self.web_port = web_port
        self.receiver = FrameReceiver(port=udp_port)
        self.app = web.Application()
        self.websockets = set()
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup HTTP and WebSocket routes."""
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/ws', self.handle_websocket)
        
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
        #frames-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
            margin-top: 20px;
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
    </style>
</head>
<body>
    <h1>🎮 Training Dashboard</h1>
    <div id="status" class="status disconnected">Connecting...</div>
    <div id="stats" class="stats">FPS: 0 | Frames: 0</div>
    <div id="frames-container"></div>
    
    <script>
        let ws = null;
        let frameCount = 0;
        let lastFrameTime = Date.now();
        let fps = 0;
        
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(protocol + '//' + window.location.host + '/ws');
            
            ws.onopen = function() {
                document.getElementById('status').textContent = 'Connected';
                document.getElementById('status').className = 'status connected';
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
                updateFrames(data.frames);
                updateStats();
            };
        }
        
        function updateFrames(frames) {
            const container = document.getElementById('frames-container');
            
            // Group frames by trainer
            const trainerGroups = {};
            for (const [key, imageData] of Object.entries(frames)) {
                const [trainer, channel] = key.split('::');
                if (!trainerGroups[trainer]) {
                    trainerGroups[trainer] = {};
                }
                trainerGroups[trainer][channel] = imageData;
            }
            
            // Update or create trainer sections
            for (const [trainer, channels] of Object.entries(trainerGroups)) {
                let trainerSection = document.getElementById('trainer-' + trainer);
                
                if (!trainerSection) {
                    // Create new trainer section
                    trainerSection = document.createElement('div');
                    trainerSection.className = 'trainer-section';
                    trainerSection.id = 'trainer-' + trainer;
                    
                    const header = document.createElement('div');
                    header.className = 'trainer-header';
                    header.textContent = '🎮 ' + trainer;
                    
                    const framesDiv = document.createElement('div');
                    framesDiv.className = 'trainer-frames';
                    framesDiv.id = 'frames-' + trainer;
                    
                    trainerSection.appendChild(header);
                    trainerSection.appendChild(framesDiv);
                    container.appendChild(trainerSection);
                }
                
                const framesDiv = document.getElementById('frames-' + trainer);
                
                // Update frames within this trainer
                for (const [channel, imageData] of Object.entries(channels)) {
                    const frameId = 'frame-' + trainer + '-' + channel;
                    let frameBox = document.getElementById(frameId);
                    
                    if (!frameBox) {
                        // Create new frame box
                        frameBox = document.createElement('div');
                        frameBox.className = 'frame-box';
                        frameBox.id = frameId;
                        
                        const title = document.createElement('div');
                        title.className = 'frame-title';
                        title.textContent = channel;
                        
                        const img = document.createElement('img');
                        img.className = 'frame-image';
                        img.id = 'img-' + trainer + '-' + channel;
                        
                        frameBox.appendChild(title);
                        frameBox.appendChild(img);
                        framesDiv.appendChild(frameBox);
                    }
                    
                    // Update image
                    const img = document.getElementById('img-' + trainer + '-' + channel);
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
        
        // Start connection
        connect();
    </script>
</body>
</html>
        """
        return web.Response(text=html, content_type='text/html')
        
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
                
                if not frames_data:
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
                message = json.dumps({'frames': frames_base64})
                
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
