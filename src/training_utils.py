"""
Training utilities including checkpoints, seeds, and episode statistics.
"""

import os
import shutil
import pickle
import random
import numpy as np
import torch
from .tb_logging import CustomSummaryWriter

# Constants
CHECKPOINT_DIR = './checkpoint'
TENSORBOARD_DIR = './runs'
SUMMARY_WRITER_MAX_QUEUE = 100000
SUMMARY_WRITER_FLUSH_SECS = 300


def set_seed(seed: int):
  """
    Set random seeds for reproducibility across all random number generators.
    
    Args:
        seed (int): The random seed value to use
        
    Note:
        This ensures deterministic behavior across Python's random, NumPy, 
        and PyTorch random number generators, including CUDA operations.
    """
  print(f"Setting random seed to: {seed}")

  # Set Python's random module seed
  random.seed(seed)

  # Set NumPy's random seed
  np.random.seed(seed)

  # Set PyTorch's random seed
  torch.manual_seed(seed)

  # Set CUDA random seed (if using GPU)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_checkpoint_directories(config):
  """
    Remove existing checkpoint and tensorboard directories for clean restart.
    
    Args:
        config (dict): Configuration dictionary containing agent name
        
    Note:
        This function is used when --restart flag is specified to ensure
        training starts from scratch without any previous state.
    """
  agent_name = config['agent']['name']
  checkpoint_path = f"{CHECKPOINT_DIR}/{agent_name}"
  tensorboard_path = f"{TENSORBOARD_DIR}/tb_{agent_name}"

  try:
    if os.path.exists(checkpoint_path):
      shutil.rmtree(checkpoint_path)
      print(f"Cleared checkpoint directory: {checkpoint_path}")
    if os.path.exists(tensorboard_path):
      shutil.rmtree(tensorboard_path)
      print(f"Cleared tensorboard directory: {tensorboard_path}")
  except OSError as e:
    print(f"Warning: Failed to clear directories - {e}")


def initialize_checkpoint_directory(config, config_file_path):
  """
    Create checkpoint directory and copy configuration file for reproducibility.
    
    Args:
        config (dict): Configuration dictionary containing agent name
        config_file_path (str): Path to the original configuration file
        
    Note:
        The configuration file is copied to the checkpoint directory to ensure
        the exact configuration used for training is preserved.
    """
  agent_name = config['agent']['name']
  checkpoint_path = f"{CHECKPOINT_DIR}/{agent_name}"

  os.makedirs(checkpoint_path, exist_ok=True)

  # Copy the config file to the checkpoint directory for reproducibility
  try:
    config_dest = f"{checkpoint_path}/config.yaml"
    if config_file_path != config_dest:  # Avoid copying to itself
      shutil.copy(config_file_path, config_dest)
      print(f"Configuration saved to: {config_dest}")
  except (shutil.SameFileError, OSError) as e:
    print(f"Warning: Could not copy config file - {e}")


def get_checkpoint_purge_step(config):
  """
    Get the global step from the last checkpoint for tensorboard log continuation.
    
    Args:
        config (dict): Configuration dictionary containing agent name
        
    Returns:
        int: The global step from the last checkpoint, or 0 if no checkpoint exists
        
    Note:
        This ensures tensorboard logs continue from the correct step when
        resuming training from a checkpoint.
    """
  checkpoint_path = os.path.join(CHECKPOINT_DIR, config['name'])
  if not os.path.exists(checkpoint_path):
    return 0

  checkpoint_state_path = os.path.join(checkpoint_path, "state.pkl")
  if not os.path.exists(checkpoint_state_path):
    return 0

  try:
    with open(checkpoint_state_path, "rb") as f:
      state = pickle.load(f)
      return state.get('global_step', 0)
  except (pickle.PickleError, KeyError, OSError) as e:
    print(f"Warning: Could not load checkpoint state - {e}")
    return 0


def record_episode_statistics(summary_writer, done, truncated, info):
  """
    Record episode statistics to tensorboard when episodes complete.
    
    Args:
        summary_writer: The summary writer for logging
        done (array): Boolean array indicating completed episodes
        truncated (array): Boolean array indicating truncated episodes  
        info (dict): Environment info containing episode statistics
        
    Note:
        Statistics are only recorded for environments where episodes
        have finished (either done or truncated).
    """
  if not (np.any(done) or np.any(truncated)):
    return

  for i in range(len(done)):
    assert info['_episode'][i] == (done[i] or truncated[i]), "Episode info mismatch"
    if done[i] or truncated[i]:
      summary_writer.add_scalar("Episode/Reward",
                                info['episode']['r'][i])
      summary_writer.add_scalar("Episode/Steps", info['episode']['l'][i])

def create_summary_writer(config):
  """
    Create a CustomSummaryWriter with appropriate configuration.
    
    Args:
        config (dict): Configuration dictionary containing agent settings
        
    Returns:
        CustomSummaryWriter: Configured summary writer for tensorboard logging
    """
  return CustomSummaryWriter(
      log_dir=f"{TENSORBOARD_DIR}/tb_{config['agent']['name']}",
      max_queue=SUMMARY_WRITER_MAX_QUEUE,
      flush_secs=SUMMARY_WRITER_FLUSH_SECS,
      purge_step=get_checkpoint_purge_step(config['agent']),
      metric_prefix_sample_mod={
          'Action/': 100,
          'Replay/': 1,
          'ReplayBuffer/': 100
      })


def setup_training_environment(config, config_file_path, restart_training):
  """
    Setup the training environment including checkpoints and random seeds.
    
    Args:
        config (dict): Configuration dictionary
        config_file_path (str): Path to the configuration file
        restart_training (bool): Whether to clear existing checkpoints
        
    Note:
        Sets up random seeds for reproducibility, manages checkpoint directories,
        and configures rendering mode based on configuration.
    """
  from .render import set_headless_mode

  # Set random seed if configured for reproducibility
  if 'seed' in config:
    set_seed(config['seed'])
    config['env']['seed'] = config['seed']

  # Handle checkpoint directory setup
  if restart_training:
    clear_checkpoint_directories(config)

  # Configure rendering mode
  set_headless_mode(config['env'].get('headless', False))

  initialize_checkpoint_directory(config, config_file_path)


def is_tqdm_disabled():
  """Return True when tqdm progress bars should be disabled."""
  value = os.environ.get('TQDM_DISABLE', '').strip().lower()
  return value in {'1', 'true', 'yes', 'on'}
