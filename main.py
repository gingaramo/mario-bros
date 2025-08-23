import argparse
import os
import gymnasium as gym
import gym_super_mario_bros  # Keep (environment registration)
import ale_py

from src.config import load_configuration, validate_config
from src.training_utils import setup_training_environment, initialize_checkpoint_directory
from src.async_training import run_async_training, run_parallel_training
from src.sync_training import run_sync_training

# Register environments
gym.register_envs(ale_py)

# Constants
DEFAULT_CONFIG_FILE = 'config.yaml'


def main(args):
  """
    Main entry point for the reinforcement learning training program.
    
    Args:
        args: Command line arguments containing:
            - config: Path to configuration file
            - restart: Whether to restart training from scratch
            - record_play: Whether to record gameplay (not implemented)
            
    Note:
        Coordinates the entire training process including configuration loading,
        environment setup, and launching either synchronous or asynchronous training.
    """
  # Load and validate configuration
  args.config = os.path.abspath(args.config)
  config = load_configuration(args.config)
  validate_config(config)

  # Setup training environment
  setup_training_environment(config, args.restart)
  initialize_checkpoint_directory(config, args.config)

  # Launch training based on mode
  execution_mode = config.get('execution_mode', 'synchronous')
  if execution_mode == 'synchronous':
    print("Running in synchronous mode.")
    run_sync_training(config)
  elif execution_mode == 'asynchronous':
    print("Running in asynchronous mode.")
    run_async_training(config)
  elif execution_mode == 'parallel':
    print("Running in parallel mode.")
    run_parallel_training(config)
  else:
    raise ValueError(f"Unknown execution mode: {execution_mode}")


def create_argument_parser():
  """
    Create and configure the command line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
  parser = argparse.ArgumentParser(
      description="Reinforcement Learning Training for Mario Bros",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--config',
                      type=str,
                      default=DEFAULT_CONFIG_FILE,
                      help='Path to the configuration YAML file')

  parser.add_argument(
      '--record_play',
      action='store_true',
      help='Record an episode and exit (not currently implemented)')

  parser.add_argument(
      '--restart',
      action='store_true',
      help='Restart training from scratch, clearing existing checkpoints')

  return parser


if __name__ == "__main__":
  parser = create_argument_parser()
  args = parser.parse_args()

  try:
    main(args)
  except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
  except Exception as e:
    print(f"Error during training: {e}")
    raise
