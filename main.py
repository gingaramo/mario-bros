import argparse
import os
import gymnasium as gym
import gym_super_mario_bros  # Keep (environment registration)
import ale_py
import torch

from src.async_training import run_async_training
from src.config import load_configuration
from src.evaluate import evaluate_agent
from src.profiler import execution_profiler_singleton
from src.sync_training import run_sync_training
from src.training_utils import setup_training_environment
from src.render import set_dashboard_mode

# Register environments
gym.register_envs(ale_py)


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

  # Debugging?
  if config.get('debug_mode', False):
    torch.autograd.set_detect_anomaly(True)

  # Setup training environment
  setup_training_environment(config, args.config, args.restart)

  # Setup profiling.
  global execution_profiler_singleton
  execution_profiler_singleton.set_name(f"{config['agent']['name']}")
  
  # Enable dashboard mode.
  if args.use_dashboard:
    set_dashboard_mode(enabled=True, trainer_name=config['agent']['name'], quality=args.jpeg_quality)

  # Launch training based on mode
  execution_mode = config.get('execution_mode', 'synchronous')
  if args.evaluate_episodes > 0:
    print(f"Evaluating the agent on {args.evaluate_episodes} episodes.")
    accumulated_reward, episode_steps = evaluate_agent(config,
                                                       args.evaluate_episodes)
    print(f"Accumulated rewards: {accumulated_reward}")
    print(f"Episode steps: {episode_steps}")

  if execution_mode == 'synchronous':
    print("Running in synchronous mode.")
    run_sync_training(config)
  elif execution_mode == 'asynchronous':
    print("Running in asynchronous mode.")
    run_async_training(config)
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
                      default="",
                      help='Path to the configuration YAML file')

  parser.add_argument(
      '--record_play',
      action='store_true',
      help='Record an episode and exit (not currently implemented)')

  parser.add_argument('--evaluate_episodes',
                      type=int,
                      default=0,
                      help='Evaluate the agent on a set of episodes and exit')

  parser.add_argument(
      '--restart',
      action='store_true',
      help='Restart training from scratch, clearing existing checkpoints')

  parser.add_argument(
      '--use_dashboard',
      action='store_true',
      help='Enable the training dashboard for real-time visualization')

  parser.add_argument(
      '--jpeg_quality',
      type=int,
      default=85,
      help='JPEG quality for the training dashboard (1-100), 85 default.')

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
