import argparse
import gymnasium as gym
import gym_super_mario_bros  # Keep (environment registration)
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import ale_py
import torch
import random
from tqdm import tqdm
import yaml
import cv2
import shutil
import os
import numpy as np

from src.agent import Agent
from src.environment import create_environment
from src.recording import Recording
from src.render import render, set_headless_mode

gym.register_envs(ale_py)


def set_seed(seed: int):
  """Set random seeds for reproducibility across all random number generators."""
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


def clear_checkpoints_dir(config):
  try:
    shutil.rmtree(f"./checkpoint/{config['agent']['name']}")
    shutil.rmtree(f"./runs/tb_{config['agent']['name']}")
  except FileNotFoundError:
    print("Failed to delete files")
    pass


def init_checkpoints_dir(config):
  os.makedirs(f"./checkpoint/{config['agent']['name']}", exist_ok=True)
  # We override the config file in the checkpoint directory.
  shutil.copy(args.config,
              f"./checkpoint/{config['agent']['name']}/config.yaml")


def main(args):
  args.config = os.path.abspath(args.config)
  print(f"Using configuration file: {args.config}")
  config = yaml.safe_load(open(args.config, 'r'))

  # Set random seed if configured
  if 'seed' in config:
    set_seed(config['seed'])

  if args.restart:
    clear_checkpoints_dir(config)
  init_checkpoints_dir(config)

  set_headless_mode(config['env'].get('headless', False))

  # Pass seed to environment config if configured
  if 'seed' in config:
    config['env']['seed'] = config['seed']

  env = create_environment(config['env'])
  action_labels = config['env']['env_action_labels']

  device = torch.device(config['device'])
  print(f"Using device: {device}")

  recording = None
  agent = Agent(env, device, config['agent'])

  if args.record_play:
    episodes = [agent.episodes_trained]
  else:
    episodes = range(agent.episodes_trained, config['env']['num_episodes'])
  pbar = tqdm(episodes, desc="Starting")
  for episode in pbar:
    if args.record_play:
      recording = Recording(f"./checkpoint/{config['agent']['name']}",
                            f"episode_{episode}")
    agent.episode_begin(recording=args.record_play)
    observation, info = env.reset()
    total_reward = 0

    if config['env'].get('max_num_steps', float('inf')) < agent.global_step:
      print(f"Maximum number of steps {config['env']['max_num_steps']}")
      break
    for timestep in range(config['env']['max_steps_per_episode']):
      action, q_values = agent.act(observation)
      next_observation, reward, done, truncated, info = env.step(action)
      # Currently ignoring truncation.

      agent.remember(observation, action, reward, next_observation, done)
      agent.replay()

      render(env, q_values, action, action_labels, recording=recording)

      observation = next_observation
      total_reward += reward
      if done or truncated:
        break
    agent.summary_writer.flush()

    if recording:
      recording.save()
      recording = None
    episode_info = {
        'episode': episode,
        'total_reward': total_reward,
        'steps': timestep + 1
    }
    agent.episode_end(episode_info)
    pbar.set_description(
        f"Episode: {episode+1}, Total Reward: {total_reward} Steps: {timestep + 1}"
    )

  env.close()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--config',
                      type=str,
                      default='config.yaml',
                      help='Path to the configuration file')
  parser.add_argument('--record_play',
                      action='store_true',
                      help='If true, we\'ll record an episode and end.')
  parser.add_argument(
      '--restart',
      action='store_true',
      help='Whether to restart training and delete checkpoints if they exist.')
  args = parser.parse_args()
  main(args)
