import argparse
import gymnasium as gym
import gym_super_mario_bros  # Keep (environment registration)
from pynput import keyboard
import threading
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
from src.render import render, set_headless_mode, set_rendering_enabled, is_headless_mode

gym.register_envs(ale_py)

FRAMES_FORWARD = -1


def on_press(key):
  global FRAMES_FORWARD
  global RENDERING_ENABLED
  try:
    if key.char == 'c':
      FRAMES_FORWARD = -1
    if key.char == 'n':
      FRAMES_FORWARD = 1
    if key.char == 's':
      set_rendering_enabled(True)
    if key.char == 'h':
      set_rendering_enabled(False)
  except AttributeError:
    pass


def start_keyboard_listener():
  listener = keyboard.Listener(on_press=on_press)
  listener.start()
  return listener


keyboard_thread = threading.Thread(target=start_keyboard_listener)
keyboard_thread.daemon = True
keyboard_thread.start()


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
  num_envs = env.unwrapped.num_envs
  action_labels = config['env']['env_action_labels']

  device = torch.device(config['device'])
  print(f"Using device: {device}")

  recording = None
  agent = Agent(env, device, config['agent'])
  # Print model summary and parameter count
  print(
      f"Model summary: {agent.model}, Parameters: {sum(p.numel() for p in agent.model.parameters())}"
  )

  if args.record_play:
    episodes = [agent.episodes_trained]
  else:
    episodes = range(agent.episodes_trained, config['env']['num_episodes'])
  episode_timesteps = [0] * num_envs
  truncated, done = [False] * num_envs, [False] * num_envs
  observation, _ = env.reset()
  total_reward = [0] * num_envs

  if args.record_play:
    recording = Recording(f"./checkpoint/{config['agent']['name']}",
                          f"episode_{agent.episodes_trained}")

  pbar = tqdm(episodes, desc="Starting")

  while True:
    if agent.episodes_trained >= config['env']['num_episodes']:
      # Stop training if we reached the maximum number of episodes
      print("Maximum number of episodes reached.")
      break
    if agent.global_step >= config['env'].get('num_steps', float('inf')):
      # Stop training if we reached the maximum number of steps
      print("Maximum number of steps reached.")
      break

    # End episodes if we reached the maximum steps per episode, or
    # if done or truncated.
    for i, episode_timestep in enumerate(episode_timesteps):
      if episode_timestep >= config['env']['max_steps_per_episode'] or (
          done[i] or truncated[i]):
        if recording:
          recording.save()
          recording = None
        episode_info = {
            'total_reward': total_reward[i],
            'steps': episode_timestep + 1
        }
        # Note we don't need to reset since environments autoreset.
        agent.episode_end(episode_info)
        pbar.set_description(
            f"Episode: {agent.episodes_trained+1}, Total Reward: {total_reward[i]} Steps: {episode_timestep + 1}"
        )
        pbar.refresh()

        # Reset the environment and prepare for the next episode
        episode_timesteps[i] = 0
        total_reward[i] = 0

        # If we are recording, we stop after the first episode.
        if args.record_play:
          return

    # This frame-by-frame step should be rewritten...
    if not is_headless_mode():
      global FRAMES_FORWARD
      while FRAMES_FORWARD >= 0:
        if FRAMES_FORWARD > 0:
          FRAMES_FORWARD -= 1
          break
        pass

    # Actual RL stuff.
    action, q_values = agent.act(observation)
    next_observation, reward, done, truncated, info = env.step(action)
    for i in range(num_envs):
      episode_timesteps[i] += 1

    if agent.curiosity_module:
      with torch.no_grad():
        frame, dense = observation.as_input(device)
        next_frame, next_dense = next_observation.as_input(device)
        curiosity_reward = agent.curiosity_module(
            (frame.unsqueeze(0).clone().detach().to(device),
             dense.clone().detach().to(device)),
            torch.tensor(action, dtype=torch.long).to(device),
            (next_frame.unsqueeze(0).clone().detach().to(device),
             next_dense.clone().detach().to(device)),
            training=False)
        reward += curiosity_reward

    agent.remember(observation, action, reward, next_observation, done)
    agent.replay()

    render(env,
           q_values,
           action,
           action_labels,
           recording=recording,
           upscale_factor=config.get('render_upscale_factor', 1),
           layout=config.get('render_layout', None))

    observation = next_observation
    total_reward += reward
  agent.summary_writer.flush()
  agent.save_checkpoint()

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
