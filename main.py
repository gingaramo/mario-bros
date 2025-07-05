import argparse
import gymnasium as gym
import gym_super_mario_bros  # Keep (environment registration)
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch
import yaml
import time
import pickle
import os
import cv2
import shutil
import numpy as np

from src.agent import Agent
from src.recording import Recording
from src.render import render, set_headless_mode

import ale_py
from tqdm import tqdm

gym.register_envs(ale_py)


def clear_checkpoints(config):
  try:
    shutil.rmtree(f"checkpoint/{config['agent']['name']}")
    shutil.rmtree(f"runs/tb_{config['agent']['name']}")
  except FileNotFoundError:
    print("Failed to delete files")
    pass


def main(args):
  print(f"Using configuration file: {args.config}")
  config = yaml.safe_load(open(args.config, 'r'))
  if args.restart:
    clear_checkpoints(config)

  set_headless_mode(config['env'].get('headless', False))
  env = gym.make(config['env']['env_name'], render_mode='rgb_array')
  if 'SuperMario' in config['env']['env_name']:
    # We only apply this for Mario. Other environments are likely just Atari
    # and thus don't need this.
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    action_labels = SIMPLE_MOVEMENT
  else:
    action_labels = config['env']['env_action_labels']

  device = torch.device(config['device'])
  print(f"Using device: {device}")

  recording = None
  agent = Agent(env.action_space.n, device, config['agent'])
  if args.record_play:
    episodes = [agent.episodes_trained]
  else:
    episodes = range(agent.episodes_trained, config['env']['num_episodes'])
  pbar = tqdm(episodes, desc="Starting")
  for episode in pbar:
    if args.record_play:
      recording = Recording(f"{config['agent']['name']}_{episode}")
    agent.episode_begin(recording=args.record_play)
    state, info = env.reset()
    state = env.render()
    total_reward = 0
    done = False
    last_score = None

    if config['env'].get('max_num_steps', float('inf')) < agent.global_step:
      print(f"Maximum number of steps {config['env']['max_num_steps']}")
      break
    for timestep in range(config['env']['max_steps_per_episode']):
      action, q_values = agent.act(state)
      next_state, reward, done, truncated, info = env.step(action)
      next_state = env.render()
      if 'SuperMario' in config['env']['env_name']:
        world, stage, score = info['world'], info['stage'], info['score']
      else:
        world, stage, score = (0, 0, 0)
      done = done or truncated

      # Amend reward with score change if configured.
      if config['env'][
          'use_score'] and last_score != None and last_score != score:
        reward += (score - last_score) / 100.0
      last_score = score

      agent.remember(action, reward, next_state, done)
      agent.replay()

      render(next_state, q_values, action, action_labels, recording=recording)

      state = next_state
      total_reward += reward
      if done:
        break

    if recording:
      recording.save()
      recording = None
    episode_info = {
        'episode': episode,
        'total_reward': total_reward,
        'world': world,
        'stage': stage,
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
