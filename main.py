import argparse
import gym
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
from src.render import render_mario_with_q_values


def main(args):
  print(f"Using configuration file: {args.config}")
  config = yaml.safe_load(open(args.config, 'r'))
  if args.restart:
    try:
      shutil.rmtree(f"checkpoint/{config['agent']['name']}")
      shutil.rmtree(f"runs/tb_{config['agent']['name']}")
    except FileNotFoundError:
      print("Failed to delete files")
      pass

  env = gym.make(config['env']['env_name'])
  env = JoypadSpace(env, SIMPLE_MOVEMENT)

  device = torch.device(config['device'])
  print(f"Using device: {device}")

  agent = Agent(env.action_space.n, device, config['agent'])
  agent_performance = []

  for episode in range(agent.episodes_trained, config['env']['num_episodes']):
    agent.episode_begin()
    state = env.reset()
    total_reward = 0
    done = False
    last_score = None

    for timestep in range(config['env']['max_steps_per_episode']):
      action, q_values = agent.act(state)
      next_state, reward, done, info = env.step(action)
      world, stage, score = info['world'], info['stage'], info['score']

      # Amend reward with score change if configured.
      if config['env'][
          'use_score'] and last_score != None and last_score != score:
        reward += (score - last_score) / 100.0
      last_score = score

      render_mario_with_q_values(next_state, q_values, SIMPLE_MOVEMENT)
      agent.remember(action, reward, next_state, done)
      agent.replay()
      state = next_state
      total_reward += reward
      if done:
        break

    episode_info = {
        'total_reward': total_reward,
        'world': world,
        'stage': stage,
        'steps': timestep + 1
    }
    agent.episode_end(episode_info)
    print(
        f"Episode {episode + 1}/{config['env']['num_episodes']} - Total Reward: {total_reward}, World: {(world, stage)}, Steps: {timestep + 1}"
    )

  env.close()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--config',
                      type=str,
                      default='config.yaml',
                      help='Path to the configuration file')
  parser.add_argument(
      '--restart',
      type=bool,
      default=False,
      help='Whether to restart training and delete checkpoints if they exist.')
  args = parser.parse_args()
  main(args)
