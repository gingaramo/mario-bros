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
from torch.utils.tensorboard import SummaryWriter  # <-- Add this import
import cv2
import numpy as np

from src.agent import Agent
from src.render import render_mario_with_q_values


def main(args):
  print(f"Using configuration file: {args.config}")
  config = yaml.safe_load(open(args.config, 'r'))

  env = gym.make(config['env']['env_name'])
  env = JoypadSpace(env, SIMPLE_MOVEMENT)

  device = torch.device(config['device'])
  print(f"Using device: {device}")

  print(f"{env.action_space.n=}")
  agent = Agent(env.action_space.n, device, config['agent'])
  agent_performance = []

  config_name = os.path.splitext(os.path.basename(args.config))[0]
  log_dir = f"runs/tensorboard_logs_{config_name}"
  writer = SummaryWriter(log_dir=log_dir)

  for episode in range(config['env']['num_episodes']):
    state = env.reset()
    world = env.unwrapped._world
    stage = env.unwrapped._stage
    score = env.unwrapped._score
    total_reward = 0
    done = False

    for timestep in range(config['env']['max_steps_per_episode']):
      action, q_values = agent.act(state)
      next_state, reward, done, info = env.step(action)

      # Perhaps I should use info['score'] instead of env.unwrapped._score?
      if config['env']['use_score'] and score != env.unwrapped._score:
        reward = (env.unwrapped._score - score) / 100.0
        score = env.unwrapped._score

      render_mario_with_q_values(next_state, q_values, SIMPLE_MOVEMENT)
      agent.remember(action, reward, next_state, done)
      agent.replay(timestep)
      state = next_state
      total_reward += reward
      if done:
        break

    print(
        f"Episode {episode + 1}/{config['env']['num_episodes']} - Total Reward: {total_reward}, World: {(world, stage)}, Steps: {timestep + 1}"
    )
    performance = {
        'total_reward': total_reward,
        'world': world,
        'stage': stage,
        'steps': timestep + 1
    }
    agent_performance.append(performance)

    writer.add_scalar('Reward/Total', total_reward, episode)

  writer.close()
  env.close()
  cv2.destroyAllWindows()

  # Pickle performance to a file
  timestamp = time.time()
  date = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
  if not os.path.exists("runs"):
    os.makedirs("runs")
  performance_file = f"runs/performance_{date}.data"
  config_file = f"runs/config_{date}.yaml"

  # Pickle agent_performance
  with open(performance_file, 'wb') as pf:
    pickle.dump(agent_performance, pf)

  # Save config yaml
  with open(config_file, 'w') as cf:
    yaml.dump(config, cf)

  print(f"Saved performance to {performance_file}")
  print(f"Saved config to {config_file}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--config',
                      type=str,
                      default='config.yaml',
                      help='Path to the configuration file')
  args = parser.parse_args()
  main(args)
