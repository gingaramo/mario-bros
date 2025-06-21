import argparse
import yaml
import gym
import gym_super_mario_bros  # Keep (environment registration)
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import torch

from agent import Agent


def main(args):
  print(f"Using configuration file: {args.config}")
  config = yaml.safe_load(open(args.config, 'r'))

  env = gym.make(config['env']['env_name'])
  # This reduces action space to a simpler set of actions.
  env = JoypadSpace(env, SIMPLE_MOVEMENT)

  device = torch.device(config['device'])
  print(f"Using device: {device}")

  print(f"{env.action_space.n=}")
  agent = Agent(env.action_space.n, device, config['agent'])

  for episode in range(config['env']['num_episodes']):
    state = env.reset()
    total_reward = 0
    done = False

    for time in range(config['env']['max_steps_per_episode']):
      action = agent.act(state)
      next_state, reward, done, info = env.step(action)
      env.render()
      agent.remember(action, reward, next_state, done)
      agent.replay(time)
      state = next_state
      total_reward += reward
      if done:
        break

    print(
        f"Episode {episode+1}/{config['env']['num_episodes']} - Total Reward: {total_reward}, Steps: {time+1}"
    )

  #agent.save(args.model_path)
  env.close()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--config',
                      type=str,
                      default='config.yaml',
                      help='Path to the configuration file')
  args = parser.parse_args()
  main(args)
