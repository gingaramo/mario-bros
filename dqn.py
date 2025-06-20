import gym_super_mario_bros
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

import agent
import preprocess

device = torch.device("mps")

# TODO
# image preprocess
# stacking frames

# Papers
# - Playing Atari with Deep Reinforcement Learning (https://arxiv.org/pdf/1312.5602)
# - Human-level control through deep reinforcement learning (https://www.nature.com/articles/nature14236)
# - Double DQN (https://arxiv.org/pdf/1509.06461)


# Define the DQN model
class DQN(nn.Module):

  def __init__(self, state_size, action_size):
    super(DQN, self).__init__()
    # Convolution layer
    # Input: (batch_size, height, width, frames)
    self.conv1 = nn.Conv2d(4, 16, 8, stride=4)
    self.conv2 = nn.Conv2d(16, 32, 4, stride=2)

    def _get_flattened_shape(state_size):
      x = torch.ones(state_size)
      x = x.unsqueeze(0)
      x = self.conv1(x)
      x = self.conv2(x)
      return x.flatten().shape[0]

    self.fc1 = nn.Linear(_get_flattened_shape(state_size), 256)
    self.fc2 = nn.Linear(256, action_size)

  def forward(self, x):
    # If we don't have a batch dimension -- add it
    if len(list(x.shape)) == 2:
      x = x.unsqueeze(0)
    x = torch.relu(self.conv1(x))
    x = torch.relu(self.conv2(x))
    x = x.flatten().unsqueeze(0)
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x


# Define the environment and agent
env_name = 'SuperMarioBrosRandomStages-v0'
env = gym_super_mario_bros.make(env_name)
state_size = (240, 256, 3)
target_state_size = (84, 110)
pixels_to_crop = 26  # Number of top pixels to crop
cropped_state_size = (target_state_size[0],
                      target_state_size[1] - pixels_to_crop)
action_size = env.action_space.n
pre = preprocess.preprocess_lambda(target_state_size, pixels_to_crop)
steps_per_action = 4  # How many steps to take before taking an action
agent = agent.Agent(
    action_size, DQN((steps_per_action, ) + cropped_state_size, action_size),
    device)
# Train the agent
num_episodes = 1000
batch_size = 32
action = random.randint(0, action_size - 1)  # Random initial action
reward = 0
done = False

print("Starting training on environment:", env_name)
print(f"Actions every {steps_per_action} steps")
for e in range(num_episodes):
  state = pre(env.reset())
  acc_state = []  # Accumulate frames for stacking
  for time in range(500):
    env.render()
    if (time % steps_per_action) == 0 and len(acc_state) > 0:
      # Combine accumulated frames into a single state
      acc_state = np.stack(acc_state, axis=0)
      agent.remember(acc_state, action, reward, next_state, done)
      action = agent.act(acc_state)
      acc_state = []
    acc_state.append(state)  # Stack frames
    next_state, reward, done, _ = env.step(action)
    next_state = pre(next_state)
    reward = reward if not done else -10
    state = next_state
    if done:
      print("episode: {}/{}, score: {}, e: {:.2}".format(
          e, num_episodes, time, agent.epsilon))
      break
    if len(agent.memory) > batch_size:
      agent.replay(batch_size)
