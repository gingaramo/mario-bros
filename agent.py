import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim


# Define the agent class
class Agent:

  def __init__(self, action_size, model, device):
    self.action_size = action_size
    self.memory = []
    self.gamma = 0.95  # discount rate
    self.epsilon = 1.0  # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.learning_rate = 0.001
    self.model = model
    self.device = device
    self.model.to(self.device)
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

  def remember(self, state, action, reward, next_state, done):
    self.memory.append(
        (torch.tensor(state.copy()).float().to(self.device), action, reward,
         torch.tensor(next_state.copy()).float().to(self.device), done))

  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return random.randrange(self.action_size)
    act_values = self.model(torch.tensor(state.copy()).float().to(self.device))
    return torch.argmax(act_values[0]).item()

  def replay(self, batch_size):
    minibatch = random.sample(self.memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
      target = reward
      if not done:
        # Stack the next state to the previous states
        next_state = torch.concat(
            [state[1:, :], next_state.unsqueeze(0)], axis=0)
        target = (reward + self.gamma * torch.max(self.model(next_state)))
      target_f = self.model(state)
      target_t = target_f.clone()
      target_t[0][action] = target
      self.optimizer.zero_grad()
      loss = nn.MSELoss()(target_f, target_t)
      loss.backward()
      self.optimizer.step()
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay
