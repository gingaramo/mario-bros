import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from src.dqn import DQN
from src.preprocess import StatePreprocess


# Define the agent class
class Agent:

  def __init__(self, action_size, device, config):
    self.action_size = action_size
    self.device = device
    self.config = config
    self.memory = []
    self.memory_size = config['memory_size']
    self.gamma = config['gamma']
    self.epsilon = config['epsilon']
    self.epsilon_min = config['epsilon_min']
    self.epsilon_decay = config['epsilon_decay']
    self.learning_rate = config['learning_rate']
    self.batch_size = config['batch_size']
    self.last_action = None
    self.last_action_repeated = 0
    self.replay_every_n_steps = config['replay_every_n_steps']
    self.preprocess = StatePreprocess(config['preprocess'])
    if 'dqn' in config['network']:
      self.model = DQN(action_size, self.preprocess(),
                       config['network']['dqn'])
    else:
      raise ValueError("Network configuration must include a valid model.")

    if config['optimizer'] == 'adam':
      self.optimizer = optim.Adam(self.model.parameters(),
                                  lr=self.learning_rate)
    else:
      raise ValueError(
          f"Unsupported optimizer: {config['optimizer']}. Supported: 'adam'.")

  def remember(self, action, reward, next_state, done):
    """Stores experience. State gathered from last state sent to act().
    """
    self.memory.append((self.preprocess(), action, reward,
                        self.preprocess.preprocess(next_state), done))
    if len(self.memory) > self.memory_size:
      self.memory.pop(random.randint(0, len(self.memory) - 1))

  def act(self, state: np.ndarray) -> int:
    """Returns the action to take based on the current state."""
    self.preprocess.add(state)
    if self.config['action_repeat_steps'] > 1:
      if self.last_action_repeated == self.config['action_repeat_steps']:
        self.last_action_repeated = 0
        # And continue with a new action
      elif self.last_action is not None:
        self.last_action_repeated += 1
        # Repeat the last action
        return self.last_action

    action = None
    if np.random.rand() <= self.epsilon:
      action = random.randrange(self.action_size)
    else:
      act_values = self.model(self.preprocess())
      action = torch.argmax(act_values).item()
    self.last_action = action
    return action

  def replay(self, timestep):
    if len(self.memory) < self.batch_size:
      return
    if timestep % self.replay_every_n_steps != 0:
      return

    print(
        f"Replaying at timestep {timestep} with memory size {len(self.memory)}"
    )
    minibatch = random.sample(self.memory, self.batch_size)
    for state, action, reward, next_state, done in minibatch:
      target = reward
      if not done:
        # Stack the next state to the previous states
        next_state = torch.concat(
            [state[1:, :], torch.Tensor(next_state).unsqueeze(0)], axis=0)
        target = (reward + self.gamma * torch.max(self.model(next_state)))
      target_f = self.model(state)
      target_t = target_f.clone()
      target_t[action] = target
      self.optimizer.zero_grad()
      loss = nn.MSELoss()(target_f, target_t)
      loss.backward()
      self.optimizer.step()
    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay
