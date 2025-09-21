from typing import Tuple, List
import numpy as np
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import os
from src.profiler import ProfileScope, ProfileLockScope
from src.dqn import DQN, DuelingDQN
from src.environment import Observation
from src.noisy_network import replace_linear_with_noisy, NoisyLinear
from src.replay_buffer import UniformExperienceReplayBuffer, PrioritizedExperienceReplayBuffer, OrderedExperienceReplayBuffer
import pickle
from torch.optim import lr_scheduler  # Keep -> see horrible exec() hack


class Agent:

  def __init__(self, env, device, summary_writer, config):
    self.config = config
    self.action_size = env.action_space.nvec[-1]
    self.num_envs = env.unwrapped.num_envs
    self.device = device
    self.summary_writer = summary_writer

    self.gamma = config['gamma']
    self.epsilon_min = config.get('epsilon_min', 0.01)
    self.epsilon_exponential_decay = config.get('epsilon_exponential_decay',
                                                None)
    self.epsilon_linear_decay = config.get('epsilon_linear_decay', None)

    # Learning parameters.
    self.learning_rate = config['learning_rate']
    self.apply_noisy_network = config.get('apply_noisy_network', False)
    assert self.apply_noisy_network or ('epsilon_exponential_decay' in config or 'epsilon_linear_decay' in config), \
        "Either 'epsilon_exponential_decay' or 'epsilon_linear_decay' must be provided in config if not using noisy networks."
    if config['loss'] == 'mse':
      self.get_loss = nn.MSELoss
    elif config['loss'] == 'smooth_l1':
      self.get_loss = nn.SmoothL1Loss
    elif config['loss'] == 'huber':
      self.get_loss = nn.HuberLoss
    else:
      raise ValueError(f"Unsupported loss function: {config['loss']}")

    # Checkpoint functionality
    _path = os.path.join("checkpoint/", config['name'])
    if not os.path.exists(_path):
      os.makedirs(_path)
    self.checkpoint_path = os.path.join(_path, 'state_dict.pkl')
    self.checkpoint_state_path = os.path.join(_path, "state.pkl")
    if (not os.path.exists(self.checkpoint_path)
        or not os.path.exists(self.checkpoint_state_path)):
      self.episodes_played = 0
      self.global_step = 0
      self.trained_experiences = 0
      self.initial_epsilon = config.get('epsilon', 1.0)
    else:
      with open(self.checkpoint_state_path, "rb") as f:
        state = pickle.load(f)
        self.episodes_played = state['episodes_played']
        self.global_step = state['global_step']
        self.trained_experiences = state['trained_experiences']
        self.initial_epsilon = state['initial_epsilon']
        print(
            f'Resuming from checkpoint. Episode {self.episodes_played}, global step {self.global_step}, trained experiences {self.trained_experiences}'
        )

    if config['optimizer'] == 'adam':
      self.optimizer = optim.Adam(self.create_models(env, config),
                                  lr=self.learning_rate)
    else:
      raise ValueError(
          f"Unsupported optimizer: {config['optimizer']}. Supported: 'adam'.")

    self.lr_scheduler = None
    if 'lr_scheduler' in config:
      exec(
          f"self.lr_scheduler = {config['lr_scheduler']['type']}(self.optimizer, **{config['lr_scheduler']['args']})"
      )

  @property
  def epsilon(self):
    if self.epsilon_exponential_decay:
      eps = self.initial_epsilon * (self.epsilon_exponential_decay**
                                    self.global_step)
    elif self.epsilon_linear_decay:
      eps = self.initial_epsilon - (self.epsilon_linear_decay *
                                    self.global_step)
    else:
      eps = 0.0
    return max(eps, self.epsilon_min)

  def save_checkpoint(self):
    self.save_models()
    with open(self.checkpoint_state_path, "wb") as f:
      pickle.dump(
          {
              'episodes_played': self.episodes_played,
              'global_step': self.global_step,
              'trained_experiences': self.trained_experiences,
              'initial_epsilon': self.initial_epsilon,
          }, f)
    self.summary_writer.flush()

  def create_models(self, env, config):
    raise NotImplementedError("Subclasses should implement this method.")

  def remember(self, observation: Tuple[List, List], action: int,
               reward: float, next_observation: Tuple[List, List], done: bool):
    raise NotImplementedError("Subclasses should implement this method.")

  def act(self, observation: Observation) -> tuple[int, np.ndarray]:
    raise NotImplementedError("Subclasses should implement this method.")

  def replay(self):
    raise NotImplementedError("Subclasses should implement this method.")

  def save_models(self):
    raise NotImplementedError("Subclasses should implement this method.")
