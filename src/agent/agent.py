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
import pickle
from torch.optim import lr_scheduler  # Keep -> see horrible exec() hack


def get_noisy_network_weights_norm(named_modules):
  weight_mu_norm = []
  weight_mu_sigma_norm = []
  for name, module in named_modules:
    if isinstance(module, NoisyLinear):
      weight_mu_norm.append(torch.norm(module.weight_mu).item())
      weight_mu_sigma_norm.append(torch.norm(module.weight_sigma).item())
  return np.array(weight_mu_norm).mean(), np.array(
      weight_mu_sigma_norm).mean().item()


class Agent:

  def __init__(self, env, device, summary_writer, config):
    self.config = config
    self.action_size = env.action_space.nvec[-1]
    self.num_envs = env.unwrapped.num_envs
    self.device = device
    self.summary_writer = summary_writer

    # Episodic or not parameters.
    self.gamma = config['gamma']

    # Checkpointing parameters
    self.replays_until_checkpoint = config['checkpoint_every_n_replays']

    # Learning parameters.
    self.batch_size = config['batch_size']
    self.learning_rate = config['learning_rate']
    if config['loss'] == 'mse':
      self.get_loss = nn.MSELoss
    elif config['loss'] == 'smooth_l1':
      self.get_loss = nn.SmoothL1Loss
    elif config['loss'] == 'huber':
      self.get_loss = nn.HuberLoss
    else:
      raise ValueError(f"Unsupported loss function: {config['loss']}")

    # Restore checkpoint dictionary.
    _path = os.path.join("checkpoint/", config['name'])
    os.makedirs(_path, exist_ok=True)
    self.checkpoint_state_path = os.path.join(_path, "state.pkl")
    checkpoint_dict = None
    if os.path.exists(self.checkpoint_state_path):
      with open(self.checkpoint_state_path, "rb") as f:
        checkpoint_dict = pickle.load(f)
        print(f"Restored checkpoint from {self.checkpoint_state_path}")

    # Initialize state from all subclasses (possibly from checkpoint).
    self.checkpoint_callbacks = []
    already_called = set()  # To avoid calling the same method multiple times.
    for subclass in type(self).__mro__:
      if (hasattr(subclass, 'load_or_init_state')
          and subclass.load_or_init_state not in already_called):
        self.checkpoint_callbacks.append(
            subclass.load_or_init_state(self, env, config, checkpoint_dict))
        already_called.add(subclass.load_or_init_state)

    # Optimizer and scheduler initialization.
    if config['optimizer'] == 'adam':
      self.optimizer = optim.Adam(self.parameters_to_optimize(),
                                  lr=self.learning_rate)
    else:
      raise ValueError(
          f"Unsupported optimizer: {config['optimizer']}. Supported: 'adam'.")

    self.lr_scheduler = None
    if 'lr_scheduler' in config:
      exec(
          f"self.lr_scheduler = {config['lr_scheduler']['type']}(self.optimizer, **{config['lr_scheduler']['args']})"
      )

  def save_checkpoint(self):
    print(f"Saving checkpoint....")
    checkpoint = {}

    for callback in self.checkpoint_callbacks:
      state = callback()
      assert state.keys().isdisjoint(checkpoint.keys(
      )), f"Checkpoint state keys overlap. {state.keys() & checkpoint.keys()}"
      checkpoint.update(state)

    with open(self.checkpoint_state_path, "wb") as f:
      pickle.dump(checkpoint, f)
    self.summary_writer.flush()

  def load_or_init_state(self, env, config, checkpoint_dict=None):
    """Initializes the agent's variables, including model parameters.
     
    This is initialization required for checkpointing, i.e. not in config and
    not static throughout training.

    **All subclasses are expected to implement this method.**

    Args:
      env: The environment to be used by the agent.
      config: The configuration dictionary.
      checkpoint_dict: If not None, a dictionary with the state to be restored.
    
    Returns:
      A callable that returns a dictionary of state key-value pairs, needed to
      resume execution by the agent (the the to-checkpoint lambda).
    """
    if checkpoint_dict is None:
      self.global_step = 0
      self.trained_experiences = 0
    else:
      self.global_step = checkpoint_dict['global_step']
      self.trained_experiences = checkpoint_dict['trained_experiences']

    return lambda: {
        'global_step': self.global_step,
        'trained_experiences': self.trained_experiences,
    }

  def parameters_to_optimize(self):
    """Returns the model parameters to be optimized."""
    raise NotImplementedError("Subclasses should implement this method.")

  def remember(self, observation: Tuple[List, List], action: int,
               reward: float, next_observation: Tuple[List, List], done: bool):
    raise NotImplementedError("Subclasses should implement this method.")

  def get_action(self, observation: Observation) -> tuple[int, np.ndarray]:
    raise NotImplementedError("Subclasses should implement this method.")

  def act(self, observation: Observation) -> tuple[int, np.ndarray]:
    """Returns the action to take based on the current observation.

    Returns:
      action: The action to take, either random or based on the model's prediction.
      act_values: The Q-values predicted by the model for the current observation.
    """
    self.global_step += self.num_envs
    self.summary_writer.set_global_step(self.global_step)

    with torch.no_grad(), ProfileScope("model_inference"):
      actions, values = self.get_action(observation)

    return actions.astype(int), values

  def clip_gradients(self, parameters):
    if 'clip_gradients' in self.config:
      return nn.utils.clip_grad_norm_(parameters,
                                      self.config['clip_gradients'])
    return torch.nn.utils.clip_grad_norm_(parameters, float('inf'))

  def train(self):
    trained_experiences = self.replay()
    if trained_experiences:
      self.trained_experiences += trained_experiences
      self.lr_scheduler.step() if self.lr_scheduler else None

      # TODO: Move checkpoint functionality into the outer training loop.
      self.replays_until_checkpoint -= 1
      if self.replays_until_checkpoint <= 0:
        with ProfileScope('checkpoint'):
          self.save_checkpoint()
        self.replays_until_checkpoint = self.config[
            'checkpoint_every_n_replays']

      self.summary_writer.add_scalar("Replay/TrainedExperiences",
                                     self.trained_experiences)
    return trained_experiences
