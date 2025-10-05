from typing import Union, Tuple, List
from collections import deque
import random
import torch
import numpy as np
import math

# Handle imports for both module-style and direct execution
try:
  import src.sum_tree as sum_tree
  from src.environment import Observation, merge_observations
except ImportError:
  import sum_tree
  from environment import Observation, merge_observations


class ReplayBuffer(object):
  """
  Holds a list of experiences used in replay.
  
  Expects input as tensors, and returns tensors in the configured device.
  """
  buffer: deque

  def __init__(self, config: dict, device: torch.device, summary_writer):
    self.config = config
    self.device = device
    self.summary_writer = summary_writer
    self.buffer = deque(maxlen=config['size'])

  def __len__(self):
    return len(self.buffer)

  def append(self, observation: Tuple[torch.Tensor, torch.Tensor], action: int,
             reward: float, next_observation: Tuple[torch.Tensor,
                                                    torch.Tensor], done: bool):
    self.buffer.append((observation, action, reward, next_observation, done))
    self.summary_writer.add_scalar('ReplayBuffer/Size', len(self.buffer))

  def from_list_to_tensors(
      self,
      all_observation: List[Observation],
      all_action: List[int],
      all_reward: List[float],
      all_next_observation: List[Observation],
      all_done: List[bool],
  ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor,
             Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Converts lists of observations, actions, rewards, next observations, and done flags
    into a tuple of tensors, in the correct device and ready to be used."""
    # Convert all of them into tensors
    all_observation = merge_observations(all_observation).as_input(self.device)
    all_action = torch.tensor(all_action,
                              dtype=torch.int64,
                              device=self.device)
    all_reward = torch.tensor(all_reward,
                              dtype=torch.float,
                              device=self.device)
    all_next_observation = merge_observations(all_next_observation).as_input(
        self.device)
    all_done = torch.tensor(all_done, dtype=torch.bool, device=self.device)

    return all_observation, all_action, all_reward, all_next_observation, all_done

  def sample(
      self, batch_size: int
  ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor,
             torch.Tensor, torch.Tensor]:
    """Samples a batch of experiences from the buffer and returns them as tensors."""
    raise NotImplementedError(
        "This method should be implemented by subclasses of ReplayBuffer.")


class OrderedExperienceReplayBuffer(ReplayBuffer):
  """A simple replay buffer that samples experiences in the order they were added."""

  def sample(self, batch_size):
    if len(self) < batch_size:
      raise ValueError(
          f"Cannot sample {batch_size} from buffer of size {len(self)}")

    minibatch = list(self.buffer)[:batch_size]

    all_observation, all_action, all_reward, all_next_observation, all_done = zip(
        *minibatch)

    all_observation, all_action, all_reward, all_next_observation, all_done = self.from_list_to_tensors(
        all_observation, all_action, all_reward, all_next_observation,
        all_done)

    return (all_observation, all_action, all_reward, all_next_observation,
            all_done)


class UniformExperienceReplayBuffer(ReplayBuffer):
  """A simple replay buffer that samples uniformly from the buffer."""

  def sample(self, batch_size):

    minibatch = random.choices(self.buffer, k=batch_size)

    all_observation, all_action, all_reward, all_next_observation, all_done = zip(
        *minibatch)

    all_observation, all_action, all_reward, all_next_observation, all_done = self.from_list_to_tensors(
        all_observation, all_action, all_reward, all_next_observation,
        all_done)

    return (all_observation, all_action, all_reward, all_next_observation,
            all_done)


class PrioritizedExperienceReplayBuffer(ReplayBuffer):
  """A replay buffer that samples based on the "surprise" of the experiences.
  
  Paper: https://arxiv.org/pdf/1511.05952

  Configured with parameters:
    - alpha: Prioritization exponent. The higher the value, the more
      importance is given to the surprise of the experiences. If zero, all
      experiences are treated equally.
    - beta: Importance sampling exponent. The higher the value, the more
      the sampling distribution is adjusted to account for the importance
      of each experience, effectively reducing the bias introduced by
      prioritization. If zero, there's no adjustment.
    - beta_annealing_steps: Number of steps over which beta is annealed
      from its initial value to 1.0.
  """

  surprise: deque

  def __init__(
      self,
      config: dict,
      device: torch.device,
      summary_writer,
      target_callback: callable,
      prediction_callback: callable,
  ):
    super().__init__(config, device, summary_writer)
    self.alpha = config['alpha']  # Prioritization exponent
    self.beta = config['beta']  # Importance sampling exponent
    self.surprise = sum_tree.SumTree(config['size'])
    self.compute_target = target_callback
    self.compute_prediction = prediction_callback
    self.eps = 1e-3  # Non-zero sampling probability.
    self.annealing_steps_taken = 0.0
    self.annealing_steps = float(config['beta_annealing_steps'])

  def get_beta(self) -> float:
    lerp = min(1.0, self.annealing_steps_taken / self.annealing_steps)
    return self.beta * (1 - lerp) + lerp * 1.0

  def compute_surprise(self, deltas):
    """Computes the surprise (TD-error) for a list of deltas."""
    return (np.abs(deltas) + self.eps)**self.alpha

  def sample(self, batch_size):
    # Since we're sampling we count this as an annealing step.
    self.annealing_steps_taken += 1
    self.summary_writer.add_scalar('ReplayBuffer/Alpha', self.alpha)
    self.summary_writer.add_scalar('ReplayBuffer/Beta', self.get_beta())

    try:
      samples = np.random.uniform(0, self.surprise.get_sum(), size=batch_size)
    except OverflowError as e:
      raise ValueError(
          f"Cannot sample, {self.surprise.get_sum()} out of bounds. {e}")
    indices = [self.surprise.find_index(sample) for sample in samples]
    min_prob = self.surprise.get_min() / self.surprise.get_sum()

    importance_sampling = np.array([
        (min_prob / (self.surprise.get_value(i) / self.surprise.get_sum())
         )**self.get_beta() for i in indices
    ])

    minibatch = [self.buffer[i] for i in indices]
    all_observation, all_action, all_reward, all_next_observation, all_done = zip(
        *minibatch)

    all_observation, all_action, all_reward, all_next_observation, all_done = self.from_list_to_tensors(
        all_observation, all_action, all_reward, all_next_observation,
        all_done)

    return (all_observation, all_action, all_reward, all_next_observation,
            all_done), importance_sampling, indices

  def update_surprise(self, indices: List[int], deltas: np.array):
    """Updates the surprise values for the given indices with the new deltas."""
    surprises = self.compute_surprise(deltas)
    for idx, surprise in zip(indices, surprises):
      self.surprise.update(idx, surprise)

  def append(self, observation: Tuple[torch.Tensor, torch.Tensor], action: int,
             reward: float, next_observation: Tuple[torch.Tensor,
                                                    torch.Tensor], done: bool):
    super().append(observation, action, reward, next_observation, done)

    # Initialize with a large value, so each new experience is visited at least once
    self.surprise.add(self.surprise.get_max() if self.surprise.get_sum() >
                      0 else 1.0)
