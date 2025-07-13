from typing import Union, Tuple, List
from collections import deque
import random
import torch
import numpy as np
import math


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
    self.summary_writer.add_scalar('Memory/Size', len(self.buffer))

  def from_list_to_tensors(
      self,
      all_observation: List[Tuple[torch.Tensor, torch.Tensor]],
      all_action: List[int],
      all_reward: List[float],
      all_next_observation: List[Tuple[torch.Tensor, torch.Tensor]],
      all_done: List[bool],
  ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor,
             Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Converts lists of observations, actions, rewards, next observations, and done flags
    into a tuple of tensors, in the correct device and ready to be used."""

    def _to_input(
        observations: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
      no_tensor = torch.tensor(())
      if len(observations[0][0].shape) == 0:
        all_frames = no_tensor
      else:
        all_frames = torch.tensor(np.stack(
            [observation[0] for observation in observations]),
                                  dtype=torch.float,
                                  device=self.device)
      if len(observations[0][1].shape) == 0:
        all_dense = no_tensor
      else:
        all_dense = torch.tensor(np.stack(
            [observation[1] for observation in observations]),
                                 dtype=torch.float,
                                 device=self.device)
      return (all_frames, all_dense)

    # Convert all of them into tensors
    all_observation = _to_input(all_observation)
    all_action = torch.tensor(all_action,
                              dtype=torch.int64,
                              device=self.device)
    all_reward = torch.tensor(all_reward,
                              dtype=torch.float,
                              device=self.device)
    all_next_observation = _to_input(all_next_observation)
    all_done = torch.tensor(all_done, dtype=torch.bool, device=self.device)

    return all_observation, all_action, all_reward, all_next_observation, all_done

  def sample(
      self, batch_size: int
  ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor,
             torch.Tensor, torch.Tensor]:
    """Samples a batch of experiences from the buffer and returns them as tensors."""
    raise NotImplementedError(
        "This method should be implemented by subclasses of ReplayBuffer.")


class UniformExperienceReplayBuffer(ReplayBuffer):
  """A simple replay buffer that samples uniformly from the buffer."""

  def sample(self, batch_size):
    if len(self) < batch_size:
      raise ValueError(
          f"Cannot sample {batch_size} from buffer of size {len(self)}")

    minibatch = random.choices(self.buffer, k=batch_size)

    all_observation, all_action, all_reward, all_next_observation, all_done = zip(
        *minibatch)

    all_observation, all_action, all_reward, all_next_observation, all_done = self.from_list_to_tensors(
        all_observation, all_action, all_reward, all_next_observation,
        all_done)

    return (all_observation, all_action, all_reward, all_next_observation,
            all_done)


class PrioritizedExperienceReplayBuffer(ReplayBuffer):
  """A replay buffer that samples based on the "surprise" of the experiences."""

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
    self.surprise = deque(maxlen=config['size'])
    self.compute_target = target_callback
    self.compute_prediction = prediction_callback
    self.eps = 1e-5  # Non-zero sampling probability.

  def sample(self, batch_size):
    if len(self) < batch_size:
      raise ValueError(
          f"Cannot sample {batch_size} from buffer of size {len(self)}")

    probs = np.array(self.surprise) / sum(self.surprise)
    indices = np.random.choice(len(self.surprise), size=batch_size, p=probs)

    minibatch = [self.buffer[i] for i in indices]
    importance_sampling = [(1.0 / (len(self.surprise) * probs[i]))**self.beta
                           for i in indices]

    all_observation, all_action, all_reward, all_next_observation, all_done = zip(
        *minibatch)

    all_observation, all_action, all_reward, all_next_observation, all_done = self.from_list_to_tensors(
        all_observation, all_action, all_reward, all_next_observation,
        all_done)

    with torch.no_grad():
      surprises = torch.abs(
          self.compute_target(all_reward, all_next_observation, all_done) -
          self.compute_prediction(all_observation, all_action)).cpu().numpy()

    # Update the surprise values for the sampled indices.
    for i, idx in enumerate(indices):
      self.surprise[idx] = (surprises[i] + self.eps)**self.alpha

    return (all_observation, all_action, all_reward, all_next_observation,
            all_done), importance_sampling

  def append(self, observation: Tuple[torch.Tensor, torch.Tensor], action: int,
             reward: float, next_observation: Tuple[torch.Tensor,
                                                    torch.Tensor], done: bool):
    super().append(observation, action, reward, next_observation, done)

    observation, action, reward, next_observation, done = self.from_list_to_tensors(
        [observation], [action], [reward], [next_observation], [done])

    with torch.no_grad():
      self.surprise.append(
          torch.abs(
              self.compute_target(reward, next_observation, done) -
              self.compute_prediction(observation, action)).cpu().numpy().item(
              ) + self.eps)
