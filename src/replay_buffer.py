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
    self.summary_writer.add_scalar('ReplayBuffer/Size', len(self.buffer))

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

      # Check if ALL observations have frame data (must be consistent)
      frame_shapes = [obs[0].shape for obs in observations]
      has_frames = [len(shape) > 0 for shape in frame_shapes]

      # Check for mixed frame types (some have frames, some don't)
      if not (all(has_frames) or not any(has_frames)):
        raise ValueError(
            "Mixed observation types not supported: some observations have frames, others don't"
        )

      if not any(has_frames):
        all_frames = no_tensor
      else:
        # Ensure all frames have the same shape
        if not all(shape == frame_shapes[0] for shape in frame_shapes):
          raise ValueError("All frame observations must have the same shape")

        # Handle both tensor and numpy array inputs efficiently
        frames_list = [observation[0] for observation in observations]
        if torch.is_tensor(frames_list[0]):
          # If already tensors, stack them and move to device
          all_frames = torch.stack(frames_list).to(device=self.device,
                                                   dtype=torch.float)
        else:
          # If numpy arrays, use np.stack then convert to tensor
          all_frames = torch.tensor(np.stack(frames_list),
                                    dtype=torch.float,
                                    device=self.device)

      # Check if ALL observations have dense data (must be consistent)
      dense_shapes = [obs[1].shape for obs in observations]
      has_dense = [len(shape) > 0 for shape in dense_shapes]

      # Check for mixed dense types (some have dense, some don't)
      if not (all(has_dense) or not any(has_dense)):
        raise ValueError(
            "Mixed observation types not supported: some observations have dense vectors, others don't"
        )

      if not any(has_dense):
        all_dense = no_tensor
      else:
        # Ensure all dense vectors have the same shape
        if not all(shape == dense_shapes[0] for shape in dense_shapes):
          raise ValueError("All dense observations must have the same shape")

        # Handle both tensor and numpy array inputs efficiently
        dense_list = [observation[1] for observation in observations]
        if torch.is_tensor(dense_list[0]):
          # If already tensors, stack them and move to device
          all_dense = torch.stack(dense_list).to(device=self.device,
                                                 dtype=torch.float)
        else:
          # If numpy arrays, use np.stack then convert to tensor
          all_dense = torch.tensor(np.stack(dense_list),
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
    self.surprise = deque(maxlen=config['size'])
    self.compute_target = target_callback
    self.compute_prediction = prediction_callback
    self.eps = 1e-3  # Non-zero sampling probability.
    self.annealing_steps_taken = 0.0
    self.annealing_steps = float(config['beta_annealing_steps'])

  def get_beta(self) -> float:
    lerp = min(1.0, self.annealing_steps_taken / self.annealing_steps)
    return self.beta * (1 - lerp) + lerp * 1.0

  def compute_surprise(self, all_observation, all_action, all_reward,
                       all_next_observation, all_done):
    """Computes the surprise (TD-error) for a batch of experiences."""
    with torch.no_grad():
      return (torch.abs(
          self.compute_target(all_reward, all_next_observation, all_done) -
          self.compute_prediction(all_observation, all_action)).cpu().numpy() +
              self.eps)**self.alpha

  def sample(self, batch_size):
    if len(self) < batch_size:
      raise ValueError(
          f"Cannot sample {batch_size} from buffer of size {len(self)}")
    # Since we're sampling we count this as an annealing step.
    self.annealing_steps_taken += 1
    self.summary_writer.add_scalar('ReplayBuffer/Alpha', self.alpha)
    self.summary_writer.add_scalar('ReplayBuffer/Beta', self.get_beta())

    # Ensure minimum surprise values to prevent zero probabilities
    surprise_array = np.array(self.surprise)
    surprise_sum = np.sum(surprise_array)

    # Handle edge case where all surprises are zero
    if surprise_sum == 0:
      # Fall back to uniform sampling if all surprises are zero
      probs = np.ones(len(self.surprise)) / len(self.surprise)
    else:
      probs = surprise_array / surprise_sum

    indices = np.random.choice(len(self.surprise), size=batch_size, p=probs)

    minibatch = [self.buffer[i] for i in indices]
    importance_sampling = np.array([
        (len(self.surprise) * probs[i])**-self.get_beta() for i in indices
    ])
    # Reduce the impact of large weights
    importance_sampling = np.clip(importance_sampling, 0, 1e3)

    # importance_sampling = np.log(importance_sampling + 1 + self.eps) -- Worked, but was a bit weird heuristic
    # From paper: for stability reasons, we always normalize weights by 1/ maxi wi so
    # that they only scale the update downwards.
    min_prob = np.min([i for i in probs])
    #if min_prob > 0:  # Avoid division by zero
    # TODO: Add back
    importance_sampling = importance_sampling / (
        (len(self.surprise) * min_prob)**-self.get_beta())
    # pass
    #else:
    #  # If min_prob is 0, use uniform importance sampling
    #  importance_sampling = np.ones_like(importance_sampling)

    all_observation, all_action, all_reward, all_next_observation, all_done = zip(
        *minibatch)

    all_observation, all_action, all_reward, all_next_observation, all_done = self.from_list_to_tensors(
        all_observation, all_action, all_reward, all_next_observation,
        all_done)

    surprises = self.compute_surprise(all_observation, all_action, all_reward,
                                      all_next_observation, all_done)

    # Update the surprise values for the sampled indices.
    for i, idx in enumerate(indices):
      self.surprise[idx] = surprises[i]

    return (all_observation, all_action, all_reward, all_next_observation,
            all_done), importance_sampling

  def append(self, observation: Tuple[torch.Tensor, torch.Tensor], action: int,
             reward: float, next_observation: Tuple[torch.Tensor,
                                                    torch.Tensor], done: bool):
    super().append(observation, action, reward, next_observation, done)

    observation, action, reward, next_observation, done = self.from_list_to_tensors(
        [observation], [action], [reward], [next_observation], [done])

    with torch.no_grad():
      surprise_value = self.compute_surprise(observation, action, reward,
                                             next_observation, done).item()
      self.surprise.append(surprise_value)

    # Ensure synchronization: both deques should always have the same length
    assert len(self.buffer) == len(self.surprise), \
        f"Buffer and surprise deques out of sync: {len(self.buffer)} vs {len(self.surprise)}"
