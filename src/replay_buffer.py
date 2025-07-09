from typing import Union, Tuple, List
import random
import torch
import numpy as np


class ReplayBuffer(object):
  """
  Holds a list of experiences used in replay.
  
  Expects input as tensors, and returns tensors in the configured device.
  """
  buffer: list

  def __init__(self, config, device, summary_writer):
    self.config = config
    self.device = device
    self.summary_writer = summary_writer
    self.buffer = []

  def __len__(self):
    return len(self.buffer)

  def append(self, observation: List[torch.Tensor], action: int, reward: float,
             next_observation: List[torch.Tensor], done: bool):
    self.buffer.append((observation, action, reward, next_observation, done))
    if len(self.buffer) > self.config['size']:
      to_remove = random.randint(0, len(self.buffer) - 1)
      self.buffer.pop(to_remove)

    self.summary_writer.add_scalar('Memory/Size', len(self.buffer))

  def sample(self, batch_size: int):
    # For now only uniform sampling
    if len(self) < batch_size:
      raise ValueError(
          f"Cannot sample {batch_size} from buffer of size {len(self)}")

    minibatch = random.choices(self.buffer, k=batch_size)

    all_observation, all_action, all_reward, all_next_observation, all_done = zip(
        *minibatch)

    def to_input(
        observations: Tuple[List[torch.Tensor], ...]
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
    all_observation = to_input(all_observation)
    all_action = torch.tensor(all_action,
                              dtype=torch.int64,
                              device=self.device)
    all_reward = torch.tensor(all_reward,
                              dtype=torch.float,
                              device=self.device)
    all_next_observation = to_input(all_next_observation)
    all_done = torch.tensor(all_done, dtype=torch.bool, device=self.device)

    return all_observation, all_action, all_reward, all_next_observation, all_done
    # TODO: add back
    # if self.memory_selection == 'prioritized':
    #   # Compute importance sampling weights, ensuring we scale by the maximum value
    #   # to ensure weights correct gradient updates downwards.
    #   N_ = float(len(self.memory))
    #   td_errors = np.array(self.memory_error, dtype=np.float32)
    #   wis_weights = (td_errors.sum() /
    #                  (td_errors * N_))**self.memory_selection_beta
    #   wis_weights_max = wis_weights.max()
    #   wis_weights = torch.tensor(wis_weights[minibatch_ids] / wis_weights_max,
    #                              device=self.device)
    # Gather memories from self.memory using indices in minibatch_ids

    # Prioritize learning from bad more than good experiences.
    # TODO: add back
    # if self.memory_selection == 'uniform':
    #   self.memory_error.append(1.0)
    # elif self.memory_selection == 'prioritized':
    #   # TODO(gingaramo): Revisit this formula below.
    #   q_estimate = reward + self.gamma * torch.argmax(
    #       self.target_model(
    #           torch.tensor(np.concatenate(
    #               (curr_state[1:], [next_state]), axis=0, dtype=np.float32),
    #                        device=self.device),
    #           torch.tensor([action], device=self.device))).item() * (not done)
    #   self.memory_error.append(
    #       td_error(self.last_q_values[action],
    #                q_estimate)**self.memory_selection_alpha)
    # else:
    #   raise ValueError(f"Invalid memory_selection {self.memory_selection}")
    # self.summary_writer.add_scalar('Memory/Estimation Error',
    #                                self.memory_error[-1], self.global_step)
