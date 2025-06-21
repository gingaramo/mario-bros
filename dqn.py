import numpy as np
import random
import torch
import torch.nn as nn

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

  def __init__(self, action_size: int, mock_input: torch.Tensor, config: dict):
    super(DQN, self).__init__()
    print(f"{action_size=}, {mock_input.shape=}, {config=}")
    self.conv1 = nn.Conv2d(mock_input.shape[0], 16, 8, stride=4)
    self.conv2 = nn.Conv2d(16, 32, 4, stride=2)

    def _get_flattened_shape(x: torch.Tensor) -> int:
      x = x.unsqueeze(0)
      x = self.conv1(x)
      x = self.conv2(x)
      return x.flatten().shape[0]

    hidden_layers = config['hidden_layers']
    # FIX: Add support for hidden_layers being multi-layer.
    self.fc1 = nn.Linear(_get_flattened_shape(mock_input), hidden_layers[0])
    self.fc2 = nn.Linear(hidden_layers[0], action_size)

  def forward(self, x):
    # Add batch dimension.
    x = x.unsqueeze(0)
    x = torch.relu(self.conv1(x))
    x = torch.relu(self.conv2(x))
    # Add it again after flattening.
    x = x.flatten().unsqueeze(0)
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    return x.squeeze(0)  # Remove batch dimension for output
