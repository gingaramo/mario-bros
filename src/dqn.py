import numpy as np
import random
import torch
import torch.nn as nn


# Define the DQN model
class DQN(nn.Module):

  def __init__(self, action_size: int, mock_input: torch.Tensor, config: dict):
    super(DQN, self).__init__()
    print(f"{action_size=}, {mock_input.shape=}, {config=}")
    # We need to ensure mock_input is on the CPU for Conv2d initialization.
    mock_input = mock_input.to(torch.device('cpu'))
    self.conv1 = nn.Conv2d(mock_input.shape[0], 16, 8, stride=4)
    self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
    self.conv3 = nn.Conv2d(32, 64, 4, stride=2)

    def _get_flattened_shape(x: torch.Tensor) -> int:
      x = x.unsqueeze(0)
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.conv3(x)
      return x.flatten().shape[0]

    hidden_layers = config['hidden_layers']
    # FIX: Add support for hidden_layers being multi-layer.
    self.fc1 = nn.Linear(_get_flattened_shape(mock_input), hidden_layers[0])
    self.fc2 = nn.Linear(hidden_layers[0], action_size)

  def forward(self, x):
    # Add batch dimension if input is (C, H, W)
    has_batch_dim = False
    if x.ndim > 3:
      has_batch_dim = True

    if not has_batch_dim:
      x = x.unsqueeze(0)
    x = torch.relu(self.conv1(x))
    x = torch.relu(self.conv2(x))
    x = torch.relu(self.conv3(x))
    # Add it again after flattening.
    x = x.flatten(start_dim=1)
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    if not has_batch_dim:
      x = x.squeeze(0)  # Remove batch dimension if it was added
    return x
