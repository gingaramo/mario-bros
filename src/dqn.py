import numpy as np
import random
import torch
import cv2
import torch.nn as nn
import src.render as render


# Define the DQN model
class DQN(nn.Module):

  def __init__(self, action_size: int, mock_input: torch.Tensor, config: dict):
    super(DQN, self).__init__()
    print(f"{action_size=}, {mock_input.shape=}, {config=}")
    # We need to ensure mock_input is on the CPU for Conv2d initialization.
    mock_input = mock_input.to(torch.device('cpu'))
    self.conv1 = nn.Conv2d(mock_input.shape[0], 64, 8, stride=4)
    self.conv2 = nn.Conv2d(64, 128, 4, stride=2)
    self.conv3 = nn.Conv2d(128, 256, 4, stride=2)

    def _get_flattened_shape(x: torch.Tensor) -> int:
      x = x.unsqueeze(0)
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.conv3(x)
      return x.flatten().shape[0]

    self.side_input_floats = 1  # Currently only last_action
    hidden_layers = [
        _get_flattened_shape(mock_input) + self.side_input_floats
    ] + config['hidden_layers'] + [action_size]
    # Add linear layers
    self.linear = []
    for in_, out_ in zip(hidden_layers[:-1], hidden_layers[1:]):
      self.linear.append(nn.Linear(in_, out_))
    self.linear = nn.ModuleList(self.linear)

  def forward(self, x, side_input):
    # Add batch dimension if input is (C, H, W)
    has_batch_dim = False
    if x.ndim > 3:
      has_batch_dim = True

    if not has_batch_dim:
      # When not trainig we may render the input frames
      render.maybe_render_dqn(x, side_input)

    if not has_batch_dim:
      x = x.unsqueeze(0)
      side_input = side_input.unsqueeze(0)
    x = torch.relu(self.conv1(x))
    x = torch.relu(self.conv2(x))
    x = torch.relu(self.conv3(x))
    # Add it again after flattening.
    x = x.flatten(start_dim=1)
    x = torch.concat([x, side_input], dim=1)
    for i, layer in enumerate(self.linear[:-1]):
      x = torch.relu(layer(x))
    # Last layer without relu
    x = self.linear[-1](x)
    if not has_batch_dim:
      x = x.squeeze(0)  # Remove batch dimension if it was added
    return x
