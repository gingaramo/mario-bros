from typing import Union, Tuple
import numpy as np
import random
import torch
import torch.nn as nn
import src.render as render
from src.environment import Observation


# Define the DQN model
class DQN(nn.Module):

  def __init__(self, action_size: int, mock_observation: Observation,
               config: dict):
    """
    Initializes the DQN model.
    """
    super(DQN, self).__init__()
    self.action_size = action_size
    has_frame_input = mock_observation.frame is not None
    has_dense_input = mock_observation.dense is not None

    print(f"Initializing DQN model with:")
    print(f" - Action size: {action_size=}")
    if has_frame_input:
      print(f" - CNN input shape: {mock_observation.frame.shape=}")
    if has_dense_input:
      print(f" - Side input shape: {mock_observation.dense.shape=}")
    print(f" - Config: {config=}")

    if has_frame_input:
      flattened_cnn_dim = self.initialize_cnn(mock_observation.frame,
                                              config['convolution'])
    else:
      self.convolutions = nn.ModuleList()

    hidden_layers_dim = [
        (flattened_cnn_dim if has_frame_input else 0) +
        (mock_observation.dense.shape[0] if has_dense_input else 0),
    ] + config['hidden_layers'] + [action_size]

    # Add hidden layers
    self.hidden_layers = nn.ModuleList()
    for in_, out_ in zip(hidden_layers_dim[:-1], hidden_layers_dim[1:]):
      self.hidden_layers.append(nn.Linear(in_, out_))

    self.activation = torch.relu

  def initialize_cnn(self, mock_frame: np.ndarray, config: dict):
    """
    Initializes the CNN layers based on the mock observation and configuration.
    """
    mock_frame = torch.Tensor(mock_frame, device=torch.device('cpu'))

    convolution_type = config.get('type', '2d')
    if convolution_type == '2d':
      make_conv = nn.Conv2d
    elif convolution_type == '3d':
      make_conv = nn.Conv3d
    else:
      raise ValueError(
          f"Unsupported convolution type: {config['type']}. Supported: '2d', '3d'."
      )
    # Technically channels_in will either be actual channels or stacked frames.
    channels_in = mock_frame.shape[0]
    self.convolutions = nn.ModuleList()
    for (channels_out, kernel_size, stride) in zip(config['channels'],
                                                   config['kernel_sizes'],
                                                   config['strides']):
      self.convolutions.append(
          make_conv(channels_in, channels_out, kernel_size, stride=stride))
      channels_in = channels_out

    def _get_flattened_shape(x: torch.Tensor) -> int:
      x = x.unsqueeze(0)
      for conv in self.convolutions:
        x = conv(x)
      return x.flatten().shape[0]

    return _get_flattened_shape(mock_frame)

  def forward(self,
              x: Tuple[torch.Tensor, torch.Tensor],
              training: bool = False):
    x, side_input = x

    if not training:
      # When not training we may render the input frames (if present)
      if x.shape[0] > 0:
        render.maybe_render_dqn(x, side_input)

      # We also add batch dimension if not training
      x = x.unsqueeze(0)
      side_input = side_input.unsqueeze(0)

    for conv in self.convolutions:
      x = torch.relu(conv(x))

    # Flatten but preserve batch dimension
    x = x.flatten(start_dim=1)
    x = torch.concat([x, side_input], dim=1)
    for hidden_layer in self.hidden_layers[:-1]:
      x = self.activation(hidden_layer(x))
    # Last layer without relu
    x = self.hidden_layers[-1](x)

    if not training:
      x = x.squeeze(0)  # Remove batch dimension if it was added
    return x
