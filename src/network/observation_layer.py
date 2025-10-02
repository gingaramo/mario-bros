from src.environment import Observation
import torch.nn as nn
import torch
import numpy as np


class CNNObservationLayer(nn.Module):

  def __init__(self, mock_observation: Observation, cnn_config: dict):
    """
    Initializes the base of a DQN model.

    Args:
      action_size (int): Number of actions the agent can take.
      mock_observation (Observation): Mock observation to initialize the model.
      config (dict): Configuration dictionary convolution parameters.
        - `convolution`: Configuration for CNN layers.
    """
    assert mock_observation.frame is not None, "Mock Observation's frame cannot be None"

    super(CNNObservationLayer, self).__init__()
    self.convolutions = nn.ModuleList()
    self.flattened_cnn_dim = self.initialize_cnn(mock_observation.frame,
                                                 cnn_config)
    self.activation = torch.nn.LeakyReLU()

  @property
  def output_dim(self):
    return self.flattened_cnn_dim

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
    # Note that dim-0 is batch dimension given vectorized environments.
    channels_in = mock_frame.shape[1] if len(mock_frame.shape) == 4 else 1
    for (channels_out, kernel_size, stride) in zip(config['channels'],
                                                   config['kernel_sizes'],
                                                   config['strides']):
      self.convolutions.append(
          make_conv(channels_in, channels_out, kernel_size, stride=stride))
      channels_in = channels_out

    def _get_flattened_shape(x: torch.Tensor) -> int:
      # We keep the first environment observation only.
      x = x[0]
      for conv in self.convolutions:
        x = conv(x)
      return x.flatten().shape[0]

    return _get_flattened_shape(mock_frame)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    "Forward pass for CNN layers only. Expects batch dimension"
    for conv in self.convolutions:
      x = self.activation(conv(x))

    # Flatten but preserve batch dimension
    return x.flatten(start_dim=1)
