from src.environment import Observation
import torch.nn as nn
import torch
import numpy as np


class CNNObservationLayer(nn.Module):

  def __init__(self, mock_observation: Observation, cnn_config: dict):
    """
    Initializes the base CNN layer for processing observations.

    Args:
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
    channels_in = mock_frame.shape[-3]
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
    return x.flatten(start_dim=-3)


class CNNTokenObservationLayer(nn.Module):
  """
  Image Patch Observation Layer implemented with CNN for turning images into sequence of patches (tokens).

  Given a batch of [B, C, H, W] images, where each C is a grayscale frame from previous steps this layer 
  generates (B, H/h x W/w) output of tokens from image patches of size (h, w) using convolutional layers.

  Note that using CNN for patch extraction is more efficient than manually extracting patches and then embedding them.
  """

  def __init__(self, mock_observation: Observation, config: dict):
    """
    Args:
      mock_observation (Observation): Mock observation to initialize the model.
      config (dict): Configuration dictionary for image patching parameters.
        - `patches_dim`: Configuration for patches (height, width).
        - `token_dimension`: Dimension of the embedding for each patch.
    """
    assert mock_observation.frame is not None, "Mock Observation's frame cannot be None"


    super(CNNTokenObservationLayer, self).__init__()
    self.initialize_cnn(mock_observation.frame, config)
    self.config = config
    self.activation = torch.nn.LeakyReLU()

  @property
  def output_dim(self):
    return self.flattened_cnn_dim

  @property
  def patch_height(self):
    return self.config['patches_dim'][0]
  
  @property
  def patch_width(self):
    return self.config['patches_dim'][1]

  def initialize_cnn(self, mock_frame: np.ndarray, config: dict):
    """
    Initializes the CNN layers based on the mock observation and configuration.
    """
    mock_frame = torch.Tensor(mock_frame, device=torch.device('cpu'))
    print(f"{mock_frame.shape=} for CNN token observation layer initialization")
    assert mock_frame.shape[-2] % config['patches_dim'][0] == 0, \
        "Frame height must be divisible by patches_dim[0]"
    assert mock_frame.shape[-1] % config['patches_dim'][1] == 0, \
        "Frame width must be divisible by patches_dim[1]"

    self.convolution = nn.Conv2d(
        in_channels=mock_frame.shape[-3],
        out_channels=config['token_dimension'],
        kernel_size=config['patches_dim'],
        stride=config['patches_dim']
    )

    # Embedding for output "token" (this self-attended token is then given to MLP
    # for Q-value prediction).
    self.value_embedding = nn.Parameter(torch.randn(config['token_dimension']))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    "Forward pass for CNN layers only. Expects batch dimension"
    #print(f"{x.shape=} input to CNN patch extraction")
    patches = self.activation(self.convolution(x))
    #print(f"{patches.shape=} after CNN patch extraction (groups = {num_frames=})")
    patches = patches.flatten(start_dim=2, end_dim=3)
    #print(f"{patches.shape=} after flatten")
    patches = patches.transpose(1, 2)  # [B, num_patches, token_dimension]
    #print(f"{patches.shape=} after transpose")

    # Flatten but preserve batch dimension, and add value embedding
    return torch.concat([patches, self.value_embedding.unsqueeze(0).unsqueeze(0).expand(patches.shape[0], -1, -1)], dim=1)
