from typing import Union, Tuple, Callable
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.render as render
from src.environment import Observation
from src.network.observation_layer import CNNObservationLayer, CNNTokenObservationLayer
from src.network.swiglu import SwiGLU
from src.noisy_network import NoisyLinear


def _create_mlp(hidden_layers_dim: list):
  hidden_layers = nn.ModuleList()
  for in_, out_ in zip(hidden_layers_dim[:-1], hidden_layers_dim[1:]):
    hidden_layers.append(nn.Linear(in_, out_))
  return hidden_layers


class BaseDQN(nn.Module):

  def __init__(self, action_size: int, mock_observation: Observation,
               config: dict):
    """
    Initializes the base of a DQN model.

    Args:
      action_size (int): Number of actions the agent can take.
      mock_observation (Observation): Mock observation to initialize the model.
      config (dict): Configuration dictionary containing model parameters.
        - `convolution`: Configuration for CNN layers.
    """
    super(BaseDQN, self).__init__()
    self.action_size = action_size
    self.convolution = None if mock_observation.frame is None else CNNObservationLayer(
        mock_observation, config['convolution'])

    cnn_output_dim = 0 if mock_observation.frame is None else self.convolution.output_dim
    # Note mock_observation.dense.shape[1] is because we run on vectorized environments,
    # so [0] is num_envs.
    dense_input_dim = 0 if mock_observation.dense is None else mock_observation.dense.shape[
        1]
    self.cnn_plus_dense_input_dim = cnn_output_dim + dense_input_dim
    self.activation = torch.nn.LeakyReLU()

  def forward_dense(self, x: torch.Tensor, training: bool) -> torch.Tensor:
    raise NotImplementedError(
        "Implement this method in subclasses. Either regular DQN or DuelingDQN."
    )

  def forward_dqn(self,
                  x: torch.Tensor,
                  side_input: torch.Tensor,
                  training: bool = False) -> torch.Tensor:
    "Forward pass for DQN model. Expects batch dimension"
    if self.convolution:
      x = self.convolution(x)
    # Side input might be an empty tensor.
    x = torch.concat([x, side_input], dim=1)
    x = self.forward_dense(x, training=training)
    return x

  def forward(self,
              x: Tuple[torch.Tensor, torch.Tensor],
              training: bool = False,
              render_input_frames: bool = False) -> torch.Tensor:
    "The actual forward call for the DQN model."
    x, side_input = x
    # Render the side input if there's a frame.
    if render_input_frames and x.numel() > 0:
      render.maybe_render_dqn(
          x[0], side_input[0] if side_input.numel() > 0 else torch.empty(()))

    return self.forward_dqn(x, side_input, training=training)


class DQN(BaseDQN):

  def __init__(self, action_size: int, mock_observation: Observation,
               config: dict):
    """
    Initializes the standard DQN model.
    """
    super(DQN, self).__init__(action_size, mock_observation, config)
    hidden_layers_dim = [self.cnn_plus_dense_input_dim]
    hidden_layers_dim += config['hidden_layers']
    hidden_layers_dim.append(action_size)

    # Add hidden layers
    self.hidden_layers = _create_mlp(hidden_layers_dim)

    self.activation = torch.nn.LeakyReLU()

    # Initialize weights properly
    self._initialize_weights()

  def _initialize_weights(self):
    """Initialize network weights to prevent initial NaN issues"""
    for module in self.modules():
      if isinstance(module, nn.Linear):
        # Xavier/Glorot initialization for linear layers
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
          nn.init.constant_(module.bias, 0)
      elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
        # He initialization for conv layers (better for ReLU activations)
        nn.init.kaiming_uniform_(module.weight, nonlinearity='leaky_relu')
        if module.bias is not None:
          nn.init.constant_(module.bias, 0)

  def forward_dense(self, x: torch.Tensor, training: bool) -> torch.Tensor:
    "Forward pass for dense layers only. Expects batch dimension"
    for hidden_layer in self.hidden_layers[:-1]:
      x = self.activation(hidden_layer(x))
    # Last layer without relu, outputs the Q-values
    return self.hidden_layers[-1](x)


class DuelingDQN(BaseDQN):
  """Dueling DQN model from https://arxiv.org/pdf/1511.06581"""

  def __init__(self, action_size: int, mock_observation: Observation,
               config: dict):
    super(DuelingDQN, self).__init__(action_size, mock_observation, config)

    # Create the MLP for the value and advantage streams
    value_hidden_layers_dim = [self.cnn_plus_dense_input_dim]
    value_hidden_layers_dim += config['value_hidden_layers']
    value_hidden_layers_dim.append(1)  # Output for value stream

    advantage_hidden_layers_dim = [self.cnn_plus_dense_input_dim]
    advantage_hidden_layers_dim += config['advantage_hidden_layers']
    advantage_hidden_layers_dim.append(
        action_size)  # Output for advantage stream

    self.value_hidden_layers = _create_mlp(value_hidden_layers_dim)
    self.advantage_hidden_layers = _create_mlp(advantage_hidden_layers_dim)

    self.activation = torch.nn.LeakyReLU()

    # Initialize weights properly
    self._initialize_weights()

  def _initialize_weights(self):
    """Initialize network weights to prevent initial NaN issues"""
    for module in self.modules():
      if isinstance(module, nn.Linear):
        # Xavier/Glorot initialization for linear layers
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
          nn.init.constant_(module.bias, 0)
      elif isinstance(module, (nn.Conv2d, nn.Conv3d)):
        # He initialization for conv layers (better for ReLU activations)
        nn.init.kaiming_uniform_(module.weight, nonlinearity='leaky_relu')
        if module.bias is not None:
          nn.init.constant_(module.bias, 0)

  def forward_dense(self,
                    x: torch.Tensor,
                    training: bool = False) -> torch.Tensor:
    "Forward pass for dense layers only. Expects batch dimension"
    value_x = x
    advantage_x = x

    for layer in self.advantage_hidden_layers[:-1]:
      advantage_x = self.activation(layer(advantage_x))
    advantage_x = self.advantage_hidden_layers[-1](advantage_x)

    for layer in self.value_hidden_layers[:-1]:
      value_x = self.activation(layer(value_x))
    value_x = self.value_hidden_layers[-1](value_x)
    return value_x + advantage_x - advantage_x.mean(dim=1, keepdim=True)


class TransformerDuelingDQN(nn.Module):
  """Transformer dueling DQN"""

  def __init__(self, action_size: int, mock_observation: Observation,
               config: dict):
    assert mock_observation.frame is not None, "Mock Observation's frame cannot be None"
    super(TransformerDuelingDQN, self).__init__()
    self.image_to_token_layer = CNNTokenObservationLayer(mock_observation, config['convolution'])
    self.action_size = action_size
    self.activation = torch.nn.LeakyReLU()

    # We need K, Q, V projections per H, and FFN for each layer
    # That means [L, 3, H, D, D/H] + [D, D]
    self.num_heads = config['num_heads']
    self.num_layers = config['num_layers']
    self.token_dimension = config['token_dimension']
    self.inverse_sqrt_dk = (self.token_dimension / self.num_heads)**-0.5
    assert self.token_dimension % self.num_heads == 0
    self.kqv_proj = nn.parameter.Parameter(torch.rand((self.num_layers, 3, self.num_heads, self.token_dimension, self.token_dimension//self.num_heads)))
    self.kqv_proj_bias = nn.parameter.Parameter(
      torch.zeros((
        self.num_layers, 3, self.num_heads,
        1, # batch -- gets broadcasted
        1, # seq_len -- gets broadcasted
        self.token_dimension//self.num_heads)))
    # Xavier initialization
    for layer in range(self.num_layers):
      for head in range(self.num_heads):
        for k_q_or_v in range(3):
          nn.init.xavier_uniform_(self.kqv_proj[layer, k_q_or_v, head])
    nn.init.constant_(self.kqv_proj_bias, 0)

    widening_factor = 2
    self.ffn_proj = nn.ModuleList([
      nn.Sequential(
          nn.Linear(self.token_dimension, self.token_dimension * widening_factor),
          nn.GELU(),
          nn.Linear(self.token_dimension * widening_factor, self.token_dimension)
      ) for _ in range(self.num_layers)
    ])
    self.layer_norm_post_mha = nn.ModuleList([nn.LayerNorm(self.token_dimension) for i in range(self.num_layers)])
    self.layer_norm_post_ffn = nn.ModuleList([nn.LayerNorm(self.token_dimension) for i in range(self.num_layers)])
    self.output_proj_advantage = nn.Sequential(
      nn.Linear(self.token_dimension, self.token_dimension * widening_factor),
      nn.GELU(),
      nn.Linear(self.token_dimension * widening_factor, action_size)
    )
    self.output_proj_value = nn.Sequential(
      nn.Linear(self.token_dimension, self.token_dimension * widening_factor),
      nn.GELU(),
      nn.Linear(self.token_dimension * widening_factor, 1),
    )
    self.dense_proj = nn.Linear(mock_observation.dense.shape[-1], self.token_dimension)

    # Assuming input shape is [B, L, H, W], where L is number of frames history
    self.seq_len = \
      (mock_observation.frame.shape[-2] // self.image_to_token_layer.patch_height) * \
      (mock_observation.frame.shape[-1] // self.image_to_token_layer.patch_width) + 2 # +1 for the value embedding +1 for dense embedding

    self.positional_embeddings = nn.Parameter(torch.zeros(self.seq_len, self.token_dimension))
    nn.init.xavier_uniform_(self.positional_embeddings)

    # Initialize weights properly
    self._initialize_weights()

  def _initialize_weights(self):
    """Initialize network weights to prevent initial NaN issues"""
    for module in self.modules():
      if isinstance(module, nn.Linear):
        # Xavier/Glorot initialization for linear layers
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
          nn.init.constant_(module.bias, 0)

  def forward(self,
              x: Tuple[torch.Tensor, torch.Tensor],
              training: bool = False,
              render_input_frames: bool = False) -> torch.Tensor:
    "The actual forward call for the DQN model."
    x, side_input = x

    # Render the side input if there's a frame.
    if render_input_frames and x.numel() > 0:
      render.maybe_render_dqn(
          x[0], side_input[0] if side_input.numel() > 0 else torch.empty(0))

    x = self.image_to_token_layer(x)
    x = torch.concat([x, self.dense_proj(side_input).unsqueeze(1)], dim=1)
    # Add positional embeddings
    x = x + self.positional_embeddings.unsqueeze(0)
    batch_size = x.shape[0]

    # self-attention layers
    for layer in range(self.num_layers):
      # Create K, Q, V projections under dimension 'c' for each head. This performs projections for every 'd' row with 'do'. 
      projections = torch.einsum('bld,chdo->chblo', x, self.kqv_proj[layer])
      # Add bias
      projections = projections + self.kqv_proj_bias[layer].expand(-1, -1, batch_size, self.seq_len, -1)

      # Extract K, Q, V
      k, q, v = projections[0], projections[1], projections[2]

      # Perform scaled dot-product attention between Q and K
      attention = torch.einsum('hblo,hbLo->hblL', q, k) * self.inverse_sqrt_dk
      # Upcast attention to float32 for numerical stability during softmax
      attention = F.softmax(attention.to(torch.float32), dim=-1)
      # Apply attention weights to V, swapping order of dimensions to make head concatennation easier
      attention = torch.einsum('hblL,hbLo->blho', attention, v)
      _b,_l,_h,_o = attention.shape
      attention = attention.reshape((_b,_l,_h * _o))
      x = self.layer_norm_post_mha[layer](attention + x)
      x_ffn_proj = self.ffn_proj[layer](x)
      x = self.layer_norm_post_ffn[layer](x_ffn_proj + x)

    assert self.seq_len == x.shape[-2], f"Expected sequence length {self.seq_len}, got {x.shape[-2]}"
    x = x[:, self.seq_len - 1, :]

    value_x = self.output_proj_value(x)
    advantage_x = self.output_proj_advantage(x)
    return_x = value_x + advantage_x - advantage_x.mean(dim=-1, keepdim=True)

    return return_x
