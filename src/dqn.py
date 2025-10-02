from typing import Union, Tuple, Callable
import numpy as np
import random
import torch
import torch.nn as nn
import src.render as render
from src.environment import Observation
from src.network.observation_layer import CNNObservationLayer


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
              training: bool = False):
    "The actual forward call for the DQN model."
    x, side_input = x
    # Render the side input if there's a frame.
    if not training:
      if x.numel() > 0:
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
