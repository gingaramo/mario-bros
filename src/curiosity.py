from typing import List, Tuple
from src.dqn import DQN
import torch.nn as nn
import torch


def _create_mlp(hidden_layers_dim: list):
  hidden_layers = nn.ModuleList()
  for in_, out_ in zip(hidden_layers_dim[:-1], hidden_layers_dim[1:]):
    hidden_layers.append(nn.Linear(in_, out_))
  return hidden_layers


class CuriosityModule(nn.Module):
  """Curiosity module for intrinsic motivation in reinforcement learning."""

  def __init__(self, config: dict, dqn: DQN):
    """
    Initialize the CuriosityModule.

    Args:
      config (dict): Configuration dictionary.
      action_size (int): Size of the action space.
      mock_observation (Observation): Mock observation for input dimensions.
    """
    super().__init__()
    self.config = config
    self.curiosity_reward_weight = float(config['curiosity_reward_weight'])
    self.curiosity_reward_exponent = float(config['curiosity_reward_exponent'])
    self.dqn = dqn

    # Calculate input dimension for prediction network (CNN + action, dense is optional)
    prediction_input_dim = dqn.flattened_cnn_dim + 1  # CNN features + action
    prediction_output_dim = dqn.flattened_cnn_dim  # Only CNN features for next state

    self.prediction_network = _create_mlp([prediction_input_dim] +
                                          config['hidden_layers_dim'] +
                                          [prediction_output_dim])

  def forward(self,
              all_observations: Tuple[torch.Tensor, torch.Tensor],
              all_actions: torch.Tensor,
              all_next_observations: Tuple[torch.Tensor, torch.Tensor],
              training: bool = True) -> torch.Tensor:
    # Get CNN features from current and next observations
    with torch.no_grad():
      cnn_features = self.dqn.forward_cnn(all_observations[0])
      next_cnn_features = self.dqn.forward_cnn(all_next_observations[0])

    # Prepare action tensor for concatenation
    if all_actions.dim() == 0:  # Scalar action
      all_actions = all_actions.unsqueeze(0).unsqueeze(0).float()
    elif all_actions.dim() == 1:  # Vector action
      all_actions = all_actions.unsqueeze(1).float()
    else:  # Already in the correct shape
      assert all_actions.dim(
      ) == 2, f"Unexpected action shape: {all_actions.shape}"

    # Combine CNN features with actions for prediction input
    features = torch.concat([cnn_features, all_actions], dim=1)

    # Pass through prediction network
    predicted_next_features = features
    for layer in self.prediction_network[:-1]:
      predicted_next_features = layer(predicted_next_features)
      predicted_next_features = torch.nn.LeakyReLU()(predicted_next_features)
    # Don't apply ReLU on the last layer
    predicted_next_features = self.prediction_network[-1](
        predicted_next_features)

    # Calculate curiosity reward based on prediction error
    prediction_error = (next_cnn_features -
                        predicted_next_features).pow(2).sum(dim=1)
    curiosity_reward = prediction_error.pow(
        self.curiosity_reward_exponent) * self.curiosity_reward_weight

    return curiosity_reward
