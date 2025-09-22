from collections import deque
import os
from typing import Union, Tuple, Callable
from src.agent.agent import Agent
from src.environment import Observation, merge_observations
import torch
import torch.nn as nn
import numpy as np


def _create_mlp(hidden_layers_dim: list):
  hidden_layers = nn.ModuleList()
  for in_, out_ in zip(hidden_layers_dim[:-1], hidden_layers_dim[1:]):
    hidden_layers.append(nn.Linear(in_, out_))
  return hidden_layers


class Policy:

  def __init__(self):
    pass

  def __call__(self, observation: Observation):
    raise NotImplementedError


class MLPPolicy(nn.Module):

  def __init__(self, action_size: int, mock_observation: Observation,
               config: dict):
    super(MLPPolicy, self).__init__()
    self.config = config
    mlp_layers_dim = [mock_observation.dense.shape[1]
                      ] + config['hidden_layers'] + [action_size]
    self.policy = _create_mlp(mlp_layers_dim)
    self.activation = torch.nn.LeakyReLU()

  def forward(self,
              x: Tuple[torch.Tensor, torch.Tensor],
              training: bool = False) -> torch.Tensor:
    image, dense = x
    assert image.numel() == 0, "MLPPolicy does not support image input (yet)"
    for layer in self.policy[:-1]:
      dense = self.activation(layer(dense))
    dense = self.policy[-1](dense)
    return dense


class REINFORCEAgent(Agent):

  def __init__(self, env, device, summary_writer, config):
    super().__init__(env, device, summary_writer, config)
    num_envs = env.unwrapped.num_envs
    self.episode_observations = {i: [] for i in range(num_envs)}
    self.episode_actions = {i: [] for i in range(num_envs)}
    self.episode_rewards = {i: [] for i in range(num_envs)}
    self.ready_actions = deque(maxlen=10000)
    self.ready_observations = deque(maxlen=10000)
    self.ready_rewards = deque(maxlen=10000)
    self.ready_episode = deque(maxlen=10000)
    self.episode_count = 0
    self.entropy_coeff = config.get('entropy_coeff', 0.0)

  def create_models(self, env, config):
    mock_observation, _ = env.reset()
    self.policy = MLPPolicy(env.action_space.nvec[-1], mock_observation,
                            config['network']['mlp_policy'])
    if os.path.exists(self.checkpoint_path):
      self.policy.load_state_dict(
          torch.load(self.checkpoint_path, map_location=self.device))
    self.policy.to(self.device)
    return self.policy.parameters()

  def get_action(self,
                 observation: Observation) -> Tuple[np.ndarray, np.ndarray]:
    x = self.policy(observation.as_input(self.device))
    probs = torch.nn.functional.softmax(x, dim=-1)
    m = torch.distributions.Categorical(probs=probs)
    actions = m.sample().cpu().numpy()

    for i, observation in enumerate(observation.as_list()):
      self.episode_observations[i].append(observation)
      self.episode_actions[i].append(actions[i])

    return actions, probs.detach().cpu().numpy()

  def remember(self, observation, action, reward, next_observation, done,
               episode_start):
    for i in range(len(reward)):
      reward_i, done_i = reward[i], done[i]
      self.episode_rewards[i].append(reward_i)
      if done_i:
        self.episode_count += 1
        # ready actions is a deque
        self.ready_actions.extend(self.episode_actions[i])
        self.ready_observations.extend(self.episode_observations[i])
        self.ready_rewards.extend(self.episode_rewards[i])
        self.ready_episode.extend([self.episode_count] *
                                  len(self.episode_rewards[i]))
        self.episode_actions[i] = []
        self.episode_observations[i] = []
        self.episode_rewards[i] = []

  def replay(self):
    if len(self.ready_rewards) < self.batch_size:
      return 0
    episode = self.ready_episode[0]
    actions = []
    observations = []
    rewards = []
    while episode == self.ready_episode[0]:
      self.ready_episode.popleft()
      actions.append(self.ready_actions.popleft())
      observations.append(self.ready_observations.popleft())
      rewards.append(self.ready_rewards.popleft())
    self.episode_learn(actions, observations, rewards)
    return len(actions)

  def episode_learn(self, actions, observations, rewards):
    R = 0

    returns = []
    # Discount future rewards back to the present using gamma
    for r in rewards[::-1]:
      R = r + self.gamma * R
      returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

    x = merge_observations(observations).as_input(self.device)
    # Calculate log_probs
    x = self.policy(x)
    probs = torch.nn.functional.softmax(x, dim=-1)
    m = torch.distributions.Categorical(probs=probs)
    log_probs = m.log_prob(torch.tensor(actions).to(self.device))

    # Normalize the returns (commented out for debugging)
    # returns = (returns - returns.mean())
    self.optimizer.zero_grad()
    entropy = m.entropy().mean()
    loss = -(log_probs *
             returns).sum() - 0.001 * entropy  # Very small entropy bonus
    loss.backward()
    pre_clip_grad_norm = self.clip_gradients(self.policy.parameters())
    self.optimizer.step()

    self.summary_writer.add_scalar('Replay/Loss', loss)
    self.summary_writer.add_scalar('Replay/LearningRate',
                                   self.optimizer.param_groups[0]['lr'])
    self.summary_writer.add_scalar(
        "Replay/ParamNorm",
        torch.nn.utils.get_total_norm(self.policy.parameters()))

    # Log action distribution for debugging
    self.summary_writer.add_scalar("Replay/ActionProb_0", probs[0, 0].item())
    self.summary_writer.add_scalar("Replay/ActionProb_1", probs[0, 1].item())
    self.summary_writer.add_scalar("Replay/ActionProb_2", probs[0, 2].item())
    self.summary_writer.add_scalar("Replay/ActionProb_3", probs[0, 3].item())
    self.summary_writer.add_scalar("Replay/MaxActionProb", probs.max().item())
    self.summary_writer.add_scalar("Replay/MinActionProb", probs.min().item())
    self.summary_writer.add_scalar("Replay/Entropy", entropy.item())
    self.summary_writer.add_scalar("Replay/PreClipGradNorm",
                                   pre_clip_grad_norm)
    #self.summary_writer.add_scalar('Replay/Entropy', entropy)
    if 'clip_gradients' in self.config:
      self.summary_writer.add_scalar(
          "Replay/GradScaleWithLr",
          min(pre_clip_grad_norm, self.config['clip_gradients']) *
          self.optimizer.param_groups[0]['lr'])

  def save_models(self):
    torch.save(self.policy.state_dict(), self.checkpoint_path)
