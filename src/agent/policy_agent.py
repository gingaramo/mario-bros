from collections import deque
import os
from typing import Union, Tuple, Callable
from src.agent.agent import Agent
from src.environment import Observation, merge_observations
import torch
import torch.nn as nn
import numpy as np

from src.profiler import ProfileScope


def _create_mlp(hidden_layers_dim: list):
  hidden_layers = nn.ModuleList()
  for in_, out_ in zip(hidden_layers_dim[:-1], hidden_layers_dim[1:]):
    hidden_layers.append(nn.Linear(in_, out_))
  return hidden_layers


class Policy(nn.Module):

  def __init__(self, action_size: int, mock_observation: Observation,
               config: dict):
    super(Policy, self).__init__()
    self.config = config

    if 'torso' in config:
      torso_layers_dim = [mock_observation.dense.shape[1]
                          ] + config['torso']['hidden_layers']
      self.torso = _create_mlp(torso_layers_dim)
      torso_output_size = torso_layers_dim[-1]
    else:
      self.torso = None
      torso_output_size = mock_observation.dense.shape[1]

    if 'critic' in config:
      mlp_layers_dim = [torso_output_size
                        ] + config['critic']['hidden_layers'] + [action_size]
      self.critic = _create_mlp(mlp_layers_dim)
    else:
      self.critic = None

    if 'actor' in config:
      mlp_layers_dim = [torso_output_size
                        ] + config['actor']['hidden_layers'] + [action_size]
      self.actor = _create_mlp(mlp_layers_dim)

    self.activation = torch.nn.LeakyReLU()

  def forward(self,
              x: Tuple[torch.Tensor, torch.Tensor],
              training: bool = False) -> torch.Tensor:
    image, dense = x
    assert image.numel() == 0, "MLPPolicy does not support image input (yet)"

    torso_dense = dense
    if self.torso:
      for layer in self.torso:
        torso_dense = self.activation(layer(torso_dense))
    critic_dense = torso_dense
    actor_dense = torso_dense

    if self.critic:
      for layer in self.critic[:-1]:
        critic_dense = self.activation(layer(critic_dense))
      critic_output = self.critic[-1](critic_dense)
    else:
      critic_output = None

    for layer in self.actor[:-1]:
      actor_dense = self.activation(layer(actor_dense))
    actor_output = self.actor[-1](actor_dense)

    return actor_output, critic_output


class PolicyAgent(Agent):

  def __init__(self, env, device, summary_writer, config):
    super().__init__(env, device, summary_writer, config)
    num_envs = env.unwrapped.num_envs
    # Running collection of episode data
    self.episode_observations = {i: [] for i in range(num_envs)}
    self.episode_actions = {i: [] for i in range(num_envs)}
    self.episode_rewards = {i: [] for i in range(num_envs)}
    # Actual experiences ready to train.
    self.ready_actions = deque(maxlen=10000)
    self.ready_observations = deque(maxlen=10000)
    self.ready_rewards = deque(maxlen=10000)
    self.ready_episode = deque(maxlen=10000)
    self.episode_count = 0
    self.entropy_coeff = config.get('entropy_coeff', 0.001)

  def create_models(self, env, config):
    mock_observation, _ = env.reset()
    self.policy = Policy(env.action_space.nvec[-1], mock_observation,
                         config['network'])
    if os.path.exists(self.checkpoint_path):
      self.policy.load_state_dict(
          torch.load(self.checkpoint_path, map_location=self.device))
    self.policy.to(self.device)
    return self.policy.parameters()

  def get_action(self, observation: Observation) -> tuple[int, np.ndarray]:
    with torch.no_grad(), ProfileScope("model_inference"):
      x, _ = self.policy(observation.as_input(self.device))
      probs = torch.nn.functional.softmax(x, dim=-1)
      m = torch.distributions.Categorical(probs=probs)
      actions = m.sample().cpu().numpy()

    for i, observation in enumerate(observation.as_list()):
      self.episode_observations[i].append(observation)
      self.episode_actions[i].append(actions[i])

    return actions, probs.detach().cpu().numpy()

  def episode_learn(self, actions, observations, rewards):
    raise NotImplementedError("Subclasses should implement this method.")

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
    while len(self.ready_episode) > 0 and episode == self.ready_episode[0]:
      self.ready_episode.popleft()
      actions.append(self.ready_actions.popleft())
      observations.append(self.ready_observations.popleft())
      rewards.append(self.ready_rewards.popleft())
    self.episode_learn(actions, observations, rewards)
    return len(actions)

  def save_models(self):
    torch.save(self.policy.state_dict(), self.checkpoint_path)


class REINFORCEAgent(PolicyAgent):

  def __init__(self, env, device, summary_writer, config):
    super().__init__(env, device, summary_writer, config)

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
    x, _ = self.policy(x)
    assert _ is None, "REINFORCE does not have a critic"
    probs = torch.nn.functional.softmax(x, dim=-1)
    m = torch.distributions.Categorical(probs=probs)
    log_probs = m.log_prob(torch.tensor(actions).to(self.device))

    # Normalize the returns (commented out for debugging)
    # returns = (returns - returns.mean())
    self.optimizer.zero_grad()
    entropy = m.entropy().mean()
    loss = -(log_probs * returns
             ).sum() - self.entropy_coeff * entropy  # Very small entropy bonus
    loss.backward()
    pre_clip_grad_norm = self.clip_gradients(self.policy.parameters())
    self.optimizer.step()

    self.summary_writer.add_scalar('Replay/Loss', loss)
    self.summary_writer.add_scalar('Replay/LearningRate',
                                   self.optimizer.param_groups[0]['lr'])
    self.summary_writer.add_scalar(
        "Replay/ParamNorm",
        torch.nn.utils.get_total_norm(self.policy.parameters()))
    self.summary_writer.add_scalar("Replay/Entropy", entropy.item())
    self.summary_writer.add_scalar("Replay/PreClipGradNorm",
                                   pre_clip_grad_norm)
    #self.summary_writer.add_scalar('Replay/Entropy', entropy)
    if 'clip_gradients' in self.config:
      self.summary_writer.add_scalar(
          "Replay/GradScaleWithLr",
          min(pre_clip_grad_norm, self.config['clip_gradients']) *
          self.optimizer.param_groups[0]['lr'])


class A2CAgent(PolicyAgent):

  def __init__(self, env, device, summary_writer, config):
    super().__init__(env, device, summary_writer, config)

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
    x, q_value = self.policy(x)
    q_value = torch.gather(q_value, 1,
                           torch.tensor(actions).unsqueeze(1).to(
                               self.device)).squeeze(1)

    advantage = returns - q_value.detach()
    probs = torch.nn.functional.softmax(x, dim=-1)
    m = torch.distributions.Categorical(probs=probs)
    log_probs = m.log_prob(torch.tensor(actions).to(self.device))

    self.optimizer.zero_grad()
    entropy = m.entropy().mean()
    actor_loss = -(log_probs * advantage).sum(
    ) - self.entropy_coeff * entropy  # Very small entropy bonus

    critic_loss = nn.MSELoss()(q_value, returns)

    # Combine both losses for shared torso network
    total_loss = actor_loss + critic_loss
    total_loss.backward()
    pre_clip_grad_norm = self.clip_gradients(self.policy.parameters())
    self.optimizer.step()

    self.summary_writer.add_scalar('Replay/ActorLoss', actor_loss)
    self.summary_writer.add_scalar('Replay/CriticLoss', critic_loss)
    self.summary_writer.add_scalar('Replay/TotalLoss', total_loss)
    self.summary_writer.add_scalar('Replay/LearningRate',
                                   self.optimizer.param_groups[0]['lr'])
    self.summary_writer.add_scalar(
        "Replay/ParamNorm",
        torch.nn.utils.get_total_norm(self.policy.parameters()))
    self.summary_writer.add_scalar("Replay/Entropy", entropy.item())
    self.summary_writer.add_scalar("Replay/PreClipGradNorm",
                                   pre_clip_grad_norm)
    #self.summary_writer.add_scalar('Replay/Entropy', entropy)
    if 'clip_gradients' in self.config:
      self.summary_writer.add_scalar(
          "Replay/GradScaleWithLr",
          min(pre_clip_grad_norm, self.config['clip_gradients']) *
          self.optimizer.param_groups[0]['lr'])
