from collections import deque
import os
from typing import Union, Tuple, Callable
from src.agent.agent import Agent
from src.environment import Observation, merge_observations
from src.network.observation_layer import CNNObservationLayer
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
    self.user_residual = config['use_residual']

    self.convolution = None if mock_observation.frame is None else CNNObservationLayer(
        mock_observation, config['convolution'])
    self.residual_parameters = nn.ParameterList()
    self.residuals = {}
    dense_input_dim = mock_observation.dense.shape[1]
    cnn_input_dim = 0 if self.convolution is None else self.convolution.output_dim

    input_dim = dense_input_dim + cnn_input_dim
    if 'torso' in config:
      torso_layers_dim = [input_dim] + config['torso']['hidden_layers']
      self.torso = _create_mlp(torso_layers_dim)
      torso_output_size = torso_layers_dim[-1]
      if self.user_residual:
        self.residuals['torso'] = nn.Parameter(
            torch.ones(len(torso_layers_dim) - 1, dtype=torch.float32) * 0.95)
        self.residual_parameters.append(self.residuals['torso'])

    else:
      self.torso = None
      torso_output_size = mock_observation.dense.shape[1]

    if 'critic' in config:
      mlp_layers_dim = [torso_output_size
                        ] + config['critic']['hidden_layers'] + [1]
      self.critic = _create_mlp(mlp_layers_dim)
      if self.user_residual:
        self.residuals['critic'] = nn.Parameter(
            torch.ones(len(mlp_layers_dim) - 1, dtype=torch.float32) * 0.95)
        self.residual_parameters.append(self.residuals['critic'])
    else:
      self.critic = None

    if 'actor' in config:
      mlp_layers_dim = [torso_output_size
                        ] + config['actor']['hidden_layers'] + [action_size]
      self.actor = _create_mlp(mlp_layers_dim)
      if self.user_residual:
        self.residuals['actor'] = nn.Parameter(
            torch.ones(len(mlp_layers_dim) - 1, dtype=torch.float32) * 0.95)
        self.residual_parameters.append(self.residuals['actor'])

    self.activation = torch.nn.LeakyReLU()

  def forward(self,
              x: Tuple[torch.Tensor, torch.Tensor],
              training: bool = False) -> torch.Tensor:
    image, dense = x
    cnn_output = self.convolution(image) if self.convolution else torch.empty(
        0, device=dense.device)

    if training and self.user_residual:
      with torch.no_grad():
        for key in self.residuals:
          self.residuals[key].clamp_(0.0, 1.0)

    torso_dense = torch.concat([cnn_output, dense], dim=1)
    if self.torso:
      for i, layer in enumerate(self.torso):
        input_dense = torso_dense
        torso_dense = self.activation(layer(torso_dense))
        if self.user_residual and input_dense.shape == torso_dense.shape:
          torso_dense = torso_dense * self.residuals['torso'][i] + \
              input_dense * (1 - self.residuals['torso'][i])
    critic_dense = torso_dense
    actor_dense = torso_dense

    if self.critic:
      for i, layer in enumerate(self.critic[:-1]):
        input_dense = critic_dense
        critic_dense = self.activation(layer(critic_dense))
        if self.user_residual and input_dense.shape == critic_dense.shape:
          critic_dense = critic_dense * self.residuals['critic'][i] + \
              input_dense * (1 - self.residuals['critic'][i])
      input_dense = critic_dense
      critic_output = self.critic[-1](critic_dense)
      if self.user_residual and input_dense.shape == critic_output.shape:
        critic_output = critic_output * self.residuals['critic'][-1] + \
            input_dense * (1 - self.residuals['critic'][-1])
    else:
      critic_output = None

    for i, layer in enumerate(self.actor[:-1]):
      input_dense = actor_dense
      actor_dense = self.activation(layer(actor_dense))
      if self.user_residual and input_dense.shape == actor_dense.shape:
        actor_dense = actor_dense * self.residuals['actor'][i] + \
            input_dense * (1 - self.residuals['actor'][i])
    input_dense = actor_dense
    actor_output = self.actor[-1](actor_dense)
    if self.user_residual and input_dense.shape == actor_output.shape:
      actor_output = actor_output * self.residuals['actor'][-1] + \
          input_dense * (1 - self.residuals['actor'][-1])

    return actor_output, critic_output


class PolicyAgent(Agent):

  def __init__(self, env, device, summary_writer, config):
    super().__init__(env, device, summary_writer, config)

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

    return actions, probs.detach().cpu().numpy()

  def save_models(self):
    torch.save(self.policy.state_dict(), self.checkpoint_path)


class REINFORCEAgent(PolicyAgent):

  def __init__(self, env, device, summary_writer, config):
    super().__init__(env, device, summary_writer, config)
    num_envs = env.unwrapped.num_envs
    # Running collection of episode data
    self.episode_observations = {i: [] for i in range(num_envs)}
    self.episode_actions = {i: [] for i in range(num_envs)}
    self.episode_rewards = {i: [] for i in range(num_envs)}
    self.episode_next_observation = {i: [] for i in range(num_envs)}
    # Actual experiences ready to train.
    self.ready_actions = deque(maxlen=10000)
    self.ready_observations = deque(maxlen=10000)
    self.ready_next_observation = deque(maxlen=10000)
    self.ready_rewards = deque(maxlen=10000)
    self.ready_episode = deque(maxlen=10000)
    self.episode_count = 0
    self.entropy_coeff = config.get('entropy_coeff', 0.001)

  def remember(self, observation, action, reward, next_observation, done,
               episode_start):
    for i, _obs in enumerate(zip(observation, next_observation)):
      obs, next_obs = _obs
      self.episode_observations[i].append(obs)
      self.episode_next_observation[i].append(next_obs)
      self.episode_actions[i].append(action[i])
      reward_i, done_i = reward[i], done[i]
      self.episode_rewards[i].append(reward_i)
      if done_i:
        self.episode_count += 1
        # ready actions is a deque
        self.ready_actions.extend(self.episode_actions[i])
        self.ready_observations.extend(self.episode_observations[i])
        self.ready_next_observation.extend(self.episode_next_observation[i])
        self.ready_rewards.extend(self.episode_rewards[i])
        self.ready_episode.extend([self.episode_count] *
                                  len(self.episode_rewards[i]))
        self.episode_actions[i] = []
        self.episode_observations[i] = []
        self.episode_next_observation[i] = []
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
    x, _ = self.policy(x, training=True)
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
    self.policy.clamp_residuals()

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
    num_envs = env.unwrapped.num_envs
    self.num_envs = num_envs
    self.n_steps = config.get('n_steps', 5)
    # Running collection of episode data
    self.episode_observations = {
        i: deque(maxlen=self.n_steps)
        for i in range(num_envs)
    }
    self.episode_actions = {
        i: deque(maxlen=self.n_steps)
        for i in range(num_envs)
    }
    self.episode_rewards = {
        i: deque(maxlen=self.n_steps)
        for i in range(num_envs)
    }
    self.finished_episodes = np.zeros(num_envs, dtype=bool)
    self.entropy_coeff = config.get('entropy_coeff', 0.001)
    self.steps = 0

  def remember(self, observation, action, reward, next_observation, done,
               episode_start):
    self.steps += 1
    # First we clean out the episodes that are done.
    for i, finished in enumerate(self.finished_episodes):
      if finished:
        self.episode_observations[i].clear()
        self.episode_actions[i].clear()
        self.episode_rewards[i].clear()
    self.finished_episodes = np.zeros(self.num_envs, dtype=bool)
    # Then we collect experiences for the new episode data
    for i in range(len(observation)):
      self.episode_observations[i].append(observation[i])
      self.episode_actions[i].append(action[i])
      self.episode_rewards[i].append(reward[i])
      if done[i]:
        self.finished_episodes[i] = True

  def replay(self):
    loss = None
    rewards = []

    valid_obs = [
        i for i in range(len(self.episode_observations))
        if len(self.episode_observations[i]) == self.n_steps
    ]
    if not valid_obs:
      return 0
    valid_finished_obs = self.finished_episodes[valid_obs]

    # observations is first and last observation of each valid episode
    observations = merge_observations(
        [self.episode_observations[i][0] for i in valid_obs] +
        [self.episode_observations[i][-1]
         for i in valid_obs]).as_input(self.device)

    policy, critic = self.policy(observations, training=True)
    critic = critic.squeeze()
    # We only need gradients for the observations that have full n_steps
    policy = policy[:len(valid_obs)]
    # But we need critic for both the first and last observation
    obs_critic = critic[:len(valid_obs)]
    next_obs_critic = critic[-len(valid_obs):]
    # zero out next_obs_critic for finished episodes
    next_obs_critic[valid_finished_obs] = 0.0

    rewards = torch.tensor([list(self.episode_rewards[i]) for i in valid_obs],
                           dtype=torch.float32,
                           device=self.device)
    gamma_powers = self.gamma**torch.arange(self.n_steps,
                                            device=self.device).float()
    # Discounted rewards
    rewards = torch.einsum('ij,j->i', rewards, gamma_powers)
    rewards += (self.gamma**self.n_steps) * next_obs_critic

    softmax = torch.nn.functional.softmax(policy, dim=-1)
    log_softmax = torch.nn.functional.log_softmax(policy, dim=-1)
    entropy = -torch.sum(softmax * log_softmax, dim=-1).mean()

    critic_loss = nn.MSELoss()(rewards, obs_critic)

    advantage = rewards - obs_critic.detach()
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    #advantage = advantage - advantage.mean()
    log_softmax_action_chosen = torch.gather(
        log_softmax, 1,
        torch.tensor([self.episode_actions[i][0] for i in valid_obs],
                     device=self.device).unsqueeze(1)).squeeze(1)

    actor_loss = -(log_softmax_action_chosen * advantage).mean()
    loss = actor_loss + critic_loss - self.entropy_coeff * entropy
    self.optimizer.zero_grad()
    loss.backward()
    pre_clip_grad_norm = self.clip_gradients(self.policy.parameters())
    self.optimizer.step()

    self.summary_writer.add_scalar('Replay/ActorLoss', actor_loss.item())
    self.summary_writer.add_scalar('Replay/CriticLoss', critic_loss.item())
    self.summary_writer.add_scalar("Replay/Entropy", entropy.item())
    self.summary_writer.add_scalar('Replay/TotalLoss', loss.item())
    self.summary_writer.add_scalar('Replay/LearningRate',
                                   self.optimizer.param_groups[0]['lr'])
    self.summary_writer.add_scalar(
        "Replay/ParamNorm",
        torch.nn.utils.get_total_norm(self.policy.parameters()))
    self.summary_writer.add_scalar("Replay/PreClipGradNorm",
                                   pre_clip_grad_norm)

    if 'clip_gradients' in self.config:
      self.summary_writer.add_scalar(
          "Replay/GradScaleWithLr",
          min(pre_clip_grad_norm, self.config['clip_gradients']) *
          self.optimizer.param_groups[0]['lr'])

    return len(valid_obs)
