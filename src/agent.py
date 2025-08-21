from typing import Union, Tuple, List
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
from src.tb_logging import CustomSummaryWriter
from src.dqn import DQN, DuelingDQN
from src.environment import Observation
from src.noisy_network import replace_linear_with_noisy, NoisyLinear
from src.state import State
from src.replay_buffer import UniformExperienceReplayBuffer, PrioritizedExperienceReplayBuffer, OrderedExperienceReplayBuffer
import pickle
from torch.optim import lr_scheduler


def get_noisy_network_weights_norm(named_modules):
  weight_mu_norm = []
  weight_mu_sigma_norm = []
  for name, module in named_modules:
    if isinstance(module, NoisyLinear):
      weight_mu_norm.append(torch.norm(module.weight_mu).item())
      weight_mu_sigma_norm.append(torch.norm(module.weight_sigma).item())
  return np.array(weight_mu_norm).mean(), np.array(
      weight_mu_sigma_norm).mean().item()


class Agent:

  def __init__(self, env, device, summary_writer, config):
    # For now this works for vectorized environments that have only one action dimension.
    self.action_size = env.action_space.nvec[-1]
    self.num_envs = env.unwrapped.num_envs
    self.device = device
    self.summary_writer = summary_writer
    self.config = config

    # Replay parameters.
    self.batch_size = config['batch_size']
    self.replays_until_target_update = config.get(
        'replays_until_target_update', 0)
    self.replays_until_checkpoint = config['checkpoint_every_n_replays']
    self.target_updates_counter = 0

    # Learning parameters.
    self.gamma = config['gamma']
    self.epsilon_min = config['epsilon_min']
    self.learning_rate = config['learning_rate']
    self.apply_noisy_network = config['apply_noisy_network']
    assert self.apply_noisy_network or ('epsilon_exponential_decay' in config or 'epsilon_linear_decay' in config), \
        "Either 'epsilon_exponential_decay' or 'epsilon_linear_decay' must be provided in config if not using noisy networks."
    self.epsilon_linear_decay = config.get('epsilon_linear_decay', None)
    self.epsilon_exponential_decay = config.get('epsilon_exponential_decay',
                                                None)

    if self.config['loss'] == 'mse':
      self.get_loss = nn.MSELoss
    elif self.config['loss'] == 'smooth_l1':
      self.get_loss = nn.SmoothL1Loss
    elif self.config['loss'] == 'huber':
      self.get_loss = nn.HuberLoss
    else:
      raise ValueError(f"Unsupported loss function: {self.config['loss']}")

    # Checkpoint functionality
    _path = os.path.join("checkpoint/", config['name'])
    if not os.path.exists(_path):
      os.makedirs(_path)
    self.checkpoint_path = os.path.join(_path, 'state_dict.pkl')
    self.checkpoint_state_path = os.path.join(_path, "state.pkl")
    if (not os.path.exists(self.checkpoint_path)
        or not os.path.exists(self.checkpoint_state_path)):
      self.episodes_played = 0
      self.global_step = 0
      self.trained_experiences = 0
      self.initial_epsilon = config.get('epsilon', 1.0)
    else:
      with open(self.checkpoint_state_path, "rb") as f:
        state = pickle.load(f)
        self.episodes_played = state['episodes_played']
        self.global_step = state['global_step']
        self.trained_experiences = state['trained_experiences']
        self.initial_epsilon = state['initial_epsilon']
        print(
            f'Resuming from checkpoint. Episode {self.episodes_played}, global step {self.global_step}, trained experiences {self.trained_experiences}'
        )

    replay_buffer_config = config['replay_buffer']
    if replay_buffer_config['type'] == 'uniform':
      self.replay_buffer = UniformExperienceReplayBuffer(
          replay_buffer_config, device, self.summary_writer)
      print(
          f"Using uniform replay buffer with size {replay_buffer_config['size']}"
      )
    elif replay_buffer_config['type'] == 'prioritized':
      self.replay_buffer = PrioritizedExperienceReplayBuffer(
          replay_buffer_config, device, self.summary_writer,
          lambda *args, **kwargs: self.compute_target(*args, **kwargs),
          lambda *args, **kwargs: self.compute_prediction(*args, **kwargs))
      print(
          f"Using prioritized replay buffer with size {replay_buffer_config['size']}, config: {replay_buffer_config}"
      )
    elif replay_buffer_config['type'] == 'ordered':
      self.replay_buffer = OrderedExperienceReplayBuffer(
          replay_buffer_config, device, self.summary_writer)
      print(
          f"Using ordered replay buffer with size {replay_buffer_config['size']}"
      )
    else:
      raise ValueError(
          f"Unsupported replay buffer type: {replay_buffer_config['type']}. Supported: 'uniform', 'prioritized'."
      )

    # Action selection parameters.
    self.action_selection = config.get('action_selection', 'max')
    self.action_selection_temperature = config.get(
        'action_selection_temperature', 1.0)

    if 'dqn' in config['network'] or 'dueling_dqn' in config['network']:
      mock_observation, _ = env.reset()
      if 'dqn' in config['network']:
        self.model = DQN(self.action_size, mock_observation,
                         config['network']['dqn'])
        self.target_model = DQN(self.action_size, mock_observation,
                                config['network']['dqn'])
      elif 'dueling_dqn' in config['network']:
        self.model = DuelingDQN(self.action_size, mock_observation,
                                config['network']['dueling_dqn'])
        self.target_model = DuelingDQN(self.action_size, mock_observation,
                                       config['network']['dueling_dqn'])

      if self.apply_noisy_network:
        self.model = replace_linear_with_noisy(self.model)
        self.target_model = replace_linear_with_noisy(self.target_model)
      if os.path.exists(self.checkpoint_path):
        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device))
      self.target_model.load_state_dict(self.model.state_dict())
      # Set the target model to eval mode and keep it there
      self.target_model.eval()
      # Ensure target model parameters are not updated
      for param in self.target_model.parameters():
        param.requires_grad = False
    else:
      raise ValueError("Network configuration must include a valid model.")
    self.model.to(self.device)
    self.target_model.to(self.device)

    # Curiosity module
    self.curiosity_module = None
    if 'curiosity' in config:
      from src.curiosity import CuriosityModule
      self.curiosity_module = CuriosityModule(config['curiosity'], self.model)
      self.curiosity_module.to(self.device)
      print("Curiosity module initialized.")

    if config['optimizer'] == 'adam':
      self.optimizer = optim.Adam(self.model.parameters(),
                                  lr=self.learning_rate)
    else:
      raise ValueError(
          f"Unsupported optimizer: {config['optimizer']}. Supported: 'adam'.")

    self.lr_scheduler = None
    if 'lr_scheduler' in config:
      exec(
          f"self.lr_scheduler = {config['lr_scheduler']['type']}(self.optimizer, **{config['lr_scheduler']['args']})"
      )

  @property
  def epsilon(self):
    if self.epsilon_exponential_decay:
      eps = self.initial_epsilon * (self.epsilon_exponential_decay**
                                    self.global_step)
    elif self.epsilon_linear_decay:
      eps = self.initial_epsilon - (self.epsilon_linear_decay *
                                    self.global_step)
    else:
      eps = 0.0
    return max(eps, self.epsilon_min)

  def remember(self, observation: Observation, action: int, reward: float,
               next_observation: Observation, done: bool):
    """Stores experience. State gathered from last state sent to act().
    """
    # Since observations come from multiple environments, we need to
    # convert them to individual observations for the replay buffer.
    observation = [
        obs.as_input(torch.device('cpu')) for obs in observation.as_list()
    ]
    next_observation = [
        obs.as_input(torch.device('cpu'))
        for obs in next_observation.as_list()
    ]
    for obs, act, rew, next_obs, done in zip(observation, action, reward,
                                             next_observation, done):
      self.replay_buffer.append(obs, act, rew, next_obs, done)

  def act(self, observation: Observation) -> tuple[int, np.ndarray]:
    """Returns the action to take based on the current observation.

    Returns:
      action: The action to take, either random or based on the model's prediction.
      act_values: The Q-values predicted by the model for the current observation.
    """
    # We only update global step when we're training, not recording.
    self.global_step += self.num_envs
    self.summary_writer.set_global_step(self.global_step)

    assert self.action_selection in [
        'max', 'random'
    ], f"Unsupported action selection method {self.action_selection}. Supported: 'max', 'random'."

    if self.action_selection == 'random':
      action = np.random.randint(low=0,
                                 high=self.action_size,
                                 size=self.num_envs)
      act_values = np.zeros((self.num_envs, self.action_size))
      return action.astype(int), act_values
    elif self.action_selection == 'max':
      with torch.no_grad():
        q_values = self.model(observation.as_input(self.device))
        q_values_np = q_values.detach().cpu().numpy()

      if self.apply_noisy_network:
        action = np.argmax(q_values_np, axis=1)

        weight_mu_norm, weight_mu_sigma_norm = get_noisy_network_weights_norm(
            self.model.named_modules())
        self.summary_writer.add_scalar("Action/NoisyNetworkWeightMuNorm",
                                       weight_mu_norm)
        self.summary_writer.add_scalar("Action/NoisyNetworkWeightSigmaNorm",
                                       weight_mu_sigma_norm)

    # Perform epsilon-greedy action selection
    action = np.zeros(self.num_envs)
    random_action_idx = np.random.rand(self.num_envs) < self.epsilon
    action[random_action_idx] = np.random.randint(
        low=0, high=self.action_size, size=self.num_envs)[random_action_idx]
    action[~random_action_idx] = np.argmax(q_values_np,
                                           axis=1)[~random_action_idx]

    self.summary_writer.add_scalar("Action/Epsilon", self.epsilon)

    return action.astype(int), q_values_np

  def clip_gradients(self):
    if 'clip_gradients' in self.config:
      return nn.utils.clip_grad_norm_(self.model.parameters(),
                                      self.config['clip_gradients'])
    return torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                          float('inf'))

  def compute_target(self, all_reward: torch.Tensor,
                     all_next_observation: Tuple[torch.Tensor, torch.Tensor],
                     all_done: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
      # Double DQN is the next line:
      if self.config.get('double_dqn', True):
        # Use the online model to select the best action for the next state
        # and then use the target model to get the Q-value for that action.
        # This helps reduce overestimation bias.
        idx = torch.argmax(self.model(all_next_observation, training=False),
                           dim=1)
        q_next = torch.gather(
            self.target_model(all_next_observation, training=False), 1,
            idx.view(-1, 1)).squeeze(1)
      else:
        # Use the target model to get the Q-value for the next state
        q_next = self.target_model(all_next_observation,
                                   training=False).max(dim=1).values
      return all_reward + self.gamma * q_next * (1.0 - all_done.float())

  def compute_prediction(self, all_observation: Tuple[torch.Tensor,
                                                      torch.Tensor],
                         all_action: torch.Tensor) -> torch.Tensor:
    q_values = self.model(all_observation, training=True)
    return torch.gather(q_values, 1, all_action.view(-1, 1)).squeeze(1)

  def replay(self):
    if len(self.replay_buffer) < max(self.batch_size,
                                     self.config['min_memory_size']):
      return False

    # Gather ids to fix memory_error later on
    # If this is importance sampling, we need to get the weights
    if isinstance(self.replay_buffer, PrioritizedExperienceReplayBuffer):
      # Sample from the replay buffer
      (all_observation, all_action, all_reward, all_next_observation, all_done
       ), importance_sampling, sampled_indices = self.replay_buffer.sample(
           self.batch_size)
      importance_sampling = torch.Tensor(importance_sampling).to(self.device)
    else:
      # Sample from the replay buffer without importance sampling
      (all_observation, all_action, all_reward, all_next_observation,
       all_done) = self.replay_buffer.sample(self.batch_size)
      importance_sampling = torch.ones(self.batch_size, device=self.device)

    target = self.compute_target(all_reward, all_next_observation, all_done)
    prediction = self.compute_prediction(all_observation, all_action)

    self.optimizer.zero_grad()
    # Compute the loss
    loss = self.get_loss(reduction='none')(prediction, target)
    if self.curiosity_module:
      curiosity_reward = self.curiosity_module(all_observation,
                                               all_action,
                                               all_next_observation,
                                               training=True)
      assert loss.shape == curiosity_reward.shape, \
          f"Loss shape {loss.shape} does not match curiosity reward shape {curiosity_reward.shape}"
      self.summary_writer.add_scalar('Replay/CuriosityReward',
                                     curiosity_reward.sum())
      self.summary_writer.add_scalar('Replay/CuriosityRewardMean',
                                     curiosity_reward.mean())
      loss = loss + curiosity_reward

    loss = loss * importance_sampling
    loss = loss.sum()
    loss.backward()
    pre_clip_grad_norm = self.clip_gradients()
    self.optimizer.step()

    self.trained_experiences += self.batch_size

    # Update the surprise values in the replay buffer
    if isinstance(self.replay_buffer, PrioritizedExperienceReplayBuffer):
      self.replay_buffer.update_surprise(
          sampled_indices, (prediction - target).detach().cpu().numpy())

    # Update target model every `target_update_frequency` steps
    self.replays_until_target_update -= 1
    if self.replays_until_target_update <= 0:
      self.replays_until_target_update = self.config.get(
          'replays_until_target_update', 0)
      self.target_updates_counter += 1
      self.summary_writer.add_scalar("Replay/TargetUpdate",
                                     self.target_updates_counter)
      # Soft update with 10% interpolation (tau = 0.1)
      tau = 0.1
      with torch.no_grad():
        for target_param, local_param in zip(self.target_model.parameters(),
                                             self.model.parameters()):
          target_param.data.copy_(tau * local_param.data +
                                  (1.0 - tau) * target_param.data)
      # Ensure target model stays in eval mode
      self.target_model.eval()

    # Checkpoint every `checkpoint_every_n_replays` replays
    self.replays_until_checkpoint -= 1
    if self.replays_until_checkpoint % self.config[
        'checkpoint_every_n_replays'] == 0:
      self.save_checkpoint()
      self.replays_until_checkpoint = self.config['checkpoint_every_n_replays']

    self.lr_scheduler.step() if self.lr_scheduler else None

    self.summary_writer.add_scalar('Replay/Loss', loss)
    self.summary_writer.add_scalar('Replay/LearningRate',
                                   self.optimizer.param_groups[0]['lr'])
    self.summary_writer.add_scalar(
        "Replay/ParamNorm",
        torch.nn.utils.get_total_norm(self.model.parameters()))
    self.summary_writer.add_scalar("Replay/PreClipGradNorm",
                                   pre_clip_grad_norm)
    self.summary_writer.add_scalar(
        "Replay/GradScaleWithLr",
        min(pre_clip_grad_norm, self.config['clip_gradients']) *
        self.optimizer.param_groups[0]['lr'])
    self.summary_writer.add_scalar("Replay/TrainedExperiences",
                                   self.trained_experiences)
    return True

  def curiosity_reward(self, observation, next_observation, action, device):
    with torch.no_grad():
      frame, dense = observation.as_input(device)
      next_frame, next_dense = next_observation.as_input(device)
      curiosity_reward = self.curiosity_module(
          (frame.unsqueeze(0).clone().detach().to(device),
           dense.clone().detach().to(device)),
          torch.tensor(action, dtype=torch.long).to(device),
          (next_frame.unsqueeze(0).clone().detach().to(device),
           next_dense.clone().detach().to(device)),
          training=False)
      return curiosity_reward

  def save_checkpoint(self):
    torch.save(self.model.state_dict(), self.checkpoint_path)
    with open(self.checkpoint_state_path, "wb") as f:
      pickle.dump(
          {
              'episodes_played': self.episodes_played,
              'global_step': self.global_step,
              'trained_experiences': self.trained_experiences,
              'initial_epsilon': self.initial_epsilon,
          }, f)
    self.summary_writer.flush()
