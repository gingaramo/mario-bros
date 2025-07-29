from typing import Union, Tuple, List
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
from src.tb_logging import GlobalStepSummaryWriter as SummaryWriter
from src.dqn import DQN, DuelingDQN
from src.environment import Observation
from src.state import State
from src.replay_buffer import UniformExperienceReplayBuffer, PrioritizedExperienceReplayBuffer
import pickle
from torch.optim import lr_scheduler


class Agent:

  def __init__(self, env, device, config):
    self.action_size = env.action_space.n
    self.device = device
    self.config = config

    # Replay parameters.
    self.replay_every_n_steps = config['replay_every_n_steps']
    self.batch_size = config['batch_size']
    self.replays_until_target_update = config.get(
        'replays_until_target_update', 0)
    self.target_updates_counter = 0

    # Learning parameters.
    self.gamma = config['gamma']
    self.epsilon_min = config['epsilon_min']
    self.learning_rate = config['learning_rate']
    assert 'epsilon_exponential_decay' in config or 'epsilon_linear_decay' in config, \
        "Either 'epsilon_exponential_decay' or 'epsilon_linear_decay' must be provided in config."
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
      self.episodes_trained = 0
      self.global_step = 0
      self.initial_epsilon = config['epsilon']
    else:
      with open(self.checkpoint_state_path, "rb") as f:
        state = pickle.load(f)
        self.episodes_trained = state['episodes_trained']
        self.global_step = state['global_step']
        self.initial_epsilon = state['initial_epsilon']
        print(
            f'Resuming from checkpoint. Episode {self.episodes_trained}, global step {self.global_step}'
        )
    self.summary_writer = SummaryWriter(log_dir=f"runs/tb_{config['name']}",
                                        max_queue=100000,
                                        flush_secs=300,
                                        purge_step=self.global_step)

    replay_buffer_config = config.get('replay_buffer', {
        'type': 'uniform',
        'size': 100000
    })
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
          f"Using prioritized replay buffer with size {replay_buffer_config['size']}"
      )
    else:
      raise ValueError(
          f"Unsupported replay buffer type: {replay_buffer_config['type']}. Supported: 'uniform', 'prioritized'."
      )

    # Action selection parameters.
    self.action_selection = config.get('action_selection', 'max')
    self.action_selection_temperature = config.get(
        'action_selection_temperature', 1.0)
    # Action repeat settings
    self.step = 0
    # Episode being recorded? Helps avoid state updates and checkpoints
    self.recording = False

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
      if os.path.exists(self.checkpoint_path):
        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device))
      self.target_model.load_state_dict(self.model.state_dict())
      # Set the target model to eval mode
      self.target_model.eval()
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
    return max(eps, self.epsilon_min)

  def remember(self, observation: Observation, action: int, reward: float,
               next_observation: Observation, done: bool):
    """Stores experience. State gathered from last state sent to act().
    """
    self.replay_buffer.append(observation.as_input(torch.device('cpu')),
                              action, reward,
                              next_observation.as_input(torch.device('cpu')),
                              done)

  def act(self, observation: Observation) -> tuple[int, np.ndarray]:
    """Returns the action to take based on the current observation.

    Returns:
      action: The action to take, either random or based on the model's prediction.
      act_values: The Q-values predicted by the model for the current observation.
    """
    if not self.recording:
      # We only update global step when we're training, not recording.
      self.global_step += 1
      self.summary_writer.set_global_step(self.global_step)

    with torch.no_grad():
      q_values = self.model(observation.as_input(self.device))
      q_values_np = q_values.cpu().detach().numpy()
      # "Act values" are q values for most cases but for softmax.
      act_values = q_values_np

    if np.random.rand() <= self.epsilon:
      action = random.randrange(self.action_size)
    else:
      if self.action_selection == 'softmax':
        # Softmax sampling with temperature
        q = q_values_np / self.action_selection_temperature
        exp_q = np.exp(q - np.max(q))
        probs = exp_q / np.sum(exp_q)
        action = np.random.choice(self.action_size, p=probs)
        act_values = probs
      elif self.action_selection == 'max':
        # Greedy action selection
        action = torch.argmax(q_values).item()
      else:
        raise ValueError(
            f"Unsupported action selection method: {self.action_selection}. Supported: 'softmax', 'max'."
        )
    return action, act_values

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
        idx = torch.argmax(self.model(all_next_observation, training=True),
                           dim=1)
        q_next = torch.gather(
            self.target_model(all_next_observation, training=True), 1,
            idx.view(-1, 1)).squeeze(1)
      else:
        # Use the target model to get the Q-value for the next state
        q_next = self.target_model(all_next_observation,
                                   training=True).max(dim=1).values
      return all_reward + self.gamma * q_next * (1.0 - all_done.float())

  def compute_prediction(self, all_observation: Tuple[torch.Tensor,
                                                      torch.Tensor],
                         all_action: torch.Tensor) -> torch.Tensor:
    q_values = self.model(all_observation, training=True)
    return torch.gather(q_values, 1, all_action.view(-1, 1)).squeeze(1)

  def replay(self):
    if self.recording:
      return
    if len(self.replay_buffer) < max(self.batch_size,
                                     self.config['min_memory_size']):
      return
    if self.global_step % self.replay_every_n_steps != 0:
      return
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
    if isinstance(self.replay_buffer, PrioritizedExperienceReplayBuffer):
      # Update the surprise values in the replay buffer
      self.replay_buffer.update_surprise(
          sampled_indices, (prediction - target).detach().cpu().numpy())
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
      for target_param, local_param in zip(self.target_model.parameters(),
                                           self.model.parameters()):
        target_param.data.copy_(tau * local_param.data +
                                (1.0 - tau) * target_param.data)

    if self.lr_scheduler:
      self.lr_scheduler.step()

    self.summary_writer.add_scalar('Replay/Loss', loss)
    self.summary_writer.add_scalar('Replay/LearningRate',
                                   self.optimizer.param_groups[0]['lr'])
    self.summary_writer.add_scalar('Replay/Epsilon', self.epsilon)
    self.summary_writer.add_scalar(
        "Replay/ParamNorm",
        torch.nn.utils.get_total_norm(self.model.parameters()))
    self.summary_writer.add_scalar("Replay/PreClipGradNorm",
                                   pre_clip_grad_norm)
    self.summary_writer.add_scalar(
        "Replay/GradScaleWithLr",
        min(pre_clip_grad_norm, self.config['clip_gradients']) *
        self.optimizer.param_groups[0]['lr'])

  def episode_begin(self, recording=False):
    self.recording = recording

  def episode_end(self, episode_info):
    if self.recording:
      return
    # Episode statistics
    self.summary_writer.add_scalar("Episode/Episode", episode_info['episode'])
    self.summary_writer.add_scalar("Episode/Reward",
                                   episode_info['total_reward'])
    self.summary_writer.add_scalar("Episode/Steps", episode_info['steps'])

    # Checkpoint
    self.episodes_trained += 1
    torch.save(self.model.state_dict(), self.checkpoint_path)
    with open(self.checkpoint_state_path, "wb") as f:
      pickle.dump(
          {
              'episodes_trained': self.episodes_trained,
              'global_step': self.global_step,
              'initial_epsilon': self.initial_epsilon,
          }, f)
    self.summary_writer.flush()
