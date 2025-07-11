from typing import Union, Tuple, List
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
from src.tb_logging import GlobalStepSummaryWriter as SummaryWriter
from src.dqn import DQN
from src.environment import Observation
from src.state import State
from src.replay_buffer import ReplayBuffer
import pickle
from torch.optim import lr_scheduler


def td_error(q_value: float, reward: float):
  error_epsilon = 1e-2  # ensures sampling for all (even zero error) experiences
  return math.fabs(q_value - reward) + error_epsilon


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
      self.epsilon = config['epsilon']
    else:
      with open(self.checkpoint_state_path, "rb") as f:
        state = pickle.load(f)
        self.episodes_trained = state['episodes_trained']
        self.global_step = state['global_step']
        self.epsilon = state['epsilon']
        print(
            f'Resuming from checkpoint. Episode {self.episodes_trained}, global step {self.global_step}'
        )
    self.summary_writer = SummaryWriter(log_dir=f"runs/tb_{config['name']}",
                                        max_queue=10000,
                                        flush_secs=60,
                                        purge_step=self.global_step)
    self.replay_buffer = ReplayBuffer(config['replay_buffer'], device,
                                      self.summary_writer)

    # Action selection parameters.
    self.action_selection = config.get('action_selection', 'max')
    self.action_selection_temperature = config.get(
        'action_selection_temperature', 1.0)
    # Action repeat settings
    self.step = 0
    # Episode being recorded? Helps avoid state updates and checkpoints
    self.recording = False

    if 'dqn' in config['network']:
      mock_observation, _ = env.reset()
      self.model = DQN(self.action_size, mock_observation,
                       config['network']['dqn'])
      self.target_model = DQN(self.action_size, mock_observation,
                              config['network']['dqn'])
      if os.path.exists(self.checkpoint_path):
        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device))
      self.target_model.load_state_dict(self.model.state_dict())
    else:
      raise ValueError("Network configuration must include a valid model.")
    self.model.to(self.device)
    self.target_model.to(self.device)
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
    self.last_q_values = q_values
    self.last_act_values = act_values
    # Log histogram of act_values (Q-values or softmax probabilities)
    self.summary_writer.add_histogram('Act/ActValues', act_values)
    return action, act_values

  def clip_gradients(self):
    if 'clip_gradients' in self.config:
      return nn.utils.clip_grad_norm_(self.model.parameters(),
                                      self.config['clip_gradients'])
    return torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                          float('inf'))

  def replay(self):
    if self.recording:
      return
    if len(self.replay_buffer) < self.batch_size:
      return
    if len(self.replay_buffer) < self.config.get('min_memory_size', 0):
      return
    if self.global_step % self.replay_every_n_steps != 0:
      return
    # Gather ids to fix memory_error later on

    (all_observation, all_action, all_reward, all_next_observation,
     all_done) = self.replay_buffer.sample(self.batch_size)
    with torch.no_grad():
      # Double DQN is the next line:
      if self.config.get('double_dqn', True):
        # Use the model to select the best action for the next state
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
      target = all_reward + self.gamma * q_next * (~all_done)

    q_values = self.model(all_observation, training=True)
    q_pred = torch.gather(q_values, 1, all_action.view(-1, 1)).squeeze(1)
    # if self.memory_selection == 'prioritized':
    #   # Update memory error with new memory error
    #   new_memory_error = [
    #       td_error(p, t)**self.memory_selection_alpha
    #       for p, t in zip(q_pred.flatten().cpu().detach().numpy(),
    #                       target.flatten().cpu().detach().numpy())
    #   ]
    #   for idx, err in zip(minibatch_ids, np.abs(new_memory_error)):
    #     self.memory_error[idx] = err
    #   per_experience_loss = self.get_loss(reduction='none')(q_pred, target)
    #   loss = torch.mean(per_experience_loss * wis_weights)
    # elif self.memory_selection == 'uniform':
    loss = self.get_loss()(q_pred, target)

    # Update target model every `target_update_frequency` steps
    self.replays_until_target_update -= 1
    if self.replays_until_target_update <= 0:
      self.replays_until_target_update = self.config.get(
          'replays_until_target_update', 0)
      self.target_updates_counter += 1
      self.summary_writer.add_scalar("Replay/TargetUpdate",
                                     self.target_updates_counter)
      self.target_model.load_state_dict(self.model.state_dict())

    self.optimizer.zero_grad()
    loss.backward()
    pre_clip_grad_norm = self.clip_gradients()
    self.optimizer.step()
    if self.lr_scheduler:
      self.lr_scheduler.step()
    if self.epsilon > self.epsilon_min:
      if self.epsilon_exponential_decay:
        self.epsilon *= self.epsilon_exponential_decay
      elif self.epsilon_linear_decay:
        self.epsilon -= self.epsilon_linear_decay

    self.summary_writer.add_scalar('Replay/Loss', loss)
    self.summary_writer.add_scalar('Replay/LearningRate',
                                   self.optimizer.param_groups[0]['lr'])
    self.summary_writer.add_scalar('Replay/Q-mean',
                                   torch.mean(q_values.flatten()))
    self.summary_writer.add_scalar('Replay/Epsilon', self.epsilon)
    self.summary_writer.add_scalar(
        "Replay/PreClipParamNorm",
        torch.nn.utils.get_total_norm(self.model.parameters()))
    self.summary_writer.add_scalar("Replay/PreClipGradNorm",
                                   pre_clip_grad_norm)

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
              'epsilon': self.epsilon
          }, f)
    self.summary_writer.flush()
