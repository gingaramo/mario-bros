from typing import Tuple, List
import numpy as np
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import os
from src.profiler import ProfileScope, ProfileLockScope
from src.dqn import DQN, DuelingDQN
from src.environment import Observation
from src.noisy_network import replace_linear_with_noisy, NoisyLinear
from src.replay_buffer import UniformExperienceReplayBuffer, PrioritizedExperienceReplayBuffer, OrderedExperienceReplayBuffer
from src.agent.agent import Agent
import pickle
from torch.optim import lr_scheduler  # Keep -> see horrible exec() hack


class ValueAgent(Agent):

  def __init__(self, env, device, summary_writer, config):
    super().__init__(env, device, summary_writer, config)

    # Replay parameters.
    self.replays_until_target_update = config.get(
        'replays_until_target_update', 0)
    self.target_updates_counter = 0

    # Epsilon parameters.
    self.initial_epsilon = config.get('initial_epsilon', 1.0)
    self.epsilon_min = config.get('epsilon_min', 0.01)
    self.epsilon_exponential_decay = config.get('epsilon_exponential_decay',
                                                None)
    self.epsilon_linear_decay = config.get('epsilon_linear_decay', None)

    # Action selection parameters.
    self.action_selection = config.get('action_selection', 'max')
    self.action_selection_temperature = config.get(
        'action_selection_temperature', 1.0)

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

    # Mutexes for resources accessed by trainer and worker threads in async execution
    self.model_lock = threading.Lock()
    self.replay_buffer_lock = threading.Lock()

  def load_or_init_state(self, env, config, checkpoint_dict):
    assert 'dqn' in config['network'] or 'dueling_dqn' in config[
        'network'], "Invalid network configuration"

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

    # Noisy network initialization.
    if config.get('apply_noisy_network', False):
      self.model = replace_linear_with_noisy(self.model)
      self.target_model = replace_linear_with_noisy(self.target_model)

    # Set the target model to eval mode and keep it there
    self.target_model.eval()
    # Ensure target model parameters are not updated
    for param in self.target_model.parameters():
      param.requires_grad = False

    # Curiosity module
    self.curiosity_module = None
    if 'curiosity' in config:
      from src.curiosity import CuriosityModule
      self.curiosity_module = CuriosityModule(config['curiosity'], self.model)
      self.curiosity_module.to(self.device)
      print("Curiosity module initialized.")

    # Checkpointing
    if checkpoint_dict is not None:
      self.model.load_state_dict(checkpoint_dict['model'])
      self.target_model.load_state_dict(checkpoint_dict['model'])

    self.model.to(self.device)
    self.target_model.to(self.device)

    return lambda: {
        'model': self.model.state_dict(),
    }

  def parameters_to_optimize(self):
    return self.model.parameters()

  def remember(self, observation: List[Tuple[List, List]], action: List[int],
               reward: List[float], next_observation: List[Tuple[List, List]],
               done: List[bool], episode_start: List[bool]):
    """Stores experience. State gathered from last state sent to act().
    """
    with ProfileLockScope("replay_buffer_append", self.replay_buffer_lock):
      for i in range(len(reward)):
        if not episode_start[i]:
          self.replay_buffer.append(observation[i], action[i], reward[i],
                                    next_observation[i], done[i])

  def compute_target(self, all_reward: torch.Tensor,
                     all_next_observation: Tuple[torch.Tensor, torch.Tensor],
                     all_done: torch.Tensor) -> torch.Tensor:
    with torch.no_grad(), ProfileScope("compute_target"):
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
    with ProfileScope("compute_prediction"):
      q_values = self.model(all_observation, training=True)
    return torch.gather(q_values, 1, all_action.view(-1, 1)).squeeze(1)

  def replay(self):
    if len(self.replay_buffer) < self.config['min_memory_size']:
      return 0

    # Gather ids to fix memory_error later on
    # If this is importance sampling, we need to get the weights
    if isinstance(self.replay_buffer, PrioritizedExperienceReplayBuffer):
      # Sample from the replay buffer
      with ProfileLockScope('replay_buffer_sample', self.replay_buffer_lock):
        (all_observation, all_action, all_reward, all_next_observation,
         all_done
         ), importance_sampling, sampled_indices = self.replay_buffer.sample(
             self.batch_size)
        importance_sampling = torch.Tensor(importance_sampling).to(self.device)

        # Given we are using indices to update the relay buffer we need to do so while
        # holding the replay buffer lock in this case.
        target = self.compute_target(all_reward, all_next_observation,
                                     all_done)
        prediction = self.compute_prediction(all_observation, all_action)

        # Update the surprise values in the replay buffer
        self.replay_buffer.update_surprise(
            sampled_indices, (prediction - target).detach().cpu().numpy())

    else:
      # Sample from the replay buffer without importance sampling
      with ProfileLockScope('replay_buffer_sample', self.replay_buffer_lock):
        (all_observation, all_action, all_reward, all_next_observation,
         all_done) = self.replay_buffer.sample(self.batch_size)
      importance_sampling = torch.ones(self.batch_size, device=self.device)
      # For the non replay buffer the target and prediction are computed without
      # replay buffer lock.
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
    pre_clip_grad_norm = self.clip_gradients(self.model.parameters())
    with self.model_lock, ProfileScope('optimizer_step'):
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
      tau = self.config.get('target_model_soft_update_tau', 0.1)
      with torch.no_grad():
        for target_param, local_param in zip(self.target_model.parameters(),
                                             self.model.parameters()):
          target_param.data.copy_(tau * local_param.data +
                                  (1.0 - tau) * target_param.data)
      # Ensure target model stays in eval mode
      self.target_model.eval()

    self.summary_writer.add_scalar('Replay/Loss', loss)
    self.summary_writer.add_scalar('Replay/LearningRate',
                                   self.optimizer.param_groups[0]['lr'])
    self.summary_writer.add_scalar(
        "Replay/ParamNorm",
        torch.nn.utils.get_total_norm(self.model.parameters()))
    self.summary_writer.add_scalar("Replay/PreClipGradNorm",
                                   pre_clip_grad_norm)
    if 'clip_gradients' in self.config:
      self.summary_writer.add_scalar(
          "Replay/GradScaleWithLr",
          min(pre_clip_grad_norm, self.config['clip_gradients']) *
          self.optimizer.param_groups[0]['lr'])
    return self.batch_size

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

  def get_action(self,
                 observation: Observation) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the action to take based on the current observation.

    Returns:
      action: The action to take, either random or based on the model's prediction.
      act_values: The Q-values predicted by the model for the current observation.
    """

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
      with torch.no_grad(), ProfileScope("model_inference"):
        q_values = self.model(observation.as_input(self.device))
        q_values_np = q_values.detach().cpu().numpy()
        best_actions = np.argmax(q_values_np, axis=1).astype(int)

    # Perform epsilon-greedy action selection
    action = np.zeros(self.num_envs)
    random_action_idx = np.random.rand(self.num_envs) < self.epsilon
    action[random_action_idx] = np.random.randint(
        low=0, high=self.action_size, size=self.num_envs)[random_action_idx]
    action[~random_action_idx] = best_actions[~random_action_idx]

    self.summary_writer.add_scalar("Action/Epsilon", self.epsilon)

    return action, q_values_np
