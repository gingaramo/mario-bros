import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
from torch.utils.tensorboard import SummaryWriter
from src.dqn import DQN
from src.state import State
import pickle


class Agent:

  def __init__(self, action_size, device, config):
    self.action_size = action_size
    self.device = device
    self.config = config

    # Replay parameters.
    self.replay_every_n_steps = config['replay_every_n_steps']
    self.memory_size = config['memory_size']
    self.memory = []
    self.memory_error = []  # Remembers error of sample
    self.batch_size = config['batch_size']
    self.target_update_frequency = config.get('target_update_frequency')
    self.replays_until_target_update = self.target_update_frequency

    # Learning parameters.
    self.gamma = config['gamma']
    self.epsilon_min = config['epsilon_min']
    self.epsilon_decay = config['epsilon_decay']
    self.learning_rate = config['learning_rate']

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
                                        purge_step=self.global_step)

    # Action selection parameters.
    self.action_selection = config.get('action_selection', 'max')
    self.action_selection_temperature = config.get(
        'action_selection_temperature', 1.0)
    # Action repeat settings
    self.action_repeat_steps = config.get('action_repeat_steps', None)
    self.step = 0
    # Episode being recorded? Helps avoid state updates and checkpoints
    self.recording = False

    self.state = State(self.device, config['state'])
    if 'dqn' in config['network']:
      self.model = DQN(action_size, self.state.current(),
                       config['network']['dqn'])
      self.target_model = DQN(action_size, self.state.current(),
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

  def remember(self, action: int, reward: int, next_state: np.ndarray,
               done: bool):
    """Stores experience. State gathered from last state sent to act().
    """
    curr_state = self.state.current().cpu().detach().numpy()
    next_state = self.state.preprocess(next_state)
    self.summary_writer.add_scalar('Memory/Size', len(self.memory),
                                   self.global_step)
    self.summary_writer.add_scalar('Memory/Reward', reward, self.global_step)

    # Skip memories that are too similar
    if len(self.memory) > 1 and math.fabs(self.memory[-1][3] - reward) < 0.1:
      return
    # Store memory with how bad we estimated
    self.memory.append(
        (curr_state, self.last_action, action, reward, next_state, done))
    self.memory_error.append(math.fabs(self.last_q_values[action] - reward))
    self.summary_writer.add_scalar('Memory/Estimation Error',
                                   self.memory_error[-1], self.global_step)

    if len(self.memory) > self.memory_size:
      to_remove = random.randint(0, len(self.memory) - 1)
      self.memory.pop(to_remove)
      self.memory_error.pop(to_remove)

  def act(self, state: np.ndarray) -> tuple[int, np.ndarray]:
    """Returns the action to take based on the current state.
    
    Returns:
      action: The action to take, either random or based on the model's prediction.
      act_values: The Q-values predicted by the model for the current state.
    """
    self.state.add(state)
    if not self.recording:
      # We only update global step when we're training, not recording.
      self.global_step += 1
    if self.action_repeat_steps and self.last_action != None and self.last_action_repeated < self.action_repeat_steps:
      self.last_action_repeated += 1
      # Repeat the last action
      return self.last_action, self.last_act_values
    self.last_action_repeated = 0

    action = None
    q_values = self.model(
        self.state.current().to(self.device),
        torch.tensor([self.last_action or 0]).to(self.device))
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
    self.last_action = action
    self.last_q_values = q_values
    self.last_act_values = act_values
    return action, act_values

  def get_loss(self):
    if self.config['loss'] == 'mse':
      self.get_loss = nn.MSELoss
    elif self.config['loss'] == 'smooth_l1':
      self.get_loss = nn.SmoothL1Loss
    else:
      raise ValueError(f"Unsupported loss function: {self.config['loss']}")
    return self.get_loss()

  def clip_gradients(self):
    if self.config['clip_gradients'] is not None:
      nn.utils.clip_grad_norm_(self.model.parameters(),
                               self.config['clip_gradients'])

  def replay(self):
    if self.recording:
      return
    if len(self.memory) < self.batch_size:
      return
    if self.global_step % self.replay_every_n_steps != 0:
      return
    # Gather ids to fix memory_error later on
    minibatch_ids = random.choices(range(len(self.memory)),
                                   weights=self.memory_error,
                                   k=self.batch_size)
    # Gather memories from self.memory using indices in minibatch_ids
    minibatch = [self.memory[i] for i in minibatch_ids]
    all_state, all_last_action, all_action, all_reward, all_next_state, all_done = zip(
        *minibatch)

    # Convert numpy arrays to tensors and move to device only here
    all_state = torch.tensor(np.stack(all_state),
                             dtype=torch.float).to(self.device)
    all_last_action = torch.tensor(all_last_action,
                                   dtype=torch.float).to(self.device)
    all_action = torch.tensor(all_action, dtype=torch.int64).to(self.device)
    all_reward = torch.tensor(all_reward, dtype=torch.float).to(self.device)
    all_next_state = torch.tensor(np.stack(all_next_state),
                                  dtype=torch.float).to(
                                      self.device).unsqueeze(1)
    all_next_state = torch.concat([all_state[:, 1:, :], all_next_state],
                                  axis=1)
    all_done = torch.tensor(all_done, dtype=torch.bool).to(self.device)

    with torch.no_grad():
      q_next, _ = torch.max(self.target_model(all_next_state,
                                              all_action.view(-1, 1)),
                            dim=1)
      target = all_reward + self.gamma * q_next * ~all_done

    q_values = self.model(all_state, all_last_action.view(-1, 1))
    q_pred = torch.gather(q_values, 1, all_action.view(-1, 1)).squeeze(1)
    new_memory_error = (q_pred.flatten().cpu().detach().numpy() -
                        target.flatten().cpu().detach().numpy())
    loss = self.get_loss()(q_pred, target)
    self.summary_writer.add_scalar('Replay/Loss', loss, self.global_step)
    self.summary_writer.add_scalar('Replay/Q-mean',
                                   torch.mean(q_values.flatten()),
                                   self.global_step)
    self.summary_writer.add_scalar('Replay/Epsilon', self.epsilon,
                                   self.global_step)

    # Update target model every `target_update_frequency` steps
    self.replays_until_target_update -= 1
    if self.replays_until_target_update <= 0:
      self.replays_until_target_update = self.target_update_frequency
      self.target_model.load_state_dict(self.model.state_dict())

    self.optimizer.zero_grad()
    loss.backward()
    self.summary_writer.add_scalar(
        "Replay/Norm", torch.nn.utils.get_total_norm(self.model.parameters()),
        self.global_step)
    self.clip_gradients()
    self.optimizer.step()

    # Update memory error with new memory error
    for idx, err in zip(minibatch_ids, np.abs(new_memory_error)):
      self.memory_error[idx] = err

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay

  def episode_begin(self, recording=False):
    self.last_action = None
    self.last_action_repeated = 0
    self.recording = recording

  def episode_end(self, episode_info):
    if self.recording:
      return
    # Episode statistics
    self.summary_writer.add_scalar("Episode/Episode", episode_info['episode'],
                                   self.global_step)
    self.summary_writer.add_scalar("Episode/Reward",
                                   episode_info['total_reward'],
                                   self.global_step)
    self.summary_writer.add_scalar("Episode/World", episode_info['world'],
                                   self.global_step)
    self.summary_writer.add_scalar("Episode/Stage", episode_info['stage'],
                                   self.global_step)
    self.summary_writer.add_scalar("Episode/Steps", episode_info['steps'],
                                   self.global_step)

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
