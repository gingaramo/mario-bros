import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import math
from src.dqn import DQN
from src.preprocess import StatePreprocess


# Define the agent class
class Agent:

  def __init__(self, action_size, device, config):
    self.action_size = action_size
    self.device = device
    self.config = config
    self.memory = []
    self.memory_size = config['memory_size']
    self.gamma = config['gamma']
    self.epsilon = config['epsilon']
    self.epsilon_min = config['epsilon_min']
    self.epsilon_decay = config['epsilon_decay']
    self.learning_rate = config['learning_rate']
    self.batch_size = config['batch_size']
    self.action_selection = config.get('action_selection', 'max')
    self.action_selection_temperature = config.get(
        'action_selection_temperature', 1.0)
    self.target_update_frequency = config.get('target_update_frequency')
    self.replays_until_target_update = self.target_update_frequency
    self.last_action = None
    self.last_action_repeated = 0
    self.replay_every_n_steps = config['replay_every_n_steps']
    self.preprocess = StatePreprocess(self.device, config['preprocess'])
    if 'dqn' in config['network']:
      self.model = DQN(action_size, self.preprocess(),
                       config['network']['dqn'])
      self.target_model = DQN(action_size, self.preprocess(),
                              config['network']['dqn'])
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

  def remember(self, action, reward, next_state, done):
    """Stores experience. State gathered from last state sent to act().
    """
    state_cpu = self.preprocess().cpu().detach().numpy()
    next_state_cpu = self.preprocess.preprocess(next_state)
    if isinstance(next_state_cpu, torch.Tensor):
      next_state_cpu = next_state_cpu.cpu().detach().numpy()
    self.memory.append((state_cpu, action, reward, next_state_cpu, done))
    if len(self.memory) > self.memory_size:
      self.memory.pop(random.randint(0, len(self.memory) - 1))

  def act(self, state: np.ndarray) -> tuple[int, np.ndarray]:
    """Returns the action to take based on the current state.
    
    Returns:
      action: The action to take, either random or based on the model's prediction.
      act_values: The Q-values predicted by the model for the current state.
    """
    self.preprocess.add(state)
    if self.config['action_repeat_steps'] > 1:
      if self.last_action_repeated == self.config['action_repeat_steps']:
        self.last_action_repeated = 0
        # And continue with a new action
      elif self.last_action is not None:
        self.last_action_repeated += 1
        # Repeat the last action
        return self.last_action, self.last_q_values

    action = None
    if np.random.rand() <= self.epsilon:
      action = random.randrange(self.action_size)
      act_values = np.zeros((self.action_size, ))
    else:
      act_values = self.model(self.preprocess().to(self.device))
      act_values_np = act_values.cpu().detach().numpy()
      if self.action_selection == 'softmax':
        # Softmax sampling
        exp_q = np.exp(act_values_np - np.max(act_values_np)
                       ) / self.action_selection_temperature
        probs = exp_q / np.sum(exp_q)
        action = np.random.choice(self.action_size, p=probs)
        act_values = probs
      elif self.action_selection == 'max':
        # Greedy action selection
        action = torch.argmax(act_values).item()
        act_values = act_values_np
      else:
        raise ValueError(
            f"Unsupported action selection method: {self.action_selection}. Supported: 'softmax', 'max'."
        )
    self.last_action = action
    self.last_q_values = act_values
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

  def replay(self, timestep):
    if len(self.memory) < self.batch_size:
      return
    if timestep % self.replay_every_n_steps != 0:
      return
    self.replays_until_target_update -= 1

    minibatch = random.sample(self.memory, self.batch_size)
    all_state, all_action, all_reward, all_next_state, all_done = zip(
        *minibatch)

    # Convert numpy arrays to tensors and move to device only here
    all_state = torch.tensor(np.stack(all_state),
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
      q_next, _ = torch.max(self.target_model(all_next_state), dim=1)
      target = all_reward + self.gamma * q_next * ~all_done

    q_pred = torch.gather(self.model(all_state), 1,
                          all_action.view(-1, 1)).squeeze(1)
    loss = self.get_loss()(q_pred, target)
    if self.replays_until_target_update <= 0:
      self.replays_until_target_update = self.target_update_frequency
      self.target_model.load_state_dict(self.model.state_dict())
    print(
        f"Log loss: {math.log(loss.item())} at timestep {timestep}. Epsilon: {self.epsilon}. Q-values: {q_pred.mean().item()}. Target: {target.mean().item()}. "
    )
    self.optimizer.zero_grad()
    loss.backward()
    self.clip_gradients()
    self.optimizer.step()

    if self.epsilon > self.epsilon_min:
      self.epsilon *= self.epsilon_decay
