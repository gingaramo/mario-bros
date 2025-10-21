"""
Synchronous training implementation with combined worker/trainer process.
"""

import numpy as np
from tqdm import tqdm
from typing import Tuple, List

from src.profiler import ProfileScope

from .environment import create_environment
from .render import render
from .keyboard_controls import wait_for_frame_step
from .training_utils import create_summary_writer
from .agent_utils import execute_agent_step, create_agent
from .tb_logging import DummySummaryWriter


def evaluate_agent(config,
                   num_episodes) -> Tuple[List[List[float]], List[int]]:
  """
    Evaluate the trained agent on a set number of episodes.

    Args:
        config (dict): Configuration dictionary
        num_episodes (int): Number of episodes to evaluate
    """
  # Create environment, summary writer, and agent
  env = create_environment(config['env'])
  summary_writer = DummySummaryWriter()
  agent = create_agent(config, env, summary_writer)

  evaluate_pbar = tqdm(total=num_episodes,
                       desc="Model Evaluation",
                       position=0,
                       unit=' episode',
                       unit_scale=True)

  observation, _ = env.reset()
  accumulated_reward = []
  episode_steps = []
  while num_episodes > 0:
    with ProfileScope("agent_act"):
      action, q_values = agent.act(observation)
    with ProfileScope("execute_agent_step"):
      experience = execute_agent_step(action, lambda action: env.step(action),
                                      observation, agent.summary_writer)

    (_, action, reward, observation, done, info) = experience

    # Store experience if we're not at the begining of an episode
    terminated_idx = 0
    for i, don in enumerate(done):
      if don:
        evaluate_pbar.update(1)
        accumulated_reward.append(info['terminated_accumulated_reward'][terminated_idx])
        episode_steps.append(info['terminated_episode_steps'][terminated_idx])
        terminated_idx += 1
        num_episodes -= 1

    # Render frames if rendering is enabled or recording is active
    render(info, q_values, action, reward, config)

  env.close()
  evaluate_pbar.close()
  return accumulated_reward, episode_steps
