"""
Core agent-environment interaction utilities.
"""

import numpy as np
import torch

from src.agent.value_agent import ValueAgent
from .training_utils import record_episode_statistics


def execute_agent_step(action, env_step, observation, summary_writer):
  """
    Execute a single step in the environment with the agent.
    
    Args:
        action: The action to take
        env_step: The environment step function
        observation: Current observation from the environment
        summary_writer: The summary writer for logging
        
    Returns:
        tuple: (experience_tuple) where experience_tuple contains
               (observation, action, reward, next_observation, done, info)

    Note:
        This function handles the core agent-environment interaction including
        curiosity rewards if enabled and episode statistics recording.
    """
  next_observation, reward, done, truncated, info = env_step(action)

  # TODO: next_observation will not be the end frame, but the first frame of the
  # next episode when reset happens. We need to pull the frame from 'info'.
  # TODO: Implement curiosity-driven exploration
  #if agent.curiosity_module:
  #  reward += agent.curiosity_reward(observation, next_observation, action,
  #                                   agent.device)

  done_or_truncated = np.logical_or(done, truncated)

  record_episode_statistics(summary_writer, done, truncated, info)

  return (observation, action, reward, next_observation, done_or_truncated,
          info)


def create_agent(config, env, summary_writer):
  """
  Create an agent for interacting with the environment.
  """
  device = torch.device(config['device'])
  print(f"Using device: {device}")
  agent = ValueAgent(env, device, summary_writer, config['agent'])
  print(f"Model summary: {agent.model}")
  print(f"Parameters: {sum(p.numel() for p in agent.model.parameters())}")
  return agent
