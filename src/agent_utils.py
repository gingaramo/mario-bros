"""
Core agent-environment interaction utilities.
"""

import numpy as np
from .training_utils import record_episode_statistics


def execute_agent_step(agent, env, observation):
  """
    Execute a single step in the environment with the agent.
    
    Args:
        agent: The RL agent
        env: The environment
        observation: Current observation from the environment
        
    Returns:
        tuple: (experience_tuple, action, q_values) where experience_tuple
               contains (observation, action, reward, next_observation, done, info)
               
    Note:
        This function handles the core agent-environment interaction including
        curiosity rewards if enabled and episode statistics recording.
    """
  action, q_values = agent.act(observation)
  next_observation, reward, done, truncated, info = env.step(action)

  # TODO: next_observation will not be the end frame, but the first frame of the
  # next episode when reset happens. We need to pull the frame from 'info'.
  if agent.curiosity_module:
    reward += agent.curiosity_reward(observation, next_observation, action,
                                     agent.device)

  done_or_truncated = np.logical_or(done, truncated)

  record_episode_statistics(agent, done, truncated, info, env.num_envs)

  experience = (observation, action, reward, next_observation,
                done_or_truncated, info)
  return experience, action, q_values
