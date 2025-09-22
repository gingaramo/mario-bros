"""
Synchronous training implementation with combined worker/trainer process.
"""

import numpy as np
from tqdm import tqdm

from src.profiler import ProfileScope

from .environment import create_environment
from .render import render
from .keyboard_controls import wait_for_frame_step
from .training_utils import create_summary_writer
from .agent_utils import execute_agent_step, create_agent


def run_sync_training(config):
  """
    Run synchronous training with combined worker/trainer in single process/thread.

    Args:
        config (dict): Configuration dictionary

    Note:
        In synchronous mode, environment interaction and training happen
        sequentially in the same process/thread. This is simpler but potentially
        slower than async mode since environment stepping blocks training.
    """
  # Create environment, summary writer, and agent
  env = create_environment(config['env'], 'asynchronous')
  summary_writer = create_summary_writer(config)
  agent = create_agent(config, env, summary_writer)

  train_pbar = tqdm(total=config['env']['num_steps'] //
                    config['env']['num_envs'],
                    desc="Synchronous Trainer",
                    position=0,
                    unit=' experiences',
                    unit_scale=True)
  env_pbar = tqdm(total=config['env']['num_steps'],
                  desc="Synchronous Worker",
                  position=1,
                  unit=' experiences',
                  unit_scale=True,
                  initial=agent.global_step)

  observation, _ = env.reset()
  episode_start = np.zeros(env.num_envs, dtype=bool)
  while agent.global_step < config['env']['num_steps']:
    with ProfileScope("agent_act"):
      action, q_values = agent.act(observation)
    with ProfileScope("execute_agent_step"):
      experience = execute_agent_step(action, lambda action: env.step(action),
                                      observation, agent.summary_writer)
      # Update progress bar
      env_pbar.update(env.num_envs)
    (observation, action, reward, next_observation, done, info) = experience

    agent.remember(observation.as_list_input('cpu'), action, reward,
                   next_observation.as_list_input('cpu'), done, episode_start)
    observation = next_observation
    episode_start = done

    # Render frames if rendering is enabled or recording is active
    render(info, q_values, action, config)
    wait_for_frame_step()  # Debug frame-by-frame stepping

    with ProfileScope("agent_train"):
      trained_experiences = agent.train()
      if trained_experiences:
        ProfileScope.add_metadata('batch_size', trained_experiences)
        train_pbar.update(trained_experiences)

  print("Maximum number of steps reached.")

  agent.save_checkpoint()
  env.close()
  env_pbar.close()
  train_pbar.close()
