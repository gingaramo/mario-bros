"""
Asynchronous training implementation with separate worker and trainer processes.
"""

import numpy as np
import time
import threading
from tqdm import tqdm

from src.agent import Agent

from .environment import create_environment
from .render import render
from .keyboard_controls import wait_for_frame_step
from .training_utils import create_summary_writer
from .agent_utils import execute_agent_step, create_agent
from .profiler import ProfileScope


def async_worker_thread(config, agent, worker_id=0):
  """
    Main function for the async worker thread that interacts with the environment.

    Args:
        config (dict): Configuration dictionary
        agent (Agent): The agent instance
        worker_id (int): The ID of the worker thread

    Note:
        This thread handles agent inference, environment interaction, rendering, and
        keyboard controls. It runs independently from the training thread for better
        performance.
    """
  pbar = tqdm(total=config['env']['num_steps'] /
              config['env'].get('num_env_workers', 1),
              desc="Asynchronous Worker",
              position=worker_id + 1,
              unit=' experiences',
              unit_scale=True)

  # We only run synchronous environment for the worker executed by the main thread
  # that is the one with access to GUI and frame rendering logic.
  env = create_environment(
      config['env'], mode='asynchronous' if worker_id > 0 else 'synchronous')
  observation, info = env.reset()
  episode_start = np.zeros(env.num_envs, dtype=bool)

  while agent.global_step < config['env']['num_steps']:
    with ProfileScope("agent_act"):
      action, q_values = agent.act(observation)
    with ProfileScope("execute_agent_step"):
      experience = execute_agent_step(action, lambda action: env.step(action),
                                      observation, agent.summary_writer)

      # Update progress bar
      pbar.update(env.num_envs)

    (observation, action, reward, next_observation, done, info) = experience

    # Store experience if we're not at the begining of an episode
    for i, (obs, act, rew, neo, don) in enumerate(
        zip(observation.as_list_input('cpu'), action, reward,
            next_observation.as_list_input('cpu'), done)):
      if not episode_start[i]:
        agent.remember(obs, act, rew, neo, don)
    observation = next_observation
    episode_start = done

    # Render frames if rendering is enabled or recording is active.
    # Note: Only worker_id 0 is run in the main process thread, and is allowed to
    # render GUI.
    if worker_id == 0:
      render(info, q_values, action, config)
    wait_for_frame_step()  # Debug frame-by-frame stepping

  print("Maximum number of steps reached.")

  agent.save_checkpoint()
  pbar.close()


def async_trainer_thread(config, agent: Agent, stop_event):
  """
    Main function for the async trainer process that handles model training.
    
    Args:
        config (dict): Configuration dictionary
        agent (Agent): The agent instance
        stop_event (threading.Event): Event to signal the thread to stop

    Note:
        This process handles all training operations including experience replay,
        model updates, and tensorboard logging. It runs independently from
        environment interaction for better GPU performance.
    """
  # Run the main training loop
  pbar = tqdm(total=None,
              desc="Trainer thread",
              position=0,
              unit=' experiences',
              unit_scale=True)

  while stop_event.is_set() is False:
    # Train the agent if replay is successful
    with ProfileScope("agent_replay"):
      if agent.replay():
        ProfileScope.add_metadata('batch_size', config['agent']['batch_size'])
        pbar.update(config['agent']['batch_size'])
      else:
        # We have not yet accumulated enough experiences.
        time.sleep(0.001)


def validate_config(config):
  assert config['env']['num_envs'] % config['env'].get(
      'num_env_workers', 1
  ) == 0, "Number of environments must be divisible by number of workers. " +\
  f"Got {config['env']['num_envs']} and {config['env'].get('num_env_workers', 1)}"


def run_async_training(config):
  """
    Coordinate async training with separate worker and trainer processes.
    
    Args:
        config (dict): Configuration dictionary

    Note:
        Creates separate processes for environment interaction (worker) and
        model training (trainer) connected via multiprocessing queues.
        The worker process handles environment stepping and rendering while
        the trainer process focuses on experience replay and model updates.
    """
  validate_config(config)

  # Update config for multiple workers:
  config['env']['num_envs'] = config['env']['num_envs'] // config['env'].get(
      'num_env_workers', 1)

  # Create and start and trainer processes.
  # 'env' is unused here, but needed for agent to initialize the agent,
  # workers will create their own copies (one sync, and many async).
  unused_env = create_environment(config['env'])
  summary_writer = create_summary_writer(config)
  agent = create_agent(config, unused_env, summary_writer)
  stop_event = threading.Event()
  trainer_thread = threading.Thread(target=async_trainer_thread,
                                    args=(config, agent, stop_event))
  trainer_thread.start()

  # If we asked for N workers, N-1 workers will run in threads.
  worker_threads = []
  for i in range(config['env'].get('num_env_workers', 1) - 1):
    worker_thread = threading.Thread(target=async_worker_thread,
                                     args=(config, agent, i + 1))
    worker_thread.start()
    worker_threads.append(worker_thread)

  # Run worker_id 0, which might render GUI, in the main process thread.
  async_worker_thread(config, agent, worker_id=0)

  # Wait for all worker threads to finish
  for worker_thread in worker_threads:
    worker_thread.join()

  # When we are done with env steps, wait for the trainer thread to finish
  stop_event.set()
  trainer_thread.join()
