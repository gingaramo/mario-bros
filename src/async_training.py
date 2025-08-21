"""
Asynchronous training implementation with separate worker and trainer processes.
"""

import torch
import threading
from multiprocessing import Queue, Process
from tqdm import tqdm

from .agent import Agent
from .environment import create_environment
from .render import render, set_headless_mode
from .tb_logging import RemoteSummaryWriterClient, RemoteSummaryWriterServer
from .keyboard_controls import setup_interactive_controls, wait_for_frame_step
from .training_utils import create_summary_writer
from .agent_utils import execute_agent_step

# Queue size constants
EXPERIENCE_QUEUE_SIZE = 1000
PARAMETER_QUEUE_SIZE = 1
SUMMARY_WRITER_QUEUE_SIZE = 1000


def setup_async_worker_environment(config):
  """
    Setup the environment and agent for the async worker process.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (env, agent, pbar, observation) - initialized environment components
    """
  set_headless_mode(config['env'].get('headless', False))
  setup_interactive_controls()

  device = torch.device(config['device'])
  print(f"Worker using device: {device}")

  env = create_environment(config['env'])
  summary_writer = RemoteSummaryWriterClient(None)  # Will be set by caller
  agent = Agent(env, device, summary_writer, config['agent'])

  pbar = tqdm(total=config['env']['num_steps'],
              desc="Agent process",
              position=0)

  observation, _ = env.reset()
  return env, agent, pbar, observation


def run_async_worker_loop(env, agent, pbar, observation, experience_queue,
                          parameter_queue, config):
  """
    Main loop for the async worker process.
    
    Args:
        env: The environment
        agent: The RL agent
        pbar: Progress bar for tracking steps
        observation: Initial observation
        experience_queue: Queue for sending experiences to trainer
        parameter_queue: Queue for receiving model updates
        config: Configuration dictionary
    """
  while True:
    experience, action, q_values = execute_agent_step(agent, env, observation)
    experience_queue.put(experience)

    # Update the observation for the next step
    (_, _, _, next_observation, _, _) = experience
    observation = next_observation

    # Update progress bar
    pbar.update(env.num_envs)

    # Render frames if rendering is enabled or recording is active
    render(env, q_values, action, config)
    wait_for_frame_step()  # Debug frame-by-frame stepping

    # Update network parameters if available
    if not parameter_queue.empty():
      agent.model.load_state_dict(parameter_queue.get())

    if agent.global_step >= config['env']['num_steps']:
      print("Maximum number of steps reached.")
      break

  agent.save_checkpoint()
  env.close()
  pbar.close()


def async_worker_process(experience_queue, parameter_queue,
                         summary_writer_queue, config):
  """
    Main function for the async worker process that interacts with the environment.
    
    Args:
        experience_queue (Queue): Queue to send experiences to the trainer
        parameter_queue (Queue): Queue to receive model updates from trainer  
        summary_writer_queue (Queue): Queue for tensorboard logging
        config (dict): Configuration dictionary
        
    Note:
        This process handles environment interaction, rendering, and keyboard controls.
        It runs independently from the training process for better performance.
    """
  env, agent, pbar, observation = setup_async_worker_environment(config)

  # Connect to remote summary writer
  agent.summary_writer = RemoteSummaryWriterClient(summary_writer_queue)

  run_async_worker_loop(env, agent, pbar, observation, experience_queue,
                        parameter_queue, config)


def setup_async_trainer_environment(config):
  """
    Setup the training environment for the async trainer process.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (device, env, summary_writer, agent) - initialized training components
    """
  device = torch.device(config['device'])
  env = create_environment(config['env'])

  # The summary writer is created here to avoid issues with multiple processes
  # trying to write to the same file
  summary_writer = create_summary_writer(config)
  agent = Agent(env, device, summary_writer, config['agent'])

  print(f"Trainer using device: {device}")
  print(f"Model summary: {agent.model}")
  print(f"Parameters: {sum(p.numel() for p in agent.model.parameters())}")

  return device, env, summary_writer, agent


def start_summary_writer_server(summary_writer_queue, summary_writer):
  """
    Start the summary writer server in a separate thread.
    
    Args:
        summary_writer_queue (Queue): Queue for receiving logging requests
        summary_writer: The summary writer instance
        
    Returns:
        threading.Thread: The summary writer server thread
    """
  summary_writer_server = RemoteSummaryWriterServer(summary_writer_queue,
                                                    summary_writer)
  summary_writer_thread = threading.Thread(target=summary_writer_server.run)
  summary_writer_thread.daemon = True
  summary_writer_thread.start()
  return summary_writer_thread


def run_trainer_loop(agent, experience_queue, parameter_queue, config):
  """
    Main training loop for processing experiences and updating the model.
    
    Args:
        agent: The RL agent
        experience_queue (Queue): Queue containing experiences from worker
        parameter_queue (Queue): Queue for sending model updates to worker
        config (dict): Configuration dictionary
    """
  pbar = tqdm(total=None, desc="Trainer process", position=1)
  replays_until_model_update = config['replays_until_model_update']

  while True:
    # Process all available experiences
    while not experience_queue.empty():
      observation, action, reward, next_observation, done, info = experience_queue.get(
      )
      agent.remember(observation, action, reward, next_observation, done)

    # Train the agent if replay is successful
    if agent.replay():
      pbar.update(config['agent']['batch_size'])

      replays_until_model_update -= 1
      if replays_until_model_update <= 0:
        replays_until_model_update = config['replays_until_model_update']
        # Create a copy of model parameters for the worker process
        state_dict_copy = {
            k: v.clone().cpu()
            for k, v in agent.model.state_dict().items()
        }
        parameter_queue.put(state_dict_copy)


def async_trainer_process(experience_queue, parameter_queue,
                          summary_writer_queue, config):
  """
    Main function for the async trainer process that handles model training.
    
    Args:
        experience_queue (Queue): Queue to receive experiences from worker
        parameter_queue (Queue): Queue to send model updates to worker
        summary_writer_queue (Queue): Queue for tensorboard logging
        config (dict): Configuration dictionary
        
    Note:
        This process handles all training operations including experience replay,
        model updates, and tensorboard logging. It runs independently from
        environment interaction for better performance.
    """
  device, env, summary_writer, agent = setup_async_trainer_environment(config)

  # Start the summary writer server thread
  summary_writer_thread = start_summary_writer_server(summary_writer_queue,
                                                      summary_writer)

  # Run the main training loop
  run_trainer_loop(agent, experience_queue, parameter_queue, config)


def run_async_training(config):
  """
    Coordinate async training with separate worker and trainer processes.
    
    Args:
        config (dict): Configuration dictionary
        
    Note:
        Creates separate processes for environment interaction (worker) and
        model training (trainer) connected via queues for better performance.
        The worker process handles environment stepping and rendering while
        the trainer process focuses on experience replay and model updates.
    """
  # Create communication queues between processes
  experience_queue = Queue(maxsize=EXPERIENCE_QUEUE_SIZE)
  parameter_queue = Queue(maxsize=PARAMETER_QUEUE_SIZE)
  summary_writer_queue = Queue(maxsize=SUMMARY_WRITER_QUEUE_SIZE)

  # Create and start worker and trainer processes
  worker_process = Process(target=async_worker_process,
                           args=(experience_queue, parameter_queue,
                                 summary_writer_queue, config))
  trainer_process = Process(target=async_trainer_process,
                            args=(experience_queue, parameter_queue,
                                  summary_writer_queue, config))

  worker_process.start()
  trainer_process.start()

  try:
    # Wait for worker to complete (reaches max steps)
    worker_process.join()
  finally:
    # Clean up resources
    experience_queue.close()
    parameter_queue.close()
    trainer_process.terminate()
