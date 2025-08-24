"""
Synchronous training implementation with combined worker/trainer process.
"""

import torch
from tqdm import tqdm

from src.profiler import ProfileScope, execution_profiler_singleton

from .agent import Agent
from .environment import create_environment
from .render import render
from .keyboard_controls import setup_interactive_controls, wait_for_frame_step
from .training_utils import create_summary_writer
from .agent_utils import execute_agent_step


def run_sync_training(config):
  """
    Run synchronous training with combined worker/trainer in single process.
  
    Args:
        config (dict): Configuration dictionary
        
    Note:
        In synchronous mode, environment interaction and training happen
        sequentially in the same process. This is simpler but potentially
        slower than async mode since environment stepping blocks training.
    """
  setup_interactive_controls()

  device = torch.device(config['device'])
  print(f"Using device: {device}")
  env = create_environment(config['env'])

  global execution_profiler_singleton
  execution_profiler_singleton.set_name(f"{config['agent']['name']}")

  # Create summary writer for logging
  summary_writer = create_summary_writer(config)
  agent = Agent(env, device, summary_writer, config['agent'])
  print(f"Model summary: {agent.model}")
  print(f"Parameters: {sum(p.numel() for p in agent.model.parameters())}")

  pbar = tqdm(total=config['env']['num_steps'],
              desc="Synchronous Worker/Trainer process",
              position=0)

  observation, _ = env.reset()
  while True:
    with ProfileScope("env_step"):
      experience, action, q_values = execute_agent_step(
          agent, env, observation)

    # Store experience and train immediately (synchronous)
    (observation, action, reward, next_observation, done, info) = experience
    agent.remember(observation, action, reward, next_observation, done)
    with ProfileScope("agent_replay"):
      agent.replay()
    observation = next_observation

    # Update progress bar
    pbar.update(env.num_envs)

    # Render frames if rendering is enabled or recording is active
    render(env, q_values, action, config)
    wait_for_frame_step()  # Debug frame-by-frame stepping

    if agent.global_step >= config['env']['num_steps']:
      print("Maximum number of steps reached.")
      break

  agent.save_checkpoint()
  env.close()
  pbar.close()
