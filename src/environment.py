import gymnasium as gym
from typing import Tuple, Optional, List
from collections import deque
import cv2
import torch
import numpy as np


class Observation(object):
  """
  Abstraction of an environment observation to manipulate through wrappers.

  An observation is composed by one or both of:
   - a pixel-frame
   - a dense vector

  For example, in Atari environments, the pixel-frame could be the RGB image
  representing the game state, and the dense vector could be the score,
  the latest action leading to the current frame.

  For non-Atari environments, like CartPole, there could be no pixel-frame,
  and the observation is just a dense vector representing the state; or it
  could be composed of the pixel-frame with dense vector of observations.

  Observations are used by the environment wrappers to return
  observations in a consistent way, regardless of the underlying environment.

  Example usage:
    obs = env.step(action)
    assert isinstance(obs, Observation)
    q_pred = model(obs.as_input())
  """

  def __init__(self,
               frame: Optional[np.ndarray] = None,
               dense: Optional[np.ndarray] = None) -> None:
    """
    Initializes an Observation instance.

    Args:
      frame: Optional pixel-frame as a numpy array in (C, H, W) format for RGB or (H, W) for grayscale.
             For history frames, can be (C, N, H, W) for RGB or (N, H, W) for grayscale (Conv3D compatible).
      dense: Optional dense vector as a numpy array (N, D) or (D,).
    """
    if frame is not None:
      # Convert from (H,W,C) to (C,H,W) if needed
      if len(
          frame.shape) == 3 and frame.shape[2] <= 16 and frame.shape[2] < min(
              frame.shape[:2]):
        # This looks like (H,W,C) format - convert to (C,H,W)
        frame = np.transpose(frame, (2, 0, 1))

      # Validate frame format - allow various shapes for different use cases
      if len(frame.shape) == 2:
        # Grayscale (H, W) is fine
        pass
      elif len(frame.shape) == 3:
        # Most common case: (C, H, W) or (N, H, W) for stacked frames
        pass
      elif len(frame.shape) == 4:
        # For batched frames from some environments
        pass
      else:
        raise ValueError(
            f"Frame must be 2D, 3D, or 4D array. Got shape {frame.shape}")

    assert (frame is not None) or (dense is not None), \
        "Observation must have at least one of frame or dense vector."

    self.frame = frame
    self.dense = dense

  def __repr__(self) -> str:
    return f"Observation(frame={self.frame.shape if self.frame is not None else None}, " \
           f"dense={self.dense.shape if self.dense is not None else None})"

  def as_input(self,
               device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the observation as a pair of numpy arrays suitable for model input.

    Returns:
      Tuple containing the frame and dense vector, using an empty vector in lieu of any missing component.
    """
    inputs = [
        torch.empty(0) if self.frame is None else self.frame,
        torch.empty(0) if self.dense is None else self.dense
    ]
    converted_inputs = [
        input.clone().detach().to(device).float() if torch.is_tensor(input)
        else torch.tensor(input, device=device, dtype=torch.float32)
        for input in inputs
    ]
    return tuple(converted_inputs)


class ObservationWrapper(gym.Wrapper):
  """
  A wrapper that returns an Observation object from the environment's step and reset methods.

  This wrapper is useful for environments that return either a pixel-frame or a dense vector,
  or both, allowing for a consistent interface across different environments.

  The wrapper converts raw environment observations (numpy arrays) into Observation objects
  based on the configured input type.

  config options:
    - input: str; "frame" (default) or "dense"
      - "frame": Treats environment output as pixel frames
      - "dense": Treats environment output as dense vectors
  """

  def __init__(self, env: gym.Env, config: dict) -> None:
    super().__init__(env)
    self.input_type = config.get('input', 'frame')
    if self.input_type not in ['frame', 'dense']:
      raise ValueError(
          f"Invalid input type '{self.input_type}'. Must be 'frame' or 'dense'."
      )

  def to_observation(self, obs: np.ndarray) -> Observation:
    """
    Converts a raw environment observation to an Observation object.
    
    Args:
      obs: Raw observation array from the environment
      
    Returns:
      Observation object with the array placed in either frame or dense field
      based on the configured input_type
    """
    if self.input_type == 'dense':
      return Observation(frame=None, dense=obs)
    elif self.input_type == 'frame':
      return Observation(frame=obs, dense=None)

  def step(self, action: int) -> Tuple[Observation, float, bool, bool, dict]:
    obs, reward, terminated, truncated, info = self.env.step(action)

    return self.to_observation(obs), reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[Observation, dict]:
    obs, info = self.env.reset(**kwargs)

    return self.to_observation(obs), info


class PreprocessFrameEnv(gym.Wrapper):
  """
  Postprocesses Observation 'frame' with common transformations.
  
  config options:
    - resize_shape: [int, int] or None ; if present, frame is resized
    - grayscale: bool
    - normalize: bool
  """

  resize_shape: Optional[Tuple[int, int]]
  grayscale: bool
  normalize: bool

  def __init__(self, env: gym.Env, config: dict) -> None:
    super().__init__(env)
    self.resize_shape = config.get('resize_shape', None)
    self.grayscale = bool(config.get('grayscale', False))
    self.normalize = bool(config.get('normalize', False))

  def preprocess(self, frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
      raise ValueError("Input frame must be a numpy ndarray.")

    # Convert from (C,H,W) to (H,W,C) for OpenCV processing if needed
    if len(frame.shape) == 3 and frame.shape[0] <= 4:
      frame = frame.transpose(1, 2, 0)

    if self.resize_shape:
      assert len(self.resize_shape) == 2
      frame = cv2.resize(frame,
                         self.resize_shape,
                         interpolation=cv2.INTER_AREA)
    if self.grayscale:
      if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if self.normalize:
      frame = frame.astype(np.float32) / 255.0

    # Convert back to (C,H,W) format if RGB, keep (H,W) for grayscale
    if len(frame.shape) == 3:
      frame = frame.transpose(2, 0, 1)

    return frame

  def step(self, action: int) -> Tuple[Observation, float, bool, bool, dict]:
    obs, reward, terminated, truncated, info = self.env.step(action)
    assert obs.frame is not None
    return Observation(frame=self.preprocess(obs.frame),
                       dense=obs.dense), reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[Observation, dict]:
    obs, info = self.env.reset(**kwargs)

    assert obs.frame is not None
    return Observation(frame=self.preprocess(obs.frame), dense=obs.dense), info


class RepeatActionEnv(gym.Wrapper):
  """
  Repeats the action given to step() as many times as configured.
  Returns early if 'truncated' or 'done'. 
  Returns latest 'state' and 'info' from step, and accumulated 'reward'.

  config options:
    - num_repeat_action: int ; actions will be repeated this many times
  """

  num_repeat_action: int

  def __init__(self, env: gym.Env, config: dict) -> None:
    super().__init__(env)
    self.num_repeat_action = config.get('num_repeat_action', 1)
    if self.num_repeat_action < 1:
      raise ValueError("num_repeat_action must be >= 1")

  def step(self, action: int) -> Tuple[Observation, float, bool, bool, dict]:
    total_reward = 0.0
    terminated = truncated = False
    obs = info = None
    for _ in range(self.num_repeat_action):
      obs, reward, terminated, truncated, info = self.env.step(action)
      total_reward += reward
      if terminated or truncated:
        break
    return obs, total_reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[Observation, dict]:
    obs, info = self.env.reset(**kwargs)

    return obs, info


class ReturnActionEnv(gym.Wrapper):
  """
  Appends the current action to the observation's dense vector.

  This wrapper takes the action that was just performed and appends it to the dense
  vector of the resulting observation. This allows the agent to have access to the
  action that led to the current state.

  config options:
    - mode: str ; "append" (default and only supported mode)
      - "append": Appends the current action as a dense parameter to the observation
  """

  def __init__(self, env: gym.Env, config: dict = None) -> None:
    super().__init__(env)
    config = config or {}
    self.mode = config.get('mode', 'append')
    if self.mode not in ['append']:
      raise ValueError(f"Invalid mode '{self.mode}'. Must be 'append'")

  def step(self, action: int) -> Tuple[Observation, float, bool, bool, dict]:
    obs, reward, terminated, truncated, info = self.env.step(action)

    if self.mode == 'append':
      # Create new observation with dense vector containing current action
      if obs.dense is not None:
        # Append current action to existing dense vector
        new_dense = np.concatenate((obs.dense, np.array([action])))
      else:
        # Create dense vector with just current action
        new_dense = np.array([action])
      return Observation(frame=obs.frame,
                         dense=new_dense), reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[Observation, dict]:
    obs, info = self.env.reset(**kwargs)

    if self.mode == 'append':
      # Create new observation with dense vector containing 0 (no previous action yet)
      if obs.dense is not None:
        # Append 0 to existing dense vector to represent no previous action
        new_dense = np.concatenate([obs.dense, np.array([0])])
      else:
        # Create dense vector with just 0 for no previous action
        new_dense = np.array([0])
      return Observation(frame=obs.frame, dense=new_dense), info


class HistoryEnv(gym.Wrapper):
  """
  Maintains a history of the last N frames.
  config options:
    - history_length: int ; if there are not enough, repeats oldest frame
  """

  history_length: int
  states: deque

  def __init__(self, env: gym.Env, config: dict) -> None:
    super().__init__(env)
    self.history_length = int(config.get('history_length', 1))
    if self.history_length < 1:
      raise ValueError("history_length must be >= 1")
    self.states = deque(maxlen=self.history_length)

  def _get_history(self) -> Observation:
    frames = list(self.states)
    if len(frames) == 0:
      raise RuntimeError("No frames in history.")

    # Extract frames and dense vectors separately
    frame_history = []
    dense_history = []

    for obs in frames:
      if obs.frame is not None:
        frame_history.append(obs.frame)
      if obs.dense is not None:
        dense_history.append(obs.dense)

    # Handle missing frames by padding with the first frame
    if frame_history:
      num_missing = self.history_length - len(frame_history)
      if num_missing > 0:
        first_frame = frame_history[0]
        pad_frames = [first_frame for _ in range(num_missing)]
        frame_history = pad_frames + frame_history

      # Stack frames for Conv3D compatibility
      if len(frame_history[0].shape) == 3:
        # RGB frames: (C, H, W) -> stack to get (C, N, H, W) for Conv3D
        stacked_frames = np.stack(frame_history, axis=1)
      else:
        # Grayscale frames: (H, W) -> stack to get (N, H, W) for Conv3D
        stacked_frames = np.stack(frame_history, axis=0)
    else:
      stacked_frames = None

    # Handle dense vectors - stack them as history too
    if dense_history:
      num_missing = self.history_length - len(dense_history)
      if num_missing > 0:
        first_dense = dense_history[0]
        pad_dense = [first_dense for _ in range(num_missing)]
        dense_history = pad_dense + dense_history
      flattened_dense = np.concatenate(dense_history)
    else:
      flattened_dense = None

    return Observation(frame=stacked_frames, dense=flattened_dense)

  def step(self, action: int) -> Tuple[Observation, float, bool, bool, dict]:
    obs, reward, terminated, truncated, info = self.env.step(action)
    self.states.append(obs)
    return self._get_history(), reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[Observation, dict]:
    self.states.clear()
    obs, info = self.env.reset(**kwargs)

    self.states.append(obs)
    return self._get_history(), info


class CaptureRenderFrameEnv(gym.Wrapper):
  """
  Captures the rendered frame from the environment after each step.

  The rendered frame can be accessed via the `rendered_frame` attribute (H,W,C).
  Returned observations always have their channel dimension first (C,H,W) to match
  the expected input format for Conv[2D|3D].

  config options:
    - mode: str ; "capture" (default) or "replace"
      - "capture": Only captures frame, returns original observation
      - "replace": Returns rendered frame as the observation, ignores original observation
      - "append": Appends rendered frame to the original observation (not implemented)
  """

  def __init__(self, env: gym.Env, config: dict = None) -> None:
    super().__init__(env)
    self.rendered_frame: Optional[np.ndarray] = None
    config = config or {}
    self.mode = config.get('mode', 'capture')
    if self.mode not in ['capture', 'replace']:
      raise ValueError(
          f"Invalid mode '{self.mode}'. Must be 'capture' or 'replace'")

  def step(self, action: int) -> Tuple[Observation, float, bool, bool, dict]:
    obs, reward, terminated, truncated, info = self.env.step(action)
    self.rendered_frame = self.env.render()
    assert self.rendered_frame is not None

    if self.mode == 'replace':
      return Observation(frame=self.rendered_frame,
                         dense=None), reward, terminated, truncated, info
    assert self.mode == 'capture'
    return obs, reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[Observation, dict]:
    obs, info = self.env.reset(**kwargs)

    self.rendered_frame = self.env.render()
    assert self.rendered_frame is not None

    if self.mode == 'replace':
      # For replace mode, we need to render the initial frame
      return Observation(frame=self.rendered_frame, dense=None), info
    else:  # mode == 'capture'
      return obs, info


def create_environment(config: dict) -> gym.Env:
  """ Creates a Gym environment with specified wrappers.
  Args:
    config: Dictionary containing environment configuration.
  Returns:
    A Gym environment instance with the specified wrappers applied.
  """
  env = gym.make(config['env_name'], render_mode='rgb_array')

  # Seed the environment if configured
  if 'seed' in config:
    env.reset(seed=config['seed'])

  # Always wrap with ObservationWrapper first to ensure Observation objects
  env = ObservationWrapper(
      env, config.get('ObservationWrapper', {'input': 'frame'}))

  for wrapper in config.get('env_wrappers', []):
    if wrapper == 'PreprocessFrameEnv':
      env = PreprocessFrameEnv(env, config.get('PreprocessFrameEnv', {}))
    elif wrapper == 'RepeatActionEnv':
      env = RepeatActionEnv(env, config.get('RepeatActionEnv', {}))
    elif wrapper == 'ReturnActionEnv':
      env = ReturnActionEnv(env, config.get('ReturnActionEnv', {}))
    elif wrapper == 'HistoryEnv':
      env = HistoryEnv(env, config.get('HistoryEnv', {}))
    elif wrapper == 'CaptureRenderFrameEnv':
      env = CaptureRenderFrameEnv(env, config.get('CaptureRenderFrameEnv', {}))
    elif wrapper == 'JoypadSpaceEnv':
      from gym_super_mario_bros.wrappers import JoypadSpace
      from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
      env = JoypadSpace(env, SIMPLE_MOVEMENT)
    else:
      raise ValueError(f"Unknown environment wrapper: {wrapper}")
  return env
