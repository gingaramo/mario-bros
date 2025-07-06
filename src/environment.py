import gymnasium as gym
from typing import Tuple, Optional, List
from collections import deque
import cv2
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
    q_pred = model(obs.frame, obs.dense)
  """

  def __init__(self,
               frame: Optional[np.ndarray] = None,
               dense: Optional[np.ndarray] = None) -> None:
    """
    Initializes an Observation instance.

    Args:
      frame: Optional pixel-frame as a numpy array (N, H, W, C) or (N, H, W) for grayscale.
      dense: Optional dense vector as a numpy array (N, D) or (D,).
    """
    self.frame = frame
    self.dense = dense
    assert (self.frame is not None) or (self.dense is not None), \
        "Observation must have at least one of frame or dense vector."

  def __repr__(self) -> str:
    return f"Observation(frame={self.frame.shape if self.frame is not None else None}, " \
           f"dense={self.dense.shape if self.dense is not None else None})"


class ObservationWrapper(gym.Wrapper):
  """
  A wrapper that returns an Observation object from the environment's step and reset methods.

  This wrapper is useful for environments that return either a pixel-frame or a dense vector,
  or both, allowing for a consistent interface across different environments.

  config options:
    - input ; str ; "frame", "dense"
  """

  def __init__(self, env: gym.Env, config: dict) -> None:
    super().__init__(env)
    self.input_type = config.get('input', 'frame')
    if self.input_type not in ['frame', 'dense']:
      raise ValueError(
          f"Invalid input type '{self.input_type}'. Must be 'frame' or 'dense'."
      )

  def step(self, action: int) -> Tuple[Observation, float, bool, bool, dict]:
    frame, reward, terminated, truncated, info = self.env.step(action)
    return Observation(frame=frame), reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[Observation, dict]:
    frame, info = self.env.reset(**kwargs)
    return Observation(frame=frame), info


class PreprocessFrameEnv(gym.Wrapper):
  """
  Postprocesses pixel-frame step() observations with common transformations.
  
  config options:
    - resize_shape: (int, int) or None ; if present, frame is resized
    - grayscale: bool
    - normalize: bool
  """

  resize_shape: Optional[Tuple[int, int]]
  grayscale: bool
  normalize: bool

  def __init__(self, env: gym.Env, config: dict) -> None:
    super().__init__(env)
    self.resize_shape = config.get('resize_shape', None)
    if self.resize_shape is not None:
      self.resize_shape = tuple(self.resize_shape)
    self.grayscale = bool(config.get('grayscale', False))
    self.normalize = bool(config.get('normalize', False))

  def preprocess(self, frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
      raise ValueError("Input frame must be a numpy ndarray.")
    if self.resize_shape is not None:
      if not (isinstance(self.resize_shape, tuple)
              and len(self.resize_shape) == 2):
        raise ValueError("resize_shape must be a tuple of (int, int)")
      frame = cv2.resize(frame,
                         self.resize_shape,
                         interpolation=cv2.INTER_AREA)
    if self.grayscale:
      if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    if self.normalize:
      frame = frame.astype(np.float32) / 255.0
    # Keep grayscale as 2D array (H, W) without channel dimension
    return frame

  def step(self, action: int) -> Tuple[Observation, float, bool, bool, dict]:
    obs, reward, terminated, truncated, info = self.env.step(action)
    # Process existing frame if it exists
    processed_frame = self.preprocess(
        obs.frame) if obs.frame is not None else None
    return Observation(frame=processed_frame,
                       dense=obs.dense), reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[Observation, dict]:
    obs, info = self.env.reset(**kwargs)
    # Process existing frame if it exists
    processed_frame = self.preprocess(
        obs.frame) if obs.frame is not None else None
    return Observation(frame=processed_frame, dense=obs.dense), info


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
    self.num_repeat_action = int(config.get('num_repeat_action', 1))
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
    return self.env.reset(**kwargs)


class ReturnActionEnv(gym.Wrapper):
  """
  Manages the previous action information.

  config options:
    - mode: str ; "capture" (default) or "append"
      - "capture": Stores previous action in info dict
      - "append": Returns tuple of (obs, prev_action) as observation
  """

  prev_action: Optional[int]

  def __init__(self, env: gym.Env, config: dict = None) -> None:
    super().__init__(env)
    self.prev_action = None
    config = config or {}
    self.mode = config.get('mode', 'capture')
    if self.mode not in ['capture', 'append']:
      raise ValueError(
          f"Invalid mode '{self.mode}'. Must be 'capture' or 'append'")

  def step(self, action: int) -> Tuple[Observation, float, bool, bool, dict]:
    obs, reward, terminated, truncated, info = self.env.step(action)
    info = dict(info) if info is not None else {}

    if self.mode == 'capture':
      info['prev_action'] = self.prev_action
      self.prev_action = action
      return obs, reward, terminated, truncated, info
    else:  # mode == 'append'
      prev_action_to_return = self.prev_action
      self.prev_action = action
      # Create new observation with dense vector containing previous action
      if obs.dense is not None:
        # Append previous action to existing dense vector
        prev_action_array = np.array(
            [prev_action_to_return]
            if prev_action_to_return is not None else [0])
        new_dense = np.concatenate([obs.dense, prev_action_array])
      else:
        # Create dense vector with just previous action
        new_dense = np.array([prev_action_to_return]
                             if prev_action_to_return is not None else [0])
      return Observation(frame=obs.frame,
                         dense=new_dense), reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[Observation, dict]:
    obs, info = self.env.reset(**kwargs)
    self.prev_action = None

    if self.mode == 'capture':
      return obs, info
    else:  # mode == 'append'
      # Create new observation with dense vector containing None (represented as 0)
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
      stacked_dense = np.stack(dense_history, axis=0)
    else:
      stacked_dense = None

    return Observation(frame=stacked_frames, dense=stacked_dense)

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

  config options:
    - mode: str ; "capture" (default), "replace", or "append"
      - "capture": Only captures frame, returns original observation
      - "replace": Returns rendered frame as the observation
      - "append": Returns tuple of (rendered_frame, original_obs)
  """

  def __init__(self, env: gym.Env, config: dict = None) -> None:
    super().__init__(env)
    self.rendered_frame: Optional[np.ndarray] = None
    config = config or {}
    self.mode = config.get('mode', 'capture')
    if self.mode not in ['capture', 'replace', 'append']:
      raise ValueError(
          f"Invalid mode '{self.mode}'. Must be 'capture', 'replace', or 'append'"
      )

  def step(self, action: int) -> Tuple[Observation, float, bool, bool, dict]:
    obs, reward, terminated, truncated, info = self.env.step(action)
    self.rendered_frame = self.env.render(mode='rgb_array')

    if self.mode == 'replace':
      return Observation(frame=self.rendered_frame.permuted(2, 0, 1),
                         dense=obs.dense), reward, terminated, truncated, info
    elif self.mode == 'append':
      # Combine rendered frame with original observation frame
      if obs.frame is not None:
        # Stack rendered frame with original frame
        combined_frame = np.stack([self.rendered_frame, obs.frame], axis=0)
      else:
        # Just use rendered frame
        combined_frame = self.rendered_frame
      return Observation(frame=combined_frame,
                         dense=obs.dense), reward, terminated, truncated, info
    else:  # mode == 'capture'
      return obs, reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[Observation, dict]:
    obs, info = self.env.reset(**kwargs)
    self.rendered_frame = None

    if self.mode == 'replace':
      # For replace mode, we need to render the initial frame
      self.rendered_frame = self.env.render()
      return Observation(frame=self.rendered_frame, dense=obs.dense), info
    elif self.mode == 'append':
      # For append mode, we need to render the initial frame and combine
      self.rendered_frame = self.env.render()
      if obs.frame is not None:
        # Stack rendered frame with original frame
        combined_frame = np.stack([self.rendered_frame, obs.frame], axis=0)
      else:
        # Just use rendered frame
        combined_frame = self.rendered_frame
      return Observation(frame=combined_frame, dense=obs.dense), info
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
    else:
      raise ValueError(f"Unknown environment wrapper: {wrapper}")
  return env
