import gymnasium as gym
from typing import Tuple, Optional, List
from collections import deque
import cv2
import numpy as np


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

  def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
    frame, reward, terminated, truncated, info = self.env.step(action)
    return self.preprocess(frame), reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
    frame, info = self.env.reset(**kwargs)
    return self.preprocess(frame), info


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

  def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
    total_reward = 0.0
    terminated = truncated = False
    state = info = None
    for _ in range(self.num_repeat_action):
      state, reward, terminated, truncated, info = self.env.step(action)
      total_reward += reward
      if terminated or truncated:
        break
    return state, total_reward, terminated, truncated, info

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)


class ReturnActionEnv(gym.Wrapper):
  """
  Appends the previous action to the returned value of step.

  config options:
    - n/a
  """

  prev_action: Optional[int]

  def __init__(self, env: gym.Env, config: dict = None) -> None:
    super().__init__(env)
    self.prev_action = None

  def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
    obs, reward, terminated, truncated, info = self.env.step(action)
    info = dict(info) if info is not None else {}
    info['prev_action'] = self.prev_action
    self.prev_action = action
    return obs, reward, terminated, truncated, info

  def reset(self, **kwargs):
    self.prev_action = None
    return self.env.reset(**kwargs)


class HistoryEnv(gym.Wrapper):
  """
  Maintains a history of the last N frames (with optional frame skipping).
  config options:
    - history_length: int ; if there are not enough, repeats oldest frame
    - num_skip_frames: int ; if > 0, each step(action) is repeated these many times
  """

  history_length: int
  num_skip_frames: int
  states: deque

  def __init__(self, env: gym.Env, config: dict) -> None:
    super().__init__(env)
    self.history_length = int(config.get('history_length', 1))
    if self.history_length < 1:
      raise ValueError("history_length must be >= 1")
    self.num_skip_frames = int(config.get('num_skip_frames', 0))
    if self.num_skip_frames < 0:
      raise ValueError("num_skip_frames must be >= 0")
    self.states = deque(maxlen=self.history_length)

  def _get_history(self) -> np.ndarray:
    frames = list(self.states)
    if len(frames) == 0:
      raise RuntimeError("No frames in history.")
    num_missing = self.history_length - len(frames)
    if num_missing > 0:
      first_frame = frames[0]
      pad_frames = [first_frame for _ in range(num_missing)]
      frames = pad_frames + frames
    return np.stack(frames, axis=0)

  def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
    step_reward = 0.0
    terminated = truncated = False
    info = None
    for _ in range(self.num_skip_frames + 1):
      state, reward, terminated, truncated, info = self.env.step(action)
      self.states.append(state)
      step_reward += reward
      if terminated or truncated:
        break
    return self._get_history(), step_reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
    self.states.clear()
    frame, info = self.env.reset(**kwargs)
    self.states.append(frame)
    return self._get_history(), info
