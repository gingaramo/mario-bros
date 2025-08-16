import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnv
from gymnasium.vector import SyncVectorEnv
from typing import Tuple, Optional, List, Union
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

  For vectorized environments, Observations frames and dense are vectorized as
  opposed having a List[Observation] of individual observations, for sake of
  efficiency. For accessing individual observations, as_list() is provided.

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

  def as_list(self) -> List['Observation']:
    """
    Converts the observation to a list of Observation objects, presumably unrolling
    vectorized environment observations.
    """
    num_envs = self.frame.shape[0] if self.frame is not None else None
    num_envs = self.dense.shape[0] if self.dense is not None else num_envs
    return [
        Observation(frame=self.frame[i] if self.frame is not None else None,
                    dense=self.dense[i] if self.dense is not None else None)
        for i in range(num_envs)
    ]


def merge_observations(observations: List[Observation]) -> Observation:
  """
  Merges a list of Observation objects into a single Observation.

  This is useful in a vectorized environment where each observation
  corresponds to a different environment instance.

  Args:
    observations: List of Observation objects to merge.

  Returns:
    A single Observation object with merged frame and dense vector.
  """
  frames = [obs.frame for obs in observations if obs.frame is not None]
  denses = [obs.dense for obs in observations if obs.dense is not None]

  merged_frame = np.stack(frames, axis=0) if frames else None
  merged_dense = np.stack(denses, axis=0) if denses else None

  return Observation(frame=merged_frame, dense=merged_dense)


class ObservationWrapper(gym.vector.VectorWrapper):
  """
  A wrapper that returns an Observation object from the environment's step and reset methods.

  This wrapper is useful for environments that return either a pixel-frame or a dense vector,
  or both, allowing for a consistent interface across different environments.

  The wrapper converts raw environment observations (numpy arrays) into Observation objects
  based on the configured input type.

  config options:
    - input: str; "frame" (default) or "dense"
      - "frame": Treats environment observations as pixels of a frame
      - "dense": Treats environment observations as a dense vector
  """

  def __init__(self, env: VectorEnv, config: dict) -> None:
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


class PreprocessFrameEnv(gym.vector.VectorWrapper):
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

  def __init__(self, env: VectorEnv, config: dict) -> None:
    super().__init__(env)
    self.resize_shape = config.get('resize_shape', None)
    self.grayscale = bool(config.get('grayscale', False))
    self.normalize = bool(config.get('normalize', False))

  def preprocess(self, frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
      raise ValueError("Input frame must be a numpy ndarray.")

    # Convert from (C,H,W) to (H,W,C) for OpenCV processing if needed
    if self.resize_shape:
      assert len(self.resize_shape) == 2
      frame = cv2.resize(frame,
                         self.resize_shape,
                         interpolation=cv2.INTER_AREA)
    if self.grayscale:
      assert len(
          frame.shape) == 3, "3 dimensions (C,H,W) needed for grayscale."
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    else:
      # Not hard to add, but won't do it now.
      raise NotImplementedError("Only grayscale images are supported.")

    if self.normalize:
      frame = frame.astype(np.float32) / 255.0
    return frame

  def step(self, action: int) -> Tuple[Observation, float, bool, bool, dict]:
    obs, reward, terminated, truncated, info = self.env.step(action)
    assert obs.frame is not None
    processed_frame = np.stack(
        [self.preprocess(obs.frame[i]) for i in range(self.env.num_envs)])
    return Observation(frame=processed_frame,
                       dense=obs.dense), reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[Observation, dict]:
    obs, info = self.env.reset(**kwargs)

    assert obs.frame is not None
    processed_frame = np.stack(
        [self.preprocess(obs.frame[i]) for i in range(self.env.num_envs)])
    return Observation(frame=processed_frame, dense=obs.dense), info


class RepeatActionEnv(gym.vector.VectorWrapper):
  """
  Repeats the action given to step() as many times as configured.
  Returns early if 'truncated' or 'done'. 
  Returns latest 'state' and 'info' from step, and accumulated 'reward'.

  config options:
    - num_repeat_action: int ; actions will be repeated this many times
  """

  num_repeat_action: int

  def __init__(self, env: VectorEnv, config: dict) -> None:
    super().__init__(env)
    self.num_repeat_action = config.get('num_repeat_action', 1)
    if self.num_repeat_action < 1:
      raise ValueError("num_repeat_action must be >= 1")
    self.observation_wrapper = self.env
    while not isinstance(self.observation_wrapper, ObservationWrapper):
      self.observation_wrapper = self.observation_wrapper.env

  def step(self, action: int) -> Tuple[Observation, float, bool, bool, dict]:
    num_envs = self.env.num_envs
    total_reward = np.zeros((num_envs, ))
    terminated = np.zeros((num_envs, ), dtype=bool)
    truncated = np.zeros((num_envs, ), dtype=bool)
    ret_obs = [None] * num_envs
    info = None
    terminated_or_truncated = set()
    for _ in range(self.num_repeat_action):
      _obs, _reward, _terminated, _truncated, info = self.env.step(action)
      for i, (o, r, t, tr) in enumerate(
          zip(_obs.as_list(), _reward, _terminated, _truncated)):
        if i in terminated_or_truncated:
          continue
        total_reward[i] += r
        terminated[i] |= t
        truncated[i] |= tr
        if t or tr:
          terminated_or_truncated.add(i)
        else:
          ret_obs[i] = o

    # Now reset environments that got terminated or truncated, so that the
    # first observation is the one after the reset.
    for i in terminated_or_truncated:
      obs, info = self.env.unwrapped.envs[i].reset()

      # Convert the observation to the expected Observation type
      ret_obs[i] = self.observation_wrapper.to_observation(obs)

    # TODO: info is not handled correctly in this impl, nobody should depend on it
    return merge_observations(
        ret_obs), total_reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[Observation, dict]:
    obs, info = self.env.reset(**kwargs)

    return obs, info


class ReturnActionEnv(gym.vector.VectorWrapper):
  """
  Appends the current action to the observation's dense vector.

  This wrapper takes the action that was just performed and appends it to the dense
  vector of the resulting observation. This allows the agent to have access to the
  action that led to the current state.

  config options:
    - mode: str ; "append" (default and only supported mode)
      - "append": Appends the current action as a dense parameter to the observation
  """

  def __init__(self, env: VectorEnv, config: dict = None) -> None:
    super().__init__(env)
    config = config or {}
    self.mode = config.get('mode', 'append')
    if self.mode not in ['append']:
      raise ValueError(f"Invalid mode '{self.mode}'. Must be 'append'")

  def step(self,
           action: List[int]) -> Tuple[Observation, float, bool, bool, dict]:
    obs, reward, terminated, truncated, info = self.env.step(action)

    if self.mode == 'append':
      num_envs = self.env.num_envs
      action = np.array(action).reshape(num_envs, -1)
      # Create new observation with dense vector containing current action
      if obs.dense is not None:
        # Append current action to existing dense vector
        new_dense = np.concatenate((obs.dense, action), axis=1)
      else:
        # Create dense vector with just current action
        new_dense = action
      return Observation(frame=obs.frame,
                         dense=new_dense), reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[Observation, dict]:
    obs, info = self.env.reset(**kwargs)

    if self.mode == 'append':
      num_envs = self.env.num_envs
      action = np.zeros((num_envs, 1))
      # Create new observation with dense vector containing 0 (no previous action yet)
      if obs.dense is not None:
        # Append 0 to existing dense vector to represent no previous action
        new_dense = np.concatenate([obs.dense, action], axis=1)
      else:
        # Create dense vector with just 0 for no previous action
        new_dense = action
      return Observation(frame=obs.frame, dense=new_dense), info


class HistoryEnv(gym.vector.VectorWrapper):
  """
  Maintains a history of the last N frames.
  config options:
    - history_length: int ; if there are not enough, repeats oldest frame
  """

  history_length: int

  def __init__(self, env: VectorEnv, config: dict) -> None:
    super().__init__(env)
    self.history_length = int(config.get('history_length', 1))
    if self.history_length < 1:
      raise ValueError("history_length must be >= 1")
    self.states = [
        deque(maxlen=self.history_length) for _ in range(self.env.num_envs)
    ]

  def _get_single_env_history(self, history: deque) -> Observation:
    history = list(history)
    if len(history) == 0:
      raise RuntimeError("No frames in history.")

    # Extract frames and dense vectors separately
    frame_history = []
    dense_history = []

    # First collect last n observations from this environment's state
    for obs in history:
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

      # Grayscale frames: (H, W) -> stack to get (N, H, W)
      stacked_frames = np.stack(frame_history, axis=0)
    else:
      stacked_frames = None

    # Dense vectors however are flattened
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

  def _get_history(self) -> Observation:
    env_observations = []
    for i in range(self.env.unwrapped.num_envs):
      env_observations.append(self._get_single_env_history(self.states[i]))
    return merge_observations(env_observations)

  def step(self, action: int) -> Tuple[Observation, float, bool, bool, dict]:
    obs, reward, terminated, truncated, info = self.env.step(action)

    for states, _obs in zip(self.states, obs.as_list()):
      states.append(_obs)
    ret_obs = self._get_history()

    # If vectorized, we need to handle the last terminated or truncated states.
    if np.any(terminated) or np.any(truncated):
      for i, terminated_or_truncated in enumerate(
          np.logical_or(terminated, truncated)):
        if terminated_or_truncated:
          self.states[i].clear()
    return ret_obs, reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[Observation, dict]:
    obs, info = self.env.reset(**kwargs)
    self.states = [
        deque(maxlen=self.history_length) for _ in range(self.env.num_envs)
    ]
    for states, _obs in zip(self.states, obs.as_list()):
      states.append(_obs)

    return self._get_history(), info


class CaptureRenderFrameEnv(gym.vector.VectorWrapper):
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
    - observation_is_frame: bool ; if True, the observation is expected to be a frame,
      otherwise we call render() on the environment.
  """

  def __init__(self, env: VectorEnv, config: dict = None) -> None:
    super().__init__(env)
    self.rendered_frame: Optional[np.ndarray] = None
    # We capture last rendered frame to render frames offset from the current step,
    # as otherwise we show the next frame with the agents action values.
    self.last_rendered_frame: Optional[np.ndarray] = None
    config = config or {}
    self.mode = config.get('mode', 'capture')
    self.observation_is_frame = config.get('observation_is_frame', False)
    if self.mode not in ['capture', 'replace']:
      raise ValueError(
          f"Invalid mode '{self.mode}'. Must be 'capture' or 'replace'")

  def step(self, action: int) -> Tuple[Observation, float, bool, bool, dict]:
    obs, reward, terminated, truncated, info = self.env.step(action)

    self.last_rendered_frame = self.rendered_frame
    if not self.observation_is_frame:
      self.rendered_frame = np.array([
          self.env.unwrapped.envs[i].render() for i in range(self.env.num_envs)
      ])
    else:
      self.rendered_frame = obs.frame

    if self.mode == 'capture':
      return obs, reward, terminated, truncated, info
    if self.mode == 'replace':
      return Observation(frame=self.rendered_frame,
                         dense=None), reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[Observation, dict]:
    obs, info = self.env.reset(**kwargs)
    if not self.observation_is_frame:
      self.rendered_frame = np.array([
          self.env.unwrapped.envs[i].render() for i in range(self.env.num_envs)
      ])
    else:
      self.rendered_frame = obs.frame

    self.last_rendered_frame = self.rendered_frame
    if self.mode == 'replace':
      # For replace mode, we need to render the initial frame
      return Observation(frame=self.rendered_frame, dense=None), info
    else:  # mode == 'capture'
      return obs, info


class ClipRewardEnv(gym.vector.VectorWrapper):
  """
  Clips the reward to be between -1 and 1.

  This is useful for environments where the reward can be very large or very small,
  and we want to normalize it to a smaller range.

  config options:
    - None
  """

  def __init__(self, env: VectorEnv, config: dict = None) -> None:
    super().__init__(env)
    self.clip_min = config.get('clip_min', -float('inf'))
    self.clip_max = config.get('clip_max', float('inf'))

  def step(self, action: int) -> Tuple[Observation, float, bool, bool, dict]:
    obs, reward, terminated, truncated, info = self.env.step(action)
    clipped_reward = np.clip(reward, self.clip_min, self.clip_max)
    return obs, clipped_reward, terminated, truncated, info


def create_environment(config: dict) -> gym.Env:
  """ Creates a Gym environment with specified wrappers.
  Args:
    config: Dictionary containing environment configuration.
  Returns:
    A Gym environment instance with the specified wrappers applied.
  """
  num_envs = config.get('num_envs', 1)
  print(
      f"Creating environment: {config['env_name']} with {num_envs} environments."
  )

  # Mario environment needs JoypadSpace which is not vectoriez, so we need to
  # create a SyncVectorEnv.
  if 'SuperMarioBros' not in config['env_name']:
    env = gym.make_vec(config['env_name'],
                       num_envs=num_envs,
                       vectorization_mode="sync",
                       render_mode="rgb_array")
  else:
    # Super Mario Bros environment is not compatible with vectorized environments.
    def _create_mario_env():
      mario_env = gym.make(config['env_name'])
      from nes_py.wrappers import JoypadSpace
      from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
      mario_env = JoypadSpace(mario_env, SIMPLE_MOVEMENT)
      return mario_env

    env = SyncVectorEnv([lambda: _create_mario_env() for _ in range(num_envs)])

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
    elif wrapper == 'ClipRewardEnv':
      env = ClipRewardEnv(env, config.get('ClipRewardEnv', {}))
    else:
      raise ValueError(f"Unknown environment wrapper: {wrapper}")
  return env
