import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector.vector_env import VectorEnv
from gymnasium.vector import AsyncVectorEnv
from typing import Any, Callable, Dict, Tuple, Optional, List
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
        torch.empty(0) if self.frame is None else self.frame / 255.0,
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
    num_envs = len(self.frame) if self.frame is not None else None
    num_envs = self.dense.shape[0] if self.dense is not None else num_envs
    return [
        Observation(frame=self.frame[i] if self.frame is not None else None,
                    dense=self.dense[i] if self.dense is not None else None)
        for i in range(num_envs)
    ]

  def as_list_input(self, device):
    """
    Converts the observation to a list of input tensors for the model.

    Args:
      device: The device to which the tensors should be moved.

    Returns:
      A list of tuples containing the frame and dense vector tensors for each environment.
    """
    return [obs.as_input(device) for obs in self.as_list()]


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


class ObservationWrapper(gym.Wrapper):
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

  def __init__(self, env, config: dict) -> None:
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

class ObservationToDictWrapper(gym.ObservationWrapper):
  """A wrapper that converts a custom 'Observation' object into a Dict space
  compatible with Gymnasium's vector environments and shared memory for AsyncVectorEnv.
  """
  def __init__(self, env: gym.Env, mock_obs: Observation) -> None:
    super().__init__(env)
    
    # This structure is easily batched by AsyncVectorEnv.
    self.observation_space = spaces.Dict({
        "frame": spaces.Box(
            low=0, high=255, shape=mock_obs.frame.shape, dtype=np.uint8
        ),
        "dense": spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(mock_obs.dense) if mock_obs.dense is not None else 0,), dtype=np.float32
        ),
    })

  def observation(self, obs: Observation) -> Dict[str, np.ndarray]:
    return {
        "frame": obs.frame,
        "dense": obs.dense,
    }

  def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    obs, info = self.env.reset(**kwargs)
    return self.observation(obs), info


class DictToObservationWrapper(gym.vector.VectorObservationWrapper):

  def observations(self, obs: Dict[str, np.ndarray]) -> Observation:
    return Observation(frame=obs["frame"], dense=obs["dense"])

class PreprocessFrameEnv(gym.Wrapper):
  """
  Postprocesses Observation 'frame' with common transformations.
  
  config options:
    - resize_shape: [int, int] or None ; if present, frame is resized
    - grayscale: bool
  """

  resize_shape: Optional[Tuple[int, int]]
  grayscale: bool

  def __init__(self, env, config: dict) -> None:
    super().__init__(env)
    self.resize_shape = config.get('resize_shape', None)
    self.grayscale = bool(config.get('grayscale', False))

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
  Truncates episodes after terminated, so observation is always first env observation. 
  Returns latest 'state' and 'info' from step, and accumulated 'reward', it also
  returns any accumulated info from terminated or truncated episodes (where info key
  contains 'terminated').

  config options:
    - num_repeat_action: int ; actions will be repeated this many times
  """

  num_repeat_action: int

  def __init__(self, env, config: dict) -> None:
    super().__init__(env)
    self.num_repeat_action = config.get('num_repeat_action', 1)
    if self.num_repeat_action < 1:
      raise ValueError("num_repeat_action must be >= 1")

  def step(self, action: int) -> Tuple[Observation, float, bool, bool, dict]:
    obs = None
    total_reward = 0
    terminated = False
    truncated = False
    info = None
    for _ in range(self.num_repeat_action):
      obs, _reward, terminated, truncated, info = self.env.step(action)
      total_reward += _reward
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
  """

  def __init__(self, env, config: dict) -> None:
    super().__init__(env)

  def step(self,
           action: List[int]) -> Tuple[Observation, float, bool, bool, dict]:
    obs, reward, terminated, truncated, info = self.env.step(action)

    if obs.dense is not None:
      # Append current action to existing dense vector
      new_dense = np.concatenate((obs.dense, action), axis=0)
    else:
      # Create dense vector with just current action
      new_dense = action

    return Observation(frame=obs.frame,
                       dense=new_dense), reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[Observation, dict]:
    obs, info = self.env.reset(**kwargs)

    if obs.dense is not None:
      # Append 0 to existing dense vector to represent no previous action
      new_dense = np.concatenate([obs.dense, 0], axis=0)
    else:
      # Create dense vector with just 0 for no previous action
      new_dense = 0
    
    return Observation(frame=obs.frame, dense=new_dense), info


class HistoryEnv(gym.Wrapper):
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
    self.states = deque(maxlen=self.history_length)

  def _get_history_observation(self) -> Observation:
    history = list(self.states)
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
    if len(frame_history) > 0:
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
    if len(dense_history) > 0:
      num_missing = self.history_length - len(dense_history)
      if num_missing > 0:
        first_dense = dense_history[0]
        pad_dense = [first_dense for _ in range(num_missing)]
        dense_history = pad_dense + dense_history
      flattened_dense = dense_history
    else:
      flattened_dense = None

    return Observation(frame=stacked_frames, dense=flattened_dense)

  def step(self, action: int) -> Tuple[Observation, float, bool, bool, dict]:
    obs, reward, terminated, truncated, info = self.env.step(action)

    self.states.append(obs)
    ret_obs = self._get_history_observation()

    if terminated or truncated:
      self.states.clear()

    return ret_obs, reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[Observation, dict]:
    obs, info = self.env.reset(**kwargs)
    self.states.clear()
    self.states.append(obs)

    return self._get_history_observation(), info


class CaptureRenderFrameEnv(gym.Wrapper):
  """Captures the rendered frame from the environment for display or for using as observation.

  If `use_for_display` is true, the rendered frame can be accessed via the `observation_frame` attribute
  in `info`.

  Args:
  - config: A dictionary containing configuration options.
    - use_for_observation: Returns rendered frame as the observation, ignores original observation.
    - use_for_display: bool ; if True, the captured frame is stored in info for display purposes
      in `info` observation_frame.
    - observation_is_frame: bool ; if True, the observation is expected to be a frame,
      otherwise we call render() on the environment.
  """

  def __init__(self, env, config: dict) -> None:
    super().__init__(env)

    self.observation_is_frame = config.get('observation_is_frame', False)
    self.use_for_observation = config.get('use_for_observation', False)
    self.use_for_display = config.get('use_for_display', False)
    
  def replace_or_capture(self, obs: Observation, info: dict) -> Observation:
    # If there is no need to replace or capture, return original observation.
    # This happens when the environment is not one of the ones displayed.
    if not (self.use_for_observation or self.use_for_display):
      return obs

    if not self.observation_is_frame:
      frame = self.env.unwrapped.render()
    else:
      frame = obs.frame

    if self.use_for_display:
      info['observation_frame'] = frame

    if self.use_for_observation:
      return Observation(frame=frame, dense=None)
    else:  # mode == 'capture'
      return obs

  def step(self, action: int) -> Tuple[Observation, float, bool, bool, dict]:
    obs, reward, terminated, truncated, info = self.env.step(action)
    return self.replace_or_capture(obs, info), reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[Observation, dict]:
    obs, info = self.env.reset(**kwargs)
    return self.replace_or_capture(obs, info), info


class ClipRewardEnv(gym.Wrapper):
  """
  Clips the reward to be between -1 and 1.

  This is useful for environments where the reward can be very large or very small,
  and we want to normalize it to a smaller range.

  config options:
    - None
  """

  def __init__(self, env, config: dict = None) -> None:
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
  num_render_envs = config['render_layout'][0] * config['render_layout'][1]

  def _make_env(index: int) -> Callable[[], gym.Env]:
    def _do_make() -> gym.Env:
      env = gym.make(config['env_name'], render_mode='rgb_array')

      # Special handling for Super Mario Bros environments
      if 'SuperMarioBros' in config['env_name']:
        from nes_py.wrappers import JoypadSpace
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
        env = JoypadSpace(env, SIMPLE_MOVEMENT)

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
          capture_render_frame_config = config.get('CaptureRenderFrameEnv', {})
          # Optimization: disable "rendering" or "capture" when not needed, this
          # drastically reduces IPC.
          capture_render_frame_config['use_for_display'] = index < num_render_envs
          env = CaptureRenderFrameEnv(env, capture_render_frame_config)
        elif wrapper == 'ClipRewardEnv':
          env = ClipRewardEnv(env, config.get('ClipRewardEnv', {}))
        else:
          raise ValueError(f"Unknown environment wrapper: {wrapper}")
      
      env = gym.wrappers.RecordEpisodeStatistics(env)
      env = ObservationToDictWrapper(
          env,
          mock_obs=env.reset()[0]
      )
      return env
    
    return _do_make

  return DictToObservationWrapper(AsyncVectorEnv([_make_env(i) for i in range(num_envs)]))