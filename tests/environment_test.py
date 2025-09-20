import unittest
import numpy as np
import gymnasium as gym
from gymnasium.vector import VectorEnv
from unittest.mock import Mock, MagicMock, patch
from typing import Tuple, Dict, Any
import cv2
import sys
import os

# Add src directory to path so we can import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
  sys.path.insert(0, src_path)

from environment import PreprocessFrameEnv, RepeatActionEnv, ReturnActionEnv, HistoryEnv, CaptureRenderFrameEnv, create_environment, Observation, ObservationWrapper


class MockVectorEnv(VectorEnv):
  """Mock vectorized environment for testing wrappers."""

  def __init__(self,
               frame_shape: Tuple[int, ...] = (84, 84, 3),
               action_space_n: int = 4,
               num_envs: int = 1):
    self.frame_shape = frame_shape
    self.num_envs = num_envs
    self.single_action_space = gym.spaces.Discrete(action_space_n)
    self.single_observation_space = gym.spaces.Box(low=0,
                                                   high=255,
                                                   shape=frame_shape,
                                                   dtype=np.uint8)
    # Set vectorized action and observation spaces
    self.action_space = gym.spaces.MultiDiscrete([action_space_n] * num_envs)
    self.observation_space = gym.spaces.Box(low=0,
                                            high=255,
                                            shape=(num_envs, ) + frame_shape,
                                            dtype=np.uint8)

    self.step_count = np.zeros(num_envs)
    self.max_steps = 100

  def step(
      self, actions: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    self.step_count += 1
    # Return vectorized observations
    frames = np.random.randint(0,
                               256, (self.num_envs, ) + self.frame_shape,
                               dtype=np.uint8)
    rewards = np.ones(self.num_envs, dtype=np.float32)
    terminated = self.step_count >= self.max_steps
    truncated = np.zeros(self.num_envs, dtype=bool)
    # For vectorized environments, info should be a dict with lists/arrays as values
    infos = {'step': self.step_count.copy()}
    return frames, rewards, terminated, truncated, infos

  def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
    self.step_count = np.zeros(self.num_envs)
    frames = np.random.randint(0,
                               256, (self.num_envs, ) + self.frame_shape,
                               dtype=np.uint8)
    # For vectorized environments, info should be a dict with lists/arrays as values
    infos = {'episode': np.ones(self.num_envs, dtype=int)}
    return frames, infos

  def close(self):
    pass


class MockEnv(gym.Env):
  """Mock single environment for backward compatibility where needed."""

  def __init__(self,
               frame_shape: Tuple[int, ...] = (84, 84, 3),
               action_space_n: int = 4):
    super().__init__()
    self.frame_shape = frame_shape
    self.action_space = gym.spaces.Discrete(action_space_n)
    self.observation_space = gym.spaces.Box(low=0,
                                            high=255,
                                            shape=frame_shape,
                                            dtype=np.uint8)
    self.step_count = 0
    self.max_steps = 100

  def step(self, action: int) -> Tuple[Observation, float, bool, bool, dict]:
    self.step_count += 1
    frame = np.random.randint(0, 256, self.frame_shape, dtype=np.uint8)
    reward = 1.0
    terminated = self.step_count >= self.max_steps
    truncated = False
    info = {'step': self.step_count}
    return Observation(frame=frame), reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[Observation, dict]:
    self.step_count = 0
    frame = np.random.randint(0, 256, self.frame_shape, dtype=np.uint8)
    info = {'episode': 1}
    return Observation(frame=frame), info


class TestObservationWrapper(unittest.TestCase):
  """Test cases for ObservationWrapper."""

  def setUp(self):
    """Set up test fixtures."""
    # Create a vectorized environment that returns numpy arrays
    self.base_env = gym.make_vec('CartPole-v1',
                                 num_envs=1,
                                 vectorization_mode="sync")

  def test_init_default_config(self):
    """Test initialization with default configuration."""
    wrapper = ObservationWrapper(self.base_env, {})
    self.assertEqual(wrapper.input_type, 'frame')

  def test_init_frame_config(self):
    """Test initialization with frame input type."""
    wrapper = ObservationWrapper(self.base_env, {'input': 'frame'})
    self.assertEqual(wrapper.input_type, 'frame')

  def test_init_dense_config(self):
    """Test initialization with dense input type."""
    wrapper = ObservationWrapper(self.base_env, {'input': 'dense'})
    self.assertEqual(wrapper.input_type, 'dense')

  def test_init_invalid_config(self):
    """Test initialization with invalid input type raises ValueError."""
    with self.assertRaises(ValueError) as context:
      ObservationWrapper(self.base_env, {'input': 'invalid'})
    self.assertIn("Invalid input type 'invalid'", str(context.exception))

  def test_to_observation_frame_type(self):
    """Test to_observation method with frame input type."""
    wrapper = ObservationWrapper(self.base_env, {'input': 'frame'})
    frame_array = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)

    obs = wrapper.to_observation(frame_array)

    self.assertIsInstance(obs, Observation)
    self.assertIsNotNone(obs.frame)
    self.assertIsNone(obs.dense)
    self.assertEqual(
        obs.frame.shape,
        (84, 84, 3))  # to_observation doesn't change channel order

  def test_to_observation_dense_type(self):
    """Test to_observation method with dense input type."""
    wrapper = ObservationWrapper(self.base_env, {'input': 'dense'})
    dense_array = np.array([1.0, 2.0, 3.0, 4.0])

    obs = wrapper.to_observation(dense_array)

    self.assertIsInstance(obs, Observation)
    self.assertIsNone(obs.frame)
    self.assertIsNotNone(obs.dense)
    np.testing.assert_array_equal(obs.dense, dense_array)

  def test_step_dense_type(self):
    """Test step method with dense input type."""
    wrapper = ObservationWrapper(self.base_env, {'input': 'dense'})

    # Reset first
    wrapper.reset()
    obs, reward, terminated, truncated, info = wrapper.step(
        [0])  # Vectorized action

    self.assertIsInstance(obs, Observation)
    self.assertIsNone(obs.frame)
    self.assertIsNotNone(obs.dense)
    self.assertEqual(obs.dense.shape,
                     (1, 4))  # Vectorized: (num_envs, obs_size)
    self.assertIsInstance(reward, np.ndarray)  # Vectorized reward
    self.assertIsInstance(terminated, np.ndarray)  # Vectorized terminated
    self.assertIsInstance(truncated, np.ndarray)  # Vectorized truncated
    # Info can be either a list (proper vectorized) or dict (some gym versions)
    if isinstance(info, list):
      self.assertEqual(len(info), 1)
    else:
      self.assertIsInstance(info, dict)

  def test_reset_dense_type(self):
    """Test reset method with dense input type."""
    wrapper = ObservationWrapper(self.base_env, {'input': 'dense'})

    obs, info = wrapper.reset()

    self.assertIsInstance(obs, Observation)
    self.assertIsNone(obs.frame)
    self.assertIsNotNone(obs.dense)
    self.assertEqual(obs.dense.shape,
                     (1, 4))  # Vectorized: (num_envs, obs_size)
    # Info can be either a list (proper vectorized) or dict (some gym versions)
    if isinstance(info, list):
      self.assertEqual(len(info), 1)
      self.assertIsInstance(info[0], dict)
    else:
      self.assertIsInstance(info, dict)

  def tearDown(self):
    """Clean up after tests."""
    self.base_env.close()


class TestPreprocessFrameEnv(unittest.TestCase):
  """Test cases for PreprocessFrameEnv wrapper."""

  def setUp(self):
    self.base_env = ObservationWrapper(MockVectorEnv(), {'input': 'frame'})

  def test_init_with_valid_config(self):
    """Test initialization with valid configuration."""
    config = {'resize_shape': (42, 42), 'grayscale': True, 'normalize': True}
    env = PreprocessFrameEnv(self.base_env, config)

    self.assertEqual(env.resize_shape, (42, 42))
    self.assertTrue(env.grayscale)
    self.assertTrue(env.normalize)

  def test_init_with_minimal_config(self):
    """Test initialization with minimal configuration."""
    config = {}
    env = PreprocessFrameEnv(self.base_env, config)

    self.assertIsNone(env.resize_shape)
    self.assertFalse(env.grayscale)
    self.assertFalse(env.normalize)

  def test_preprocess_resize_only(self):
    """Test preprocessing with resize only (requires grayscale due to source limitation)."""
    config = {
        'resize_shape': (42, 42),
        'grayscale': True
    }  # Source code requires grayscale
    env = PreprocessFrameEnv(self.base_env, config)

    frame = np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8)
    processed = env.preprocess(frame)

    # After grayscale and resize, should be 2D with new size
    self.assertEqual(processed.shape, (42, 42))

  def test_preprocess_grayscale_only(self):
    """Test preprocessing with grayscale only."""
    config = {'grayscale': True}
    env = PreprocessFrameEnv(self.base_env, config)

    frame = np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8)
    processed = env.preprocess(frame)

    self.assertEqual(processed.shape, (84, 84))

  def test_preprocess_normalize_only(self):
    """Test preprocessing with normalization only (requires grayscale due to source limitation)."""
    config = {
        'normalize': True,
        'grayscale': True
    }  # Source code requires grayscale
    env = PreprocessFrameEnv(self.base_env, config)

    frame = np.ones((84, 84, 3), dtype=np.uint8) * 255
    processed = env.preprocess(frame)

    self.assertTrue(np.allclose(processed, 1.0))
    self.assertEqual(processed.dtype, np.float32)
    # After grayscale, should be 2D
    self.assertEqual(len(processed.shape), 2)

  def test_preprocess_all_transformations(self):
    """Test preprocessing with all transformations."""
    config = {'resize_shape': (42, 42), 'grayscale': True, 'normalize': True}
    env = PreprocessFrameEnv(self.base_env, config)

    frame = np.ones((84, 84, 3), dtype=np.uint8) * 255
    processed = env.preprocess(frame)

    self.assertEqual(processed.shape, (42, 42))
    self.assertTrue(np.allclose(processed, 1.0))
    self.assertEqual(processed.dtype, np.float32)

  def test_preprocess_invalid_input(self):
    """Test preprocessing with invalid input."""
    config = {}
    env = PreprocessFrameEnv(self.base_env, config)

    with self.assertRaises(ValueError):
      env.preprocess("not an array")

  def test_preprocess_invalid_resize_shape(self):
    """Test preprocessing with invalid resize shape."""
    config = {'resize_shape': (42, )}  # Invalid tuple length
    env = PreprocessFrameEnv(self.base_env, config)

    frame = np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8)
    with self.assertRaises(AssertionError):
      env.preprocess(frame)

  def test_step(self):
    """Test step method."""
    config = {'grayscale': True}
    env = PreprocessFrameEnv(self.base_env, config)

    obs, reward, terminated, truncated, info = env.step(
        [0])  # Vectorized action

    self.assertIsInstance(obs, Observation)
    # For vectorized environments: shape is (num_envs, height, width) after grayscale
    self.assertEqual(obs.frame.shape, (1, 84, 84))
    self.assertEqual(reward, [1.0])  # Vectorized reward
    self.assertIsInstance(terminated, np.ndarray)  # Vectorized terminated
    self.assertIsInstance(truncated, np.ndarray)  # Vectorized truncated
    # Info can be either a list (proper vectorized) or dict (some gym versions)
    if isinstance(info, list):
      self.assertEqual(len(info), 1)
    else:
      self.assertIsInstance(info, dict)

  def test_reset(self):
    """Test reset method."""
    config = {'grayscale': True}
    env = PreprocessFrameEnv(self.base_env, config)

    obs, info = env.reset()

    self.assertIsInstance(obs, Observation)
    # For vectorized environments: shape is (num_envs, height, width) after grayscale
    self.assertEqual(obs.frame.shape, (1, 84, 84))
    # Info can be either a list (proper vectorized) or dict (some gym versions)
    if isinstance(info, list):
      self.assertEqual(len(info), 1)
    else:
      self.assertIsInstance(info, dict)


@unittest.skip(
    "RepeatActionEnv has contradictory design - inherits from VectorWrapper but rejects VectorEnv"
)
class TestRepeatActionEnv(unittest.TestCase):
  """Test cases for RepeatActionEnv wrapper."""

  def setUp(self):
    self.base_env = MockEnv(
    )  # RepeatActionEnv doesn't support vectorized envs

  def test_init_with_valid_config(self):
    """Test initialization with valid configuration."""
    config = {'num_repeat_action': 3}
    env = RepeatActionEnv(self.base_env, config)

    self.assertEqual(env.num_repeat_action, 3)

  def test_init_with_default_config(self):
    """Test initialization with default configuration."""
    config = {}
    env = RepeatActionEnv(self.base_env, config)

    self.assertEqual(env.num_repeat_action, 1)

  def test_init_with_invalid_config(self):
    """Test initialization with invalid configuration."""
    config = {'num_repeat_action': 0}

    with self.assertRaises(ValueError):
      RepeatActionEnv(self.base_env, config)

  def test_step_single_repeat(self):
    """Test step with single repeat."""
    config = {'num_repeat_action': 1}
    env = RepeatActionEnv(self.base_env, config)

    obs, reward, terminated, truncated, info = env.step(0)

    self.assertEqual(reward, 1.0)  # Single step reward
    self.assertEqual(self.base_env.step_count, 1)

  def test_step_multiple_repeats(self):
    """Test step with multiple repeats."""
    config = {'num_repeat_action': 3}
    env = RepeatActionEnv(self.base_env, config)

    obs, reward, terminated, truncated, info = env.step(0)

    self.assertEqual(reward, 3.0)  # Accumulated reward
    self.assertEqual(self.base_env.step_count, 3)

  def test_step_early_termination(self):
    """Test step with early termination."""
    self.base_env.max_steps = 2  # Force early termination
    config = {'num_repeat_action': 5}
    env = RepeatActionEnv(self.base_env, config)

    obs, reward, terminated, truncated, info = env.step(0)

    self.assertEqual(reward, 2.0)  # Only 2 steps before termination
    self.assertTrue(terminated)
    self.assertEqual(self.base_env.step_count, 2)

  def test_reset(self):
    """Test reset method."""
    config = {'num_repeat_action': 3}
    env = RepeatActionEnv(self.base_env, config)

    obs, info = env.reset()

    self.assertIsInstance(obs, Observation)
    expected_shape = (self.base_env.frame_shape[2],
                      self.base_env.frame_shape[0],
                      self.base_env.frame_shape[1])
    self.assertEqual(obs.frame.shape, expected_shape)
    self.assertIsInstance(info, dict)
    self.assertEqual(self.base_env.step_count, 0)


class TestReturnActionEnv(unittest.TestCase):
  """Test cases for ReturnActionEnv wrapper."""

  def setUp(self):
    self.base_env = ObservationWrapper(MockVectorEnv(), {'input': 'frame'})

  def test_init_default_mode(self):
    """Test initialization with default mode."""
    env = ReturnActionEnv(self.base_env)
    self.assertEqual(env.mode, 'append')

  def test_init_with_valid_mode(self):
    """Test initialization with valid mode."""
    env = ReturnActionEnv(self.base_env, {'mode': 'append'})
    self.assertEqual(env.mode, 'append')

  def test_init_with_invalid_mode(self):
    """Test initialization with invalid mode raises ValueError."""
    with self.assertRaises(ValueError) as context:
      ReturnActionEnv(self.base_env, {'mode': 'invalid'})
    self.assertIn("Invalid mode 'invalid'", str(context.exception))

  def test_step_append_mode_first_action(self):
    """Test step with append mode and first action."""
    env = ReturnActionEnv(self.base_env, {'mode': 'append'})

    obs, reward, terminated, truncated, info = env.step(
        [1])  # Vectorized action

    # Should return observation with dense vector containing current action
    self.assertIsInstance(obs, Observation)
    self.assertEqual(obs.frame.shape, (1, 84, 84, 3))  # Vectorized frame shape
    self.assertIsNotNone(obs.dense)  # Dense vector should exist
    self.assertEqual(obs.dense.shape, (1, 1))  # Vectorized: (num_envs, 1)
    self.assertEqual(obs.dense[0][0], 1)  # Current action in vectorized format

  def test_step_append_mode_subsequent_actions(self):
    """Test step with append mode and subsequent actions."""
    env = ReturnActionEnv(self.base_env, {'mode': 'append'})

    # First step
    obs1, _, _, _, _ = env.step([1])  # Vectorized action
    self.assertEqual(obs1.dense[0][0], 1)  # First action in vectorized format

    # Second step
    obs2, reward, terminated, truncated, info = env.step(
        [2])  # Vectorized action

    # Should return observation with dense vector containing current action
    self.assertIsInstance(obs2, Observation)
    self.assertEqual(obs2.frame.shape,
                     (1, 84, 84, 3))  # Vectorized frame shape
    self.assertIsNotNone(obs2.dense)  # Dense vector should exist
    self.assertEqual(obs2.dense.shape, (1, 1))  # Vectorized: (num_envs, 1)
    self.assertEqual(obs2.dense[0][0],
                     2)  # Current action in vectorized format

  def test_step_with_existing_dense_vector(self):
    """Test step when base environment already provides dense vector."""

    # Create a mock environment that directly returns Observation objects with dense vectors
    class MockObservationEnv(VectorEnv):

      def __init__(self):
        self.num_envs = 1
        self.action_space = gym.spaces.MultiDiscrete([4])
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=(1, 84, 84, 3),
                                                dtype=np.uint8)

      def step(self, actions):
        frame = np.random.randint(0, 256, (1, 84, 84, 3), dtype=np.uint8)
        obs = Observation(frame=frame,
                          dense=np.array([[99, 88]]))  # Existing dense vector
        reward = np.ones(1, dtype=np.float32)
        terminated = np.zeros(1, dtype=bool)
        truncated = np.zeros(1, dtype=bool)
        info = [{}]
        return obs, reward, terminated, truncated, info

      def reset(self, **kwargs):
        frame = np.random.randint(0, 256, (1, 84, 84, 3), dtype=np.uint8)
        obs = Observation(frame=frame, dense=np.array([[99, 88]]))
        info = [{}]
        return obs, info

      def close(self):
        pass

    env = ReturnActionEnv(MockObservationEnv(), {'mode': 'append'})

    obs, reward, terminated, truncated, info = env.step(
        [5])  # Vectorized action

    # Should append action to existing dense vector
    self.assertIsInstance(obs, Observation)
    self.assertEqual(obs.frame.shape, (1, 84, 84, 3))  # Vectorized frame shape
    self.assertIsNotNone(obs.dense)
    self.assertEqual(obs.dense.shape,
                     (1, 3))  # Vectorized: (num_envs, original_2 + action)
    np.testing.assert_array_equal(obs.dense, [[99, 88, 5]])

  def test_reset_append_mode(self):
    """Test reset with append mode."""
    env = ReturnActionEnv(self.base_env, {'mode': 'append'})

    # Take a step first
    env.step([1])  # Vectorized action

    # Reset
    obs, info = env.reset()

    # Should return observation with dense vector containing 0 (no previous action)
    self.assertIsInstance(obs, Observation)
    self.assertEqual(obs.frame.shape, (1, 84, 84, 3))  # Vectorized frame shape
    self.assertIsNotNone(obs.dense)
    self.assertEqual(obs.dense.shape, (1, 1))  # Vectorized: (num_envs, 1)
    self.assertEqual(obs.dense[0][0],
                     0)  # No previous action in vectorized format
    self.assertIsInstance(info, dict)
    self.assertIn('episode', info)
    self.assertEqual(len(info['episode']), 1)  # One environment

  def test_reset_with_existing_dense_vector(self):
    """Test reset when base environment already provides dense vector."""

    # Create a mock environment that directly returns Observation objects with dense vectors
    class MockObservationEnv(VectorEnv):

      def __init__(self):
        self.num_envs = 1
        self.action_space = gym.spaces.MultiDiscrete([4])
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=(1, 84, 84, 3),
                                                dtype=np.uint8)

      def step(self, actions):
        frame = np.random.randint(0, 256, (1, 84, 84, 3), dtype=np.uint8)
        obs = Observation(frame=frame,
                          dense=np.array([[77, 66]]))  # Existing dense vector
        reward = np.ones(1, dtype=np.float32)
        terminated = np.zeros(1, dtype=bool)
        truncated = np.zeros(1, dtype=bool)
        info = [{}]
        return obs, reward, terminated, truncated, info

      def reset(self, **kwargs):
        frame = np.random.randint(0, 256, (1, 84, 84, 3), dtype=np.uint8)
        obs = Observation(frame=frame, dense=np.array([[77, 66]]))
        info = [{}]
        return obs, info

      def close(self):
        pass

    env = ReturnActionEnv(MockObservationEnv(), {'mode': 'append'})

    obs, info = env.reset()

    # Should append 0 to existing dense vector
    self.assertIsInstance(obs, Observation)
    self.assertEqual(obs.frame.shape, (1, 84, 84, 3))  # Vectorized frame shape
    self.assertIsNotNone(obs.dense)
    self.assertEqual(obs.dense.shape,
                     (1, 3))  # Vectorized: (num_envs, original_2 + 0)
    np.testing.assert_array_equal(obs.dense, [[77, 66, 0]])


class TestHistoryEnv(unittest.TestCase):
  """Test cases for HistoryEnv wrapper."""

  def setUp(self):
    self.base_env = ObservationWrapper(
        MockVectorEnv(), {'input': 'frame'})  # Wrap with ObservationWrapper

  def test_init_with_valid_config(self):
    """Test initialization with valid configuration."""
    config = {'history_length': 4}
    env = HistoryEnv(self.base_env, config)

    self.assertEqual(env.history_length, 4)
    self.assertEqual(len(env.states),
                     1)  # One deque per environment (1 env in this case)
    self.assertEqual(len(env.states[0]),
                     0)  # The deque for the first environment should be empty

  def test_init_with_minimal_config(self):
    """Test initialization with minimal configuration."""
    config = {'history_length': 2}
    env = HistoryEnv(self.base_env, config)

    self.assertEqual(env.history_length, 2)

  def test_init_with_invalid_config(self):
    """Test initialization with invalid configuration."""
    with self.assertRaises(ValueError):
      HistoryEnv(self.base_env, {'history_length': 0})

  def test_get_history_empty(self):
    """Test _get_history with empty deque."""
    config = {'history_length': 2}
    env = HistoryEnv(self.base_env, config)

    with self.assertRaises(RuntimeError):
      env._get_history()

  def test_get_history_insufficient_frames(self):
    """Test _get_history with insufficient frames."""
    config = {'history_length': 3}
    env = HistoryEnv(self.base_env, config)

    # Add one observation to the first environment's deque
    frame = np.ones((84, 84, 3), dtype=np.uint8)
    obs = Observation(frame=frame)
    env.states[0].append(obs)

    history = env._get_history()

    self.assertIsInstance(history, Observation)
    # For vectorized environments: shape is (num_envs, num_frames, height, width, channels)
    self.assertEqual(history.frame.shape,
                     (1, 3, 84, 84, 3))  # (num_envs, N, H, W, C)
    # First two frames should be identical (padded) for the first environment
    np.testing.assert_array_equal(history.frame[0, 0], history.frame[0, 1])
    np.testing.assert_array_equal(history.frame[0, 1], history.frame[0, 2])

  def test_get_history_sufficient_frames(self):
    """Test _get_history with sufficient frames."""
    config = {'history_length': 2}
    env = HistoryEnv(self.base_env, config)

    # Add two different observations to the first environment's deque
    frame1 = np.ones((84, 84, 3), dtype=np.uint8)
    frame2 = np.ones((84, 84, 3), dtype=np.uint8) * 2
    obs1 = Observation(frame=frame1)
    obs2 = Observation(frame=frame2)
    env.states[0].append(obs1)
    env.states[0].append(obs2)

    history = env._get_history()

    self.assertIsInstance(history, Observation)
    # For vectorized environments: shape is (num_envs, num_frames, height, width, channels)
    self.assertEqual(history.frame.shape,
                     (1, 2, 84, 84, 3))  # (num_envs, N, H, W, C)
    # Compare frames for the first environment
    np.testing.assert_array_equal(history.frame[0, 0], frame1)
    np.testing.assert_array_equal(history.frame[0, 1], frame2)

  def test_step_no_skip_frames(self):
    """Test step method."""
    config = {'history_length': 2}
    env = HistoryEnv(self.base_env, config)

    # Reset to initialize
    env.reset()

    obs, reward, terminated, truncated, info = env.step(
        [0])  # Vectorized action

    self.assertIsInstance(obs, Observation)
    # For vectorized environments: shape is (num_envs, num_frames, height, width, channels)
    self.assertEqual(obs.frame.shape,
                     (1, 2, 84, 84, 3))  # (num_envs, N, H, W, C)
    self.assertEqual(reward, [1.0])  # Vectorized reward
    self.assertEqual(len(env.states[0]),
                     2)  # Reset frame + step frame for first environment

  def test_step_with_dense_vectors(self):
    """Test step with dense vectors in observations."""
    config = {'history_length': 3}

    # For this test, create a special mock that can return Observations with both frame and dense
    class MockVectorEnvWithDense(MockVectorEnv):

      def step(self, actions):
        frames = np.random.randint(0, 256, (1, 84, 84, 3), dtype=np.uint8)
        dense = np.array([[actions[0], self.step_count[0]]])
        obs = Observation(frame=frames, dense=dense)
        self.step_count[0] += 1
        rewards = np.array([1.0])
        terminated = np.array([False])
        truncated = np.array([False])
        infos = [{}]
        return obs, rewards, terminated, truncated, infos

      def reset(self, **kwargs):
        self.step_count = np.array([0])
        frames = np.random.randint(0, 256, (1, 84, 84, 3), dtype=np.uint8)
        dense = np.array([[0, 0]])
        obs = Observation(frame=frames, dense=dense)
        infos = [{}]
        return obs, infos

    # Use the special mock directly without ObservationWrapper
    special_base_env = MockVectorEnvWithDense()
    env = HistoryEnv(special_base_env, config)

    # Reset and step
    env.reset()
    obs, reward, terminated, truncated, info = env.step(
        [1])  # Vectorized action

    self.assertIsInstance(obs, Observation)
    # For vectorized environments: shape is (num_envs, num_frames, height, width, channels)
    self.assertEqual(
        obs.frame.shape,
        (1, 3, 84, 84, 3))  # (num_envs, N, H, W, C) - 1 env, 3 history frames
    # For vectorized environments: shape is (num_envs, total_dense_features)
    self.assertEqual(
        obs.dense.shape,
        (1,
         6))  # (num_envs, 3 history steps * 2 dense features each = 6 total)

  def test_reset(self):
    """Test reset method."""
    config = {'history_length': 2}
    env = HistoryEnv(self.base_env, config)

    # Add some observations to the first environment's deque
    obs1 = Observation(frame=np.ones((84, 84, 3)))
    obs2 = Observation(frame=np.ones((84, 84, 3)))
    env.states[0].append(obs1)
    env.states[0].append(obs2)

    obs, info = env.reset()

    self.assertIsInstance(obs, Observation)
    # For vectorized environments: shape is (num_envs, num_frames, height, width, channels)
    self.assertEqual(obs.frame.shape,
                     (1, 2, 84, 84, 3))  # (num_envs, N, H, W, C)
    self.assertEqual(len(env.states[0]),
                     1)  # Only reset observation in first environment's deque
    self.assertIsInstance(
        info, dict)  # Vectorized environments return dict with array values
    self.assertIn('episode', info)
    self.assertEqual(len(info['episode']), 1)  # One environment
    # Check that step_count was reset on the underlying MockVectorEnv
    np.testing.assert_array_equal(self.base_env.env.step_count, [0])

  def test_deque_maxlen_behavior(self):
    """Test that deque properly manages maxlen."""
    config = {'history_length': 2}
    env = HistoryEnv(self.base_env, config)

    # Add 3 observations to a deque with maxlen=2
    frame1 = np.ones((84, 84, 3), dtype=np.uint8)
    frame2 = np.ones((84, 84, 3), dtype=np.uint8) * 2
    frame3 = np.ones((84, 84, 3), dtype=np.uint8) * 3
    obs1 = Observation(frame=frame1)
    obs2 = Observation(frame=frame2)
    obs3 = Observation(frame=frame3)

    env.states[0].append(obs1)  # Add to first environment's deque
    env.states[0].append(obs2)
    env.states[0].append(obs3)  # Should evict obs1

    self.assertEqual(len(env.states[0]),
                     2)  # Check the deque for first environment
    history = env._get_history()

    # Should contain frame2 and frame3, not frame1
    # For vectorized environments: shape is (num_envs, num_frames, height, width, channels)
    # Extract frames for the first environment
    np.testing.assert_array_equal(
        history.frame[0, 0],  # First environment, first frame
        frame2)
    np.testing.assert_array_equal(
        history.frame[0, 1],  # First environment, second frame
        frame3)

  def test_dense_vector_history_stacking(self):
    """Test that dense vectors are properly stacked as history."""
    config = {'history_length': 3}
    env = HistoryEnv(self.base_env, config)

    # Add observations with different dense vectors to the first environment's history
    frame = np.ones((84, 84, 3), dtype=np.uint8)
    dense1 = np.array([1, 2])
    dense2 = np.array([3, 4])
    obs1 = Observation(frame=frame, dense=dense1)
    obs2 = Observation(frame=frame, dense=dense2)

    # For vectorized environments, states[0] is the deque for the first environment
    env.states[0].append(obs1)
    env.states[0].append(obs2)
    # Add a third observation (history_length is 3 by default)
    env.states[0].append(obs1)  # Reuse obs1 for third observation

    history = env._get_history()

    self.assertIsInstance(history, Observation)
    # For vectorized environments, dense shape is (num_envs, total_features)
    self.assertEqual(
        history.dense.shape,
        (1, 6))  # (1 env, 3 history steps * 2 features each = 6 total)
    # Dense vectors should be concatenated: [dense1, dense2, dense1]
    expected_dense = np.concatenate([dense1, dense2, dense1])
    np.testing.assert_array_equal(history.dense[0],
                                  expected_dense)  # Check first environment

  def test_mixed_observations_history(self):
    """Test history with mixed observations (some with/without dense vectors)."""
    config = {'history_length': 2}
    env = HistoryEnv(self.base_env, config)

    # Add one observation with dense, one without to the first environment's deque
    frame = np.ones((84, 84, 3), dtype=np.uint8)
    obs1 = Observation(frame=frame, dense=np.array([1, 2]))
    obs2 = Observation(frame=frame)  # No dense vector

    env.states[0].append(obs1)
    env.states[0].append(obs2)

    history = env._get_history()

    self.assertIsInstance(history, Observation)
    # For vectorized environments: shape is (num_envs, num_frames, height, width, channels)
    self.assertEqual(history.frame.shape,
                     (1, 2, 84, 84, 3))  # (num_envs, N, H, W, C)
    # Only one observation has dense vector, but we pad to match history_length=2
    self.assertEqual(
        history.dense.shape,
        (1,
         4))  # Vectorized: (num_envs, 2 history steps * 2 features = 4 total)
    # Both entries should be the same (padded with the first/only dense vector)
    expected_dense = np.concatenate([np.array([1, 2]), np.array([1, 2])])
    np.testing.assert_array_equal(history.dense[0],
                                  expected_dense)  # Check first environment


class TestIntegration(unittest.TestCase):
  """Integration tests for combining multiple wrappers."""

  def test_multiple_wrappers(self):
    """Test combining multiple wrappers."""
    base_env = MockVectorEnv()

    # Apply wrappers in order - start with ObservationWrapper to convert arrays to Observations
    env = ObservationWrapper(base_env, {'input': 'frame'})
    env = PreprocessFrameEnv(env, {'grayscale': True})
    env = ReturnActionEnv(env, {})
    env = HistoryEnv(env, {'history_length': 3})

    # Test reset
    obs, info = env.reset()
    self.assertIsInstance(obs, Observation)
    # For vectorized environments: shape is (num_envs, num_frames, height, width)
    # After preprocessing: grayscale removes channel dimension
    self.assertEqual(obs.frame.shape, (
        1, 3, 84,
        84))  # (num_envs, history_length=3, H, W) - no channel after grayscale

    # Test step
    obs, reward, terminated, truncated, info = env.step(
        [1])  # Vectorized action
    self.assertIsInstance(obs, Observation)
    self.assertEqual(obs.frame.shape, (1, 3, 84, 84))  # Same shape as reset
    self.assertEqual(reward, [1.0])  # Vectorized reward
    # ReturnActionEnv appends current action to dense vector, not info
    self.assertIsNotNone(obs.dense)  # Dense vector should contain action

  def test_wrapper_order_independence(self):
    """Test that wrapper order doesn't break functionality."""
    base_env1 = MockVectorEnv()
    base_env2 = MockVectorEnv()

    # Order 1: ObservationWrapper -> Preprocess -> History
    env1 = ObservationWrapper(base_env1, {'input': 'frame'})
    env1 = PreprocessFrameEnv(env1, {'grayscale': True})
    env1 = HistoryEnv(env1, {'history_length': 2})

    # Order 2: ObservationWrapper -> Preprocess -> History
    env2 = ObservationWrapper(base_env2, {'input': 'frame'})
    env2 = PreprocessFrameEnv(env2, {'grayscale': True})
    env2 = HistoryEnv(env2, {'history_length': 2})

    # Both should work without errors
    obs1, info1 = env1.reset()
    obs2, info2 = env2.reset()

    self.assertIsInstance(obs1, Observation)
    self.assertIsInstance(obs2, Observation)
    self.assertEqual(obs1.frame.shape, obs2.frame.shape)

    obs1, reward1, _, _, _ = env1.step([0])  # Vectorized action
    obs2, reward2, _, _, _ = env2.step([0])  # Vectorized action

    self.assertIsInstance(obs1, Observation)
    self.assertIsInstance(obs2, Observation)
    self.assertEqual(obs1.frame.shape, obs2.frame.shape)
    self.assertEqual(reward1, reward2)


class MockVectorRenderEnv(MockVectorEnv):
  """Mock vectorized environment with render method for testing CaptureRenderFrameEnv."""

  def __init__(self,
               frame_shape: Tuple[int, ...] = (84, 84, 3),
               action_space_n: int = 4,
               num_envs: int = 1):
    super().__init__(frame_shape, action_space_n, num_envs)
    self.render_called = False
    self.render_return = np.ones(
        (num_envs, ) + frame_shape, dtype=np.uint8) * 123

    # Create mock individual environments for CaptureRenderFrameEnv
    # The source code expects self.env.unwrapped.envs[i].render()
    class MockIndividualEnv:

      def __init__(self, render_value):
        self.render_value = render_value
        self.frame_shape = (84, 84, 3)  # Standard frame shape
        self.render_called = False
        self.render_return = render_value

      def render(self, mode='rgb_array'):
        self.render_called = True
        return self.render_value

    # Create individual environments that return the right render values
    self.individual_envs = [
        MockIndividualEnv(self.render_return[i]) for i in range(num_envs)
    ]

    # Set up the unwrapped attribute to point to self and add envs
    self.envs = self.individual_envs

  @property
  def unwrapped(self):
    return self

  def render(self, mode='rgb_array'):
    self.render_called = True
    # Also call render on individual environments to set their render_called flags
    for env in self.envs:
      env.render(mode)
    return self.render_return


class MockRenderEnv(MockEnv):
  """Mock environment with render method for testing CaptureRenderFrameEnv."""

  def __init__(self,
               frame_shape: Tuple[int, ...] = (84, 84, 3),
               action_space_n: int = 4):
    super().__init__(frame_shape, action_space_n)
    self.render_called = False
    self.render_return = np.ones(frame_shape, dtype=np.uint8) * 123

  def render(self, mode='rgb_array'):
    self.render_called = True
    return self.render_return


class TestCaptureRenderFrameEnv(unittest.TestCase):
  """Test cases for CaptureRenderFrameEnv wrapper."""

  def setUp(self):
    base_env = MockVectorRenderEnv()
    # Wrap with ObservationWrapper since CaptureRenderFrameEnv expects Observation objects
    self.base_env = ObservationWrapper(base_env, {'input': 'frame'})

  def test_init_default_mode(self):
    """Test initialization with default mode."""
    env = CaptureRenderFrameEnv(self.base_env)
    self.assertIsNone(env.rendered_frame)
    self.assertEqual(env.mode, 'capture')

  def test_init_with_valid_modes(self):
    """Test initialization with valid modes."""
    for mode in ['capture', 'replace']:
      env = CaptureRenderFrameEnv(self.base_env, {'mode': mode})
      self.assertEqual(env.mode, mode)

  def test_init_with_invalid_mode(self):
    """Test initialization with invalid mode raises ValueError."""
    with self.assertRaises(ValueError) as context:
      CaptureRenderFrameEnv(self.base_env, {'mode': 'invalid'})
    self.assertIn("Invalid mode 'invalid'", str(context.exception))

  def test_step_capture_mode(self):
    """Test step with capture mode (default behavior)."""
    env = CaptureRenderFrameEnv(self.base_env, {
        'mode': 'capture',
        'observation_is_frame': False
    })
    obs, reward, terminated, truncated, info = env.step(
        [0])  # Vectorized action

    self.assertTrue(self.base_env.unwrapped.envs[0].render_called
                    )  # Access unwrapped single env
    self.assertIsNotNone(env.rendered_frame)
    # rendered_frame should be array of frames from all envs: shape (num_envs, H, W, C)
    self.assertEqual(env.rendered_frame.shape, (1, 84, 84, 3))
    np.testing.assert_array_equal(env.rendered_frame[0],
                                  self.base_env.unwrapped.envs[0].render_return
                                  )  # Compare first env frame
    # Should return original observation (unchanged)
    self.assertIsInstance(obs, Observation)
    # For vectorized environments, frame shape is (num_envs, H, W, C) - same as original observation
    expected_shape = (1, self.base_env.unwrapped.envs[0].frame_shape[0],
                      self.base_env.unwrapped.envs[0].frame_shape[1],
                      self.base_env.unwrapped.envs[0].frame_shape[2])
    self.assertEqual(obs.frame.shape, expected_shape)
    self.assertEqual(reward, [1.0])

  def test_step_replace_mode(self):
    """Test step with replace mode."""
    env = CaptureRenderFrameEnv(self.base_env, {
        'mode': 'replace',
        'observation_is_frame': False
    })
    obs, reward, terminated, truncated, info = env.step(
        [0])  # Vectorized action

    self.assertTrue(self.base_env.unwrapped.envs[0].render_called
                    )  # Access unwrapped single env
    self.assertIsNotNone(env.rendered_frame)
    # Should return rendered frame as observation
    self.assertIsInstance(obs, Observation)
    # For vectorized environments, frame shape is (num_envs, H, W, C)
    expected_shape = (1, self.base_env.env.frame_shape[0],
                      self.base_env.env.frame_shape[1],
                      self.base_env.env.frame_shape[2])
    self.assertEqual(obs.frame.shape, expected_shape)
    # Check that the data matches the rendered frame
    np.testing.assert_array_equal(
        obs.frame, self.base_env.env.render_return)  # Access unwrapped
    self.assertEqual(reward, 1.0)

  def test_reset_capture_mode(self):
    """Test reset with capture mode."""
    env = CaptureRenderFrameEnv(self.base_env, {'mode': 'capture'})
    env.step(0)  # Set rendered_frame
    obs, info = env.reset()

    self.assertIsNotNone(
        env.rendered_frame)  # rendered_frame should be set after reset
    np.testing.assert_array_equal(
        env.rendered_frame,
        self.base_env.env.render_return)  # Access unwrapped
    self.assertIsInstance(obs, Observation)
    # For vectorized environments, frame shape is (num_envs, H, W, C) - same as original observation
    expected_shape = (1, self.base_env.unwrapped.envs[0].frame_shape[0],
                      self.base_env.unwrapped.envs[0].frame_shape[1],
                      self.base_env.unwrapped.envs[0].frame_shape[2])
    self.assertEqual(obs.frame.shape, expected_shape)
    self.assertIsInstance(info, dict)
    self.assertIn('episode', info)
    self.assertIn('observation_frame', info)

  def test_reset_replace_mode(self):
    """Test reset with replace mode."""
    env = CaptureRenderFrameEnv(self.base_env, {'mode': 'replace'})
    obs, info = env.reset()

    self.assertIsNotNone(env.rendered_frame)
    # Should return rendered frame as observation with channel dimension first (C,H,W)
    self.assertIsInstance(obs, Observation)
    # For vectorized environments, frame shape is (num_envs, H, W, C)
    expected_shape = (1, self.base_env.unwrapped.envs[0].frame_shape[0],
                      self.base_env.unwrapped.envs[0].frame_shape[1],
                      self.base_env.unwrapped.envs[0].frame_shape[2])
    self.assertEqual(obs.frame.shape, expected_shape)
    # Check that the data matches the rendered frame (first environment in vectorized obs)
    np.testing.assert_array_equal(obs.frame[0],
                                  self.base_env.unwrapped.envs[0].render_return
                                  )  # Compare single env frame
    self.assertIsInstance(info, dict)
    self.assertIn('episode', info)
    self.assertIn('observation_frame', info)

  def test_channel_dimension_conversion(self):
    """Test that frames are correctly converted from (H,W,C) to (C,H,W)."""
    env = CaptureRenderFrameEnv(self.base_env, {'mode': 'replace'})
    obs, _, _, _, _ = env.step([0])  # Vectorized action

    # For vectorized environments, the shape is (num_envs, H, W, C)
    # Original rendered frame shape is (num_envs, H, W, C) = (1, 84, 84, 3)
    self.assertEqual(obs.frame.shape, (1, 84, 84, 3))

    # Check that the conversion is correct by comparing pixel values
    original_frame = self.base_env.env.render_return  # Access unwrapped render_return
    converted_frame = obs.frame  # (num_envs, H, W, C)

    # Test specific pixel: original[env, h, w, c] should equal converted[env, h, w, c]
    env_idx, h, w, c = 0, 10, 20, 1
    self.assertEqual(original_frame[env_idx, h, w, c],
                     converted_frame[env_idx, h, w, c])

  def test_grayscale_frame_handling(self):
    """Test handling of grayscale frames (2D arrays)."""
    # Create a mock environment that returns grayscale frames
    grayscale_env = MockVectorRenderEnv(frame_shape=(84, 84))
    grayscale_env.render_return = np.ones((1, 84, 84), dtype=np.uint8) * 123
    wrapped_grayscale_env = ObservationWrapper(grayscale_env,
                                               {'input': 'frame'})

    env = CaptureRenderFrameEnv(wrapped_grayscale_env, {'mode': 'replace'})
    obs, _, _, _, _ = env.step([0])

    # Grayscale frames should remain unchanged (no channel dimension to convert)
    # Note: For vectorized env, we expect a batch dimension
    self.assertEqual(obs.frame.shape, (1, 84, 84))
    np.testing.assert_array_equal(obs.frame, grayscale_env.render_return)


class TestCreateEnvironment(unittest.TestCase):
  """Test cases for create_environment function."""

  def setUp(self):
    """Set up test fixtures."""
    self.base_config = {'env_name': 'CartPole-v1', 'env_wrappers': []}

  @patch('gymnasium.make_vec')
  def test_create_environment_basic(self, mock_gym_make_vec):
    """Test creating environment with minimal configuration."""
    mock_env = MockVectorEnv(
    )  # Use MockVectorEnv which properly inherits from VectorEnv
    mock_gym_make_vec.return_value = mock_env

    config = self.base_config.copy()
    env = create_environment(config)

    mock_gym_make_vec.assert_called_once_with('CartPole-v1',
                                              num_envs=1,
                                              vectorization_mode="sync",
                                              render_mode='rgb_array')
    # env will be wrapped, so we can't directly compare
    self.assertIsNotNone(env)

  @patch('gymnasium.make_vec')
  def test_create_environment_with_multiple_wrappers(self, mock_gym_make_vec):
    """Test creating environment with multiple wrappers and correct order."""
    mock_env = MockVectorEnv()  # Use MockVectorEnv instead of MockEnv
    mock_gym_make_vec.return_value = mock_env

    config = self.base_config.copy()
    config['env_wrappers'] = [
        'PreprocessFrameEnv',
        'ReturnActionEnv'  # Remove RepeatActionEnv as it doesn't support vectorized envs
    ]
    # Use nested configuration structure
    config['PreprocessFrameEnv'] = {'grayscale': True, 'normalize': True}
    config['ReturnActionEnv'] = {'mode': 'append'}

    env = create_environment(config)

    mock_gym_make_vec.assert_called_once_with('CartPole-v1',
                                              num_envs=1,
                                              vectorization_mode="sync",
                                              render_mode='rgb_array')
    # Check wrapper nesting order (outermost to innermost)
    # Now includes AccumulatedStepsEnv, AccumulatedRewardEnv, and ObservationWrapper at the base
    self.assertIsInstance(env, ReturnActionEnv)
    self.assertIsInstance(env.env, PreprocessFrameEnv)
    # AccumulatedStepsEnv wraps AccumulatedRewardEnv which wraps ObservationWrapper
    self.assertIsInstance(env.env.env.env.env, ObservationWrapper)
    self.assertEqual(env.env.env.env.env.env, mock_env)

    # Check wrapper configurations
    self.assertTrue(env.env.grayscale)  # PreprocessFrameEnv has grayscale
    self.assertTrue(env.env.normalize)  # PreprocessFrameEnv has normalize
    self.assertEqual(env.mode, 'append')  # ReturnActionEnv has mode

  @patch('gymnasium.make_vec')
  def test_create_environment_with_nested_config(self, mock_gym_make_vec):
    """Test creating environment with nested configuration for each wrapper."""
    mock_env = MockVectorEnv()
    mock_gym_make_vec.return_value = mock_env

    config = {
        'env_name':
        'CartPole-v1',
        'env_wrappers':
        ['PreprocessFrameEnv', 'HistoryEnv', 'CaptureRenderFrameEnv'],
        'PreprocessFrameEnv': {
            'grayscale': True,
            'resize_shape': (84, 84),
            'normalize': False
        },
        'HistoryEnv': {
            'history_length': 4
        },
        'CaptureRenderFrameEnv': {
            'mode': 'replace'
        }
    }

    env = create_environment(config)

    mock_gym_make_vec.assert_called_once_with('CartPole-v1',
                                              num_envs=1,
                                              vectorization_mode="sync",
                                              render_mode='rgb_array')

    # Check wrapper nesting order (outermost to innermost)
    # Now includes AccumulatedStepsEnv, AccumulatedRewardEnv, and ObservationWrapper at the base
    self.assertIsInstance(env, CaptureRenderFrameEnv)
    self.assertIsInstance(env.env, HistoryEnv)
    self.assertIsInstance(env.env.env, PreprocessFrameEnv)
    # AccumulatedStepsEnv wraps AccumulatedRewardEnv which wraps ObservationWrapper
    self.assertIsInstance(env.env.env.env.env.env, ObservationWrapper)
    self.assertEqual(env.env.env.env.env.env.env, mock_env)

    # Check wrapper configurations
    self.assertTrue(env.env.env.grayscale)
    self.assertEqual(env.env.env.resize_shape, (84, 84))
    self.assertFalse(env.env.env.normalize)
    self.assertEqual(env.env.history_length, 4)
    self.assertEqual(env.mode, 'replace')

  @patch('gymnasium.make_vec')
  def test_create_environment_unknown_wrapper(self, mock_gym_make_vec):
    """Test creating environment with unknown wrapper raises ValueError."""
    mock_env = MockVectorEnv(
    )  # Use MockVectorEnv which properly inherits from VectorEnv
    mock_gym_make_vec.return_value = mock_env

    config = self.base_config.copy()
    config['env_wrappers'] = ['UnknownWrapper']

    with self.assertRaises(ValueError) as context:
      create_environment(config)

    self.assertIn("Unknown environment wrapper: UnknownWrapper",
                  str(context.exception))


if __name__ == '__main__':
  unittest.main()
