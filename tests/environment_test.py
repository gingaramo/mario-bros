import unittest
import numpy as np
import gymnasium as gym
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


class MockEnv(gym.Env):
  """Mock environment for testing wrappers."""

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
    # Create a simple gym environment that returns numpy arrays
    self.base_env = gym.make('CartPole-v1')

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
    self.assertEqual(obs.frame.shape,
                     (3, 84, 84))  # Should be converted to (C,H,W)

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
    obs, reward, terminated, truncated, info = wrapper.step(0)

    self.assertIsInstance(obs, Observation)
    self.assertIsNone(obs.frame)
    self.assertIsNotNone(obs.dense)
    self.assertEqual(obs.dense.shape,
                     (4, ))  # CartPole has 4-dimensional observation space
    self.assertIsInstance(reward, float)
    self.assertIsInstance(terminated, bool)
    self.assertIsInstance(truncated, bool)
    self.assertIsInstance(info, dict)

  def test_reset_dense_type(self):
    """Test reset method with dense input type."""
    wrapper = ObservationWrapper(self.base_env, {'input': 'dense'})

    obs, info = wrapper.reset()

    self.assertIsInstance(obs, Observation)
    self.assertIsNone(obs.frame)
    self.assertIsNotNone(obs.dense)
    self.assertEqual(obs.dense.shape,
                     (4, ))  # CartPole has 4-dimensional observation space
    self.assertIsInstance(info, dict)

  def tearDown(self):
    """Clean up after tests."""
    self.base_env.close()


class TestPreprocessFrameEnv(unittest.TestCase):
  """Test cases for PreprocessFrameEnv wrapper."""

  def setUp(self):
    self.base_env = MockEnv()

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
    """Test preprocessing with resize only."""
    config = {'resize_shape': (42, 42)}
    env = PreprocessFrameEnv(self.base_env, config)

    frame = np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8)
    processed = env.preprocess(frame)

    self.assertEqual(processed.shape, (3, 42, 42))

  def test_preprocess_grayscale_only(self):
    """Test preprocessing with grayscale only."""
    config = {'grayscale': True}
    env = PreprocessFrameEnv(self.base_env, config)

    frame = np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8)
    processed = env.preprocess(frame)

    self.assertEqual(processed.shape, (84, 84))

  def test_preprocess_normalize_only(self):
    """Test preprocessing with normalization only."""
    config = {'normalize': True}
    env = PreprocessFrameEnv(self.base_env, config)

    frame = np.ones((84, 84, 3), dtype=np.uint8) * 255
    processed = env.preprocess(frame)

    self.assertTrue(np.allclose(processed, 1.0))
    self.assertEqual(processed.dtype, np.float32)

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

    obs, reward, terminated, truncated, info = env.step(0)

    self.assertIsInstance(obs, Observation)
    self.assertEqual(obs.frame.shape, (84, 84))
    self.assertEqual(reward, 1.0)
    self.assertIsInstance(terminated, bool)
    self.assertIsInstance(truncated, bool)
    self.assertIsInstance(info, dict)

  def test_reset(self):
    """Test reset method."""
    config = {'grayscale': True}
    env = PreprocessFrameEnv(self.base_env, config)

    obs, info = env.reset()

    self.assertIsInstance(obs, Observation)
    self.assertEqual(obs.frame.shape, (84, 84))
    self.assertIsInstance(info, dict)


class TestRepeatActionEnv(unittest.TestCase):
  """Test cases for RepeatActionEnv wrapper."""

  def setUp(self):
    self.base_env = MockEnv()

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
    self.base_env = MockEnv()

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

    obs, reward, terminated, truncated, info = env.step(1)

    # Should return observation with dense vector containing current action
    self.assertIsInstance(obs, Observation)
    self.assertEqual(obs.frame.shape, (3, 84, 84))  # Original frame unchanged
    self.assertIsNotNone(obs.dense)  # Dense vector should exist
    self.assertEqual(obs.dense.shape, (1, ))  # Single value (current action)
    self.assertEqual(obs.dense[0], 1)  # Current action

  def test_step_append_mode_subsequent_actions(self):
    """Test step with append mode and subsequent actions."""
    env = ReturnActionEnv(self.base_env, {'mode': 'append'})

    # First step
    obs1, _, _, _, _ = env.step(1)
    self.assertEqual(obs1.dense[0], 1)  # First action

    # Second step
    obs2, reward, terminated, truncated, info = env.step(2)

    # Should return observation with dense vector containing current action
    self.assertIsInstance(obs2, Observation)
    self.assertEqual(obs2.frame.shape, (3, 84, 84))  # Original frame unchanged
    self.assertIsNotNone(obs2.dense)  # Dense vector should exist
    self.assertEqual(obs2.dense.shape, (1, ))  # Single value (current action)
    self.assertEqual(obs2.dense[0], 2)  # Current action

  def test_step_with_existing_dense_vector(self):
    """Test step when base environment already provides dense vector."""

    # Create a base env that returns observations with existing dense vectors
    class MockEnvWithDense(MockEnv):

      def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # Add existing dense vector to observation
        obs_with_dense = Observation(frame=obs.frame, dense=np.array([99, 88]))
        return obs_with_dense, reward, terminated, truncated, info

    base_env = MockEnvWithDense()
    env = ReturnActionEnv(base_env, {'mode': 'append'})

    obs, reward, terminated, truncated, info = env.step(5)

    # Should append action to existing dense vector
    self.assertIsInstance(obs, Observation)
    self.assertEqual(obs.frame.shape, (3, 84, 84))
    self.assertIsNotNone(obs.dense)
    self.assertEqual(obs.dense.shape, (3, ))  # Original 2 + action
    np.testing.assert_array_equal(obs.dense, [99, 88, 5])

  def test_reset_append_mode(self):
    """Test reset with append mode."""
    env = ReturnActionEnv(self.base_env, {'mode': 'append'})

    # Take a step first
    env.step(1)

    # Reset
    obs, info = env.reset()

    # Should return observation with dense vector containing 0 (no previous action)
    self.assertIsInstance(obs, Observation)
    self.assertEqual(obs.frame.shape, (3, 84, 84))
    self.assertIsNotNone(obs.dense)
    self.assertEqual(obs.dense.shape, (1, ))
    self.assertEqual(obs.dense[0], 0)  # No previous action
    self.assertIsInstance(info, dict)

  def test_reset_with_existing_dense_vector(self):
    """Test reset when base environment already provides dense vector."""

    # Create a base env that returns observations with existing dense vectors
    class MockEnvWithDense(MockEnv):

      def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # Add existing dense vector to observation
        obs_with_dense = Observation(frame=obs.frame, dense=np.array([77, 66]))
        return obs_with_dense, info

    base_env = MockEnvWithDense()
    env = ReturnActionEnv(base_env, {'mode': 'append'})

    obs, info = env.reset()

    # Should append 0 to existing dense vector
    self.assertIsInstance(obs, Observation)
    self.assertEqual(obs.frame.shape, (3, 84, 84))
    self.assertIsNotNone(obs.dense)
    self.assertEqual(obs.dense.shape, (3, ))  # Original 2 + 0
    np.testing.assert_array_equal(obs.dense, [77, 66, 0])


class TestHistoryEnv(unittest.TestCase):
  """Test cases for HistoryEnv wrapper."""

  def setUp(self):
    self.base_env = MockEnv()

  def test_init_with_valid_config(self):
    """Test initialization with valid configuration."""
    config = {'history_length': 4}
    env = HistoryEnv(self.base_env, config)

    self.assertEqual(env.history_length, 4)
    self.assertEqual(len(env.states), 0)

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

    # Add one observation
    frame = np.ones((84, 84, 3), dtype=np.uint8)
    obs = Observation(frame=frame)
    env.states.append(obs)

    history = env._get_history()

    self.assertIsInstance(history, Observation)
    self.assertEqual(history.frame.shape, (3, 3, 84, 84))  # (C, N, H, W)
    # First two frames should be identical (padded)
    np.testing.assert_array_equal(history.frame[:, 0], history.frame[:, 1])
    np.testing.assert_array_equal(history.frame[:, 1], history.frame[:, 2])

  def test_get_history_sufficient_frames(self):
    """Test _get_history with sufficient frames."""
    config = {'history_length': 2}
    env = HistoryEnv(self.base_env, config)

    # Add two different observations
    frame1 = np.ones((84, 84, 3), dtype=np.uint8)
    frame2 = np.ones((84, 84, 3), dtype=np.uint8) * 2
    obs1 = Observation(frame=frame1)
    obs2 = Observation(frame=frame2)
    env.states.append(obs1)
    env.states.append(obs2)

    history = env._get_history()

    self.assertIsInstance(history, Observation)
    self.assertEqual(history.frame.shape, (3, 2, 84, 84))  # (C, N, H, W)
    # Convert to (H, W, C) for comparison with original frames
    np.testing.assert_array_equal(history.frame[:, 0].transpose(1, 2, 0),
                                  frame1)
    np.testing.assert_array_equal(history.frame[:, 1].transpose(1, 2, 0),
                                  frame2)

  def test_step_no_skip_frames(self):
    """Test step method."""
    config = {'history_length': 2}
    env = HistoryEnv(self.base_env, config)

    # Reset to initialize
    env.reset()

    obs, reward, terminated, truncated, info = env.step(0)

    self.assertIsInstance(obs, Observation)
    self.assertEqual(obs.frame.shape, (3, 2, 84, 84))  # (C, N, H, W)
    self.assertEqual(reward, 1.0)  # Single step reward
    self.assertEqual(len(env.states), 2)  # Reset frame + step frame

  def test_step_with_dense_vectors(self):
    """Test step with dense vectors in observations."""
    config = {'history_length': 3}
    env = HistoryEnv(self.base_env, config)

    # Mock the base environment to return observations with dense vectors
    def mock_step(action):
      frame = np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8)
      dense = np.array([action, self.base_env.step_count])
      obs = Observation(frame=frame, dense=dense)
      self.base_env.step_count += 1
      return obs, 1.0, False, False, {}

    def mock_reset(**kwargs):
      self.base_env.step_count = 0
      frame = np.random.randint(0, 256, (84, 84, 3), dtype=np.uint8)
      dense = np.array([0, 0])
      obs = Observation(frame=frame, dense=dense)
      return obs, {}

    self.base_env.step = mock_step
    self.base_env.reset = mock_reset

    # Reset and step
    env.reset()
    obs, reward, terminated, truncated, info = env.step(1)

    self.assertIsInstance(obs, Observation)
    self.assertEqual(
        obs.frame.shape,
        (3, 3, 84, 84))  # (C, N, H, W) - 3 channels, 3 history frames
    self.assertEqual(
        obs.dense.shape,
        (6, ))  # 3 history steps * 2 dense features each = 6 total

  def test_reset(self):
    """Test reset method."""
    config = {'history_length': 2}
    env = HistoryEnv(self.base_env, config)

    # Add some observations to states
    obs1 = Observation(frame=np.ones((84, 84, 3)))
    obs2 = Observation(frame=np.ones((84, 84, 3)))
    env.states.append(obs1)
    env.states.append(obs2)

    obs, info = env.reset()

    self.assertIsInstance(obs, Observation)
    self.assertEqual(obs.frame.shape, (3, 2, 84, 84))  # (C, N, H, W)
    self.assertEqual(len(env.states), 1)  # Only reset observation
    self.assertIsInstance(info, dict)
    self.assertEqual(self.base_env.step_count, 0)

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

    env.states.append(obs1)
    env.states.append(obs2)
    env.states.append(obs3)  # Should evict obs1

    self.assertEqual(len(env.states), 2)
    history = env._get_history()

    # Should contain frame2 and frame3, not frame1
    # Convert to (H, W, C) for comparison with original frames
    np.testing.assert_array_equal(history.frame[:, 0].transpose(1, 2, 0),
                                  frame2)
    np.testing.assert_array_equal(history.frame[:, 1].transpose(1, 2, 0),
                                  frame3)

  def test_dense_vector_history_stacking(self):
    """Test that dense vectors are properly stacked as history."""
    config = {'history_length': 3}
    env = HistoryEnv(self.base_env, config)

    # Add observations with different dense vectors
    frame = np.ones((84, 84, 3), dtype=np.uint8)
    dense1 = np.array([1, 2])
    dense2 = np.array([3, 4])
    obs1 = Observation(frame=frame, dense=dense1)
    obs2 = Observation(frame=frame, dense=dense2)

    env.states.append(obs1)
    env.states.append(obs2)

    history = env._get_history()

    self.assertIsInstance(history, Observation)
    self.assertEqual(history.dense.shape,
                     (6, ))  # 3 history steps * 2 features each = 6 total
    # Dense vectors should be concatenated: [dense1, dense1, dense2]
    expected_dense = np.concatenate([dense1, dense1, dense2])
    np.testing.assert_array_equal(history.dense, expected_dense)

  def test_mixed_observations_history(self):
    """Test history with mixed observations (some with/without dense vectors)."""
    config = {'history_length': 2}
    env = HistoryEnv(self.base_env, config)

    # Add one observation with dense, one without
    frame = np.ones((84, 84, 3), dtype=np.uint8)
    obs1 = Observation(frame=frame, dense=np.array([1, 2]))
    obs2 = Observation(frame=frame)  # No dense vector

    env.states.append(obs1)
    env.states.append(obs2)

    history = env._get_history()

    self.assertIsInstance(history, Observation)
    self.assertEqual(history.frame.shape, (3, 2, 84, 84))  # (C, N, H, W)
    # Only one observation has dense vector, but we pad to match history_length=2
    self.assertEqual(history.dense.shape,
                     (4, ))  # 2 history steps * 2 features = 4 total
    # Both entries should be the same (padded with the first/only dense vector)
    expected_dense = np.concatenate([np.array([1, 2]), np.array([1, 2])])
    np.testing.assert_array_equal(history.dense, expected_dense)


class TestIntegration(unittest.TestCase):
  """Integration tests for combining multiple wrappers."""

  def test_multiple_wrappers(self):
    """Test combining multiple wrappers."""
    base_env = MockEnv()

    # Apply wrappers in order
    env = PreprocessFrameEnv(base_env, {'grayscale': True})
    env = RepeatActionEnv(env, {'num_repeat_action': 2})
    env = ReturnActionEnv(env, {})
    env = HistoryEnv(env, {'history_length': 3})

    # Test reset
    obs, info = env.reset()
    self.assertIsInstance(obs, Observation)
    self.assertEqual(obs.frame.shape,
                     (3, 84, 84))  # history_length=3, grayscale

    # Test step
    obs, reward, terminated, truncated, info = env.step(1)
    self.assertIsInstance(obs, Observation)
    self.assertEqual(obs.frame.shape, (3, 84, 84))
    self.assertEqual(reward, 2.0)  # 2 repeated actions
    # ReturnActionEnv appends current action to dense vector, not info
    self.assertIsNotNone(obs.dense)  # Dense vector should contain action

  def test_wrapper_order_independence(self):
    """Test that wrapper order doesn't break functionality."""
    base_env1 = MockEnv()
    base_env2 = MockEnv()

    # Order 1: Preprocess -> Repeat -> History
    env1 = PreprocessFrameEnv(base_env1, {'grayscale': True})
    env1 = RepeatActionEnv(env1, {'num_repeat_action': 2})
    env1 = HistoryEnv(env1, {'history_length': 2})

    # Order 2: Repeat -> Preprocess -> History
    env2 = RepeatActionEnv(base_env2, {'num_repeat_action': 2})
    env2 = PreprocessFrameEnv(env2, {'grayscale': True})
    env2 = HistoryEnv(env2, {'history_length': 2})

    # Both should work without errors
    obs1, info1 = env1.reset()
    obs2, info2 = env2.reset()

    self.assertIsInstance(obs1, Observation)
    self.assertIsInstance(obs2, Observation)
    self.assertEqual(obs1.frame.shape, obs2.frame.shape)

    obs1, reward1, _, _, _ = env1.step(0)
    obs2, reward2, _, _, _ = env2.step(0)

    self.assertIsInstance(obs1, Observation)
    self.assertIsInstance(obs2, Observation)
    self.assertEqual(obs1.frame.shape, obs2.frame.shape)
    self.assertEqual(reward1, reward2)


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
    self.base_env = MockRenderEnv()

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
    env = CaptureRenderFrameEnv(self.base_env, {'mode': 'capture'})
    obs, reward, terminated, truncated, info = env.step(0)

    self.assertTrue(self.base_env.render_called)
    self.assertIsNotNone(env.rendered_frame)
    np.testing.assert_array_equal(env.rendered_frame,
                                  self.base_env.render_return)
    # Should return original observation (unchanged but converted to C,H,W format)
    self.assertIsInstance(obs, Observation)
    expected_shape = (self.base_env.frame_shape[2],
                      self.base_env.frame_shape[0],
                      self.base_env.frame_shape[1])
    self.assertEqual(obs.frame.shape, expected_shape)
    self.assertEqual(reward, 1.0)

  def test_step_replace_mode(self):
    """Test step with replace mode."""
    env = CaptureRenderFrameEnv(self.base_env, {'mode': 'replace'})
    obs, reward, terminated, truncated, info = env.step(0)

    self.assertTrue(self.base_env.render_called)
    self.assertIsNotNone(env.rendered_frame)
    # Should return rendered frame as observation with channel dimension first (C,H,W)
    self.assertIsInstance(obs, Observation)
    expected_shape = (self.base_env.frame_shape[2],
                      self.base_env.frame_shape[0],
                      self.base_env.frame_shape[1])
    self.assertEqual(obs.frame.shape, expected_shape)
    # Check that the data is correctly transposed
    expected_frame = np.transpose(self.base_env.render_return, (2, 0, 1))
    np.testing.assert_array_equal(obs.frame, expected_frame)
    self.assertEqual(reward, 1.0)

  def test_reset_capture_mode(self):
    """Test reset with capture mode."""
    env = CaptureRenderFrameEnv(self.base_env, {'mode': 'capture'})
    env.step(0)  # Set rendered_frame
    obs, info = env.reset()

    self.assertIsNotNone(
        env.rendered_frame)  # rendered_frame should be set after reset
    np.testing.assert_array_equal(env.rendered_frame,
                                  self.base_env.render_return)
    self.assertIsInstance(obs, Observation)
    expected_shape = (self.base_env.frame_shape[2],
                      self.base_env.frame_shape[0],
                      self.base_env.frame_shape[1])
    self.assertEqual(obs.frame.shape, expected_shape)
    self.assertIsInstance(info, dict)

  def test_reset_replace_mode(self):
    """Test reset with replace mode."""
    env = CaptureRenderFrameEnv(self.base_env, {'mode': 'replace'})
    obs, info = env.reset()

    self.assertIsNotNone(env.rendered_frame)
    # Should return rendered frame as observation with channel dimension first (C,H,W)
    self.assertIsInstance(obs, Observation)
    expected_shape = (self.base_env.frame_shape[2],
                      self.base_env.frame_shape[0],
                      self.base_env.frame_shape[1])
    self.assertEqual(obs.frame.shape, expected_shape)
    # Check that the data is correctly transposed
    expected_frame = np.transpose(self.base_env.render_return, (2, 0, 1))
    np.testing.assert_array_equal(obs.frame, expected_frame)
    self.assertIsInstance(info, dict)

  def test_channel_dimension_conversion(self):
    """Test that frames are correctly converted from (H,W,C) to (C,H,W)."""
    env = CaptureRenderFrameEnv(self.base_env, {'mode': 'replace'})
    obs, _, _, _, _ = env.step(0)

    # Original rendered frame shape is (H,W,C) = (84,84,3)
    # After conversion should be (C,H,W) = (3,84,84)
    self.assertEqual(obs.frame.shape, (3, 84, 84))

    # Check that the conversion is correct by comparing pixel values
    original_frame = self.base_env.render_return  # (H,W,C)
    converted_frame = obs.frame  # (C,H,W)

    # Test specific pixel: original[h,w,c] should equal converted[c,h,w]
    h, w, c = 10, 20, 1
    self.assertEqual(original_frame[h, w, c], converted_frame[c, h, w])

  def test_grayscale_frame_handling(self):
    """Test handling of grayscale frames (2D arrays)."""
    # Create a mock environment that returns grayscale frames
    grayscale_env = MockRenderEnv(frame_shape=(84, 84))
    grayscale_env.render_return = np.ones((84, 84), dtype=np.uint8) * 123

    env = CaptureRenderFrameEnv(grayscale_env, {'mode': 'replace'})
    obs, _, _, _, _ = env.step(0)

    # Grayscale frames should remain unchanged (no channel dimension to convert)
    self.assertEqual(obs.frame.shape, (84, 84))
    np.testing.assert_array_equal(obs.frame, grayscale_env.render_return)


class TestCreateEnvironment(unittest.TestCase):
  """Test cases for create_environment function."""

  def setUp(self):
    """Set up test fixtures."""
    self.base_config = {'env_name': 'CartPole-v1', 'env_wrappers': []}

  @patch('gymnasium.make')
  def test_create_environment_basic(self, mock_gym_make):
    """Test creating environment with minimal configuration."""
    mock_env = MockEnv()  # Use MockEnv which properly inherits from gym.Env
    mock_gym_make.return_value = mock_env

    config = self.base_config.copy()
    env = create_environment(config)

    mock_gym_make.assert_called_once_with('CartPole-v1',
                                          render_mode='rgb_array')
    # env will be wrapped, so we can't directly compare
    self.assertIsNotNone(env)

  @patch('gymnasium.make')
  def test_create_environment_with_multiple_wrappers(self, mock_gym_make):
    """Test creating environment with multiple wrappers and correct order."""
    mock_env = MockEnv()  # Use MockEnv instead of Mock()
    mock_gym_make.return_value = mock_env

    config = self.base_config.copy()
    config['env_wrappers'] = [
        'PreprocessFrameEnv', 'RepeatActionEnv', 'ReturnActionEnv'
    ]
    # Use nested configuration structure
    config['PreprocessFrameEnv'] = {'grayscale': True, 'normalize': True}
    config['RepeatActionEnv'] = {'num_repeat_action': 3}
    config['ReturnActionEnv'] = {'mode': 'append'}

    env = create_environment(config)

    mock_gym_make.assert_called_once_with('CartPole-v1',
                                          render_mode='rgb_array')
    # Check wrapper nesting order (outermost to innermost)
    # Now includes ObservationWrapper at the base
    self.assertIsInstance(env, ReturnActionEnv)
    self.assertIsInstance(env.env, RepeatActionEnv)
    self.assertIsInstance(env.env.env, PreprocessFrameEnv)
    self.assertIsInstance(env.env.env.env, ObservationWrapper)
    self.assertEqual(env.env.env.env.env, mock_env)

    # Check wrapper configurations
    self.assertTrue(env.env.env.grayscale)
    self.assertTrue(env.env.env.normalize)
    self.assertEqual(env.env.num_repeat_action, 3)
    self.assertEqual(env.mode, 'append')
    self.assertTrue(env.env.env.normalize)
    self.assertEqual(env.env.num_repeat_action, 3)

  @patch('gymnasium.make')
  def test_create_environment_with_nested_config(self, mock_gym_make):
    """Test creating environment with nested configuration for each wrapper."""
    mock_env = MockEnv()
    mock_gym_make.return_value = mock_env

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

    mock_gym_make.assert_called_once_with('CartPole-v1',
                                          render_mode='rgb_array')

    # Check wrapper nesting order (outermost to innermost)
    # Now includes ObservationWrapper at the base
    self.assertIsInstance(env, CaptureRenderFrameEnv)
    self.assertIsInstance(env.env, HistoryEnv)
    self.assertIsInstance(env.env.env, PreprocessFrameEnv)
    self.assertIsInstance(env.env.env.env, ObservationWrapper)
    self.assertEqual(env.env.env.env.env, mock_env)

    # Check wrapper configurations
    self.assertTrue(env.env.env.grayscale)
    self.assertEqual(env.env.env.resize_shape, (84, 84))
    self.assertFalse(env.env.env.normalize)
    self.assertEqual(env.env.history_length, 4)
    self.assertEqual(env.mode, 'replace')

  @patch('gymnasium.make')
  def test_create_environment_unknown_wrapper(self, mock_gym_make):
    """Test creating environment with unknown wrapper raises ValueError."""
    mock_env = MockEnv()  # Use MockEnv which properly inherits from gym.Env
    mock_gym_make.return_value = mock_env

    config = self.base_config.copy()
    config['env_wrappers'] = ['UnknownWrapper']

    with self.assertRaises(ValueError) as context:
      create_environment(config)

    self.assertIn("Unknown environment wrapper: UnknownWrapper",
                  str(context.exception))


if __name__ == '__main__':
  unittest.main()
