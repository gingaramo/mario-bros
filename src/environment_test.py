import unittest
import numpy as np
import gymnasium as gym
from unittest.mock import Mock, MagicMock, patch
from typing import Tuple, Dict, Any
import cv2

from environment import PreprocessFrameEnv, RepeatActionEnv, ReturnActionEnv, HistoryEnv, CaptureRenderFrameEnv


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

  def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
    self.step_count += 1
    frame = np.random.randint(0, 256, self.frame_shape, dtype=np.uint8)
    reward = 1.0
    terminated = self.step_count >= self.max_steps
    truncated = False
    info = {'step': self.step_count}
    return frame, reward, terminated, truncated, info

  def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
    self.step_count = 0
    frame = np.random.randint(0, 256, self.frame_shape, dtype=np.uint8)
    info = {'episode': 1}
    return frame, info


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

    self.assertEqual(processed.shape, (42, 42, 3))

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
    with self.assertRaises(ValueError):
      env.preprocess(frame)

  def test_step(self):
    """Test step method."""
    config = {'grayscale': True}
    env = PreprocessFrameEnv(self.base_env, config)

    obs, reward, terminated, truncated, info = env.step(0)

    self.assertEqual(obs.shape, (84, 84))
    self.assertEqual(reward, 1.0)
    self.assertIsInstance(terminated, bool)
    self.assertIsInstance(truncated, bool)
    self.assertIsInstance(info, dict)

  def test_reset(self):
    """Test reset method."""
    config = {'grayscale': True}
    env = PreprocessFrameEnv(self.base_env, config)

    obs, info = env.reset()

    self.assertEqual(obs.shape, (84, 84))
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

    self.assertEqual(obs.shape, (84, 84, 3))
    self.assertIsInstance(info, dict)
    self.assertEqual(self.base_env.step_count, 0)


class TestReturnActionEnv(unittest.TestCase):
  """Test cases for ReturnActionEnv wrapper."""

  def setUp(self):
    self.base_env = MockEnv()

  def test_init(self):
    """Test initialization."""
    env = ReturnActionEnv(self.base_env)

    self.assertIsNone(env.prev_action)

  def test_step_first_action(self):
    """Test step with first action."""
    env = ReturnActionEnv(self.base_env)

    obs, reward, terminated, truncated, info = env.step(1)

    self.assertIsNone(info['prev_action'])  # No previous action
    self.assertEqual(env.prev_action, 1)

  def test_step_subsequent_actions(self):
    """Test step with subsequent actions."""
    env = ReturnActionEnv(self.base_env)

    # First step
    env.step(1)

    # Second step
    obs, reward, terminated, truncated, info = env.step(2)

    self.assertEqual(info['prev_action'], 1)  # Previous action
    self.assertEqual(env.prev_action, 2)  # Current action stored

  def test_step_preserves_existing_info(self):
    """Test that step preserves existing info."""
    env = ReturnActionEnv(self.base_env)

    obs, reward, terminated, truncated, info = env.step(1)

    self.assertIn('step', info)  # From base env
    self.assertIn('prev_action', info)  # Added by wrapper

  def test_reset(self):
    """Test reset method."""
    env = ReturnActionEnv(self.base_env)

    # Take a step to set prev_action
    env.step(1)
    self.assertEqual(env.prev_action, 1)

    # Reset
    obs, info = env.reset()

    self.assertIsNone(env.prev_action)
    self.assertEqual(obs.shape, (84, 84, 3))
    self.assertIsInstance(info, dict)


class TestHistoryEnv(unittest.TestCase):
  """Test cases for HistoryEnv wrapper."""

  def setUp(self):
    self.base_env = MockEnv()

  def test_init_with_valid_config(self):
    """Test initialization with valid configuration."""
    config = {'history_length': 4, 'num_skip_frames': 2}
    env = HistoryEnv(self.base_env, config)

    self.assertEqual(env.history_length, 4)
    self.assertEqual(env.num_skip_frames, 2)
    self.assertEqual(len(env.states), 0)

  def test_init_with_minimal_config(self):
    """Test initialization with minimal configuration."""
    config = {'history_length': 2}
    env = HistoryEnv(self.base_env, config)

    self.assertEqual(env.history_length, 2)
    self.assertEqual(env.num_skip_frames, 0)

  def test_init_with_invalid_config(self):
    """Test initialization with invalid configuration."""
    with self.assertRaises(ValueError):
      HistoryEnv(self.base_env, {'history_length': 0})

    with self.assertRaises(ValueError):
      HistoryEnv(self.base_env, {'history_length': 2, 'num_skip_frames': -1})

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

    # Add one frame
    frame = np.ones((84, 84, 3), dtype=np.uint8)
    env.states.append(frame)

    history = env._get_history()

    self.assertEqual(history.shape, (3, 84, 84, 3))
    # First two frames should be identical (padded)
    np.testing.assert_array_equal(history[0], history[1])
    np.testing.assert_array_equal(history[1], history[2])

  def test_get_history_sufficient_frames(self):
    """Test _get_history with sufficient frames."""
    config = {'history_length': 2}
    env = HistoryEnv(self.base_env, config)

    # Add two different frames
    frame1 = np.ones((84, 84, 3), dtype=np.uint8)
    frame2 = np.ones((84, 84, 3), dtype=np.uint8) * 2
    env.states.append(frame1)
    env.states.append(frame2)

    history = env._get_history()

    self.assertEqual(history.shape, (2, 84, 84, 3))
    np.testing.assert_array_equal(history[0], frame1)
    np.testing.assert_array_equal(history[1], frame2)

  def test_step_no_skip_frames(self):
    """Test step with no frame skipping."""
    config = {'history_length': 2, 'num_skip_frames': 0}
    env = HistoryEnv(self.base_env, config)

    # Reset to initialize
    env.reset()

    obs, reward, terminated, truncated, info = env.step(0)

    self.assertEqual(obs.shape, (2, 84, 84, 3))
    self.assertEqual(reward, 1.0)  # Single step reward
    self.assertEqual(len(env.states), 2)  # Reset frame + step frame

  def test_step_with_skip_frames(self):
    """Test step with frame skipping."""
    config = {'history_length': 3, 'num_skip_frames': 2}
    env = HistoryEnv(self.base_env, config)

    # Reset to initialize
    env.reset()

    obs, reward, terminated, truncated, info = env.step(0)

    self.assertEqual(obs.shape, (3, 84, 84, 3))
    self.assertEqual(reward, 3.0)  # 3 steps (1 + 2 skipped)
    self.assertEqual(len(env.states), 3)  # Maxlen=3, so only last 3 frames
    self.assertEqual(self.base_env.step_count, 3)

  def test_step_early_termination_with_skip(self):
    """Test step with early termination during frame skipping."""
    self.base_env.max_steps = 2
    config = {'history_length': 2, 'num_skip_frames': 5}
    env = HistoryEnv(self.base_env, config)

    # Reset to initialize
    env.reset()

    obs, reward, terminated, truncated, info = env.step(0)

    self.assertTrue(terminated)
    self.assertEqual(reward, 2.0)  # Only 2 steps before termination
    self.assertEqual(self.base_env.step_count, 2)

  def test_reset(self):
    """Test reset method."""
    config = {'history_length': 2}
    env = HistoryEnv(self.base_env, config)

    # Add some frames to states
    env.states.append(np.ones((84, 84, 3)))
    env.states.append(np.ones((84, 84, 3)))

    obs, info = env.reset()

    self.assertEqual(obs.shape, (2, 84, 84, 3))
    self.assertEqual(len(env.states), 1)  # Only reset frame
    self.assertIsInstance(info, dict)
    self.assertEqual(self.base_env.step_count, 0)

  def test_deque_maxlen_behavior(self):
    """Test that deque properly manages maxlen."""
    config = {'history_length': 2}
    env = HistoryEnv(self.base_env, config)

    # Add 3 frames to a deque with maxlen=2
    frame1 = np.ones((84, 84, 3), dtype=np.uint8)
    frame2 = np.ones((84, 84, 3), dtype=np.uint8) * 2
    frame3 = np.ones((84, 84, 3), dtype=np.uint8) * 3

    env.states.append(frame1)
    env.states.append(frame2)
    env.states.append(frame3)  # Should evict frame1

    self.assertEqual(len(env.states), 2)
    history = env._get_history()

    # Should contain frame2 and frame3, not frame1
    np.testing.assert_array_equal(history[0], frame2)
    np.testing.assert_array_equal(history[1], frame3)


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
    self.assertEqual(obs.shape, (3, 84, 84))  # history_length=3, grayscale

    # Test step
    obs, reward, terminated, truncated, info = env.step(1)
    self.assertEqual(obs.shape, (3, 84, 84))
    self.assertEqual(reward, 2.0)  # 2 repeated actions
    self.assertIn('prev_action', info)

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

    self.assertEqual(obs1.shape, obs2.shape)

    obs1, reward1, _, _, _ = env1.step(0)
    obs2, reward2, _, _, _ = env2.step(0)

    self.assertEqual(obs1.shape, obs2.shape)
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

  def test_init(self):
    """Test initialization."""
    env = CaptureRenderFrameEnv(self.base_env)
    self.assertIsNone(env.rendered_frame)

  def test_step_captures_rendered_frame(self):
    """Test that step captures the rendered frame."""
    env = CaptureRenderFrameEnv(self.base_env)
    obs, reward, terminated, truncated, info = env.step(0)
    self.assertTrue(self.base_env.render_called)
    self.assertIsNotNone(env.rendered_frame)
    np.testing.assert_array_equal(env.rendered_frame,
                                  self.base_env.render_return)
    self.assertEqual(obs.shape, self.base_env.frame_shape)
    self.assertEqual(reward, 1.0)

  def test_step_multiple_calls(self):
    """Test that rendered_frame updates on each step."""
    env = CaptureRenderFrameEnv(self.base_env)
    env.step(0)
    first_frame = env.rendered_frame.copy()
    # Change what render returns
    self.base_env.render_return = np.ones(self.base_env.frame_shape,
                                          dtype=np.uint8) * 77
    env.step(1)
    np.testing.assert_array_equal(env.rendered_frame,
                                  self.base_env.render_return)
    self.assertFalse(np.array_equal(first_frame, env.rendered_frame))

  def test_reset_clears_rendered_frame(self):
    """Test that reset clears the rendered frame."""
    env = CaptureRenderFrameEnv(self.base_env)
    env.step(0)
    self.assertIsNotNone(env.rendered_frame)
    obs, info = env.reset()
    self.assertIsNone(env.rendered_frame)
    self.assertEqual(obs.shape, self.base_env.frame_shape)
    self.assertIsInstance(info, dict)


if __name__ == '__main__':
  unittest.main()
