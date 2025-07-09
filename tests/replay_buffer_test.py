import unittest
import torch
import numpy as np
from unittest.mock import Mock, MagicMock
import sys
import os

# Add src directory to path so we can import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
  sys.path.insert(0, src_path)

from replay_buffer import ReplayBuffer


class TestReplayBuffer(unittest.TestCase):
  """End-to-end test cases for ReplayBuffer."""

  def setUp(self):
    """Set up test fixtures."""
    self.device = torch.device('cpu')
    self.mock_summary_writer = Mock()
    self.config = {'size': 100}
    self.replay_buffer = ReplayBuffer(config=self.config,
                                      device=self.device,
                                      summary_writer=self.mock_summary_writer)

  def _create_mock_observation_tensors(self,
                                       has_frame=True,
                                       has_dense=True,
                                       batch_size=1):
    """Helper to create mock observation tensors similar to Observation.as_input()."""
    observations = []
    for _ in range(batch_size):
      frame = torch.randn(3, 84, 84) if has_frame else torch.empty(0)
      dense = torch.randn(4) if has_dense else torch.empty(0)
      observations.append([frame, dense])
    return observations

  def test_basic_workflow_with_frame_and_dense_observations(self):
    """Test complete workflow: append experiences and sample batches with both frame and dense data."""
    # Add multiple experiences with both frame and dense observations
    experiences = []
    for i in range(10):
      obs = self._create_mock_observation_tensors(has_frame=True,
                                                  has_dense=True)[0]
      next_obs = self._create_mock_observation_tensors(has_frame=True,
                                                       has_dense=True)[0]
      action = i % 4  # Actions 0-3
      reward = float(i * 0.1)
      done = (i == 9)  # Last experience is terminal

      self.replay_buffer.append(obs, action, reward, next_obs, done)
      experiences.append((obs, action, reward, next_obs, done))

    # Verify buffer state
    self.assertEqual(len(self.replay_buffer), 10)
    self.assertEqual(len(self.replay_buffer.buffer), 10)

    # Verify summary writer was called
    self.assertEqual(self.mock_summary_writer.add_scalar.call_count, 10)

    # Sample a batch and verify format
    batch_size = 5
    all_obs, all_actions, all_rewards, all_next_obs, all_done = self.replay_buffer.sample(
        batch_size)

    # Check observation tensors (frame and dense)
    frames, dense = all_obs
    next_frames, next_dense = all_next_obs

    self.assertEqual(frames.shape, (batch_size, 3, 84, 84))
    self.assertEqual(dense.shape, (batch_size, 4))
    self.assertEqual(next_frames.shape, (batch_size, 3, 84, 84))
    self.assertEqual(next_dense.shape, (batch_size, 4))

    # Check other tensors
    self.assertEqual(all_actions.shape, (batch_size, ))
    self.assertEqual(all_rewards.shape, (batch_size, ))
    self.assertEqual(all_done.shape, (batch_size, ))

    # Check tensor types and devices
    self.assertEqual(frames.dtype, torch.float)
    self.assertEqual(frames.device, self.device)
    self.assertEqual(all_actions.dtype, torch.int64)
    self.assertEqual(all_rewards.dtype, torch.float)
    self.assertEqual(all_done.dtype, torch.bool)

  def test_frame_only_observations(self):
    """Test workflow with observations that only have frame data (no dense vector)."""
    # Add experiences with frame-only observations
    for i in range(5):
      obs = self._create_mock_observation_tensors(has_frame=True,
                                                  has_dense=False)[0]
      next_obs = self._create_mock_observation_tensors(has_frame=True,
                                                       has_dense=False)[0]

      self.replay_buffer.append(obs, i, 1.0, next_obs, False)

    # Sample and verify
    all_obs, all_actions, all_rewards, all_next_obs, all_done = self.replay_buffer.sample(
        3)

    frames, dense = all_obs
    next_frames, next_dense = all_next_obs

    self.assertEqual(frames.shape, (3, 3, 84, 84))
    self.assertEqual(dense.numel(), 0)  # Empty tensor
    self.assertEqual(next_frames.shape, (3, 3, 84, 84))
    self.assertEqual(next_dense.numel(), 0)  # Empty tensor

  def test_dense_only_observations(self):
    """Test workflow with observations that only have dense data (no frame)."""
    # Add experiences with dense-only observations
    for i in range(5):
      obs = self._create_mock_observation_tensors(has_frame=False,
                                                  has_dense=True)[0]
      next_obs = self._create_mock_observation_tensors(has_frame=False,
                                                       has_dense=True)[0]

      self.replay_buffer.append(obs, i, 1.0, next_obs, False)

    # Sample and verify
    all_obs, all_actions, all_rewards, all_next_obs, all_done = self.replay_buffer.sample(
        3)

    frames, dense = all_obs
    next_frames, next_dense = all_next_obs

    self.assertEqual(frames.numel(), 0)  # Empty tensor
    self.assertEqual(dense.shape, (3, 4))
    self.assertEqual(next_frames.numel(), 0)  # Empty tensor
    self.assertEqual(next_dense.shape, (3, 4))

  def test_buffer_size_limit_and_random_eviction(self):
    """Test that buffer respects size limit and randomly evicts old experiences."""
    # Set small buffer size
    small_config = {'size': 5}
    small_buffer = ReplayBuffer(small_config, self.device,
                                self.mock_summary_writer)

    # Add more experiences than buffer can hold
    experiences = []
    for i in range(10):
      obs = self._create_mock_observation_tensors()[0]
      next_obs = self._create_mock_observation_tensors()[0]
      action = i
      reward = float(i)
      done = False

      small_buffer.append(obs, action, reward, next_obs, done)
      experiences.append(
          (action, reward))  # Store action and reward for tracking

    # Buffer should not exceed max size
    self.assertEqual(len(small_buffer), 5)
    self.assertLessEqual(len(small_buffer.buffer), 5)

    # Sample and verify we get valid data
    all_obs, all_actions, all_rewards, all_next_obs, all_done = small_buffer.sample(
        3)

    # All actions should be from our original set (0-9)
    for action in all_actions:
      self.assertIn(action.item(), list(range(10)))

    # All rewards should be from our original set (0.0-9.0)
    for reward in all_rewards:
      self.assertIn(reward.item(), [float(i) for i in range(10)])

  def test_insufficient_samples_error(self):
    """Test that sampling raises error when buffer has insufficient experiences."""
    # Add only 2 experiences
    for i in range(2):
      obs = self._create_mock_observation_tensors()[0]
      next_obs = self._create_mock_observation_tensors()[0]
      self.replay_buffer.append(obs, i, 1.0, next_obs, False)

    # Try to sample more than available
    with self.assertRaises(ValueError) as context:
      self.replay_buffer.sample(5)

    self.assertIn("Cannot sample 5 from buffer of size 2",
                  str(context.exception))

  def test_training_scenario_with_consistent_observations(self):
    """Test realistic training scenario with consistent observation types and various rewards/actions."""
    # Simulate a training scenario with diverse experiences but consistent observation structure
    np.random.seed(42)  # For reproducible tests
    torch.manual_seed(42)

    # Use consistent observation type (both frame and dense) throughout training
    experiences_data = []
    for episode in range(3):
      for step in range(20):
        obs = self._create_mock_observation_tensors(has_frame=True,
                                                    has_dense=True)[0]
        next_obs = self._create_mock_observation_tensors(has_frame=True,
                                                         has_dense=True)[0]

        action = np.random.randint(0, 4)
        reward = np.random.uniform(-1.0, 1.0)
        done = (step == 19)  # Terminal state at end of episode

        self.replay_buffer.append(obs, action, reward, next_obs, done)
        experiences_data.append({
            'action': action,
            'reward': reward,
            'done': done
        })

    self.assertEqual(len(self.replay_buffer), 60)

    # Sample multiple batches to test consistency
    for batch_num in range(5):
      batch_size = np.random.randint(8, 16)

      all_obs, all_actions, all_rewards, all_next_obs, all_done = self.replay_buffer.sample(
          batch_size)

      frames, dense = all_obs
      next_frames, next_dense = all_next_obs

      # Verify tensor shapes are consistent
      self.assertEqual(all_actions.shape[0], batch_size)
      self.assertEqual(all_rewards.shape[0], batch_size)
      self.assertEqual(all_done.shape[0], batch_size)
      self.assertEqual(frames.shape[0], batch_size)
      self.assertEqual(dense.shape[0], batch_size)
      self.assertEqual(next_frames.shape[0], batch_size)
      self.assertEqual(next_dense.shape[0], batch_size)

      # Verify actions are in valid range
      self.assertTrue(torch.all(all_actions >= 0))
      self.assertTrue(torch.all(all_actions <= 3))

      # Verify rewards are in expected range
      self.assertTrue(torch.all(all_rewards >= -1.0))
      self.assertTrue(torch.all(all_rewards <= 1.0))

      # Verify data types
      self.assertEqual(all_actions.dtype, torch.int64)
      self.assertEqual(all_rewards.dtype, torch.float)
      self.assertEqual(all_done.dtype, torch.bool)

      # Verify observation shapes
      self.assertEqual(frames.shape[1:], (3, 84, 84))
      self.assertEqual(dense.shape[1:], (4, ))


if __name__ == '__main__':
  unittest.main()
