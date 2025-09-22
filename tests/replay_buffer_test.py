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

from replay_buffer import UniformExperienceReplayBuffer, PrioritizedExperienceReplayBuffer


class TestUniformReplayBuffer(unittest.TestCase):
  """End-to-end test cases for UniformExperienceReplayBuffer."""

  def setUp(self):
    """Set up test fixtures."""
    self.device = torch.device('cpu')
    self.mock_summary_writer = Mock()
    self.config = {'size': 100}
    self.replay_buffer = UniformExperienceReplayBuffer(
        config=self.config,
        device=self.device,
        summary_writer=self.mock_summary_writer)

  def _create_mock_observation_tensors(self,
                                       has_frame=True,
                                       has_dense=True,
                                       batch_size=1):
    """Helper to create mock observation tensors similar to Observation.as_input()."""
    observations = []
    for _ in range(batch_size):
      frame = torch.randn(3, 84, 84) if has_frame else torch.tensor(())
      dense = torch.randn(4) if has_dense else torch.tensor(())
      observations.append((frame, dense))
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
    small_buffer = UniformExperienceReplayBuffer(small_config, self.device,
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

  def test_sample_with_repeats_when_insufficient(self):
    """Test that sampling with repeats works when buffer has insufficient experiences."""
    # Add only 2 experiences
    for i in range(2):
      obs = self._create_mock_observation_tensors()[0]
      next_obs = self._create_mock_observation_tensors()[0]
      self.replay_buffer.append(obs, i, 1.0, next_obs, False)

    # Try to sample more than available (should allow repeats, no error)
    all_obs, all_actions, all_rewards, all_next_obs, all_done = self.replay_buffer.sample(
        5)
    # All actions should be from our original set (0, 1)
    for action in all_actions:
      self.assertIn(action.item(), [0, 1])

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


class TestPrioritizedReplayBuffer(unittest.TestCase):
  """End-to-end test cases for PrioritizedExperienceReplayBuffer."""

  def setUp(self):
    """Set up test fixtures."""
    self.device = torch.device('cpu')
    self.mock_summary_writer = Mock()

    # Mock the target and prediction callbacks required for prioritized replay
    self.mock_target_callback = Mock(
        return_value=torch.tensor([1.0, 2.0, 3.0]))
    self.mock_prediction_callback = Mock(
        return_value=torch.tensor([1.1, 1.9, 3.2]))

    self.config = {
        'size': 100,
        'alpha': 0.6,  # Prioritization exponent
        'beta': 0.4,  # Importance sampling exponent
        'beta_annealing_steps': 1000  # Beta annealing steps
    }
    self.replay_buffer = PrioritizedExperienceReplayBuffer(
        config=self.config,
        device=self.device,
        summary_writer=self.mock_summary_writer,
        target_callback=self.mock_target_callback,
        prediction_callback=self.mock_prediction_callback)

  def _create_mock_observation_tensors(self,
                                       has_frame=True,
                                       has_dense=True,
                                       batch_size=1):
    """Helper to create mock observation tensors similar to Observation.as_input()."""
    observations = []
    for _ in range(batch_size):
      frame = torch.randn(3, 84, 84) if has_frame else torch.tensor(())
      dense = torch.randn(4) if has_dense else torch.tensor(())
      observations.append((frame, dense))
    return observations

  def test_prioritized_buffer_initialization(self):
    """Test that prioritized buffer initializes correctly with required parameters."""
    self.assertEqual(self.replay_buffer.alpha, 0.6)
    self.assertEqual(self.replay_buffer.beta, 0.4)
    self.assertEqual(len(self.replay_buffer.surprise), 0)
    self.assertIsNotNone(self.replay_buffer.compute_target)
    self.assertIsNotNone(self.replay_buffer.compute_prediction)

  def test_prioritized_append_computes_surprise(self):
    """Test that appending experiences initializes surprise values."""
    obs = self._create_mock_observation_tensors()[0]
    next_obs = self._create_mock_observation_tensors()[0]

    self.replay_buffer.append(obs, 1, 0.5, next_obs, False)

    # Verify surprise was initialized
    self.assertEqual(len(self.replay_buffer.surprise), 1)
    self.assertGreater(self.replay_buffer.surprise[0], 0)

    # In the current implementation, callbacks are not called during append
    # They are used later during sampling/updating

  def test_prioritized_sampling_with_importance_weights(self):
    """Test that prioritized sampling returns importance sampling weights."""
    # Add several experiences with different mock surprise values
    experiences = []
    for i in range(10):
      obs = self._create_mock_observation_tensors()[0]
      next_obs = self._create_mock_observation_tensors()[0]

      # Mock different target/prediction values to create different surprises
      self.mock_target_callback.return_value = torch.tensor([float(i + 1)])
      self.mock_prediction_callback.return_value = torch.tensor([float(i)])

      self.replay_buffer.append(obs, i % 4, float(i) * 0.1, next_obs, i == 9)
      experiences.append((i % 4, float(i) * 0.1, i == 9))

    # Sample batch and verify it returns importance weights
    batch_size = 5

    # Mock the callbacks to return tensors of the right size for sampling
    self.mock_target_callback.return_value = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0])
    self.mock_prediction_callback.return_value = torch.tensor(
        [0.9, 1.9, 2.9, 3.9, 4.9])

    result = self.replay_buffer.sample(batch_size)

    # Should return tuple with (experiences, importance_weights, indices)
    self.assertIsInstance(result, tuple)
    self.assertEqual(len(result), 3)

    experiences_tuple, importance_weights, sampled_indices = result
    all_obs, all_actions, all_rewards, all_next_obs, all_done = experiences_tuple

    # Verify experience tensors
    self.assertEqual(len(importance_weights), batch_size)
    self.assertEqual(all_actions.shape[0], batch_size)

    # Verify importance weights are positive numbers
    for weight in importance_weights:
      self.assertIsInstance(weight, float)
      self.assertGreater(weight, 0)

  def test_prioritized_sampling_updates_surprises(self):
    """Test that sampling updates surprise values based on current TD errors."""
    # Add experiences
    for i in range(5):
      obs = self._create_mock_observation_tensors()[0]
      next_obs = self._create_mock_observation_tensors()[0]

      self.mock_target_callback.return_value = torch.tensor([1.0])
      self.mock_prediction_callback.return_value = torch.tensor([0.5])

      self.replay_buffer.append(obs, i, 0.1, next_obs, False)

    # Store initial surprise values
    initial_surprises = list(self.replay_buffer.surprise)

    # Sample experiences
    result = self.replay_buffer.sample(3)
    experiences_tuple, importance_weights, sampled_indices = result

    # Simulate TD errors for updating surprise values
    td_errors = np.array([0.5, 1.0, 1.5])  # Different TD errors
    self.replay_buffer.update_surprise(sampled_indices, td_errors)

    # At least some surprise values should have changed
    current_surprises = list(self.replay_buffer.surprise)
    self.assertNotEqual(initial_surprises, current_surprises)

  def test_prioritized_error_handling(self):
    """Test that prioritized buffer allows sampling with repeats when insufficient experiences (legacy test, now expects no error)."""
    # Add only 2 experiences
    for i in range(2):
      obs = self._create_mock_observation_tensors()[0]
      next_obs = self._create_mock_observation_tensors()[0]
      self.mock_target_callback.return_value = torch.tensor([1.0])
      self.mock_prediction_callback.return_value = torch.tensor([0.5])
      self.replay_buffer.append(obs, i, 1.0, next_obs, False)

    # Try to sample more than available (should allow repeats, no error)
    (all_obs, all_actions, all_rewards, all_next_obs,
     all_done), importance_sampling, indices = self.replay_buffer.sample(5)
    # All actions should be from our original set (0, 1)
    for action in all_actions:
      self.assertIn(action.item(), [0, 1])

  def test_zero_surprise_handling(self):
    """Test that prioritized buffer handles experiences with zero surprise correctly."""
    obs = self._create_mock_observation_tensors()[0]
    next_obs = self._create_mock_observation_tensors()[0]

    self.replay_buffer.append(obs, 1, 0.5, next_obs, False)

    # New experiences are initialized with a default value (1.0 for first experience)
    self.assertEqual(len(self.replay_buffer.surprise), 1)
    self.assertEqual(self.replay_buffer.surprise[0], 1.0)

    # Test updating with zero TD error
    sampled_indices = [0]
    td_errors = np.array([0.0])  # Zero TD error
    self.replay_buffer.update_surprise(sampled_indices, td_errors)

    # After update, surprise should be (0 + eps)^alpha = (1e-3)^0.6 ≈ 0.0158
    expected_surprise = (1e-3)**0.6
    self.assertAlmostEqual(self.replay_buffer.surprise[0],
                           expected_surprise,
                           places=3)

  def test_buffer_overflow_maintains_surprise_sync(self):
    """Test that when buffer overflows, surprise values stay synchronized."""
    # Set small buffer size to test overflow
    small_config = {
        'size': 3,
        'alpha': 0.6,
        'beta': 0.4,
        'beta_annealing_steps': 1000
    }
    small_buffer = PrioritizedExperienceReplayBuffer(
        config=small_config,
        device=self.device,
        summary_writer=self.mock_summary_writer,
        target_callback=self.mock_target_callback,
        prediction_callback=self.mock_prediction_callback)

    # Add more experiences than buffer can hold
    for i in range(5):
      obs = self._create_mock_observation_tensors()[0]
      next_obs = self._create_mock_observation_tensors()[0]

      self.mock_target_callback.return_value = torch.tensor([float(i + 1)])
      self.mock_prediction_callback.return_value = torch.tensor([float(i)])

      small_buffer.append(obs, i, float(i), next_obs, False)

    # Buffer and surprise should both be limited to size 3
    self.assertEqual(len(small_buffer), 3)
    self.assertEqual(len(small_buffer.surprise), 3)


class TestReplayBufferEdgeCases(unittest.TestCase):
  """Test edge cases and error conditions for replay buffers."""

  def setUp(self):
    """Set up test fixtures."""
    self.device = torch.device('cpu')
    self.mock_summary_writer = Mock()
    self.config = {'size': 10}

  def test_empty_tensor_handling(self):
    """Test that replay buffer handles empty tensors correctly."""
    replay_buffer = UniformExperienceReplayBuffer(
        config=self.config,
        device=self.device,
        summary_writer=self.mock_summary_writer)

    # Test with completely empty tensors
    empty_frame = torch.tensor(())
    empty_dense = torch.tensor(())
    obs = (empty_frame, empty_dense)
    next_obs = (empty_frame, empty_dense)

    replay_buffer.append(obs, 0, 1.0, next_obs, False)
    self.assertEqual(len(replay_buffer), 1)

    # Sample and verify empty tensors are handled
    all_obs, all_actions, all_rewards, all_next_obs, all_done = replay_buffer.sample(
        1)
    frames, dense = all_obs
    next_frames, next_dense = all_next_obs

    self.assertEqual(frames.numel(), 0)
    self.assertEqual(dense.numel(), 0)
    self.assertEqual(next_frames.numel(), 0)
    self.assertEqual(next_dense.numel(), 0)

  def test_mixed_observation_types_error_prevention(self):
    """Test handling of mixed observation types within single buffer."""
    replay_buffer = UniformExperienceReplayBuffer(
        config=self.config,
        device=self.device,
        summary_writer=self.mock_summary_writer)

    # Add multiple observations to ensure we get mixed types when sampling
    frame_dense_obs = (torch.randn(3, 84, 84), torch.randn(4))
    frame_only_obs = (torch.randn(3, 84, 84), torch.tensor(()))

    # Add several of each type to increase chances of sampling mixed types
    for _ in range(5):
      replay_buffer.append(frame_dense_obs, 0, 1.0, frame_dense_obs, False)
    for _ in range(5):
      replay_buffer.append(frame_only_obs, 1, 1.0, frame_only_obs, False)

    # Try sampling many times - should eventually get mixed types and raise error
    error_raised = False
    for attempt in range(100):  # Try many times to get mixed sampling
      try:
        all_obs, all_actions, all_rewards, all_next_obs, all_done = replay_buffer.sample(
            6)
        # If we get here without an error, the batch happened to be homogeneous
      except ValueError as e:
        if "Mixed observation types not supported" in str(e):
          error_raised = True
          break
        else:
          raise  # Re-raise if it's a different ValueError

    # Should have raised the mixed types error at least once
    self.assertTrue(
        error_raised,
        "Expected ValueError for mixed observation types was never raised")

  def test_large_batch_sampling(self):
    """Test sampling when batch size equals buffer size."""
    replay_buffer = UniformExperienceReplayBuffer(
        config={'size': 5},
        device=self.device,
        summary_writer=self.mock_summary_writer)

    # Fill buffer to capacity
    for i in range(5):
      obs = (torch.randn(3, 84, 84), torch.randn(4))
      replay_buffer.append(obs, i, float(i), obs, False)

    # Sample entire buffer - note that random.choices allows duplicates
    all_obs, all_actions, all_rewards, all_next_obs, all_done = replay_buffer.sample(
        5)

    self.assertEqual(all_actions.shape[0], 5)
    # Verify all actions are in the valid range (0-4)
    self.assertTrue(torch.all(all_actions >= 0))
    self.assertTrue(torch.all(all_actions <= 4))

  def test_summary_writer_logging(self):
    """Test that summary writer is called correctly for memory logging."""
    mock_writer = Mock()
    replay_buffer = UniformExperienceReplayBuffer(config=self.config,
                                                  device=self.device,
                                                  summary_writer=mock_writer)

    # Add experiences and verify logging
    for i in range(3):
      obs = (torch.randn(3, 84, 84), torch.randn(4))
      replay_buffer.append(obs, i, float(i), obs, False)

    # Verify summary writer was called for each append
    self.assertEqual(mock_writer.add_scalar.call_count, 3)

    # Verify the calls were for replay buffer size logging
    calls = mock_writer.add_scalar.call_args_list
    for i, call in enumerate(calls):
      args, kwargs = call
      self.assertEqual(args[0], 'ReplayBuffer/Size')
      self.assertEqual(args[1], i + 1)  # Size should increment

  def test_device_consistency(self):
    """Test that replay buffer respects device settings for all tensors."""
    # Test with different device if available
    device = torch.device('cpu')
    replay_buffer = UniformExperienceReplayBuffer(
        config=self.config,
        device=device,
        summary_writer=self.mock_summary_writer)

    # Add experiences with tensors on different devices
    obs = (torch.randn(3, 84, 84), torch.randn(4))
    next_obs = (torch.randn(3, 84, 84), torch.randn(4))

    replay_buffer.append(obs, 0, 1.0, next_obs, False)

    # Sample and verify all tensors are on correct device
    all_obs, all_actions, all_rewards, all_next_obs, all_done = replay_buffer.sample(
        1)

    frames, dense = all_obs
    next_frames, next_dense = all_next_obs

    self.assertEqual(frames.device, device)
    self.assertEqual(dense.device, device)
    self.assertEqual(all_actions.device, device)
    self.assertEqual(all_rewards.device, device)
    self.assertEqual(all_done.device, device)
    self.assertEqual(next_frames.device, device)
    self.assertEqual(next_dense.device, device)

  def test_batch_consistency_across_multiple_samples(self):
    """Test that multiple sampling calls return consistent batch formats."""
    replay_buffer = UniformExperienceReplayBuffer(
        config={'size': 50},
        device=self.device,
        summary_writer=self.mock_summary_writer)

    # Add many experiences
    for i in range(20):
      obs = (torch.randn(3, 84, 84), torch.randn(4))
      next_obs = (torch.randn(3, 84, 84), torch.randn(4))
      replay_buffer.append(obs, i % 4, float(i), next_obs, i % 5 == 0)

    # Sample different batch sizes and verify consistency
    for batch_size in [1, 4, 8, 16]:
      all_obs, all_actions, all_rewards, all_next_obs, all_done = replay_buffer.sample(
          batch_size)

      frames, dense = all_obs
      next_frames, next_dense = all_next_obs

      # Verify all tensors have correct batch dimension
      self.assertEqual(frames.shape[0], batch_size)
      self.assertEqual(dense.shape[0], batch_size)
      self.assertEqual(all_actions.shape[0], batch_size)
      self.assertEqual(all_rewards.shape[0], batch_size)
      self.assertEqual(all_done.shape[0], batch_size)
      self.assertEqual(next_frames.shape[0], batch_size)
      self.assertEqual(next_dense.shape[0], batch_size)

      # Verify tensor shapes are consistent
      self.assertEqual(frames.shape[1:], (3, 84, 84))
      self.assertEqual(dense.shape[1:], (4, ))
      self.assertEqual(next_frames.shape[1:], (3, 84, 84))
      self.assertEqual(next_dense.shape[1:], (4, ))
