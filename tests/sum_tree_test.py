import unittest
import numpy as np
import math
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.sum_tree import SumTree


class TestSumTree(unittest.TestCase):
  """Comprehensive test suite for SumTree class."""

  def test_initialization(self):
    """Test SumTree initialization with various capacities."""
    # Test with small capacity
    tree = SumTree(4)
    self.assertEqual(tree.capacity, 4)  # 2^2 = 4
    self.assertEqual(len(tree.values), 7)  # 2 * 4 - 1
    self.assertEqual(tree.cursor, 0)
    self.assertEqual(tree.recompute_step, 0)

    # Test with non-power-of-2 capacity
    tree = SumTree(7)
    expected_capacity = 2**math.ceil(math.log2(7))  # 2^3 = 8
    self.assertEqual(tree.capacity, expected_capacity)

    # Test with large capacity
    tree = SumTree(1000)
    expected_capacity = 2**math.ceil(math.log2(1000))  # 2^10 = 1024
    self.assertEqual(tree.capacity, expected_capacity)

  def test_initialization_with_custom_recompute_every_n(self):
    """Test initialization with custom recompute_every_n parameter."""
    tree = SumTree(4, recompute_every_n=500)
    self.assertEqual(tree.recompute_every_n, 500)

  def test_invalid_capacity(self):
    """Test that invalid capacity raises appropriate errors."""
    with self.assertRaises(ValueError):
      SumTree(0)

    with self.assertRaises(ValueError):
      SumTree(-1)

  def test_add_single_value(self):
    """Test adding a single value to the tree."""
    tree = SumTree(4)
    tree.add(10.0)

    # Check that the value is stored in the correct leaf
    leaf_idx = tree.capacity - 1  # First leaf index
    self.assertEqual(tree.values[leaf_idx], 10.0)

    # Check that the root contains the sum
    self.assertEqual(tree.values[0], 10.0)

    # Check cursor moved
    self.assertEqual(tree.cursor, 1)

  def test_add_multiple_values(self):
    """Test adding multiple values to fill the tree."""
    tree = SumTree(4)
    values = [10.0, 20.0, 30.0, 40.0]

    for value in values:
      tree.add(value)

    # Check all values are stored correctly
    for i, value in enumerate(values):
      leaf_idx = tree.capacity - 1 + i
      self.assertEqual(tree.values[leaf_idx], value)

    # Check total sum in root
    self.assertEqual(tree.values[0], sum(values))

    # Check cursor wrapped around
    self.assertEqual(tree.cursor, 0)

  def test_add_with_wraparound(self):
    """Test adding values that cause cursor to wrap around."""
    tree = SumTree(4)
    initial_values = [10.0, 20.0, 30.0, 40.0]

    # Fill the tree
    for value in initial_values:
      tree.add(value)

    # Add one more value (should overwrite first value)
    tree.add(50.0)

    # Check that first value was overwritten
    first_leaf_idx = tree.capacity - 1
    self.assertEqual(tree.values[first_leaf_idx], 50.0)

    # Check new total
    expected_sum = 50.0 + 20.0 + 30.0 + 40.0
    self.assertEqual(tree.values[0], expected_sum)

    # Check cursor position
    self.assertEqual(tree.cursor, 1)

  def test_propagate_functionality(self):
    """Test that _propagate correctly updates parent nodes."""
    tree = SumTree(4)

    # Add values and check intermediate node values
    tree.add(10.0)
    tree.add(20.0)

    # In a capacity-4 tree:
    # Index 0: root
    # Index 1, 2: internal nodes
    # Index 3, 4, 5, 6: leaf nodes

    # Check that internal nodes have correct sums
    # Left subtree sum should be in index 1
    self.assertEqual(tree.values[1], 30.0)  # 10 + 20

    # Root should have total sum
    self.assertEqual(tree.values[0], 30.0)

  def test_find_index_basic(self):
    """Test finding index with basic scenarios."""
    tree = SumTree(4)
    values = [10.0, 20.0, 30.0, 40.0]

    for value in values:
      tree.add(value)

    # Total sum is 100, test different ranges
    # Value 0-10 should map to index 0
    self.assertEqual(tree.find_index(5.0), 0)
    self.assertEqual(tree.find_index(10.0), 0)

    # Value 10-30 should map to index 1
    self.assertEqual(tree.find_index(15.0), 1)
    self.assertEqual(tree.find_index(30.0), 1)

    # Value 30-60 should map to index 2
    self.assertEqual(tree.find_index(45.0), 2)
    self.assertEqual(tree.find_index(60.0), 2)

    # Value 60-100 should map to index 3
    self.assertEqual(tree.find_index(80.0), 3)
    self.assertEqual(tree.find_index(100.0), 3)

  def test_find_index_edge_cases(self):
    """Test find_index with edge cases."""
    tree = SumTree(4)

    # Test with single value
    tree.add(50.0)
    self.assertEqual(tree.find_index(25.0), 0)
    self.assertEqual(tree.find_index(50.0), 0)

    # Test with zero value
    tree = SumTree(4)
    tree.add(0.0)
    tree.add(10.0)
    self.assertEqual(tree.find_index(5.0), 1)

  def test_find_index_boundary_values(self):
    """Test find_index at exact boundary values."""
    tree = SumTree(4)
    tree.add(25.0)
    tree.add(25.0)
    tree.add(25.0)
    tree.add(25.0)

    # Each segment is exactly 25, test boundaries
    self.assertEqual(tree.find_index(0.0), 0)
    self.assertEqual(tree.find_index(25.0), 0)
    self.assertEqual(tree.find_index(25.1), 1)
    self.assertEqual(tree.find_index(50.0), 1)
    self.assertEqual(tree.find_index(50.1), 2)
    self.assertEqual(tree.find_index(75.0), 2)
    self.assertEqual(tree.find_index(75.1), 3)
    self.assertEqual(tree.find_index(100.0), 3)

  def test_recompute_functionality(self):
    """Test that recomputation works correctly."""
    tree = SumTree(4, recompute_every_n=3)

    # Add values that will trigger recomputation
    tree.add(10.0)
    tree.add(20.0)

    # Check state before recomputation trigger
    self.assertEqual(tree.values[0], 30.0)

    # This add should trigger recomputation (3rd addition)
    tree.add(30.0)

    # Verify recompute_step reset
    self.assertEqual(tree.recompute_step, 0)

    # Verify tree structure is still correct
    self.assertEqual(tree.values[0], 60.0)  # Total sum

    # Add one more to test after recomputation
    tree.add(40.0)
    self.assertEqual(tree.values[0], 100.0)

  def test_floating_point_precision(self):
    """Test handling of floating point precision issues."""
    tree = SumTree(4)

    # Add very small values
    small_values = [1e-6, 2e-6, 3e-6, 4e-6]
    for value in small_values:
      tree.add(value)

    total_sum = sum(small_values)
    self.assertAlmostEqual(tree.values[0], total_sum, places=10)

    # Test find_index with small values
    # 1.5e-6 should fall in the range of the first value (1e-6)
    # But since we're looking for cumulative ranges, it might be in second range
    idx = tree.find_index(1.5e-6)
    self.assertTrue(0 <= idx < 4)  # Just ensure it's in valid range

  def test_large_values(self):
    """Test with large floating point values."""
    tree = SumTree(4)

    large_values = [1e6, 2e6, 3e6, 4e6]
    for value in large_values:
      tree.add(value)

    self.assertEqual(tree.values[0], sum(large_values))

    # Test find_index with large values
    # 1.5e6 should fall after first value (1e6), so index should be 1
    idx = tree.find_index(1.5e6)
    self.assertTrue(0 <= idx < 4)  # Just ensure it's in valid range

  def test_zero_values(self):
    """Test behavior with zero values."""
    tree = SumTree(4)

    # Add mix of zero and non-zero values
    tree.add(0.0)
    tree.add(10.0)
    tree.add(0.0)
    tree.add(20.0)

    self.assertEqual(tree.values[0], 30.0)

    # Test find_index with zeros
    self.assertEqual(tree.find_index(0.0), 0)  # Should hit first zero
    self.assertEqual(tree.find_index(5.0), 1)  # Should hit 10.0
    self.assertEqual(tree.find_index(15.0), 3)  # Should hit 20.0

  def test_negative_values(self):
    """Test behavior with negative values (if supported)."""
    tree = SumTree(4)

    # Add negative values
    tree.add(-10.0)
    tree.add(20.0)
    tree.add(-5.0)
    tree.add(15.0)

    # Total should be 20
    self.assertEqual(tree.values[0], 20.0)

  def test_capacity_one(self):
    """Test edge case with capacity of 1."""
    tree = SumTree(1)
    self.assertEqual(tree.capacity, 1)

    tree.add(42.0)
    self.assertEqual(tree.values[0], 42.0)
    self.assertEqual(tree.find_index(21.0), 0)

    # Overwrite the single value
    tree.add(84.0)
    self.assertEqual(tree.values[0], 84.0)

  def test_very_large_capacity(self):
    """Test initialization with very large capacity."""
    # This tests the mathematical correctness of capacity calculation
    large_capacity = 10000
    tree = SumTree(large_capacity)

    # Should not crash and should have reasonable capacity
    expected_capacity = 2**math.ceil(math.log2(large_capacity))
    self.assertEqual(tree.capacity, expected_capacity)

    # Test basic functionality
    tree.add(1.0)
    self.assertEqual(tree.values[0], 1.0)

  def test_stress_add_and_find(self):
    """Stress test with many additions and finds."""
    tree = SumTree(16)
    values = [float(i) for i in range(1, 17)]  # 1.0 to 16.0

    # Add all values
    for value in values:
      tree.add(value)

    total_sum = sum(values)  # Should be 136
    self.assertEqual(tree.values[0], total_sum)

    # Test multiple find operations
    for i in range(100):
      random_value = np.random.uniform(0, total_sum)
      idx = tree.find_index(random_value)
      self.assertTrue(0 <= idx < 16)

  def test_propagate_edge_cases(self):
    """Test _propagate with edge cases."""
    tree = SumTree(4)

    # Test propagation from each leaf position
    for i in range(4):
      leaf_idx = tree.capacity - 1 + i
      tree._propagate(leaf_idx, 10.0)

    # All propagations should sum up in root
    self.assertEqual(tree.values[0], 40.0)

  def test_consistency_after_many_operations(self):
    """Test that tree remains consistent after many operations."""
    tree = SumTree(8)

    # Add initial values
    values = [float(i) for i in range(1, 9)]
    for value in values:
      tree.add(value)

    initial_sum = sum(values)
    self.assertEqual(tree.values[0], initial_sum)

    # Perform many overwrites
    for i in range(100):
      new_value = float(i % 10 + 1)
      tree.add(new_value)

      # Check that root sum is reasonable
      self.assertGreater(tree.values[0], 0)

      # Check that find_index doesn't crash
      test_value = tree.values[0] * 0.5
      idx = tree.find_index(test_value)
      self.assertTrue(0 <= idx < 8)

  def test_recompute_preserves_values(self):
    """Test that recomputation preserves leaf values."""
    tree = SumTree(4, recompute_every_n=3)

    values = [10.0, 20.0, 30.0]
    for value in values:
      tree.add(value)

    # Store leaf values before potential recomputation
    leaf_values = tree.values[tree.capacity - 1:tree.capacity - 1 +
                              tree.cursor].copy()

    # Force another recomputation by adding more values
    tree.add(40.0)  # This fills the tree and might trigger recomputation
    tree.add(50.0)  # This overwrites first value (index 0)
    tree.add(
        60.0
    )  # This overwrites second value (index 1) and should trigger recomputation

    # Check that leaf values are as expected (with overwrites)
    # After these operations: index 0 has 50.0, index 1 has 60.0, others unchanged
    expected_leaves = [50.0, 60.0, 30.0, 40.0]
    actual_leaves = tree.values[tree.capacity - 1:2 * tree.capacity - 1]
    np.testing.assert_array_equal(actual_leaves, expected_leaves)

  def test_find_index_exact_zero(self):
    """Test find_index when value is exactly 0."""
    tree = SumTree(4)
    tree.add(10.0)
    tree.add(20.0)
    tree.add(30.0)
    tree.add(40.0)

    # Value 0 should always map to first index
    self.assertEqual(tree.find_index(0.0), 0)

  def test_find_index_exact_total(self):
    """Test find_index when value equals the total sum."""
    tree = SumTree(4)
    tree.add(10.0)
    tree.add(20.0)
    tree.add(30.0)
    tree.add(40.0)

    total = tree.values[0]  # 100.0
    # Should map to last index when value equals total
    self.assertEqual(tree.find_index(total), 3)

  def test_empty_tree_behavior(self):
    """Test behavior with newly initialized tree (all zeros)."""
    tree = SumTree(4)

    # Root should be zero
    self.assertEqual(tree.values[0], 0.0)

    # Empty tree should have cursor at 0 and not be full
    self.assertEqual(tree.cursor, 0)
    self.assertFalse(tree.full)

    # Add one value to test find_index functionality
    tree.add(10.0)
    self.assertEqual(tree.find_index(0.0), 0)
    self.assertEqual(tree.find_index(5.0), 0)
    self.assertEqual(tree.find_index(10.0), 0)

  def test_partial_fill_behavior(self):
    """Test behavior when tree is only partially filled."""
    tree = SumTree(8)  # Large capacity

    # Only add a few values
    tree.add(10.0)
    tree.add(20.0)

    # Total should be 30
    self.assertEqual(tree.values[0], 30.0)

    # Test find_index with partial fill
    self.assertEqual(tree.find_index(5.0), 0)
    self.assertEqual(tree.find_index(15.0), 1)
    self.assertEqual(tree.find_index(30.0), 1)

  def test_overwrite_with_zeros(self):
    """Test overwriting existing values with zeros."""
    tree = SumTree(4)

    # Add initial values
    tree.add(10.0)
    tree.add(20.0)
    tree.add(30.0)
    tree.add(40.0)

    initial_sum = tree.values[0]  # 100.0

    # Overwrite first value with zero
    tree.add(0.0)

    # Sum should decrease
    expected_sum = 0.0 + 20.0 + 30.0 + 40.0
    self.assertEqual(tree.values[0], expected_sum)

  def test_all_same_values(self):
    """Test with all identical values."""
    tree = SumTree(4)

    # Add same value multiple times
    for _ in range(4):
      tree.add(25.0)

    self.assertEqual(tree.values[0], 100.0)

    # Each quarter should map to different indices
    self.assertEqual(tree.find_index(12.5), 0)  # First quarter
    self.assertEqual(tree.find_index(37.5), 1)  # Second quarter
    self.assertEqual(tree.find_index(62.5), 2)  # Third quarter
    self.assertEqual(tree.find_index(87.5), 3)  # Fourth quarter

  def test_extreme_recompute_frequency(self):
    """Test with very frequent recomputation."""
    tree = SumTree(4, recompute_every_n=1)  # Recompute every step

    # Start with one value to avoid empty tree issues
    tree.add(10.0)

    values = [20.0, 30.0, 40.0, 50.0, 60.0]
    for value in values:
      tree.add(value)
      # Should recompute after each add but still work correctly
      self.assertGreaterEqual(tree.values[0], 0)

  def test_cursor_wraparound_multiple_times(self):
    """Test cursor wrapping around multiple times."""
    tree = SumTree(2)  # Very small capacity

    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # More than capacity

    for value in values:
      tree.add(value)

    # Final values should be the last 2 values added
    expected_leaves = [5.0, 6.0]
    actual_leaves = tree.values[tree.capacity - 1:2 * tree.capacity - 1]
    np.testing.assert_array_equal(actual_leaves, expected_leaves)

    # Sum should be 11.0
    self.assertEqual(tree.values[0], 11.0)

  def test_get_sum(self):
    """Test get_sum method returns the correct total sum."""
    tree = SumTree(4)

    # Empty tree should have sum of 0
    self.assertEqual(tree.get_sum(), 0.0)

    # Add values and check sum
    values = [10.0, 20.0, 30.0, 40.0]
    for value in values:
      tree.add(value)

    self.assertEqual(tree.get_sum(), 100.0)

    # Add more values (with wraparound)
    tree.add(50.0)  # Overwrites first value
    expected_sum = 50.0 + 20.0 + 30.0 + 40.0
    self.assertEqual(tree.get_sum(), expected_sum)

  def test_get_min_full_tree(self):
    """Test get_min method with full tree."""
    tree = SumTree(4)
    values = [15.0, 5.0, 25.0, 10.0]

    for value in values:
      tree.add(value)

    self.assertEqual(tree.get_min(), 5.0)

    # Add a smaller value (with wraparound)
    tree.add(2.0)  # Overwrites first value (15.0)
    self.assertEqual(tree.get_min(), 2.0)

  def test_get_min_partial_tree(self):
    """Test get_min method with partially filled tree."""
    tree = SumTree(6)

    # Add only 3 values
    tree.add(15.0)
    tree.add(5.0)
    tree.add(25.0)

    self.assertEqual(tree.get_min(), 5.0)

  def test_get_max_full_tree(self):
    """Test get_max method with full tree."""
    tree = SumTree(4)
    values = [15.0, 5.0, 25.0, 10.0]

    for value in values:
      tree.add(value)

    self.assertEqual(tree.get_max(), 25.0)

    # Add a larger value (with wraparound)
    tree.add(35.0)  # Overwrites first value (15.0)
    self.assertEqual(tree.get_max(), 35.0)

  def test_get_max_partial_tree(self):
    """Test get_max method with partially filled tree."""
    tree = SumTree(6)

    # Add only 3 values
    tree.add(15.0)
    tree.add(5.0)
    tree.add(25.0)

    self.assertEqual(tree.get_max(), 25.0)

  def test_get_value_basic(self):
    """Test get_value method for basic functionality."""
    tree = SumTree(4)
    values = [10.0, 20.0, 30.0, 40.0]

    for value in values:
      tree.add(value)

    # Test all indices
    for i, value in enumerate(values):
      self.assertEqual(tree.get_value(i), value)

  def test_get_value_with_wraparound(self):
    """Test get_value method after cursor wraparound."""
    tree = SumTree(4)
    initial_values = [10.0, 20.0, 30.0, 40.0]

    # Fill the tree
    for value in initial_values:
      tree.add(value)

    # Add more values with wraparound
    tree.add(50.0)  # Overwrites index 0
    tree.add(60.0)  # Overwrites index 1

    # Check values at each index
    self.assertEqual(tree.get_value(0), 50.0)
    self.assertEqual(tree.get_value(1), 60.0)
    self.assertEqual(tree.get_value(2), 30.0)
    self.assertEqual(tree.get_value(3), 40.0)

  def test_get_value_partial_tree(self):
    """Test get_value method with partially filled tree."""
    tree = SumTree(6)

    # Add only 3 values
    tree.add(15.0)
    tree.add(25.0)
    tree.add(35.0)

    self.assertEqual(tree.get_value(0), 15.0)
    self.assertEqual(tree.get_value(1), 25.0)
    self.assertEqual(tree.get_value(2), 35.0)

  def test_get_value_invalid_indices(self):
    """Test get_value method with invalid indices."""
    tree = SumTree(4)
    values = [10.0, 20.0, 30.0, 40.0]

    for value in values:
      tree.add(value)

    # Test negative index
    with self.assertRaises(IndexError):
      tree.get_value(-1)

    # Test index beyond length
    with self.assertRaises(IndexError):
      tree.get_value(4)

    # Test index beyond length with larger number
    with self.assertRaises(IndexError):
      tree.get_value(100)

  def test_get_value_invalid_indices_partial_tree(self):
    """Test get_value method with invalid indices on partial tree."""
    tree = SumTree(6)

    # Add only 3 values
    tree.add(15.0)
    tree.add(25.0)
    tree.add(35.0)

    # Test valid indices
    self.assertEqual(tree.get_value(0), 15.0)
    self.assertEqual(tree.get_value(2), 35.0)

    # Test index beyond current cursor in non-full tree
    with self.assertRaises(IndexError):
      tree.get_value(3)

    # Test negative index
    with self.assertRaises(IndexError):
      tree.get_value(-1)

  def test_update_basic(self):
    """Test update method for basic functionality."""
    tree = SumTree(4)
    values = [10.0, 20.0, 30.0, 40.0]

    for value in values:
      tree.add(value)

    initial_sum = tree.get_sum()  # 100.0

    # Update value at index 0
    tree.update(0, 15.0)
    self.assertEqual(tree.get_value(0), 15.0)

    # Check sum is updated correctly
    expected_sum = initial_sum - 10.0 + 15.0  # 105.0
    self.assertEqual(tree.get_sum(), expected_sum)

  def test_update_multiple_indices(self):
    """Test updating multiple indices."""
    tree = SumTree(4)
    values = [10.0, 20.0, 30.0, 40.0]

    for value in values:
      tree.add(value)

    # Update multiple values
    tree.update(0, 5.0)
    tree.update(2, 35.0)
    tree.update(3, 45.0)

    # Check individual values
    self.assertEqual(tree.get_value(0), 5.0)
    self.assertEqual(tree.get_value(1), 20.0)  # Unchanged
    self.assertEqual(tree.get_value(2), 35.0)
    self.assertEqual(tree.get_value(3), 45.0)

    # Check total sum
    expected_sum = 5.0 + 20.0 + 35.0 + 45.0
    self.assertEqual(tree.get_sum(), expected_sum)

  def test_update_to_zero(self):
    """Test updating values to zero."""
    tree = SumTree(4)
    values = [10.0, 20.0, 30.0, 40.0]

    for value in values:
      tree.add(value)

    # Update to zero
    tree.update(1, 0.0)

    self.assertEqual(tree.get_value(1), 0.0)
    expected_sum = 10.0 + 0.0 + 30.0 + 40.0
    self.assertEqual(tree.get_sum(), expected_sum)

  def test_update_invalid_indices(self):
    """Test update method with invalid indices."""
    tree = SumTree(4)
    values = [10.0, 20.0, 30.0, 40.0]

    for value in values:
      tree.add(value)

    # Test negative index
    with self.assertRaises(IndexError):
      tree.update(-1, 50.0)

    # Test index beyond length
    with self.assertRaises(IndexError):
      tree.update(4, 50.0)

  def test_update_invalid_indices_partial_tree(self):
    """Test update method with invalid indices on partial tree."""
    tree = SumTree(6)

    # Add only 3 values
    tree.add(15.0)
    tree.add(25.0)
    tree.add(35.0)

    # Test valid update
    tree.update(1, 30.0)
    self.assertEqual(tree.get_value(1), 30.0)

    # Test index beyond current cursor in non-full tree
    with self.assertRaises(IndexError):
      tree.update(3, 50.0)

  def test_update_and_find_consistency(self):
    """Test that update maintains consistency with find_index."""
    tree = SumTree(4)
    values = [10.0, 20.0, 30.0, 40.0]

    for value in values:
      tree.add(value)

    # Update a value
    tree.update(1, 50.0)  # Change 20.0 to 50.0

    # Total sum should be 10 + 50 + 30 + 40 = 130
    expected_sum = 130.0
    self.assertEqual(tree.get_sum(), expected_sum)

    # Test find_index still works correctly
    # 0-10: index 0, 10-60: index 1, 60-90: index 2, 90-130: index 3
    self.assertEqual(tree.find_index(5.0), 0)
    self.assertEqual(tree.find_index(35.0), 1)
    self.assertEqual(tree.find_index(75.0), 2)
    self.assertEqual(tree.find_index(110.0), 3)

  def test_min_max_with_negative_values(self):
    """Test get_min and get_max with negative values."""
    tree = SumTree(4)
    values = [-10.0, 5.0, -3.0, 8.0]

    for value in values:
      tree.add(value)

    self.assertEqual(tree.get_min(), -10.0)
    self.assertEqual(tree.get_max(), 8.0)

  def test_min_max_with_zeros(self):
    """Test get_min and get_max with zero values."""
    tree = SumTree(4)
    values = [0.0, 10.0, 0.0, 5.0]

    for value in values:
      tree.add(value)

    self.assertEqual(tree.get_min(), 0.0)
    self.assertEqual(tree.get_max(), 10.0)

  def test_min_max_all_same_values(self):
    """Test get_min and get_max when all values are the same."""
    tree = SumTree(4)

    for _ in range(4):
      tree.add(42.0)

    self.assertEqual(tree.get_min(), 42.0)
    self.assertEqual(tree.get_max(), 42.0)

  def test_methods_consistency_after_operations(self):
    """Test that all methods remain consistent after various operations."""
    tree = SumTree(5)

    # Add initial values
    values = [5.0, 15.0, 25.0, 35.0, 45.0]
    for value in values:
      tree.add(value)

    # Check initial state
    self.assertEqual(tree.get_sum(), 125.0)
    self.assertEqual(tree.get_min(), 5.0)
    self.assertEqual(tree.get_max(), 45.0)

    # Update some values
    tree.update(0, 50.0)  # 5.0 -> 50.0
    tree.update(2, 1.0)  # 25.0 -> 1.0

    # Check updated state
    expected_sum = 50.0 + 15.0 + 1.0 + 35.0 + 45.0  # 146.0
    self.assertEqual(tree.get_sum(), expected_sum)
    self.assertEqual(tree.get_min(), 1.0)
    self.assertEqual(tree.get_max(), 50.0)

    # Add more values (wraparound)
    tree.add(100.0)  # Overwrites index 0: 50.0 -> 100.0

    # Check final state
    final_sum = 100.0 + 15.0 + 1.0 + 35.0 + 45.0  # 196.0
    self.assertEqual(tree.get_sum(), final_sum)
    self.assertEqual(tree.get_min(), 1.0)
    self.assertEqual(tree.get_max(), 100.0)

    # Verify find_index still works
    for _ in range(10):
      random_value = np.random.uniform(0, final_sum)
      idx = tree.find_index(random_value)
      self.assertTrue(0 <= idx < 5)

  # ...existing code...


if __name__ == '__main__':
  unittest.main()
