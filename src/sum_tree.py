import math
import numpy as np


class SumTree(object):
  """
  A circular buffer implemented as a binary tree to allow for efficient sampling.

  # Initialize
  stree = SumTree(length) # circular buffer of length `length`
  for sample in samples:
    stree.add(sample)

  # Finds index of node that falls within the CDF of values added.
  sampled_idx = stree.find_index(np.random.uniform(0, stree.values[0]))
  """

  def __init__(self, length: int, recompute_every_n: int = 1000):
    # Ensure capacity is a power of 2
    if length <= 0:
      raise ValueError("Length must be positive")
    self.length = length
    self.capacity = 2**math.ceil(math.log2(max(1, length)))
    self.values = np.zeros(2 * self.capacity - 1, dtype=np.float32)

    # Cursor will wrap around when it reaches the capacity.
    self.cursor = 0
    self.full = False

    # Recompute the tree every recompute_every_n steps to ensure fp32
    # precision is not messing things up too much.
    self.recompute_every_n = recompute_every_n
    self.recompute_step = 0

  def _propagate(self, idx: int, value: float):
    """Propagate the value up the tree."""
    # Idx is a number in the range [capacity - 1, 2 * capacity - 2].
    assert 0 <= idx - self.capacity + 1 < self.length, \
        f"Index {idx} out of bounds for length {self.length}"

    while True:
      if idx == 0:
        break
      parent = (idx - 1) // 2
      self.values[parent] += value
      idx = parent

  def add(self, value: float):
    """Add a new value to the tree."""
    self.recompute_step += 1
    if self.recompute_step >= self.recompute_every_n:
      self.recompute_step = 0
      # Recompute the tree
      self.values[:self.capacity - 1].fill(0.0)
      for i in range(self.length):
        idx = i + self.capacity - 1
        self._propagate(idx, self.values[idx])

    idx = self.cursor + self.capacity - 1
    # Substract the old value from the current node
    self._propagate(idx, -self.values[idx])
    self.values[idx] = value
    # Now propagate the value up the tree.
    self._propagate(idx, value)

    # Wrap around the cursor if needed
    self.cursor += 1
    if self.cursor >= self.length:
      self.cursor = 0
      # From now on, the tree is full
      self.full = True

  def update(self, idx: int, value: float):
    """Update the value at the given index."""
    if not self.full and idx >= self.cursor:
      raise IndexError(f"Index out of bounds for non-full tree {idx}")
    if idx < 0 or idx >= self.length:
      raise IndexError(f"Index out of bounds {idx}")

    idx += self.capacity - 1
    # Substract the old value from the current node
    self._propagate(idx, -self.values[idx])
    self.values[idx] = value
    # Now propagate the value up the tree.
    self._propagate(idx, value)

  def find_index(self, value: float) -> int:
    """Find the index of the value in the tree."""
    idx = 0
    while idx < self.capacity - 1:
      left = 2 * idx + 1
      right = left + 1
      if value <= self.values[left]:
        idx = left
      else:
        value -= self.values[left]
        idx = right

    # Convert from tree index to user index
    idx -= (self.capacity - 1)

    # When the tree is not full, we might navigate to indices beyond cursor
    # due to floating point precision issues. Clamp to valid range.
    if not self.full and idx >= self.cursor:
      idx = self.cursor - 1 if self.cursor > 0 else 0
    if self.full and idx >= self.length:
      idx = self.length - 1

    if self.full:
      assert 0 <= idx < self.length, f"Index {idx} for {value} found out of bounds, sum {self.values[0]}"
    else:
      assert 0 <= idx < self.cursor, f"Index {idx} for {value} found out of bounds, sum {self.values[0]}"
    return idx

  def get_min(self) -> float:
    """Get the minimum value in the tree."""
    if not self.full:
      # If the tree is not full, return the minimum of the current values
      return self.values[self.capacity - 1:self.capacity - 1 +
                         self.cursor].min()
    return self.values[self.capacity - 1:self.capacity - 1 + self.length].min()

  def get_max(self) -> float:
    """Get the maximum value in the tree."""
    if not self.full:
      # If the tree is not full, return the maximum of the current values
      return self.values[self.capacity - 1:self.capacity - 1 +
                         self.cursor].max()
    return self.values[self.capacity - 1:self.capacity - 1 + self.length].max()

  def get_sum(self) -> float:
    """Get the sum of all values in the tree."""
    return self.values[0]

  def get_value(self, idx: int) -> float:
    """Get the value at the given index."""
    if not self.full and idx >= self.cursor:
      raise IndexError(f"Index out of bounds for non-full tree {idx}")
    if idx < 0 or idx >= self.length:
      raise IndexError(f"Index out of bounds {idx} for length {self.length}")
    return self.values[self.capacity - 1 + idx]

  def __len__(self):
    """Return the number of values currently stored in the tree."""
    if self.full:
      return self.length
    else:
      return self.cursor

  def __iter__(self):
    """Iterate over the values currently stored in the tree."""
    end = self.length if self.full else self.cursor
    for i in range(end):
      yield self.values[self.capacity - 1 + i]

  def __getitem__(self, idx: int) -> float:
    """Get the value at the given index (for compatibility with deque-style access)."""
    return self.get_value(idx)
