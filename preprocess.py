import cv2
import numpy as np
import torch


class StatePreprocess(object):
  """Class to preprocess the state for the Network."""

  def __init__(self, config: dict):
    self.stack_frames = config['stack_frames']
    self.stacked_frames = [np.zeros(config['resize_shape'], dtype=np.float32)
                           ] * self.stack_frames
    self.resize_shape = tuple(config['resize_shape'])
    self.grayscale = config['grayscale']
    self.normalize = config['normalize']

  def preprocess(self, state: np.ndarray) -> np.ndarray:
    """Preprocesses the state by resizing, converting to grayscale, and normalizing."""
    state = cv2.resize(state, self.resize_shape, interpolation=cv2.INTER_AREA)
    if self.grayscale:
      state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    if self.normalize:
      state = state / 255.0
    return state

  def add(self, state: np.ndarray) -> None:
    """Adds a new frame to the stack of frames."""
    self.stacked_frames.pop(0)
    self.stacked_frames.append(self.preprocess(state))

  def __call__(self) -> torch.tensor:
    """Returns the input for the model."""
    return torch.Tensor(self.stacked_frames)
