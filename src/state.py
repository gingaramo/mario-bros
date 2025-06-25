import cv2
import numpy as np
import torch


class State(object):
  """Class to preprocess and feed state for the Network.
  
  Captures teh logic for stacking frames, resizing, converting to grayscale,
  and normalizing the input state for the model.
  """

  def __init__(self, device, config: dict):
    self.device = device
    self.stack_frames = config['stack_frames']
    self.stacked_frames = [
        torch.zeros(config['resize_shape']).to(self.device)
        for _ in range(self.stack_frames)
    ]
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
    self.stacked_frames.append(
        torch.Tensor(self.preprocess(state)).to(self.device))

  def current(self) -> torch.tensor:
    """Returns the input for the model."""
    # Stack the list of tensors along a new dimension (e.g., dim=0 for channels).
    return torch.stack(self.stacked_frames, dim=0)
