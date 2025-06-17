import numpy as np
import cv2


def to_grayscale(img: np.ndarray) -> np.ndarray:
  """Takes an image as a np array of 3 dimensions (W, H, C) and turn into greyscale"""
  return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def to_size(img: np.ndarray, size: tuple) -> np.ndarray:
  """Takes an image as a np array of 3 dimensions (W, H, C) and resizes it to the given size"""
  return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def preprocess_lambda(target_state_size):
  """Returns a function that preprocesses the state to grayscale and resizes it."""

  def preprocess_state(state):
    """Preprocess the state to grayscale and resize it."""
    state = to_grayscale(state)
    state = to_size(state, target_state_size)
    return state

  return preprocess_state
