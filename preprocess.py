import numpy as np
import cv2


def to_grayscale(img: np.ndarray) -> np.ndarray:
  """Takes an image as a np array of 3 dimensions (W, H, C) and turn into greyscale"""
  return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def to_size(img: np.ndarray, size: tuple) -> np.ndarray:
  """Takes an image as a np array of 3 dimensions (W, H, C) and resizes it to the given size"""
  return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def crop_top_pixels(image: np.ndarray, pixels: int) -> np.ndarray:
  """Crops the top 'pixels' rows from a grayscale image of shape (H, W)."""
  return image[pixels:, :]


def preprocess_lambda(target_state_size, top_pixels_to_crop):
  """Returns a function that preprocesses the state by converting it to grayscale
  resizing it to target_state_size, and then cropping the specified number of 
  top pixels.

  Note: target_state_size refers to the size before cropping the top pixels.
  """

  def preprocess_state(state):
    """Preprocess the state to grayscale and resize it."""
    # Following approach in https://arxiv.org/pdf/1312.5602
    state = to_grayscale(state)
    state = to_size(state, target_state_size)
    state = crop_top_pixels(state, top_pixels_to_crop)
    return state

  return preprocess_state
