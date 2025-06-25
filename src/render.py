import cv2
import numpy as np


def render_q_values_bar(q_values_norm, width=256, height=40, labels=None):
  n_actions = len(q_values_norm)
  bar_img = np.zeros((height + 16, width, 3),
                     dtype=np.uint8)  # Extra space for labels
  bar_width = width // n_actions
  for i, q in enumerate(q_values_norm):
    # Height proportional to Q-value
    bar_h = int(q * height)
    y_start = height - bar_h
    # Color: red (low) to green (high)
    color = (0, int(255 * q), int(255 * (1 - q)))
    cv2.rectangle(bar_img, (i * bar_width, y_start),
                  ((i + 1) * bar_width - 2, height), color, -1)
    # Draw label if provided
    if labels is not None:
      label = '+'.join(labels[i]) if isinstance(labels[i], list) else str(
          labels[i])
      font_scale = 0.4
      thickness = 1
      text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                  thickness)[0]
      text_x = i * bar_width + (bar_width - text_size[0]) // 2
      text_y = height + 12
      cv2.putText(bar_img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                  font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
  return bar_img


def stack_frame_with_q_bar(frame, q_bar):
  return np.vstack((frame, q_bar))


def render_mario_with_q_values(next_state, q_values, labels):
  """
  Render the Mario frame with Q-values overlay.
  """
  # Convert next_state from RGB to BGR for correct OpenCV display
  next_state_bgr = cv2.cvtColor(next_state, cv2.COLOR_RGB2BGR)
  # Upscale the frame to 2x before rendering Q-values
  next_state_up = cv2.resize(
      next_state_bgr,
      (next_state_bgr.shape[1] * 2, next_state_bgr.shape[0] * 2),
      interpolation=cv2.INTER_NEAREST)

  # Convert next_state from RGB to BGR for correct OpenCV display
  next_state_bgr = cv2.cvtColor(next_state, cv2.COLOR_RGB2BGR)
  # Upscale the frame to 2x before rendering Q-values
  next_state_up = cv2.resize(
      next_state_bgr,
      (next_state_bgr.shape[1] * 2, next_state_bgr.shape[0] * 2),
      interpolation=cv2.INTER_NEAREST)

  # Normalize and render Q-values
  q_values_norm = (q_values - np.min(q_values)) / (np.ptp(q_values) + 1e-8)
  q_bar = render_q_values_bar(q_values_norm,
                              width=next_state_up.shape[1],
                              labels=labels)
  frame_with_q = stack_frame_with_q_bar(next_state_up, q_bar)
  cv2.imshow("Mario with Q-values", frame_with_q)
  cv2.waitKey(1)  # Allow OpenCV to update the display
