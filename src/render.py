import cv2
import numpy as np

# By default rendering is enabled
RENDERING_ENABLED = True


def maybe_render_dqn(x, side_input: int):
  global RENDERING_ENABLED
  if not RENDERING_ENABLED:
    return
  num_frames = x.shape[0]
  frames = x.cpu().detach().numpy()
  # Normalize and convert to uint8 for display if needed
  if frames.max() <= 1.0:
    frames = (frames * 255).astype(np.uint8)
  else:
    frames = frames.astype(np.uint8)
  # Stack frames horizontally for visualization
  stacked = np.concatenate([frames[i] for i in range(num_frames)], axis=1)
  # Scale stacked 2x the size
  stacked = cv2.resize(stacked, (stacked.shape[1] * 2, stacked.shape[0] * 2),
                       interpolation=cv2.INTER_NEAREST)

  # If grayscale, add channel dimension for cv2
  if stacked.ndim == 2:
    stacked = cv2.cvtColor(stacked, cv2.COLOR_GRAY2BGR)
  # Render side_input value below the stacked frames
  text = f"side_input: {side_input.item()}"
  # Calculate new image height to add space for text
  text_height = 40
  new_height = stacked.shape[0] + text_height
  new_img = np.zeros((new_height, stacked.shape[1], 3), dtype=stacked.dtype)
  new_img[:stacked.shape[0], :, :] = stacked
  # Put text in the new area below the frames
  cv2.putText(new_img, text, (10, stacked.shape[0] + 30),
              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
  stacked = new_img
  cv2.imshow("Stacked Frames", stacked)


def frame_q_values_bar(q_values_norm,
                       action,
                       width=256,
                       height=40,
                       labels=None):
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
    if action == i:
      color = (int(255), int(255), int(255))
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


def frame_with_q_values(next_state, q_values, action, labels):
  """
  Render the frame with Q-values overlay.
  """
  # Convert next_state from RGB to BGR for correct OpenCV display
  next_state_bgr = cv2.cvtColor(next_state, cv2.COLOR_RGB2BGR)
  # Upscale the frame to 2x before rendering Q-values
  next_state_up = cv2.resize(
      next_state_bgr,
      (next_state_bgr.shape[1] * 4, next_state_bgr.shape[0] * 4),
      interpolation=cv2.INTER_NEAREST)

  # Normalize and render Q-values
  q_values_norm = (q_values - np.min(q_values)) / (np.ptp(q_values) + 1e-8)
  q_bar = frame_q_values_bar(q_values_norm,
                             action,
                             width=next_state_up.shape[1],
                             labels=labels)
  frame_with_q = np.vstack((next_state_up, q_bar))
  return frame_with_q


def render(next_state, q_values, action, labels, recording=None):
  global RENDERING_ENABLED
  key = cv2.pollKey()
  if key != -1:
    if key == ord('h') or key == ord('H'):
      RENDERING_ENABLED = False
      print("Rendering hidden (press 'S' to show again)")
    elif key == ord('s') or key == ord('S'):
      RENDERING_ENABLED = True
      print("Rendering shown (press 'H' to hide)")

  if RENDERING_ENABLED or recording:
    frame = frame_with_q_values(next_state, q_values, action, labels)
    if recording:
      recording.add_frame(frame)
    if RENDERING_ENABLED:
      cv2.imshow("Frame with Q-values", frame)
