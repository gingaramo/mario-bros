import cv2
import numpy as np
import torch
import math

HEADLESS_MODE = False
RENDERING_ENABLED = True
USE_DASHBOARD = False
_dashboard_sender = None
_trainer_name = None


def set_headless_mode(value):
  global HEADLESS_MODE
  print(f"Headless mode: {value}")
  HEADLESS_MODE = value


def is_headless_mode():
  global HEADLESS_MODE
  return HEADLESS_MODE


def set_rendering_enabled(value):
  global RENDERING_ENABLED
  print(f"Rendering enabled: {value}")
  RENDERING_ENABLED = value


def set_dashboard_mode(enabled, trainer_name=None, host='localhost', port=9999, quality=85):
  """
  Enable or disable dashboard mode for non-blocking rendering.
  
  Args:
    enabled: True to send frames to dashboard, False to use cv2.imshow
    trainer_name: Name of the trainer (e.g., config['agent']['name'])
    host: Dashboard server hostname
    port: Dashboard UDP port
    quality: JPEG compression quality (1-100, default 85)
  """
  global USE_DASHBOARD
  global _dashboard_sender
  global _trainer_name
  
  USE_DASHBOARD = enabled
  
  if enabled:
    if trainer_name is None:
      raise ValueError("trainer_name is required when enabling dashboard mode")
    
    _trainer_name = trainer_name
    
    if _dashboard_sender is None or _dashboard_sender.trainer_name != trainer_name:
      if _dashboard_sender is not None:
        _dashboard_sender.close()
      
      from src.dashboard import FrameSender
      _dashboard_sender = FrameSender(trainer_name=trainer_name, host=host, port=port, quality=quality)
      print(f"Dashboard mode enabled for trainer '{trainer_name}' - sending frames to {host}:{port}")
      print(f"  JPEG quality: {quality}")
  elif not enabled and _dashboard_sender is not None:
    _dashboard_sender.close()
    _dashboard_sender = None
    _trainer_name = None
    print("Dashboard mode disabled")


def _display_frame(window_name, frame):
  """
  Display a frame either via cv2.imshow or dashboard.
  
  Args:
    window_name: Name/channel for the frame
    frame: numpy array (BGR image)
  """
  global USE_DASHBOARD
  global _dashboard_sender
  
  if USE_DASHBOARD and _dashboard_sender is not None:
    _dashboard_sender.send_frame(frame, channel=window_name)
  else:
    cv2.imshow(window_name, frame)
    cv2.waitKey(1)


def should_render():
  global HEADLESS_MODE
  global RENDERING_ENABLED
  return (not HEADLESS_MODE) and RENDERING_ENABLED


def tile_frames_in_grid(frames, n, m):
  """
  Tile frames in an n x m grid pattern.

  Args:
    frames: List of frames (numpy arrays) to tile
    
  Returns:
    numpy array: Combined frame with all frames tiled in a grid
  """
  if len(frames) == 0:
    # If no frames, return empty image
    return np.zeros((100, 100, 3), dtype=np.uint8)

  # Get dimensions of a single frame
  frame_height, frame_width = frames[0].shape[:2]

  # Create empty grid image
  grid_height = n * frame_height
  grid_width = m * frame_width
  final_frame = np.zeros((grid_height, grid_width, 3), dtype=frames[0].dtype)

  # Place each frame in the grid
  for i, frame in enumerate(frames):
    row = i // m
    col = i % m
    y_start = row * frame_height
    y_end = y_start + frame_height
    x_start = col * frame_width
    x_end = x_start + frame_width
    final_frame[y_start:y_end, x_start:x_end] = frame

  return final_frame


def maybe_render_dqn(x, side_input: torch.Tensor):
  if not should_render():
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
  if side_input.numel() > 0:
    text = f"side_input: {side_input}"
    # Calculate new image height to add space for text
    text_height = 40
    new_height = stacked.shape[0] + text_height
    new_img = np.zeros((new_height, stacked.shape[1], 3), dtype=stacked.dtype)
    new_img[:stacked.shape[0], :, :] = stacked
    # Put text in the new area below the frames
    cv2.putText(new_img, text, (10, stacked.shape[0] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    stacked = new_img
  _display_frame("Stacked Frames", stacked)


def frame_q_values_bar(q_values_norm,
                       action,
                       width=256,
                       height=40,
                       labels=None,
                       q_values_orig=None):
  n_actions = len(q_values_norm)
  bar_img = np.zeros((height + 32, width, 3),
                     dtype=np.uint8)  # Extra space for labels and Q-values
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

    # Draw Q-value if provided
    if q_values_orig is not None:
      q_val_text = f"{q_values_orig[i]:.2f}"
      font_scale = 0.35
      thickness = 1
      text_size = cv2.getTextSize(q_val_text, cv2.FONT_HERSHEY_SIMPLEX,
                                  font_scale, thickness)[0]
      text_x = i * bar_width + (bar_width - text_size[0]) // 2
      text_y = height + 28
      cv2.putText(bar_img, q_val_text, (text_x, text_y),
                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0),
                  thickness, cv2.LINE_AA)  # Yellow color
  return bar_img


def frame_with_q_values(next_state, q_values, action, labels, reward, upscale_factor):
  """
  Render the frame with Q-values overlay.
  """
  # Convert next_state from RGB to BGR for correct OpenCV display
  next_state_bgr = cv2.cvtColor(next_state, cv2.COLOR_RGB2BGR)
  # Upscale before rendering Q-values
  next_state_up = cv2.resize(next_state_bgr,
                             (int(next_state_bgr.shape[1] * upscale_factor),
                              int(next_state_bgr.shape[0] * upscale_factor)),
                             interpolation=cv2.INTER_NEAREST)

  # Write reward text on the top right part of the frame
  reward_text = f"Reward: {reward:.2f}"
  cv2.putText(next_state_up, reward_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
              (0, 255, 255), 2)  # Yellow color

  # Normalize and render Q-values
  q_values_norm = (q_values - np.min(q_values)) / (np.ptp(q_values) + 1e-8)
  q_bar = frame_q_values_bar(q_values_norm,
                             action,
                             width=next_state_up.shape[1],
                             labels=labels,
                             q_values_orig=q_values)
  frame_with_q = np.vstack((next_state_up, q_bar))
  return frame_with_q


def render(info, q_values, action, rewards, config, recording=None):
  labels = config['env']['env_action_labels']
  upscale_factor = config.get('render_upscale_factor', 1)
  layout = config.get('render_layout', None)
  if not should_render() and not recording:
    return

  assert 'observation_frame' in info, "Missing observation frame in info"
  frame = info['observation_frame']
  # Take the first environment's data, for now. Might be good to render all.

  frames = []
  for frame, q_values, action, reward in zip(frame, q_values, action, rewards):
    # Render the frame with Q-values and action labels
    frame = frame_with_q_values(frame, q_values, action, labels, reward,
                                upscale_factor)
    frames.append(frame)

  if recording:
    recording.add_frame(frame[0])

  if should_render():
    if layout is None:
      # Render square by default.
      layout = (int(math.ceil(q_values.shape[0]**0.5)),
                int(math.ceil(q_values.shape[0]**0.5)))
    final_frame = tile_frames_in_grid(frames[:layout[0] * layout[1]],
                                      layout[0], layout[1])
    _display_frame("Frame with Q-values", final_frame)


def render_model_weights(model):
  """
  Render neural network weights as a square heatmap image.
  
  Args:
    model: PyTorch model whose weights will be visualized
  """
  if not should_render():
    return

  # Collect all weight parameters (excluding biases)
  all_weights = []
  for name, param in model.named_parameters():
    if 'weight' in name and param.requires_grad:
      # Flatten the weight tensor and convert to numpy
      weights_flat = param.detach().cpu().numpy().flatten()
      all_weights.extend(weights_flat)

  if not all_weights:
    return

  # Convert to numpy array
  weights_array = np.array(all_weights)

  # Calculate square image size
  total_params = len(weights_array)
  img_size = int(math.ceil(math.sqrt(total_params)))

  # Pad with zeros if necessary to make it square
  if total_params < img_size * img_size:
    padding = img_size * img_size - total_params
    weights_array = np.pad(weights_array, (0, padding),
                           mode='constant',
                           constant_values=0)

  # Reshape to square image
  weights_img = weights_array.reshape(img_size, img_size)

  # Normalize to [0, 1] range
  min_weight = weights_img.min()
  max_weight = weights_img.max()

  if max_weight != min_weight:
    weights_normalized = (weights_img - min_weight) / (max_weight - min_weight)
  else:
    weights_normalized = np.zeros_like(weights_img)

  # Convert to 8-bit image
  weights_uint8 = (weights_normalized * 255).astype(np.uint8)

  # Apply green-to-red colormap
  # OpenCV uses BGR, so we create a colormap where:
  # Green (low values): [0, 255, 0]
  # Red (high values): [0, 0, 255]
  colored_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

  # Red channel increases with weight value
  colored_img[:, :, 2] = weights_uint8
  # Green channel decreases with weight value
  colored_img[:, :, 1] = 255 - weights_uint8
  # Blue channel stays at 0

  # Scale up the image for better visibility (10x)
  scale_factor = 10
  display_img = cv2.resize(colored_img,
                           (img_size * scale_factor, img_size * scale_factor),
                           interpolation=cv2.INTER_NEAREST)

  # Add text overlay with statistics
  text_info = f"Weights: {total_params} | Range: [{min_weight:.4f}, {max_weight:.4f}]"
  cv2.putText(display_img, text_info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
              (255, 255, 255), 2)

  # Display the image
  _display_frame("Model Weights Heatmap", display_img)
