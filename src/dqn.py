import numpy as np
import random
import torch
import cv2
import torch.nn as nn


# Define the DQN model
class DQN(nn.Module):

  def __init__(self, action_size: int, mock_input: torch.Tensor, config: dict):
    super(DQN, self).__init__()
    print(f"{action_size=}, {mock_input.shape=}, {config=}")
    # We need to ensure mock_input is on the CPU for Conv2d initialization.
    mock_input = mock_input.to(torch.device('cpu'))
    self.conv1 = nn.Conv2d(mock_input.shape[0], 64, 8, stride=4)
    self.conv2 = nn.Conv2d(64, 128, 4, stride=2)
    self.conv3 = nn.Conv2d(128, 256, 4, stride=2)

    def _get_flattened_shape(x: torch.Tensor) -> int:
      x = x.unsqueeze(0)
      x = self.conv1(x)
      x = self.conv2(x)
      x = self.conv3(x)
      return x.flatten().shape[0]

    self.side_input_floats = 1  # Currently only last_action
    hidden_layers = [
        _get_flattened_shape(mock_input) + self.side_input_floats
    ] + config['hidden_layers'] + [action_size]
    # Add linear layers
    self.linear = []
    for in_, out_ in zip(hidden_layers[:-1], hidden_layers[1:]):
      self.linear.append(nn.Linear(in_, out_))
    self.linear = nn.ModuleList(self.linear)

  def render(self, x, side_input: int):
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
    cv2.waitKey(1)
    #cv2.destroyAllWindows()

  def forward(self, x, side_input):
    # Add batch dimension if input is (C, H, W)
    has_batch_dim = False
    if x.ndim > 3:
      has_batch_dim = True

    if not has_batch_dim:
      # When not trainig we render the input frames
      self.render(x, side_input)

    if not has_batch_dim:
      x = x.unsqueeze(0)
      side_input = side_input.unsqueeze(0)
    x = torch.relu(self.conv1(x))
    x = torch.relu(self.conv2(x))
    x = torch.relu(self.conv3(x))
    # Add it again after flattening.
    x = x.flatten(start_dim=1)
    x = torch.concat([x, side_input], dim=1)
    for i, layer in enumerate(self.linear[:-1]):
      x = torch.relu(layer(x))
    # Last layer without relu
    x = self.linear[-1](x)
    if not has_batch_dim:
      x = x.squeeze(0)  # Remove batch dimension if it was added
    return x
