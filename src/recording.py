import cv2
import os


class Recording():
  "Collects frames and writes a mp4 video file when done."

  def __init__(self, filename, fps=30, frame_size=(640, 480)):
    output_dir = "./recording"
    os.makedirs(output_dir, exist_ok=True)
    self.output_path = os.path.join(output_dir, filename + ".mp4")
    self.fps = fps
    self.frame_size = frame_size
    self.frames = []

  def add_frame(self, frame):
    """Add a frame (numpy array) to the recording."""
    assert frame.shape[:-1] == (
        self.frame_size[1], self.frame_size[0]
    ), f"{frame.shape[:-1]} != {(self.frame_size[1], self.frame_size[0])}"
    self.frames.append(frame)

  def save(self):
    """Write all collected frames to an mp4 file."""
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)
    for frame in self.frames:
      out.write(frame)
    out.release()
    print(f"Recorded {self.output_path}")
