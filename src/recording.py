import cv2
import os


class Recording():
  "Collects frames and writes a mp4 video file when done."

  def __init__(self, filename, fps=40):
    output_dir = "./recording"
    os.makedirs(output_dir, exist_ok=True)
    self.output_path = os.path.join(output_dir, filename + ".mp4")
    self.fps = fps
    self.frame_size = None
    self.frames = []

  def add_frame(self, frame):
    """Add a frame (numpy array) to the recording."""
    if not self.frame_size:
      self.frame_size = frame.shape[:-1]
      print(f"using {self.frame_size=}")
    self.frames.append(frame)

  def save(self):
    """Write all collected frames to an mp4 file."""
    if not self.frames:
      print("No frames to save")
      return

    # Try different codecs until one works
    codecs_to_try = [
        cv2.VideoWriter_fourcc(*'mp4v'),
        cv2.VideoWriter_fourcc(*'XVID'),
        cv2.VideoWriter_fourcc(*'MJPG'),
        cv2.VideoWriter_fourcc(*'X264'),
    ]

    print(f"using {self.frame_size=}")
    # Note: frame_size should be (width, height), not (height, width)
    frame_size_wh = (self.frame_size[1], self.frame_size[0])

    out = None
    for fourcc in codecs_to_try:
      out = cv2.VideoWriter(self.output_path, fourcc, self.fps, frame_size_wh)
      if out.isOpened():
        print(f"Successfully opened video writer with fourcc: {fourcc}")
        break
      else:
        out.release()
        out = None

    if out is None:
      print(
          f"Error: Could not open video writer for {self.output_path} with any codec"
      )
      return

    print(
        f"Writing {len(self.frames)} frames to {self.output_path} at {self.fps} FPS"
    )
    for frame in self.frames:
      out.write(frame)
    out.release()
    print(f"Recorded {self.output_path}")
