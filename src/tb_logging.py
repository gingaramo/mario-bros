from torch.utils.tensorboard import SummaryWriter


class GlobalStepSummaryWriter(SummaryWriter):
  """A summary writer that allows setting a global step for logging."""

  def __init__(self,
               log_dir,
               max_queue=10000,
               flush_secs=60,
               purge_step=0,
               global_step=0):
    super().__init__(log_dir, max_queue, flush_secs, purge_step)
    self.global_step = global_step

  def add_scalar(self, tag, scalar_value):
    super().add_scalar(tag, scalar_value, self.global_step)

  def add_histogram(self, tag, values):
    super().add_histogram(tag, values, self.global_step)

  def set_global_step(self, global_step):
    """Set the global step for logging."""
    self.global_step = global_step
