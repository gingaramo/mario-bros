from torch.utils.tensorboard import SummaryWriter


class CustomSummaryWriter(SummaryWriter):
  """
  A custom summary writer.
   
  It allows setting a global step for logging and sampled logging.
  """

  def __init__(self,
               log_dir,
               max_queue=10000,
               flush_secs=60,
               purge_step=0,
               global_step=0,
               metric_prefix_sample_mod={}):
    super().__init__(log_dir, max_queue, flush_secs, purge_step)
    self.global_step = global_step
    self.metric_prefix_sample_mod = metric_prefix_sample_mod

  def add_scalar(self, tag, scalar_value):
    for prefix, sample_mod in self.metric_prefix_sample_mod.items():
      if tag.startswith(prefix):
        if self.global_step % sample_mod == 0:
          super().add_scalar(tag, scalar_value, self.global_step)
        return
    super().add_scalar(tag, scalar_value, self.global_step)

  def add_histogram(self, tag, values):
    super().add_histogram(tag, values, self.global_step)

  def set_global_step(self, global_step):
    """Set the global step for logging."""
    self.global_step = global_step


class DummySummaryWriter(CustomSummaryWriter):
  """
  A dummy summary writer that does not log anything.
  """

  def __init__(self, *args, **kwargs):
    pass

  def add_scalar(self, tag, scalar_value):
    pass

  def add_histogram(self, tag, values):
    pass

  def set_global_step(self, global_step):
    pass
