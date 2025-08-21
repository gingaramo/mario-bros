from torch.utils.tensorboard import SummaryWriter


class RemoteSummaryWriterClient():
  """
  A fake custom summary writer that allows inter-process communication.
  
  It takes as input a multiprocessing Queue and forwards writes to it.
  """

  def __init__(self, queue):
    self.queue = queue

  def add_scalar(self, tag, scalar_value):
    self.queue.put(('scalar', tag, scalar_value))

  def add_histogram(self, tag, values):
    self.queue.put(('histogram', tag, values))

  def flush(self):
    """Flush the queue."""
    self.queue.put(('flush', ))

  def set_global_step(self, global_step):
    """Set the global step for logging."""
    self.queue.put(('global_step', global_step))


class RemoteSummaryWriterServer():
  """
  A fake custom summary writer that allows inter-process communication.

  It takes as input a multiprocessing Queue processes forwards calls to it.
  """

  def __init__(self, queue, summary_writer):
    self.queue = queue
    self.summary_writer = summary_writer

  def run(self):
    while True:
      item = self.queue.get()
      if item[0] == 'scalar':
        self.summary_writer.add_scalar(item[1], item[2])
      elif item[0] == 'histogram':
        self.summary_writer.add_histogram(item[1], item[2])
      elif item[0] == 'global_step':
        self.summary_writer.set_global_step(item[1])
      elif item[0] == 'flush':
        self.summary_writer.flush()
      else:
        raise ValueError(f"Unknown item type: {item[0]}")


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
