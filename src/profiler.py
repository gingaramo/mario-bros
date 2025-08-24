class ExecutionProfile:

  def __init__(self):
    self.enabled = False
    self.records = {}
    # Name of the profiling session is set later on (worker/trainer, and agent name)
    self.name = None

  def start(self):
    self.enabled = True

  def stop(self):
    self.enabled = False

  def record(self, name, start, finish):
    if not self.enabled:
      return
    if name not in self.records:
      self.records[name] = []
    self.records[name].append((start, finish))

  def set_name(self, name):
    self.name = name

  def save(self):
    import json
    import os
    filename = f"{self.name}_profile_{os.getpid()}.json"
    with open(filename, 'w') as f:
      json.dump(self.records, f, indent=2)


execution_profiler_singleton = ExecutionProfile()


class ProfileScope:
  """
  Context manager for profiling a code block.

  Usage:
      with ProfileScope('block_name'):
          # code to profile
  """

  def __init__(self, name, profiler=execution_profiler_singleton):
    self.name = name
    self.profiler = profiler

  def __enter__(self):
    from time import time
    self.start_time = time()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    from time import time
    end_time = time()
    self.profiler.record(self.name, self.start_time, end_time)
