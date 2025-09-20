import threading


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

  def record(self, name, start, finish, metadata={}):
    if not self.enabled:
      return
    if name not in self.records:
      self.records[name] = []
    if len(metadata) > 0:
      self.records[name].append((start, finish, metadata))
    else:
      self.records[name].append((start, finish))

  def set_name(self, name):
    self.name = name

  def save(self):
    import json
    import os
    filename = f"{self.name}_profile_{os.getpid()}.json"
    with open(filename, 'w') as f:
      json.dump(self.records, f, indent=2)
    self.records = {}  # Clear records after saving


execution_profiler_singleton = ExecutionProfile()
_thread_local_scope = threading.local()


class ProfileScope:
  """
  Context manager for profiling a code block.

  Usage:
      with ProfileScope('block_name'):
          # code to profile
  """

  def __init__(self, name, profiler=execution_profiler_singleton):
    self.name = name + '_' + threading.current_thread().native_id.__str__()
    self.profiler = profiler
    self.metadata = {}
    _thread_local_scope.profile_scope = self

  @staticmethod
  def add_metadata(key, value):
    if not hasattr(_thread_local_scope, 'profile_scope'):
      raise RuntimeError("No active profile scope found")
    _thread_local_scope.profile_scope._add_metadata(key, value)

  def _add_metadata(self, key, value):
    if not self.profiler.enabled:
      return
    self.metadata[key] = value

  def __enter__(self):
    from time import time
    self.start_time = time()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    from time import time
    end_time = time()
    self.profiler.record(self.name, self.start_time, end_time, self.metadata)


class ProfileLockScope(ProfileScope):

  def __init__(self, name, lock, profiler=execution_profiler_singleton):
    super().__init__(name, profiler)
    self.lock = lock

  def __enter__(self):
    with ProfileScope(self.name + '_lock'):
      self.lock.acquire()
    super().__enter__()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.lock.release()
    super().__exit__(exc_type, exc_value, traceback)
