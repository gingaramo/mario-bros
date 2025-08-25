"""
Keyboard controls and interactive debugging utilities.
"""

import os
import threading
from pynput import keyboard
from .render import set_rendering_enabled, is_headless_mode

# Keyboard control constants
KEY_CONTINUE = 'C'
KEY_NEXT_FRAME = 'N'
KEY_ENABLE_RENDERING = 'S'
KEY_DISABLE_RENDERING = 'H'
KEY_ENABLE_PROFILING = 'P'
KEY_DISABLE_PROFILING = 'O'

# Frame stepping control (global state for keyboard interaction)
FRAMES_FORWARD = -1


def handle_keyboard_input(key):
  """
    Handle keyboard input for controlling frame stepping and rendering.
    
    Args:
        key: The keyboard key pressed
        
    Keyboard shortcuts:
        - 'C': Continue normal execution
        - 'N': Step forward one frame
        - 'S': Enable rendering
        - 'H': Disable rendering (headless mode)
    """
  global FRAMES_FORWARD
  try:
    # Single key presses (with shift modifier)
    if key.char == KEY_CONTINUE:
      FRAMES_FORWARD = -1
    elif key.char == KEY_NEXT_FRAME:
      FRAMES_FORWARD = 1
    elif key.char == KEY_ENABLE_RENDERING:
      set_rendering_enabled(True)
    elif key.char == KEY_DISABLE_RENDERING:
      set_rendering_enabled(False)
    elif key.char == KEY_ENABLE_PROFILING:
      from .profiler import execution_profiler_singleton
      execution_profiler_singleton.start()
      import os
      pid = os.getpid()
      print("Profiling enabled. PID:", pid)
    elif key.char == KEY_DISABLE_PROFILING:
      from .profiler import execution_profiler_singleton
      execution_profiler_singleton.save()
      execution_profiler_singleton.stop()
      print("Profiling disabled.")
  except AttributeError:
    # Handle special keys that don't have a char attribute
    pass


def start_keyboard_listener():
  """
    Start a keyboard listener thread for interactive debugging controls.
    
    Returns:
        keyboard.Listener: The keyboard listener object
    """
  listener = keyboard.Listener(on_press=handle_keyboard_input)
  listener.start()
  return listener


def setup_interactive_controls():
  """
    Setup interactive keyboard controls for debugging and frame stepping.
    Only runs when not in headless mode.
    """
  if not is_headless_mode():
    keyboard_thread = threading.Thread(target=start_keyboard_listener)
    keyboard_thread.daemon = True
    keyboard_thread.start()
    print("Interactive controls enabled:\n"
          f"  - Shift+{KEY_CONTINUE}: Continue normal execution\n"
          f"  - Shift+{KEY_NEXT_FRAME}: Step forward one frame\n"
          f"  - Shift+{KEY_ENABLE_RENDERING}: Enable rendering\n"
          f"  - Shift+{KEY_DISABLE_RENDERING}: Disable rendering")


def wait_for_frame_step():
  """
    Handle frame-by-frame stepping for debugging purposes.
    
    This function blocks execution when frame stepping is enabled, allowing
    users to step through the environment one frame at a time for debugging.
    Only active when not in headless mode.
    """
  if not is_headless_mode():
    global FRAMES_FORWARD
    while FRAMES_FORWARD >= 0:
      if FRAMES_FORWARD > 0:
        FRAMES_FORWARD -= 1
        break
