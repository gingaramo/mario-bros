import threading
from pynput import keyboard


class KeyboardListener(object):
  "Keyboard listener that starts and stops pynput listener"

  def __init__(self, keys):
    self._pressed_event = {key: threading.Event() for key in keys}

    def _on_release(key):
      try:
        self._pressed_event[key.char].clear()
        return True
      except Exception as e:
        pass

    def _on_press(key):
      try:
        self._pressed_event[key.char].set()
        return True
      except AttributeError:
        pass

    self._listener = keyboard.Listener(on_press=_on_press,
                                       on_release=_on_release)
    self._listener.start()

  def stop(self):
    self._listener.stop()
    self._listener.join()

  def is_pressed(self, key):
    assert key in self._pressed_event
    return self._pressed_event[key].is_set()
