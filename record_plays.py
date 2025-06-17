import gym_super_mario_bros
import numpy as np  # For saving states efficiently
import pickle
import gzip
import time
import time
import traceback
import keyboard
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

import preprocess

env = gym_super_mario_bros.make('SuperMarioBrosRandomStages-v2')
# See https://pypi.org/project/gym-super-mario-bros/
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# --- Data Storage ---
recorded_traces = []
current_episode_traces = []

# --- Game Loop ---
print("Starting Super Mario Bros!")
print("Controls: ")
print("  WAD: Movement")
print("  E: Record Episode and Start Over")
print("  Q: Quit and Save Traces")
print("-----------------------------------------")

pre = preprocess.preprocess_lambda((120, 128))  # Preprocessing function
current_state = env.reset()
# Check if it returned a tuple (state, info) for newer gym with compatibility
if isinstance(current_state, tuple) and len(current_state) == 2:
  current_state, info = current_state
current_state = pre(current_state)  # Preprocess the initial state

episode_count = 0

listener = keyboard.KeyboardListener(['q', 'e', 'd', 'a', 'w'])

try:
  while True:
    env.render()  # Show the game screen

    # Determine action from keyboard input
    action = 0  # Default to NOOP (No Operation)

    is_right_pressed = listener.is_pressed('d')
    is_left_pressed = listener.is_pressed('a')
    is_jump_pressed = listener.is_pressed('w')
    end_episode = listener.is_pressed('e')
    is_quitting = listener.is_pressed('q')

    # See https://github.com/Kautenja/gym-super-mario-bros/blob/master/gym_super_mario_bros/actions.py
    if is_right_pressed and is_jump_pressed:
      action = 2  # right + jump
    elif is_right_pressed:
      action = 1  # right
    elif is_left_pressed:  # Note: SIMPLE_MOVEMENT doesn't have 'left' + 'jump' directly
      action = 6  # left
    elif is_jump_pressed:
      action = 5  # jump
    else:
      action = 0  # NOOP

    # Store the state before taking the action
    next_state, reward, done, info = env.step(action)
    next_state = pre(next_state)  # Preprocess the next state

    state_to_record = np.array(current_state,
                               dtype=np.uint8)  # Ensure state is storable
    next_state_to_record = np.array(next_state, dtype=np.uint8)
    current_state = next_state  # Update current state for the next loop

    # Record the experience
    current_episode_traces.append(
        (state_to_record, action, float(reward), next_state_to_record,
         bool(done), info))

    if done or end_episode or is_quitting:
      episode_count += 1
      print(
          f"Episode {episode_count} finished. Score: {info.get('score', 0)}. Traces in episode: {len(current_episode_traces)}"
      )
      recorded_traces.extend(current_episode_traces)
      print(f"Total traces collected so far: {len(recorded_traces)}")
      current_episode_traces = []  # Reset for the next episode

      # Reset environment for the next episode
      try:
        current_state = env.reset()
        if isinstance(current_state,
                      tuple) and len(current_state) == 2:  # For newer gym
          current_state, info = current_state
        current_state = pre(current_state)  # Preprocess the new state
      except Exception as e:
        print(f"Error during env.reset() after episode: {e}")
        break  # Exit loop if reset fails

      if is_quitting:
        print("Quitting and saving...")
        break
      print("New game started. Press Q to quit and save.")

    time.sleep(0.02)

except KeyboardInterrupt:
  print("\nRecording interrupted by user! Saving any collected traces...")
except Exception as e:
  print(f"\nAn error occurred: {e}")
  print(traceback.format_exc())
  print("Saving any collected traces...")
finally:
  env.close()
  if recorded_traces:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"mario_traces_{timestamp}.pkl.gz"
    try:
      with gzip.open(filename, 'wb') as f:
        pickle.dump(recorded_traces, f)
      print(f"Traces successfully saved to {filename}")
      print(f"Total episodes recorded: {episode_count}")
      print(
          f"Total (state, action, reward, next_state, done) tuples: {len(recorded_traces)}"
      )
    except Exception as e:
      print(f"Error saving traces: {e}")
  else:
    print("No traces were recorded or an error prevented saving.")

  print("Recording script finished.")
  listener.stop()
