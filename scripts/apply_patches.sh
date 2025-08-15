#!/bin/bash

# Save the current directory
START_DIR=$(pwd)

# Activate the environment
conda activate MarioBros

# Find the installed gym-super-mario-bros directory
GYM_MARIO_PATH=$(python -c "import gym_super_mario_bros; import os; print(os.path.dirname(gym_super_mario_bros.__file__))")

echo "Applying patch to gym-super-mario-bros at $GYM_MARIO_PATH"
cd "$GYM_MARIO_PATH"

# The --ignore-whitespace is often helpful to prevent errors
# The -p1 means strip one directory component from paths in the patch file
# This assumes your patch file paths start with `a/gym_super_mario_bros/...` or `b/gym_super_mario_bros/...`
git apply --ignore-whitespace -p1 "$START_DIR/scripts/gym_super_mario_bros_my_fix.patch"

if [ $? -eq 0 ]; then
  echo "Patch applied successfully!"
else
  echo "Error applying patch. You might need to manually resolve conflicts or inspect the patch file."
  # Return to the original directory before exiting
  cd "$START_DIR"
  # exit 1
fi

# Return to the original directory
cd "$START_DIR"