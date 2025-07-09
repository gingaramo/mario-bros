#!/usr/bin/env python3
"""
Simple test runner that runs unit tests from the tests/ directory.
"""

import os
import sys
import subprocess


def main():
  # Change to tests directory where tests are now located
  script_dir = os.path.dirname(os.path.abspath(__file__))
  tests_dir = script_dir

  if not os.path.exists(tests_dir):
    print("Error: tests/ directory not found")
    sys.exit(1)

  os.chdir(tests_dir)

  # Forward all arguments to run_tests.py
  cmd = [sys.executable, 'run_tests.py'] + sys.argv[1:]

  try:
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
  except KeyboardInterrupt:
    print("\nTest run interrupted")
    sys.exit(1)


if __name__ == "__main__":
  main()
