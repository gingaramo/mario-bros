#!/usr/bin/env python3
"""
Test runner script that executes tests from the tests/ directory.
"""

import os
import sys
import subprocess


def main():
  # Get the directory containing this script (project root)
  project_root = os.path.dirname(os.path.abspath(__file__))
  tests_dir = os.path.join(project_root, 'tests')

  if not os.path.exists(tests_dir):
    print("Error: tests/ directory not found")
    sys.exit(1)

  # Change to tests directory
  os.chdir(tests_dir)

  # Run the test runner
  cmd = [sys.executable, 'run_tests.py'] + sys.argv[1:]

  try:
    result = subprocess.run(cmd)
    sys.exit(result.returncode)
  except KeyboardInterrupt:
    print("\nTest run interrupted")
    sys.exit(1)


if __name__ == "__main__":
  main()
