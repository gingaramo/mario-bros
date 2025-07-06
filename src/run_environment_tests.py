#!/usr/bin/env python3
"""
Test runner for environment wrapper classes.

Usage:
    python run_environment_tests.py
    python run_environment_tests.py -v  # verbose output
"""

import sys
import unittest
import argparse


def main():
  parser = argparse.ArgumentParser(description='Run environment wrapper tests')
  parser.add_argument('-v',
                      '--verbose',
                      action='store_true',
                      help='Verbose output')
  parser.add_argument('--pattern',
                      type=str,
                      default='test_*.py',
                      help='Pattern for test discovery')

  args = parser.parse_args()

  # Discover and run tests
  loader = unittest.TestLoader()
  suite = loader.discover('.', pattern='environment_test.py')

  verbosity = 2 if args.verbose else 1
  runner = unittest.TextTestRunner(verbosity=verbosity)
  result = runner.run(suite)

  # Exit with error code if tests failed
  sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
  main()
