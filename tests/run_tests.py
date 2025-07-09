#!/usr/bin/env python3
"""
Simple test runner using Python's unittest framework.
Discovers and runs all test files matching '*test*.py' pattern.
"""

import unittest
import sys
import os
import argparse

# Add src directory to path so tests can import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
  sys.path.insert(0, src_path)


def main():
  """Main entry point for the test runner."""
  parser = argparse.ArgumentParser(
      description="Run unit tests using unittest framework",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog="""
Examples:
  python run_tests.py           # Run all tests
  python run_tests.py -v        # Run with verbose output
  python run_tests.py --failfast # Stop on first failure
        """)

  parser.add_argument("-v",
                      "--verbose",
                      action="store_true",
                      help="Verbose output")
  parser.add_argument("--failfast",
                      action="store_true",
                      help="Stop on first failure")

  args = parser.parse_args()

  # Use unittest's built-in test discovery
  loader = unittest.TestLoader()
  suite = loader.discover('.', pattern='*test*.py')

  # Count total tests
  test_count = suite.countTestCases()
  if test_count == 0:
    print("No tests found matching pattern '*test*.py'")
    sys.exit(1)

  print(f"Running {test_count} test cases...")

  # Run tests
  runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1,
                                   failfast=args.failfast)

  result = runner.run(suite)

  # Exit with appropriate code
  if result.wasSuccessful():
    print("\nALL TESTS PASSED")
    sys.exit(0)
  else:
    print("\nTESTS FAILED")
    sys.exit(1)


if __name__ == "__main__":
  main()
