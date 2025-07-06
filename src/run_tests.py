#!/usr/bin/env python3
"""
Test runner for all test files in the src directory.
Automatically discovers and runs all test files matching the pattern '*test*.py'.
"""

import unittest
import sys
import os
from pathlib import Path
import argparse
from typing import List, Optional


def discover_tests(test_dir: str = ".",
                   pattern: str = "*test*.py",
                   verbose: bool = False) -> unittest.TestSuite:
  """
    Discover all test files in the given directory.
    
    Args:
        test_dir: Directory to search for test files
        pattern: Pattern to match test files
        verbose: Whether to print discovered test files
        
    Returns:
        TestSuite containing all discovered tests
    """
  if verbose:
    print(f"Discovering tests in: {os.path.abspath(test_dir)}")
    print(f"Pattern: {pattern}")

  loader = unittest.TestLoader()
  suite = loader.discover(test_dir, pattern=pattern, top_level_dir=test_dir)

  if verbose:
    test_count = suite.countTestCases()
    print(f"Discovered {test_count} test cases")

    # Print discovered test files
    for test_group in suite:
      if hasattr(test_group, '_tests'):
        for test_case in test_group._tests:
          if hasattr(test_case, '_testMethodName'):
            module_name = test_case.__class__.__module__
            class_name = test_case.__class__.__name__
            method_name = test_case._testMethodName
            print(f"  {module_name}.{class_name}.{method_name}")

  return suite


def run_specific_tests(test_files: List[str],
                       verbose: bool = False) -> unittest.TestResult:
  """
    Run specific test files.
    
    Args:
        test_files: List of test file paths
        verbose: Whether to run in verbose mode
        
    Returns:
        TestResult object
    """
  loader = unittest.TestLoader()
  suite = unittest.TestSuite()

  for test_file in test_files:
    if verbose:
      print(f"Loading tests from: {test_file}")

    # Convert file path to module name
    module_name = Path(test_file).stem
    try:
      module = __import__(module_name)
      tests = loader.loadTestsFromModule(module)
      suite.addTest(tests)
    except ImportError as e:
      print(f"Error importing {module_name}: {e}")
      continue

  runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
  return runner.run(suite)


def run_all_tests(test_dir: str = ".",
                  pattern: str = "*test*.py",
                  verbose: bool = False) -> unittest.TestResult:
  """
    Run all discovered tests.
    
    Args:
        test_dir: Directory to search for test files
        pattern: Pattern to match test files
        verbose: Whether to run in verbose mode
        
    Returns:
        TestResult object
    """
  suite = discover_tests(test_dir, pattern, verbose)

  if suite.countTestCases() == 0:
    print(
        f"No tests found matching pattern '{pattern}' in directory '{test_dir}'"
    )
    return None

  runner = unittest.TextTestRunner(
      verbosity=2 if verbose else 1,
      stream=sys.stdout,
      buffer=True  # Capture stdout/stderr during tests
  )

  print(f"\nRunning {suite.countTestCases()} test cases...\n")
  return runner.run(suite)


def main():
  """Main entry point for the test runner."""
  parser = argparse.ArgumentParser(
      description="Run tests in the current directory",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py -v                 # Run all tests with verbose output
  python run_tests.py -f environment_test.py  # Run specific test file
  python run_tests.py -p "*env*test*.py" # Run tests matching pattern
  python run_tests.py --discover         # Just discover tests, don't run
        """)

  parser.add_argument("-v",
                      "--verbose",
                      action="store_true",
                      help="Verbose output")

  parser.add_argument("-f",
                      "--files",
                      nargs="*",
                      help="Specific test files to run")

  parser.add_argument("-p",
                      "--pattern",
                      default="*test*.py",
                      help="Pattern to match test files (default: *test*.py)")

  parser.add_argument(
      "-d",
      "--directory",
      default=".",
      help="Directory to search for tests (default: current directory)")

  parser.add_argument("--discover",
                      action="store_true",
                      help="Only discover tests, don't run them")

  parser.add_argument("--failfast",
                      action="store_true",
                      help="Stop on first failure")

  args = parser.parse_args()

  # Ensure we're in the right directory
  if not os.path.exists(args.directory):
    print(f"Error: Directory '{args.directory}' does not exist")
    sys.exit(1)

  # Change to test directory
  original_dir = os.getcwd()
  os.chdir(args.directory)

  try:
    if args.discover:
      # Just discover and list tests
      suite = discover_tests(".", args.pattern, verbose=True)
      print(f"\nTotal test cases discovered: {suite.countTestCases()}")
      return

    if args.files:
      # Run specific test files
      result = run_specific_tests(args.files, args.verbose)
    else:
      # Run all discovered tests
      result = run_all_tests(".", args.pattern, args.verbose)

    if result is None:
      sys.exit(1)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(
        f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")

    if result.failures:
      print(f"\nFAILURES:")
      for test, traceback in result.failures:
        print(
            f"  {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}"
        )

    if result.errors:
      print(f"\nERRORS:")
      for test, traceback in result.errors:
        print(
            f"  {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}"
        )

    # Exit with appropriate code
    if result.failures or result.errors:
      print(f"\nTESTS FAILED")
      sys.exit(1)
    else:
      print(f"\nALL TESTS PASSED")
      sys.exit(0)

  except KeyboardInterrupt:
    print("\nTest run interrupted by user")
    sys.exit(1)
  except Exception as e:
    print(f"Error running tests: {e}")
    sys.exit(1)
  finally:
    # Restore original directory
    os.chdir(original_dir)


if __name__ == "__main__":
  main()
