# Testing Guide

This directory contains unit tests using Python's `unittest` framework.

## Quick Start

```bash
# Run all tests (from project root)
./test

# Run with verbose output
./test -v

# Stop on first failure
./test --failfast

# Run from src/ directory
cd src
python run_tests.py
```

## Test Files

- `environment_test.py` - Unit tests for environment wrappers
- `run_tests.py` - Simple unittest-based test runner

## Test Runner

The project uses a single, simple test runner based on Python's built-in `unittest` framework:

```bash
# Run all tests
python run_tests.py

# Run with verbose output
python run_tests.py -v

# Stop on first failure
python run_tests.py --failfast
```

The test runner automatically discovers all files matching the pattern `*test*.py` in the current directory.

## Test Structure

### Environment Wrappers Tests

The `environment_test.py` file contains comprehensive tests for:

- **PreprocessFrameEnv**: Tests frame preprocessing (resize, grayscale, normalize)
- **RepeatActionEnv**: Tests action repetition functionality
- **ReturnActionEnv**: Tests action history tracking
- **HistoryEnv**: Tests frame history management
- **Integration Tests**: Tests wrapper combinations

### Test Coverage

Each wrapper class includes tests for:
- ✅ Initialization with valid/invalid configurations
- ✅ Core functionality (step, reset methods)
- ✅ Error handling and edge cases
- ✅ Type safety and input validation
- ✅ Integration with other wrappers

## Adding New Tests

Create a new test file following the `*test*.py` naming pattern:

```python
# my_module_test.py
import unittest
from my_module import MyClass

class TestMyClass(unittest.TestCase):
    def test_something(self):
        obj = MyClass()
        self.assertEqual(obj.method(), expected_result)

if __name__ == '__main__':
    unittest.main()
```

The test runner will automatically discover and run your new test file.

## Pre-commit Hooks

Tests are automatically run before each commit via pre-commit hooks. The hooks use:

```bash
python -m unittest discover -s src -p "*test*.py" -v
```

This ensures all tests pass before code is committed.

## Troubleshooting

### Import Errors
Make sure you're running tests from the `src/` directory:
```bash
cd src
python run_tests.py
```

### Module Not Found
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt  # if requirements file exists
```

### Test Discovery Issues
Check test file naming - they should match `*test*.py` pattern:
```bash
python run_tests.py  # Will show discovered test count
```
