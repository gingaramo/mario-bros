# Tests Directory

This directory contains all unit tests for the Mario Bros project.

## Running Tests

From the project root:
```bash
python run_tests.py
```

From this directory:
```bash
python run_tests.py
```

## Test Files

- `environment_test.py` - Tests for environment wrappers and preprocessing
- `run_tests.py` - Main test runner script
- `test_runner.py` - Alternative test runner

## Adding New Tests

1. Create new test files with the naming pattern `*test*.py`
2. Use Python's `unittest` framework
3. Import source modules by adding the path setup at the top:

```python
import sys
import os

# Add src directory to path so tests can import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now you can import from src
from your_module import YourClass
```
