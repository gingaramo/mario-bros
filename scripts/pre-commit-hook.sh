#!/bin/sh
#
# Pre-commit hook script to run unit tests
# This script runs all unit tests in the tests/ directory before allowing a commit
#
# To install this hook:
# 1. Copy this file to .git/hooks/pre-commit
# 2. Make it executable: chmod +x .git/hooks/pre-commit
#
# Alternatively, use pre-commit framework (recommended)

set -e

echo "Running pre-commit checks..."

# Change to the repository root directory
cd "$(git rev-parse --show-toplevel)"

# Check if we're in the right directory (should have tests/ folder)
if [ ! -d "tests" ]; then
    echo "Error: tests/ directory not found. Are you in the correct repository?"
    exit 1
fi

echo "Running unit tests..."

# Run the test runner
python run_tests.py

# Check if tests passed
if [ $? -eq 0 ]; then
    echo "✅ All tests passed! Proceeding with commit."
    exit 0
else
    echo "❌ Tests failed! Commit aborted."
    echo "Please fix the failing tests before committing."
    exit 1
fi
