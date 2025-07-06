#!/bin/sh
#
# Pre-commit hook script to run unit tests
# This script runs all unit tests in the src/ directory before allowing a commit
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

# Check if we're in the right directory (should have src/ folder)
if [ ! -d "src" ]; then
    echo "Error: src/ directory not found. Are you in the correct repository?"
    exit 1
fi

echo "Running unit tests in src/ directory..."

# Run all unit tests in src/ directory
cd src
python -m unittest discover -s . -p "*test*.py" -v

# Check if tests passed
if [ $? -eq 0 ]; then
    echo "✅ All tests passed! Proceeding with commit."
    exit 0
else
    echo "❌ Tests failed! Commit aborted."
    echo "Please fix the failing tests before committing."
    exit 1
fi
