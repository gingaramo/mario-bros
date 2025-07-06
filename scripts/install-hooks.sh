#!/bin/bash
#
# Installation script for pre-commit hooks
# This script sets up pre-commit hooks to run unit tests before each commit
#

set -e

echo "Setting up pre-commit hooks for MarioBros project..."

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "Error: This is not a Git repository. Please run this script from the repository root."
    exit 1
fi

# Method 1: Install using pre-commit framework (recommended)
echo "Checking for pre-commit framework..."
if command -v pre-commit >/dev/null 2>&1; then
    echo "✅ pre-commit framework found. Installing hooks..."
    pre-commit install
    echo "✅ Pre-commit hooks installed successfully!"
    echo ""
    echo "You can test the hooks with: pre-commit run --all-files"
    echo ""
else
    echo "⚠️  pre-commit framework not found."
    echo "Installing pre-commit is recommended. You can install it with:"
    echo "  pip install pre-commit"
    echo "  or"
    echo "  conda install -c conda-forge pre-commit"
    echo ""
    echo "Falling back to manual Git hook installation..."
    
    # Method 2: Manual Git hook installation
    if [ -f "scripts/pre-commit-hook.sh" ]; then
        echo "Installing manual Git pre-commit hook..."
        cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
        chmod +x .git/hooks/pre-commit
        echo "✅ Manual pre-commit hook installed successfully!"
    else
        echo "❌ Error: scripts/pre-commit-hook.sh not found!"
        exit 1
    fi
fi

echo ""
echo "🎉 Pre-commit hooks are now active!"
echo ""
echo "What happens now:"
echo "- Before each commit, all unit tests in src/ will run automatically"
echo "- If any test fails, the commit will be blocked"
echo "- You'll need to fix failing tests before you can commit"
echo ""
echo "To test the setup, try making a commit or run:"
if command -v pre-commit >/dev/null 2>&1; then
    echo "  pre-commit run --all-files"
else
    echo "  git commit --dry-run"
fi
