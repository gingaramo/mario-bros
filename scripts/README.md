# Pre-commit Hooks Setup

This directory contains scripts and configuration for setting up pre-commit hooks that automatically run unit tests before each commit.

## Quick Setup

Run the installation script:
```bash
./scripts/install-hooks.sh
```

This will automatically detect whether you have the `pre-commit` framework installed and configure the appropriate hooks.

## What Gets Checked

Before each commit, the following checks will run:
- **All unit tests** in the `tests/` directory
- Code formatting checks (if using pre-commit framework)
- Python syntax validation
- Import sorting validation

## Manual Installation

### Option 1: Using pre-commit framework (Recommended)

1. Install pre-commit:
   ```bash
   pip install pre-commit
   # or
   conda install -c conda-forge pre-commit
   ```

2. Install the hooks:
   ```bash
   pre-commit install
   ```

3. Test the setup:
   ```bash
   pre-commit run --all-files
   ```

### Option 2: Manual Git Hook

If you prefer not to use the pre-commit framework:

1. Copy the hook script:
   ```bash
   cp scripts/pre-commit-hook.sh .git/hooks/pre-commit
   chmod +x .git/hooks/pre-commit
   ```

## Testing the Setup

Try making a commit to see the hooks in action:
```bash
git add .
git commit -m "Test commit"
```

You should see the tests running before the commit is allowed to proceed.

## Bypassing Hooks (Emergency Use Only)

If you need to commit without running the hooks (not recommended):
```bash
git commit --no-verify -m "Emergency commit"
```

## Files

- `.pre-commit-config.yaml` - Configuration for the pre-commit framework
- `scripts/pre-commit-hook.sh` - Manual Git hook script
- `scripts/install-hooks.sh` - Installation script that chooses the best method
- `scripts/README.md` - This documentation file
