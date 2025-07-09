# Test Pre-commit Hook Functionality

This script helps verify that the pre-commit hook correctly blocks commits when tests fail.

## Testing the Setup

The pre-commit hook has been successfully installed and tested. Here's what happens:

✅ **When all tests pass**: The commit proceeds normally
❌ **When tests fail**: The commit is blocked and you'll see an error message

## Verification

The hook was tested by running it manually:
```bash
.git/hooks/pre-commit
```

Result: All 58 tests passed successfully!

## Files Created

1. **`.pre-commit-config.yaml`** - Configuration for the pre-commit framework (for future use)
2. **`scripts/pre-commit-hook.sh`** - The actual hook script that runs tests
3. **`scripts/install-hooks.sh`** - Installation script that sets up the hooks
4. **`scripts/README.md`** - Documentation for the pre-commit setup
5. **`.git/hooks/pre-commit`** - The active Git hook (copy of the script)

## How It Works

Before every commit, Git will automatically:

1. Run all unit tests in the `tests/` directory
2. If any test fails, the commit is blocked
3. If all tests pass, the commit proceeds normally

## Bypassing (Emergency Only)

If you absolutely need to commit without running tests:
```bash
git commit --no-verify -m "Emergency commit"
```

## Upgrading to pre-commit Framework

For enhanced functionality (code formatting, linting, etc.), install the pre-commit framework:

```bash
pip install pre-commit
pre-commit install
```

This will automatically use the `.pre-commit-config.yaml` configuration file.
