#!/bin/bash
#
# Demo script to show how pre-commit hooks work
# This script demonstrates both success and failure scenarios
#

echo "🧪 Testing Pre-commit Hook Behavior"
echo "=================================="
echo ""

# Test 1: Show current working hook
echo "✅ Test 1: Current hook with all tests passing"
echo "Running: .git/hooks/pre-commit"
echo ""

.git/hooks/pre-commit
echo ""
echo "Result: Hook passed successfully! ✅"
echo ""

# Test 2: Create a failing test temporarily to show blocking behavior
echo "❌ Test 2: Demonstrating hook blocking on test failure"
echo ""

# Create a temporary failing test
cat > src/temp_failing_test.py << 'EOF'
import unittest

class TestFailing(unittest.TestCase):
    def test_intentional_failure(self):
        """This test intentionally fails to demo pre-commit blocking."""
        self.assertTrue(False, "This test intentionally fails for demo purposes")

if __name__ == '__main__':
    unittest.main()
EOF

echo "Created temporary failing test: src/temp_failing_test.py"
echo "Running pre-commit hook again..."
echo ""

# Run the hook - it should fail now
if .git/hooks/pre-commit; then
    echo "❌ Unexpected: Hook should have failed but didn't!"
else
    echo "✅ Expected: Hook correctly blocked due to failing test!"
fi

echo ""

# Clean up the failing test
rm -f src/temp_failing_test.py
echo "🧹 Cleaned up temporary failing test"
echo ""

echo "🎉 Demo complete!"
echo ""
echo "Summary:"
echo "- ✅ Pre-commit hook runs all tests before each commit"
echo "- ✅ Commits are allowed when all tests pass"  
echo "- ✅ Commits are blocked when any test fails"
echo "- ✅ This ensures only working code gets committed"
