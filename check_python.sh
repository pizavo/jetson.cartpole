#!/bin/bash
# Quick verification script for Jetson Nano setup
# This helps identify Python version issues

echo "================================================"
echo "Jetson Nano - Python Version Check"
echo "================================================"
echo ""

echo "Checking Python installations..."
echo ""

# Check Python 2
if command -v python &> /dev/null; then
    PY2_VERSION=$(python --version 2>&1)
    echo "❌ 'python' command found: $PY2_VERSION"
    echo "   → This is TOO OLD for AI training"
    echo "   → DO NOT USE 'python' or 'pip'"
else
    echo "✓ 'python' command not found (good!)"
fi

echo ""

# Check Python 3
if command -v python3 &> /dev/null; then
    PY3_VERSION=$(python3 --version 2>&1)
    echo "✅ 'python3' command found: $PY3_VERSION"
    echo "   → USE THIS for all commands"
    echo "   → Use 'python3 -m pip' for package management"
else
    echo "❌ 'python3' command NOT FOUND"
    echo "   → Install with: sudo apt-get install python3 python3-pip python3-dev"
    exit 1
fi

echo ""

# Check pip3
if command -v pip3 &> /dev/null; then
    PIP3_VERSION=$(pip3 --version 2>&1)
    echo "✅ 'pip3' found: $PIP3_VERSION"
    echo "   → Prefer using: python3 -m pip (avoids wrapper warnings)"
else
    echo "❌ 'pip3' not found"
    echo "   → Install with: sudo apt-get install python3-pip"
fi

echo ""
echo "================================================"
echo "Summary"
echo "================================================"
echo ""
echo "CORRECT commands to use:"
echo "  python3 --version"
echo "  python3 train_ai.py"
echo "  python3 train_dqn.py"
echo "  python3 -m pip install numpy --user"
echo "  python3 -m pip list"
echo ""
echo "WRONG commands (DO NOT USE):"
echo "  python --version    ❌"
echo "  python train_ai.py  ❌"
echo "  pip install numpy   ❌"
echo "  pip3 install numpy  ⚠️  (works but may show warnings)"
echo ""

# Check if in cartpole directory
if [ -f "Cargo.toml" ]; then
    echo "✓ You are in the cartpole directory"
    echo ""
    echo "Next steps:"
    echo "  1. Build: ./build_jetson.sh"
    echo "  2. Test: python3 test_setup.py"
    echo "  3. Train: python3 train_ai.py"
else
    echo "⚠ Warning: Cargo.toml not found"
    echo "  Make sure you're in the cartpole directory"
    echo "  Run: cd ~/cartpole"
fi

echo ""

