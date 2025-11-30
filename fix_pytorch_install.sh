#!/bin/bash
# Fix PyTorch installation issues on Jetson Nano

echo "=========================================="
echo "PyTorch Installation Fix"
echo "=========================================="
echo ""

# Check current Python versions
echo "Checking Python installations..."
echo ""

if command -v python2.7 &> /dev/null; then
    echo "Python 2.7: $(python2.7 --version 2>&1)"
fi

if command -v python3.6 &> /dev/null; then
    echo "Python 3.6: $(python3.6 --version 2>&1)"
fi

if command -v python3.8 &> /dev/null; then
    echo "Python 3.8: $(python3.8 --version 2>&1)"
fi

echo ""
echo "Current PYTHONPATH: $PYTHONPATH"
echo ""

# The issue: PYTHONPATH points to Python 3.8 packages
# but python3.6 and python2.7 are trying to use them

echo "=========================================="
echo "Problem Identified"
echo "=========================================="
echo ""
echo "1. PyTorch 1.10.0 wheel is for Python 3.8 (cp38)"
echo "2. Your PYTHONPATH points to Python 3.8 packages"
echo "3. python3.6 and python2.7 are trying to use Python 3.8 packages"
echo "4. This causes syntax errors and import failures"
echo ""

echo "=========================================="
echo "Solution"
echo "=========================================="
echo ""
echo "Option 1: ONLY use python3.8 for everything (recommended)"
echo "  - Remove Python 2.7 from PATH or stop using it"
echo "  - Always use 'python3.8' command explicitly"
echo "  - Update all scripts to use python3.8"
echo ""
echo "Option 2: Install PyTorch separately for each Python version"
echo "  - Not recommended - wastes space and causes confusion"
echo ""

read -p "Do you want to fix this by configuring only Python 3.8? (y/n): " response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo ""
    echo "Fixing environment for Python 3.8 only..."

    # Clean up PYTHONPATH to only apply when using Python 3.8
    # Remove the global PYTHONPATH from bashrc if it exists
    if grep -q "PYTHONPATH=/mnt/microsd/python-packages" ~/.bashrc; then
        echo "Removing global PYTHONPATH from ~/.bashrc..."
        sed -i '/PYTHONPATH=\/mnt\/microsd\/python-packages/d' ~/.bashrc
    fi

    # Create a wrapper script for python3.8
    cat > ~/use_python38.sh << 'EOF'
#!/bin/bash
# Source this before using Python 3.8
export PYTHONUSERBASE="/mnt/microsd/python-packages"
export PYTHONPATH="/mnt/microsd/python-packages/lib/python3.8/site-packages:$PYTHONPATH"
export PYO3_PYTHON=$(which python3.8)
export PATH="$PYTHONUSERBASE/bin:$PATH"
echo "✓ Python 3.8 environment activated"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  PYTHONUSERBASE: $PYTHONUSERBASE"
EOF

    chmod +x ~/use_python38.sh

    echo ""
    echo "✓ Created ~/use_python38.sh"
    echo ""
    echo "=========================================="
    echo "How to Use"
    echo "=========================================="
    echo ""
    echo "Before using Python 3.8 or PyTorch, run:"
    echo "  source ~/use_python38.sh"
    echo ""
    echo "Then use:"
    echo "  python3.8 -c 'import torch; print(torch.__version__)'"
    echo "  python3.8 train_ai.py"
    echo ""
    echo "Do NOT use 'python' or 'python3' or 'python3.6' - only 'python3.8'"
    echo ""

    # Test it
    echo "Testing PyTorch with Python 3.8..."
    source ~/use_python38.sh
    cd ~
    python3.8 -c "import torch; print('✓ PyTorch', torch.__version__, 'works!')" 2>/dev/null

    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "✓ SUCCESS!"
        echo "=========================================="
        echo ""
        echo "PyTorch is working with Python 3.8"
        echo ""
        echo "Add this to your workflow:"
        echo "  1. Open terminal"
        echo "  2. Run: source ~/use_python38.sh"
        echo "  3. Use python3.8 for all commands"
        echo ""
    else
        echo ""
        echo "⚠ PyTorch still has issues. Let's check the installation..."
        echo ""
        echo "Checking PyTorch files..."
        ls -la /mnt/microsd/python-packages/lib/python3.8/site-packages/torch/ | head -20
        echo ""
        echo "Try reinstalling PyTorch:"
        echo "  pip3.8 install --force-reinstall torch-1.10.0-cp38-cp38-linux_aarch64.whl"
    fi
else
    echo ""
    echo "No changes made."
    echo ""
    echo "Current issues:"
    echo "  - python (2.7) tries to use Python 3.8 packages → SyntaxError"
    echo "  - python3.6 tries to use Python 3.8 packages → SyntaxError"
    echo "  - python3.8 might work but environment needs to be set correctly"
    echo ""
    echo "To fix manually:"
    echo "  1. Only use 'python3.8' command"
    echo "  2. Set PYTHONPATH only when using Python 3.8"
    echo "  3. Don't set PYTHONPATH globally in ~/.bashrc"
fi

echo ""

