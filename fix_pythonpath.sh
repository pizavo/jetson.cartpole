#!/bin/bash
# Fix PYTHONPATH that has python3.8 mixed in
# Run this on Jetson Nano if your PYTHONPATH shows python3.8 paths

echo "=========================================="
echo "Fix PYTHONPATH - Remove Python 3.8"
echo "=========================================="
echo ""

echo "Current PYTHONPATH:"
echo "$PYTHONPATH"
echo ""

if echo "$PYTHONPATH" | grep -q "python3.8"; then
    echo "⚠ Found python3.8 in PYTHONPATH!"
    echo ""
    echo "This will cause import conflicts with Python 3.6."
    echo ""

    read -p "Fix it now? (y/n): " response

    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo ""
        echo "Step 1: Cleaning ~/.bashrc..."

        if grep -q "python3.8" ~/.bashrc; then
            echo "Found python3.8 references in ~/.bashrc"
            cp ~/.bashrc ~/.bashrc.backup
            sed -i '/python3\.8/d' ~/.bashrc
            echo "✓ Removed python3.8 lines (backup at ~/.bashrc.backup)"
        else
            echo "✓ No python3.8 in ~/.bashrc"
        fi

        echo ""
        echo "Step 2: Setting correct PYTHONPATH..."
        unset PYTHONPATH
        export PYTHONPATH="/mnt/microsd/python-packages/lib/python3.6/site-packages"

        echo ""
        echo "Step 3: Sourcing correct environment..."
        if [ -f "setup_jetson_env.sh" ]; then
            source setup_jetson_env.sh
        fi

        echo ""
        echo "✓ Fixed!"
        echo ""
        echo "New PYTHONPATH:"
        echo "$PYTHONPATH"
        echo ""

        if echo "$PYTHONPATH" | grep -q "python3.8"; then
            echo "⚠ Still has python3.8! Check ~/.bashrc manually:"
            echo "  nano ~/.bashrc"
            echo "  # Remove all lines with python3.8"
            echo "  # Save and exit"
            echo "  source ~/.bashrc"
        else
            echo "✓ Clean! Only python3.6 in PYTHONPATH"
        fi

        echo ""
        echo "To make permanent:"
        echo "  1. Exit and restart terminal"
        echo "  2. Run: source setup_jetson_env.sh"
        echo "  3. Verify: echo \$PYTHONPATH"
        echo ""
    fi
else
    echo "✓ PYTHONPATH looks clean - no python3.8 found"
    echo ""
    echo "Current path:"
    echo "$PYTHONPATH"
fi

echo ""

