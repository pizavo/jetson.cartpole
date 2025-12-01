#!/bin/bash
# Environment setup for Jetson Nano with microSD installation
# Configured for Python 3.6 (stock JetPack 4.6)

# Custom installation paths on microSD
export CARGO_HOME="/mnt/microsd/cargo"
export RUSTUP_HOME="/mnt/microsd/rustup"
export PYTHONUSERBASE="/mnt/microsd/python-packages"

# Use Python 3.6 (stock Jetson Nano Python)
export PYO3_PYTHON=$(which python3.6 || which python3)

# Add to PATH
export PATH="$PYTHONUSERBASE/bin:$CARGO_HOME/bin:$PATH"

# Set PYTHONPATH for Python 3.6
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.6/site-packages:$PYTHONPATH"

# Add PyTorch library path
export LD_LIBRARY_PATH="$PYTHONUSERBASE/lib/python3.6/site-packages/torch/lib:$LD_LIBRARY_PATH"

echo "=========================================="
echo "Python 3.6 Environment Activated"
echo "=========================================="
echo "  CARGO_HOME=$CARGO_HOME"
echo "  RUSTUP_HOME=$RUSTUP_HOME"
echo "  PYTHONUSERBASE=$PYTHONUSERBASE"
echo "  PYTHONPATH=$PYTHONPATH"
echo ""
echo "✓ Using Python 3.6 (stock JetPack 4.6)"
echo "  python3.6 train_ai.py"
echo "  python3.6 train_dqn.py"
echo ""

