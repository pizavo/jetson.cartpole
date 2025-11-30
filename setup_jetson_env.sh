#!/bin/bash
# Environment setup for Jetson Nano with microSD installation
# IMPORTANT: This sets up Python 3.8 environment ONLY
# DO NOT add this to ~/.bashrc globally - source it when needed

# Custom installation paths on microSD
export CARGO_HOME="/mnt/microsd/cargo"
export RUSTUP_HOME="/mnt/microsd/rustup"
export PYTHONUSERBASE="/mnt/microsd/python-packages"

# Force PyO3 to use Python 3.8 (not 3.6)
export PYO3_PYTHON=$(which python3.8)

# Add to PATH
export PATH="$PYTHONUSERBASE/bin:$CARGO_HOME/bin:$PATH"

# CRITICAL: This PYTHONPATH is for Python 3.8 ONLY
# Do not use 'python', 'python3', or 'python3.6' after sourcing this
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.8/site-packages:$PYTHONPATH"

echo "=========================================="
echo "Python 3.8 Environment Activated"
echo "=========================================="
echo "  CARGO_HOME=$CARGO_HOME"
echo "  RUSTUP_HOME=$RUSTUP_HOME"
echo "  PYTHONUSERBASE=$PYTHONUSERBASE"
echo "  PYTHONPATH=$PYTHONPATH"
echo ""
echo "⚠ WARNING: Use ONLY 'python3.8' command"
echo "  ✅ python3.8 train_ai.py"
echo "  ❌ python train_ai.py"
echo "  ❌ python3 train_ai.py"
echo "  ❌ python3.6 train_ai.py"
echo ""

