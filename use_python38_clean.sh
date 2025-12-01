#!/bin/bash
echo ""
echo "⚠ Use ONLY 'python3.8' command!"
echo ""
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PYTHONUSERBASE: $PYTHONUSERBASE"
echo "PYTHONPATH: $PYTHONPATH"
echo "=========================================="
echo "✓ Clean Python 3.8 Environment"
echo "=========================================="

export LD_LIBRARY_PATH="/mnt/microsd/python-packages/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"
# Also set LD_LIBRARY_PATH for PyTorch C extensions

export PATH="$PYTHONUSERBASE/bin:$CARGO_HOME/bin:$PATH"
# Add to PATH

export PYTHONPATH="/mnt/microsd/python-packages/lib/python3.8/site-packages"
# Set PYTHONPATH for Python 3.8 ONLY (no duplicates, no Python 3.6)

unset PYTHONPATH
# CRITICAL: Clear any existing PYTHONPATH first to avoid conflicts

export PYO3_PYTHON=/usr/bin/python3.8
# Force PyO3 to use Python 3.8

export PYTHONUSERBASE="/mnt/microsd/python-packages"
export RUSTUP_HOME="/mnt/microsd/rustup"
export CARGO_HOME="/mnt/microsd/cargo"
# Custom installation paths on microSD

# Clean Python 3.8 environment setup (replaces ~/use_python38.sh)

