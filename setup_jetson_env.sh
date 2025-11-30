#!/bin/bash
# Environment setup for Jetson Nano with microSD installation
# Add this to your ~/.bashrc to make it permanent

# Custom installation paths on microSD
export CARGO_HOME="/mnt/microsd/cargo"
export RUSTUP_HOME="/mnt/microsd/rustup"
export PYTHONUSERBASE="/mnt/microsd/python-packages"

# Force PyO3 to use Python 3.8 (not 3.6)
export PYO3_PYTHON=$(which python3.8 || which python3)

# Add to PATH
export PATH="$PYTHONUSERBASE/bin:$CARGO_HOME/bin:$PATH"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.8/site-packages:$PYTHONPATH"

echo "Environment configured for microSD installation:"
echo "  CARGO_HOME=$CARGO_HOME"
echo "  RUSTUP_HOME=$RUSTUP_HOME"
echo "  PYTHONUSERBASE=$PYTHONUSERBASE"

