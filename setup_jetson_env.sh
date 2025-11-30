#!/bin/bash
# Environment setup for Jetson Nano with microSD installation
# Add this to your ~/.bashrc to make it permanent

# Custom installation paths on microSD
export CARGO_HOME="/mnt/microsd/cargo"
export RUSTUP_HOME="/mnt/microsd/rustup"
export PYTHONUSERBASE="/mnt/microsd/python-packages"

# Add to PATH
export PATH="$CARGO_HOME/bin:$PATH"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.6/site-packages:$PYTHONPATH"

echo "Environment configured for microSD installation:"
echo "  CARGO_HOME=$CARGO_HOME"
echo "  RUSTUP_HOME=$RUSTUP_HOME"
echo "  PYTHONUSERBASE=$PYTHONUSERBASE"

