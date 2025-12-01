#!/bin/bash
EOF
========================================

# Should show ONLY python3.6, no python3.8!
echo $PYTHONPATH
# Verify

source setup_jetson_env.sh
# Re-source the correct environment

source ~/.bashrc
# Remove any lines with python3.8 or /python3.8/
nano ~/.bashrc
# Clean your ~/.bashrc if it has python3.8 paths

export PYTHONPATH="/mnt/microsd/python-packages/lib/python3.6/site-packages"
unset PYTHONPATH
# If PYTHONPATH is wrong (shows python3.8)
-------------------
Quick Fix Commands:

PATH=$PYTHONUSERBASE/bin:$CARGO_HOME/bin:$PATH
LD_LIBRARY_PATH=/mnt/microsd/python-packages/lib/python3.6/site-packages/torch/lib
PYTHONPATH=/mnt/microsd/python-packages/lib/python3.6/site-packages
PYO3_PYTHON=/usr/bin/python3.6
PYTHONUSERBASE=/mnt/microsd/python-packages
RUSTUP_HOME=/mnt/microsd/rustup
CARGO_HOME=/mnt/microsd/cargo
----------------------
Environment Variables:

✅ Source setup_jetson_env.sh which sets python3.6 paths
❌ Setting PYTHONPATH globally in ~/.bashrc with python3.8 paths

✅ python3.6 train_ai.py       # Correct!
❌ python3.8 train_ai.py       # Wrong! We use Python 3.6
❌ python3 train_ai.py         # May use wrong Python
❌ python train_ai.py          # Wrong! Uses Python 2.7
----------------
Common Mistakes:

echo $PYO3_PYTHON # Should be /usr/bin/python3.6
which python3.6   # Should be /usr/bin/python3.6
# Check what Python is being used

# If it shows python3.8, fix your ~/.bashrc
# Should ONLY show: /mnt/microsd/python-packages/lib/python3.6/site-packages
echo $PYTHONPATH
# If you get import errors or version conflicts

python3.6 -c "import torch"
source setup_jetson_env.sh
cd ~
# If PyTorch doesn't work
----------------
Troubleshooting:

./build_all.sh
cargo clean
# Clean rebuild

./build_all.sh
# Build everything

./setup_python.sh
# Setup Python only

./build_rust.sh
# Build Rust only
--------------
Build/Rebuild:

python3.6 train_dqn.py
# Train DQN with GPU (if PyTorch + CUDA working)

python3.6 train_ai.py
# Train simple agent

python3.6 test_setup.py
# Performance test

./target/release/cartpole
# Visual game (if display available)
-------------
Run CartPole:

python3.6 -c "import cartpole; print('CartPole: OK')"
python3.6 -c "import torch; print('PyTorch:', torch.__version__)"
python3.6 -c "import numpy; print('NumPy:', numpy.__version__)"
-------------
Test Imports:

export LD_LIBRARY_PATH="$PYTHONUSERBASE/lib/python3.6/site-packages/torch/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/mnt/microsd/python-packages/lib/python3.6/site-packages"
export PATH="$PYTHONUSERBASE/bin:$CARGO_HOME/bin:$PATH"
export PYO3_PYTHON=/usr/bin/python3.6
export PYTHONUSERBASE="/mnt/microsd/python-packages"
export RUSTUP_HOME="/mnt/microsd/rustup"
export CARGO_HOME="/mnt/microsd/cargo"
OR manually set environment:

source setup_jetson_env.sh
----------------------------------
Setup (once per terminal session):

CRITICAL: Always use 'python3.6' explicitly!

========================================
CartPole - Quick Command Reference
========================================
cat << 'EOF'

# All commands use Python 3.6 (stock JetPack 4.6)
# Quick command reference for CartPole on Jetson Nano

