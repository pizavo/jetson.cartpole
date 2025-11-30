#!/bin/bash
# Quick command reference for CartPole on Jetson Nano

cat << 'EOF'
========================================
CartPole - Quick Command Reference
========================================

CRITICAL: Always use 'python3.8' explicitly!

Setup (once per terminal session):
----------------------------------
source ~/use_python38.sh

OR manually:
export PYTHONPATH="/mnt/microsd/python-packages/lib/python3.8/site-packages:$PYTHONPATH"
export PYTHONUSERBASE="/mnt/microsd/python-packages"

Test Imports:
-------------
python3.8 -c "import numpy; print('NumPy:', numpy.__version__)"
python3.8 -c "import torch; print('PyTorch:', torch.__version__)"
python3.8 -c "import cartpole; print('CartPole: OK')"

Run CartPole:
-------------
# Visual game (if display available)
./target/release/cartpole

# Performance test
python3.8 test_setup.py

# Train simple agent
python3.8 train_ai.py

# Train DQN with GPU (if PyTorch + CUDA working)
python3.8 train_dqn.py

Build/Rebuild:
--------------
# Build Rust only
./build_rust.sh

# Setup Python only
./setup_python.sh

# Build everything
./build_all.sh

# Clean rebuild
cargo clean
./build_all.sh

Troubleshooting:
----------------
# If PyTorch doesn't work
cd ~
python3.8 -c "import torch"

# If you get syntax errors
echo $PYTHONPATH  # Should only be set when using Python 3.8
which python      # Should NOT use this
which python3     # Should NOT use this
which python3.8   # Should use THIS

# Fix PyTorch issues
./fix_pytorch_install.sh

Common Mistakes:
----------------
❌ python train_ai.py          # Wrong! Uses Python 2.7
❌ python3 train_ai.py         # Wrong! Uses Python 3.6
❌ python3.6 train_ai.py       # Wrong! Uses Python 3.6
✅ python3.8 train_ai.py       # Correct!

❌ Setting PYTHONPATH globally in ~/.bashrc
✅ Source ~/use_python38.sh before using Python 3.8

Environment Variables:
----------------------
CARGO_HOME=/mnt/microsd/cargo
RUSTUP_HOME=/mnt/microsd/rustup
PYTHONUSERBASE=/mnt/microsd/python-packages
PYO3_PYTHON=/usr/bin/python3.8
PYTHONPATH=/mnt/microsd/python-packages/lib/python3.8/site-packages
PATH=$PYTHONUSERBASE/bin:$CARGO_HOME/bin:$PATH

Quick Fix Commands:
-------------------
# If things are broken
cd /mnt/microsd/projects/jetson.cartpole
./fix_pytorch_install.sh
source ~/use_python38.sh
cd ~
python3.8 -c "import torch; print('OK')"

========================================
EOF

