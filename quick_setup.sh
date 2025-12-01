#!/bin/bash
# Quick commands to run NOW on Jetson Nano

echo "=========================================="
echo "Quick Setup - Python 3.6 Only"
echo "=========================================="
echo ""

cd /mnt/microsd/projects/jetson.cartpole

echo "Current PYTHONPATH:"
echo "$PYTHONPATH"
echo ""

# Clean set PYTHONPATH
echo "Setting clean PYTHONPATH (no duplicates, no python3.8)..."
unset PYTHONPATH
export PYTHONPATH="/mnt/microsd/python-packages/lib/python3.6/site-packages"
export PYTHONUSERBASE="/mnt/microsd/python-packages"
export PYO3_PYTHON=/usr/bin/python3.6
export CARGO_HOME="/mnt/microsd/cargo"
export RUSTUP_HOME="/mnt/microsd/rustup"
export PATH="$PYTHONUSERBASE/bin:$CARGO_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$PYTHONUSERBASE/lib/python3.6/site-packages/torch/lib:$LD_LIBRARY_PATH"

echo ""
echo "✓ Environment set!"
echo ""
echo "PYTHONPATH: $PYTHONPATH"
echo "PYO3_PYTHON: $PYO3_PYTHON"
echo ""

# Verify
echo "Verifying Python 3.6..."
python3.6 --version

echo ""
echo "Testing imports..."
python3.6 -c "import numpy; print('✓ NumPy:', numpy.__version__)" 2>&1

echo ""
echo "Testing PyTorch (if installed)..."
python3.6 -c "import torch; print('✓ PyTorch:', torch.__version__); print('  CUDA available:', torch.cuda.is_available())" 2>&1

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "1. If PyTorch import failed, install it:"
echo "   cd /mnt/microsd"
echo "   python3.6 -m pip install --user torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
echo ""
echo "2. Rebuild CartPole:"
echo "   cd /mnt/microsd/projects/jetson.cartpole"
echo "   cargo clean"
echo "   cargo build --release --features python"
echo ""
echo "3. Test CartPole:"
echo "   python3.6 -c 'import cartpole; print(\"CartPole OK\")'"
echo ""
echo "4. Train:"
echo "   python3.6 train_ai.py"
echo ""

