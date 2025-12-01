#!/bin/bash
# Complete PyTorch installation fix for Jetson Nano

echo "=========================================="
echo "PyTorch Complete Fix"
echo "=========================================="
echo ""

# Step 1: Clean up broken installation
echo "Step 1: Cleaning up broken PyTorch installation..."
rm -rf /mnt/microsd/python-packages/lib/python3.8/site-packages/torch*
rm -rf /mnt/microsd/python-packages/lib/python3.8/site-packages/typing_extensions*

echo "✓ Removed old PyTorch files"
echo ""

# Step 2: Check wheel file
echo "Step 2: Verifying PyTorch wheel file..."
WHEEL_FILE="torch-1.10.0-cp38-cp38-linux_aarch64.whl"

if [ -f "/mnt/microsd/$WHEEL_FILE" ]; then
    echo "✓ Found wheel at /mnt/microsd/$WHEEL_FILE"
elif [ -f "$WHEEL_FILE" ]; then
    echo "✓ Found wheel in current directory"
    WHEEL_FILE="./$WHEEL_FILE"
else
    echo "✗ PyTorch wheel not found!"
    echo "Download it first:"
    echo "  cd /mnt/microsd"
    echo "  wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp38-cp38-linux_aarch64.whl"
    exit 1
fi

echo ""

# Step 3: Clean environment and reinstall
echo "Step 3: Installing PyTorch with clean environment..."

# Clear all Python paths
unset PYTHONPATH
export PYTHONUSERBASE="/mnt/microsd/python-packages"

# Install
cd /mnt/microsd
python3.8 -m pip install --user --no-deps torch-1.10.0-cp38-cp38-linux_aarch64.whl

if [ $? -ne 0 ]; then
    echo "✗ Installation failed!"
    exit 1
fi

echo "✓ PyTorch installed"
echo ""

# Step 4: Install dependencies
echo "Step 4: Installing PyTorch dependencies..."
python3.8 -m pip install --user typing-extensions

echo ""

# Step 5: Create clean environment script
echo "Step 5: Creating clean environment script..."
cat > ~/use_python38.sh << 'EOF'
#!/bin/bash
# Clean Python 3.8 environment

export CARGO_HOME="/mnt/microsd/cargo"
export RUSTUP_HOME="/mnt/microsd/rustup"
export PYTHONUSERBASE="/mnt/microsd/python-packages"
export PYO3_PYTHON=/usr/bin/python3.8

# Clear and set PYTHONPATH (no duplicates!)
unset PYTHONPATH
export PYTHONPATH="/mnt/microsd/python-packages/lib/python3.8/site-packages"

# Add library paths for PyTorch
export LD_LIBRARY_PATH="/mnt/microsd/python-packages/lib/python3.8/site-packages/torch/lib:$LD_LIBRARY_PATH"

export PATH="$PYTHONUSERBASE/bin:$CARGO_HOME/bin:$PATH"

echo "✓ Python 3.8 environment activated"
EOF

chmod +x ~/use_python38.sh

echo "✓ Created ~/use_python38.sh"
echo ""

# Step 6: Remove problematic PYTHONPATH from bashrc
echo "Step 6: Cleaning ~/.bashrc..."
if grep -q "PYTHONPATH.*python-packages" ~/.bashrc; then
    echo "Found PYTHONPATH in ~/.bashrc, creating backup..."
    cp ~/.bashrc ~/.bashrc.backup
    sed -i '/PYTHONPATH.*python-packages/d' ~/.bashrc
    echo "✓ Removed PYTHONPATH from ~/.bashrc (backup at ~/.bashrc.backup)"
else
    echo "✓ ~/.bashrc is clean"
fi

echo ""

# Step 7: Test installation
echo "Step 7: Testing PyTorch installation..."
echo ""

# Source clean environment
source ~/use_python38.sh

# Test from home directory
cd ~

echo "Testing NumPy..."
python3.8 -c "import numpy; print('  NumPy:', numpy.__version__)" 2>&1

echo ""
echo "Testing PyTorch..."
python3.8 -c "import torch; print('  PyTorch:', torch.__version__); print('  CUDA available:', torch.cuda.is_available())" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ SUCCESS! PyTorch is working!"
    echo "=========================================="
    echo ""
    echo "Usage:"
    echo "  1. Start new terminal"
    echo "  2. Run: source ~/use_python38.sh"
    echo "  3. Use: python3.8 train_ai.py"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ PyTorch test failed"
    echo "=========================================="
    echo ""
    echo "Debug info:"
    echo "PYTHONPATH: $PYTHONPATH"
    echo ""
    echo "PyTorch files:"
    ls -la /mnt/microsd/python-packages/lib/python3.8/site-packages/torch/ | head -20
    echo ""
    echo "Try running:"
    echo "  cd ~"
    echo "  source ~/use_python38.sh"
    echo "  python3.8 -c 'import torch'"
fi

echo ""

