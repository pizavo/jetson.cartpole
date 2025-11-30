#!/bin/bash
# Python setup script for Jetson Nano AI training

echo "=========================================="
echo "CartPole - Python Setup"
echo "=========================================="
echo ""

# Force Python 3.8
if command -v python3.8 &> /dev/null; then
    PYTHON_CMD="python3.8"
    echo "✓ Using Python 3.8"
else
    PYTHON_CMD="python3"
    echo "⚠ Python 3.8 not found, using default python3"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version)
echo "✓ Python found: $PYTHON_VERSION"

PY_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")
PY_VERSION="$PY_MAJOR.$PY_MINOR"
echo "  Python version: $PY_VERSION"

# PyO3 0.27.2 requires Python 3.7+
if [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 7 ]]; then
    echo "  ✗ Error: Python $PY_VERSION is too old"
    echo "  PyO3 0.27.2 requires Python 3.7 or newer"
    echo "  Please upgrade Python to 3.8+"
    exit 1
fi

echo "  ✓ Python version compatible with PyO3 0.27.2"

# Check for Python dev headers
if command -v ${PYTHON_CMD}-config &> /dev/null; then
    echo "✓ Python development headers found"
elif pkg-config --exists python3; then
    echo "✓ Python development headers found"
else
    echo "⚠ Warning: Python development headers not found"
    echo "  Installing now..."
    sudo apt-get update
    sudo apt-get install -y python3.8-dev python3-dev
fi

echo ""
echo "Installing Python dependencies..."
echo "=========================================="

# Set up Python user base for microSD
export PYTHONUSERBASE="/mnt/microsd/python-packages"
export PATH="$PYTHONUSERBASE/bin:$PATH"
mkdir -p "$PYTHONUSERBASE"
echo "✓ Python packages will install to: $PYTHONUSERBASE"

echo ""
echo "Would you like to install Python dependencies for AI training? (y/n)"
read -r response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo ""

    # Install system packages first (provides optimized binaries)
    echo "Installing system Python packages..."
    sudo apt-get install -y python3.8-dev libopenblas-base libopenmpi-dev

    # Install NumPy
    echo ""
    echo "Installing NumPy..."
    $PYTHON_CMD -m pip install --user --ignore-installed numpy

    if [ $? -eq 0 ]; then
        echo "✓ NumPy installed successfully"

        # Verify NumPy
        $PYTHON_CMD -c "import numpy; print('  NumPy version:', numpy.__version__)"
    else
        echo "⚠ NumPy installation failed, trying with Cython..."
        $PYTHON_CMD -m pip install --user Cython
        $PYTHON_CMD -m pip install --user --no-cache-dir numpy
    fi

    # Test CartPole module
    echo ""
    echo "Testing CartPole module..."
    if [ -f "cartpole.so" ]; then
        $PYTHON_CMD -c "import cartpole; env = cartpole.PyCartPole(); print('✓ CartPole module works!')" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "✓ CartPole Python module loaded successfully"
        else
            echo "⚠ CartPole module test failed"
        fi
    else
        echo "⚠ cartpole.so not found - run build_rust.sh first"
    fi

    # Optional PyTorch installation
    echo ""
    echo "Would you like to install PyTorch for Jetson? (y/n)"
    echo "(This will download ~500MB to microSD)"
    read -r pytorch_response

    if [[ "$pytorch_response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo ""
        echo "PyTorch Installation Guide:"
        echo "=========================================="
        echo ""
        echo "For Python 3.8, you'll need to:"
        echo "1. Download PyTorch wheel for your JetPack version from:"
        echo "   https://forums.developer.nvidia.com/t/pytorch-for-jetson"
        echo ""
        echo "2. Install with:"
        echo "   $PYTHON_CMD -m pip install --user <pytorch-wheel-file>"
        echo ""
        echo "Note: Pre-built wheels are usually for Python 3.6."
        echo "For Python 3.8, you may need to build from source or use Python 3.6."
        echo ""
    fi
else
    echo "Skipping Python dependencies installation"
fi

echo ""
echo "=========================================="
echo "Python Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Test Python module: $PYTHON_CMD -c 'import cartpole; print(\"Success!\")'"
echo "  2. Run performance test: $PYTHON_CMD test_setup.py"
echo "  3. Train AI: $PYTHON_CMD train_ai.py"
echo ""

