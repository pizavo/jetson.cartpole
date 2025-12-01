#!/bin/bash
# Python setup script for Jetson Nano AI training

echo "=========================================="
echo "CartPole - Python Setup"
echo "=========================================="
echo ""

# Use Python 3.6 (stock JetPack 4.6)
PYTHON_CMD="python3.6"
echo "✓ Using Python 3.6 (stock JetPack 4.6)"

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version)
echo "✓ Python found: $PYTHON_VERSION"

PY_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")
PY_VERSION="$PY_MAJOR.$PY_MINOR"
echo "  Python version: $PY_VERSION"

# PyO3 0.15.2 supports Python 3.6+
if [[ "$PY_MAJOR" -eq 3 ]] && [[ "$PY_MINOR" -lt 6 ]]; then
    echo "  ✗ Error: Python $PY_VERSION is too old"
    echo "  PyO3 0.15.2 requires Python 3.6 or newer"
    exit 1
fi

echo "  ✓ Python version $PY_VERSION compatible with PyO3 0.15.2"

# Check for Python dev headers
if command -v python3-config &> /dev/null; then
    echo "✓ Python development headers found"
elif pkg-config --exists python3; then
    echo "✓ Python development headers found"
else
    echo "⚠ Warning: Python development headers not found"
    echo "  Installing now..."
    sudo apt-get update
    sudo apt-get install -y python3-dev
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
    sudo apt-get install -y python3-numpy python3-dev libopenblas-base libopenmpi-dev libblas-dev

    # Check if system NumPy works
    echo ""
    echo "Checking system NumPy..."
    if $PYTHON_CMD -c "import numpy; print('System NumPy:', numpy.__version__)" 2>/dev/null; then
        echo "✓ Using system NumPy (optimized for Jetson)"
        echo "  Location: /usr/lib/python3/dist-packages"
        echo ""
        echo "Note: System NumPy is recommended for Jetson Nano"
        echo "      It's pre-compiled and optimized by NVIDIA"
    else
        # Try installing user NumPy with specific version that has wheels
        echo "System NumPy not found, installing user NumPy..."
        echo ""

        # Try NumPy 1.19.2 which has better wheel support for Python 3.6
        echo "Attempting NumPy 1.19.2 (has pre-built wheels)..."
        $PYTHON_CMD -m pip install --user "numpy==1.19.2" 2>/dev/null

        if [ $? -eq 0 ]; then
            echo "✓ NumPy 1.19.2 installed successfully"
            $PYTHON_CMD -c "import numpy; print('  NumPy version:', numpy.__version__)"
        else
            # Last resort: try latest numpy with binary install only
            echo "Trying latest NumPy (binary only, no build)..."
            $PYTHON_CMD -m pip install --user --only-binary :all: numpy 2>/dev/null

            if [ $? -eq 0 ]; then
                echo "✓ NumPy installed successfully"
                $PYTHON_CMD -c "import numpy; print('  NumPy version:', numpy.__version__)"
            else
                echo "✗ NumPy installation failed"
                echo ""
                echo "Recommendation: Use system NumPy instead"
                echo "  sudo apt-get install python3-numpy"
                echo "  System NumPy is already available and works fine"
            fi
        fi
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
        echo "For Python 3.6 (JetPack 4.6):"
        echo "1. Download PyTorch wheel from:"
        echo "   https://forums.developer.nvidia.com/t/pytorch-for-jetson"
        echo ""
        echo "2. For JetPack 4.6, use:"
        echo "   wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
        echo "   $PYTHON_CMD -m pip install --user torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
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

