#!/bin/bash
# Quick fix script to ensure PyO3 uses Python 3.8 on Jetson Nano

echo "=========================================="
echo "Fix PyO3 to use Python 3.8"
echo "=========================================="
echo ""

# Check if Python 3.8 is installed
if ! command -v python3.8 &> /dev/null; then
    echo "✗ Error: python3.8 not found"
    echo "  Install with: sudo apt-get install python3.8 python3.8-dev"
    exit 1
fi

echo "✓ Python 3.8 found: $(python3.8 --version)"
echo ""

# Set environment variable
export PYO3_PYTHON=$(which python3.8)
echo "Setting PYO3_PYTHON=$PYO3_PYTHON"
echo ""

# Add to current shell session
echo "export PYO3_PYTHON=$PYO3_PYTHON" >> ~/.bashrc
echo "✓ Added to ~/.bashrc"
echo ""

# Install python3.8-dev if not already installed
if ! dpkg -l | grep -q python3.8-dev; then
    echo "Installing python3.8-dev..."
    sudo apt-get update
    sudo apt-get install -y python3.8-dev
fi

# Update alternatives for python3-config
if [ -f "/usr/bin/python3.8-config" ]; then
    echo "Setting up python3-config alternatives..."
    sudo update-alternatives --install /usr/bin/python3-config python3-config /usr/bin/python3.6-config 1 2>/dev/null || true
    sudo update-alternatives --install /usr/bin/python3-config python3-config /usr/bin/python3.8-config 2
    sudo update-alternatives --set python3-config /usr/bin/python3.8-config
    echo "✓ python3-config now points to Python 3.8"
else
    echo "✗ python3.8-config not found even after installation"
    echo "  Continuing anyway - PYO3_PYTHON environment variable will handle this"
fi

echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="
echo ""

echo "Python versions:"
python3 --version
python3.8 --version

echo ""
echo "python3-config:"
python3-config --prefix

echo ""
echo "PYO3_PYTHON environment:"
echo "  PYO3_PYTHON=$PYO3_PYTHON"

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Source bashrc: source ~/.bashrc"
echo "2. Clean build: cargo clean"
echo "3. Rebuild: ./build_jetson.sh"
echo ""

