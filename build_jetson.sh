#!/bin/bash
# Build script for Jetson Nano deployment

echo "=========================================="
echo "CartPole - Jetson Nano Build Script"
echo "=========================================="
echo ""

# Check if running on Jetson (ARM architecture)
ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "arm"* ]]; then
    echo "✓ Detected ARM architecture (Jetson compatible)"
else
    echo "⚠ Warning: Not running on ARM architecture"
    echo "  This script is optimized for Jetson Nano"
    echo "  Current architecture: $ARCH"
    echo ""
fi

# Check for CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "✓ CUDA found: version $CUDA_VERSION"
else
    echo "⚠ Warning: CUDA not found"
    echo "  CUDA is not required for CartPole but needed for GPU-accelerated AI training"
fi

echo ""

# Check for Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Python found: $PYTHON_VERSION"
else
    echo "✗ Error: Python3 not found"
    echo "  Please install: sudo apt-get install python3 python3-dev"
    exit 1
fi

# Check for Python dev headers
if pkg-config --exists python3; then
    echo "✓ Python development headers found"
else
    echo "⚠ Warning: Python development headers not found"
    echo "  Installing now..."
    sudo apt-get update
    sudo apt-get install -y python3-dev
fi

echo ""
echo "Building CartPole..."
echo "=========================================="

# Build the visual game
echo ""
echo "1. Building visual game (no Python bindings)..."
cargo build --release

if [ $? -eq 0 ]; then
    echo "✓ Visual game built successfully"
    echo "  Run with: ./target/release/cartpole"
else
    echo "✗ Build failed"
    exit 1
fi

# Build with Python support
echo ""
echo "2. Building with Python bindings..."
cargo build --release --features python

if [ $? -eq 0 ]; then
    echo "✓ Python module built successfully"

    # Copy the Python module
    if [ -f "target/release/libcartpole.so" ]; then
        cp target/release/libcartpole.so cartpole.so
        echo "✓ Python module copied to: cartpole.so"
    elif [ -f "target/release/libcartpole_lib.so" ]; then
        cp target/release/libcartpole_lib.so cartpole.so
        echo "✓ Python module copied to: cartpole.so"
    else
        echo "⚠ Warning: Could not find .so file"
        echo "  Check target/release/ directory"
    fi
else
    echo "✗ Python build failed"
    echo "  This is optional - the visual game still works"
fi

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run visual game: ./target/release/cartpole"
echo "  2. Test Python: python3 -c 'import cartpole; print(\"Success!\")'"
echo "  3. Train AI: python3 train_ai.py"
echo ""

# Optional: Install Python dependencies for AI training
echo "Would you like to install Python dependencies for AI training? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo ""
    echo "Installing Python dependencies..."
    pip3 install numpy

    echo ""
    echo "Would you like to install PyTorch for Jetson? (y/n)"
    echo "(This will download ~500MB)"
    read -r pytorch_response
    if [[ "$pytorch_response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Please follow NVIDIA's official guide for PyTorch on Jetson:"
        echo "https://forums.developer.nvidia.com/t/pytorch-for-jetson"
        echo ""
        echo "Quick install (for JetPack 4.6):"
        echo "wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
        echo "pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl"
    fi
fi

echo ""
echo "All done! Happy training! 🚀"

