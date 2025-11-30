#!/bin/bash
# Rust build script for Jetson Nano

echo "=========================================="
echo "CartPole - Rust Build"
echo "=========================================="
echo ""

# Force PyO3 to use Python 3.8 (not 3.6)
if command -v python3.8 &> /dev/null; then
    export PYO3_PYTHON=$(which python3.8)
    export PYTHONPATH="/mnt/microsd/python-packages/lib/python3.8/site-packages:$PYTHONPATH"
    echo "✓ PYO3_PYTHON set to: $PYO3_PYTHON"
else
    echo "⚠ Warning: python3.8 not found, using default python3"
    if command -v python3 &> /dev/null; then
        export PYO3_PYTHON=$(which python3)
    fi
fi

# Check if running on Jetson (ARM architecture)
ARCH=$(uname -m)
if [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "arm"* ]]; then
    echo "✓ Detected ARM architecture (Jetson compatible)"
else
    echo "⚠ Warning: Not running on ARM architecture"
    echo "  This script is optimized for Jetson Nano"
    echo "  Current architecture: $ARCH"
fi

echo ""

# Check for CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "✓ CUDA found: version $CUDA_VERSION"
else
    echo "⚠ Warning: CUDA not found"
    echo "  CUDA is not required for CartPole but needed for GPU-accelerated AI training"
fi

echo ""

# Check for Rust
if command -v rustc &> /dev/null; then
    RUST_VERSION=$(rustc --version)
    echo "✓ Rust found: $RUST_VERSION"
else
    echo "✗ Error: Rust not found"
    echo "  Install with: ./install_rust_microsd.sh"
    exit 1
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
    exit 1
fi

echo ""
echo "=========================================="
echo "Rust Build Complete!"
echo "=========================================="
echo ""

