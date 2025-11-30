#!/bin/bash
echo ""
echo "=========================================="
echo "Rust Build Complete!"
echo "=========================================="
echo ""

fi
    exit 1
    echo "  This is optional - the visual game still works"
    echo "✗ Python build failed"
else
    fi
        echo "  Check target/release/ directory"
        echo "⚠ Warning: Could not find .so file"
    else
        echo "✓ Python module copied to: cartpole.so"
        cp target/release/libcartpole_lib.so cartpole.so
    elif [ -f "target/release/libcartpole_lib.so" ]; then
        echo "✓ Python module copied to: cartpole.so"
        cp target/release/libcartpole.so cartpole.so
    if [ -f "target/release/libcartpole.so" ]; then
    # Copy the Python module

    echo "✓ Python module built successfully"
if [ $? -eq 0 ]; then

cargo build --release --features python
echo "2. Building with Python bindings..."
echo ""
# Build with Python support

fi
    exit 1
    echo "✗ Build failed"
else
    echo "  Run with: ./target/release/cartpole"
    echo "✓ Visual game built successfully"
if [ $? -eq 0 ]; then

cargo build --release
echo "1. Building visual game (no Python bindings)..."
echo ""
# Build the visual game

echo "=========================================="
echo "Building CartPole..."
echo ""

fi
    exit 1
    echo "  Install with: ./install_rust_microsd.sh"
    echo "✗ Error: Rust not found"
else
    echo "✓ Rust found: $RUST_VERSION"
    RUST_VERSION=$(rustc --version)
if command -v rustc &> /dev/null; then
# Check for Rust

echo ""

fi
    echo "  CUDA is not required for CartPole but needed for GPU-accelerated AI training"
    echo "⚠ Warning: CUDA not found"
else
    echo "✓ CUDA found: version $CUDA_VERSION"
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
if command -v nvcc &> /dev/null; then
# Check for CUDA

echo ""

fi
    echo "  Current architecture: $ARCH"
    echo "  This script is optimized for Jetson Nano"
    echo "⚠ Warning: Not running on ARM architecture"
else
    echo "✓ Detected ARM architecture (Jetson compatible)"
if [[ "$ARCH" == "aarch64" ]] || [[ "$ARCH" == "arm"* ]]; then
ARCH=$(uname -m)
# Check if running on Jetson (ARM architecture)

fi
    echo "✓ PYO3_PYTHON set to: $PYO3_PYTHON"
    export PYTHONPATH="/mnt/microsd/python-packages/lib/python3.8/site-packages:$PYTHONPATH"
    export PYO3_PYTHON=$(which python3.8)
if command -v python3.8 &> /dev/null; then
# Force PyO3 to use Python 3.8 (not 3.6)

echo ""
echo "=========================================="
echo "CartPole - Rust Build"
echo "=========================================="

# Rust build script for Jetson Nano

