#!/bin/bash
# Master build script - runs both Rust and Python setup

echo "=========================================="
echo "CartPole - Complete Build Script"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Build Rust CartPole (visual game + Python module)"
echo "  2. Set up Python environment and dependencies"
echo ""

# Force PyO3 to use Python 3.8
if command -v python3.8 &> /dev/null; then
    export PYO3_PYTHON=$(which python3.8)
    export PYTHONUSERBASE="/mnt/microsd/python-packages"
    export PYTHONPATH="$PYTHONUSERBASE/lib/python3.8/site-packages:$PYTHONPATH"
fi

# Custom paths for microSD installation
export CARGO_HOME="/mnt/microsd/cargo"
export RUSTUP_HOME="/mnt/microsd/rustup"
export PATH="$CARGO_HOME/bin:$PATH"

echo "Installation paths:"
echo "  Rust/Cargo: $CARGO_HOME"
echo "  Python packages: $PYTHONUSERBASE"
echo "  Project: $(pwd)"
echo ""

# Check if scripts exist
if [ ! -f "build_rust.sh" ]; then
    echo "✗ Error: build_rust.sh not found"
    exit 1
fi

if [ ! -f "setup_python.sh" ]; then
    echo "✗ Error: setup_python.sh not found"
    exit 1
fi

# Make scripts executable
chmod +x build_rust.sh
chmod +x setup_python.sh

# Run Rust build
echo "=========================================="
echo "Step 1: Building Rust Components"
echo "=========================================="
echo ""

./build_rust.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "✗ Rust build failed!"
    echo "  Fix the errors above and try again"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Setting Up Python Environment"
echo "=========================================="
echo ""

./setup_python.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "⚠ Python setup had issues, but Rust build succeeded"
    echo "  You can still use the visual game"
fi

echo ""
echo "=========================================="
echo "Build Complete!"
echo "=========================================="
echo ""
echo "What you can do now:"
echo "  1. Run visual game: ./target/release/cartpole"
echo "  2. Test Python: python3.8 -c 'import cartpole; print(\"Success!\")'"
echo "  3. Run tests: python3.8 test_setup.py"
echo "  4. Train AI: python3.8 train_ai.py"
echo ""
echo "Environment variables to add to ~/.bashrc:"
echo "  export CARGO_HOME=\"/mnt/microsd/cargo\""
echo "  export RUSTUP_HOME=\"/mnt/microsd/rustup\""
echo "  export PYTHONUSERBASE=\"/mnt/microsd/python-packages\""
echo "  export PYO3_PYTHON=\$(which python3.8)"
echo "  export PATH=\"\$CARGO_HOME/bin:\$PATH\""
echo "  export PYTHONPATH=\"\$PYTHONUSERBASE/lib/python3.8/site-packages:\$PYTHONPATH\""
echo ""
echo "Happy training! 🚀"
echo ""

