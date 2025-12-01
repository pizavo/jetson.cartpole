#!/bin/bash
# Download and install correct PyTorch for Python 3.8 on Jetson Nano

echo "=========================================="
echo "PyTorch for Python 3.8 - Correct Version"
echo "=========================================="
echo ""

echo "IMPORTANT: The issue is that you have a Python 3.6 wheel (cp36)"
echo "but you're using Python 3.8. The .so files are incompatible."
echo ""

# Check current wheel
if [ -f "/mnt/microsd/torch-1.10.0-cp38-cp38-linux_aarch64.whl" ]; then
    echo "Checking if wheel is actually for Python 3.8..."
    unzip -l /mnt/microsd/torch-1.10.0-cp38-cp38-linux_aarch64.whl | grep "_C.cpython" | head -1

    if unzip -l /mnt/microsd/torch-1.10.0-cp38-cp38-linux_aarch64.whl | grep -q "cpython-36"; then
        echo ""
        echo "✗ ERROR: This wheel is actually for Python 3.6, not 3.8!"
        echo "  The filename says cp38 but contains cp36 files."
        echo "  This is why PyTorch fails to load."
        echo ""
        WRONG_WHEEL=true
    elif unzip -l /mnt/microsd/torch-1.10.0-cp38-cp38-linux_aarch64.whl | grep -q "cpython-38"; then
        echo ""
        echo "✓ Wheel is correct for Python 3.8"
        WRONG_WHEEL=false
    else
        echo ""
        echo "⚠ Cannot determine Python version in wheel"
        WRONG_WHEEL=unknown
    fi
fi

echo ""
echo "=========================================="
echo "Solution Options"
echo "=========================================="
echo ""

if [ "$WRONG_WHEEL" = "true" ]; then
    echo "Your current wheel is for Python 3.6. You have 3 options:"
    echo ""
    echo "Option 1: Build PyTorch from source for Python 3.8 (SLOW - 6+ hours)"
    echo "  - Most reliable for Python 3.8"
    echo "  - Instructions: https://forums.developer.nvidia.com/t/pytorch-for-jetson"
    echo ""
    echo "Option 2: Use Python 3.6 instead (EASIER)"
    echo "  - Keep PyTorch 1.10.0 wheel you have"
    echo "  - Rebuild CartPole for Python 3.6 (change PyO3 version)"
    echo "  - Simpler but uses older Python"
    echo ""
    echo "Option 3: Try finding a Python 3.8 wheel (MAY NOT EXIST)"
    echo "  - NVIDIA usually only provides Python 3.6 wheels for JetPack 4.6"
    echo "  - You may not find one"
    echo ""

    read -p "Which option do you prefer? (1/2/3): " choice

    if [ "$choice" = "2" ]; then
        echo ""
        echo "=========================================="
        echo "Switching to Python 3.6"
        echo "=========================================="
        echo ""
        echo "This will:"
        echo "1. Use the existing PyTorch wheel (works with Python 3.6)"
        echo "2. Update CartPole to use Python 3.6 (PyO3 0.16.6)"
        echo "3. Install packages for Python 3.6"
        echo ""

        read -p "Continue with Python 3.6 setup? (y/n): " confirm

        if [ "$confirm" = "y" ]; then
            # Clean Python 3.8 installation
            echo "Removing Python 3.8 packages..."
            rm -rf /mnt/microsd/python-packages/lib/python3.8

            # Install PyTorch for Python 3.6
            echo "Installing PyTorch for Python 3.6..."
            export PYTHONUSERBASE="/mnt/microsd/python-packages"
            unset PYTHONPATH

            cd /mnt/microsd
            python3.6 -m pip install --user torch-1.10.0-cp36-cp36m-linux_aarch64.whl
            python3.6 -m pip install --user numpy typing-extensions

            # Create Python 3.6 environment script
            cat > ~/use_python36.sh << 'EOF'
#!/bin/bash
export CARGO_HOME="/mnt/microsd/cargo"
export RUSTUP_HOME="/mnt/microsd/rustup"
export PYTHONUSERBASE="/mnt/microsd/python-packages"
export PYO3_PYTHON=/usr/bin/python3.6

unset PYTHONPATH
export PYTHONPATH="/mnt/microsd/python-packages/lib/python3.6/site-packages"
export LD_LIBRARY_PATH="/mnt/microsd/python-packages/lib/python3.6/site-packages/torch/lib:$LD_LIBRARY_PATH"
export PATH="$PYTHONUSERBASE/bin:$CARGO_HOME/bin:$PATH"

echo "✓ Python 3.6 environment activated"
EOF
            chmod +x ~/use_python36.sh

            echo ""
            echo "✓ Python 3.6 environment created"
            echo ""
            echo "Next steps:"
            echo "1. Update Cargo.toml: pyo3 = { version = \"0.16.6\", ... }"
            echo "2. Rebuild CartPole: cargo clean && cargo build --release --features python"
            echo "3. Test: source ~/use_python36.sh && python3.6 -c 'import torch'"
            echo ""

            # Test
            source ~/use_python36.sh
            cd ~
            python3.6 -c "import torch; print('PyTorch:', torch.__version__)" 2>&1

            if [ $? -eq 0 ]; then
                echo ""
                echo "✓ PyTorch works with Python 3.6!"
            fi
        fi

    elif [ "$choice" = "1" ]; then
        echo ""
        echo "To build PyTorch from source for Python 3.8:"
        echo "Visit: https://forums.developer.nvidia.com/t/pytorch-for-jetson"
        echo "This will take 6+ hours on Jetson Nano"
        echo ""

    else
        echo ""
        echo "Checking NVIDIA forums for Python 3.8 wheels..."
        echo "Visit: https://forums.developer.nvidia.com/t/pytorch-for-jetson"
        echo ""
    fi
else
    echo "Cannot proceed - please check your PyTorch wheel file"
fi

echo ""

