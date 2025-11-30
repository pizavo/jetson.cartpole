#!/bin/bash
# Install Rust to microSD card for Jetson Nano

# Custom paths on microSD
export CARGO_HOME="/mnt/microsd/cargo"
export RUSTUP_HOME="/mnt/microsd/rustup"

echo "Installing Rust to microSD..."
echo "  CARGO_HOME: $CARGO_HOME"
echo "  RUSTUP_HOME: $RUSTUP_HOME"
echo ""

# Create directories
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME"

# Install Rust (won't modify PATH automatically)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --no-modify-path -y

# Verify installation
if [ -f "$CARGO_HOME/bin/rustc" ]; then
    echo ""
    echo "✓ Rust installed successfully!"
    echo ""
    echo "Add these lines to your ~/.bashrc:"
    echo ""
    echo "export CARGO_HOME=\"$CARGO_HOME\""
    echo "export RUSTUP_HOME=\"$RUSTUP_HOME\""
    echo "export PATH=\"\$CARGO_HOME/bin:\$PATH\""
    echo ""
    echo "Then run: source ~/.bashrc"
else
    echo "✗ Installation failed"
    exit 1
fi

