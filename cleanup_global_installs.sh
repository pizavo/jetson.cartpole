#!/bin/bash
# Cleanup script to remove globally installed Rust and Python packages
# Run this to free up eMMC space after moving to microSD

echo "=========================================="
echo "Cleanup Global Installations"
echo "=========================================="
echo ""
echo "This will remove:"
echo "  - Rust installation from ~/.cargo and ~/.rustup"
echo "  - Python packages from ~/.local"
echo ""
echo "⚠️  WARNING: This cannot be undone!"
echo ""
read -p "Continue? (yes/no): " confirm

if [[ "$confirm" != "yes" ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Starting cleanup..."

# Remove Rust
if [ -d "$HOME/.cargo" ] || [ -d "$HOME/.rustup" ]; then
    echo ""
    echo "Removing Rust from home directory..."
    rm -rf "$HOME/.cargo"
    rm -rf "$HOME/.rustup"
    echo "✓ Removed ~/.cargo and ~/.rustup"

    # Clean up PATH modifications in bashrc
    if [ -f "$HOME/.bashrc" ]; then
        echo "Cleaning up ~/.bashrc..."
        sed -i '/\.cargo\/env/d' "$HOME/.bashrc"
        echo "✓ Cleaned ~/.bashrc"
    fi
else
    echo "✓ No Rust installation found in home directory"
fi

# Remove Python user packages
if [ -d "$HOME/.local/lib" ]; then
    echo ""
    echo "Python packages in ~/.local:"
    python3 -m pip list --user 2>/dev/null || echo "  (none found)"
    echo ""
    read -p "Remove all Python packages from ~/.local? (yes/no): " remove_python

    if [[ "$remove_python" == "yes" ]]; then
        # Get list of user-installed packages
        packages=$(python3 -m pip list --user --format=freeze 2>/dev/null | cut -d= -f1)

        if [ -n "$packages" ]; then
            echo "Uninstalling packages..."
            echo "$packages" | xargs python3 -m pip uninstall -y
            echo "✓ Removed Python packages"
        fi

        # Remove the directory
        rm -rf "$HOME/.local/lib/python3.6/site-packages"
        echo "✓ Removed ~/.local/lib/python3.6/site-packages"
    fi
else
    echo "✓ No Python packages found in ~/.local"
fi

# Show disk space freed
echo ""
echo "=========================================="
echo "Cleanup complete!"
echo "=========================================="
echo ""
echo "Disk space on eMMC:"
df -h / | tail -1

echo ""
echo "Don't forget to add these to ~/.bashrc:"
echo ""
echo "export CARGO_HOME=\"/mnt/microsd/cargo\""
echo "export RUSTUP_HOME=\"/mnt/microsd/rustup\""
echo "export PYTHONUSERBASE=\"/mnt/microsd/python-packages\""
echo "export PATH=\"\$CARGO_HOME/bin:\$PATH\""
echo "export PYTHONPATH=\"\$PYTHONUSERBASE/lib/python3.6/site-packages:\$PYTHONPATH\""
echo ""

