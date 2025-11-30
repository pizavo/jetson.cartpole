# CartPole on Jetson Nano - Complete Setup Guide

This guide will help you deploy and run the CartPole game on NVIDIA Jetson Nano for AI training and testing with CUDA acceleration.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Building CartPole](#building-cartpole)
4. [Python Setup](#python-setup)
5. [CUDA and PyTorch](#cuda-and-pytorch)
6. [Running Training](#running-training)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### Hardware
- NVIDIA Jetson Nano (2GB or 4GB model)
- MicroSD card (32GB+ recommended)
- Power supply (5V 4A barrel jack recommended for 4GB model)
- (Optional) Monitor, keyboard, mouse for initial setup

### Software
- JetPack SDK 4.6+ (comes with Ubuntu 18.04)
- Internet connection for downloading dependencies

## Initial Setup

### 1. Flash JetPack to SD Card

Download and flash JetPack from NVIDIA:
```bash
# Download from: https://developer.nvidia.com/jetpack-sdk-46
# Use Etcher or similar tool to flash to SD card
```

### 2. First Boot Configuration

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install essential tools
sudo apt-get install -y build-essential git curl wget
```

### 3. Install Rust

**⚠️ IMPORTANT: For Jetson Nano with limited eMMC space:**

If your eMMC is tight, install Rust to microSD card instead:

```bash
# Install to microSD (recommended for space-constrained systems)
cd /mnt/microsd/projects/jetson.cartpole
chmod +x install_rust_microsd.sh
./install_rust_microsd.sh

# Add to ~/.bashrc (copy the lines shown by the script)
echo 'export CARGO_HOME="/mnt/microsd/cargo"' >> ~/.bashrc
echo 'export RUSTUP_HOME="/mnt/microsd/rustup"' >> ~/.bashrc
echo 'export PATH="$CARGO_HOME/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

**OR** for standard installation to home directory:

```bash
# Install Rust toolchain (requires ~2GB in home directory)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

Verify installation:
```bash
rustc --version
cargo --version
```

## Important: Python Version Compatibility

**⚠️ Jetson Nano ships with Python 3.6**

This project uses PyO3 0.20.3 which is the last version to support Python 3.6. 
- PyO3 0.21+ requires Python 3.7+
- If you see "Python interpreter version (3.6) is lower than PyO3's minimum supported version (3.7)", the Cargo.toml has the wrong PyO3 version

The current configuration is optimized for Jetson Nano's Python 3.6.

## Building CartPole

### 1. Transfer Files to Jetson

Option A - Using Git (if hosted):
```bash
git clone <your-repo-url>
cd cartpole
```

Option B - Using SCP from your development machine:
```bash
# On your Windows machine
scp -r C:\Projects\IDE\RustRover\cartpole username@jetson-ip:~/
```

Option C - USB drive or SD card transfer

### 2. Build the Project

```bash
cd ~/cartpole

# Make the build script executable
chmod +x build_jetson.sh

# Run the automated build script
./build_jetson.sh
```

Or build manually:

```bash
# Build visual game (headless on Jetson, but works with display)
cargo build --release

# Build Python module
cargo build --release --features python

# Copy Python module
cp target/release/libcartpole.so cartpole.so
```

### 3. Verify Build

```bash
# Test the Rust library
cargo test

# Test Python import
python3 -c "import cartpole; print('CartPole module loaded successfully!')"
```

## Python Setup

### Important: Python 2.7 vs Python 3

**⚠️ CRITICAL:** Jetson Nano ships with both Python 2.7 and Python 3.6+

- `python` → Python 2.7.17 (default, DON'T USE)
- `python3` → Python 3.6+ (USE THIS)

**Always use `python3` and `pip3` for all commands!**

### 1. Verify Python 3 is Installed

```bash
python3 --version  # Should show Python 3.6 or newer
pip3 --version     # Should show pip for Python 3
```

If Python 3 is not installed:
```bash
sudo apt-get install -y python3 python3-pip python3-dev
```

### 2. Install Python Dependencies

**⚠️ For microSD installation (saves eMMC space):**

```bash
# Set custom Python package location
export PYTHONUSERBASE="/mnt/microsd/python-packages"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.6/site-packages:$PYTHONPATH"
mkdir -p "$PYTHONUSERBASE"

# Add to ~/.bashrc to make permanent
echo 'export PYTHONUSERBASE="/mnt/microsd/python-packages"' >> ~/.bashrc
echo 'export PYTHONPATH="$PYTHONUSERBASE/lib/python3.6/site-packages:$PYTHONPATH"' >> ~/.bashrc

# Install packages
python3 -m pip install --user --upgrade pip
python3 -m pip install --user numpy
```

**OR** for standard installation:

```bash
# Install system packages (provides pre-built binaries)
sudo apt-get install -y python3-pip python3-dev python3-numpy libopenblas-base libopenmpi-dev

# Upgrade pip (use --user to avoid permission issues)
python3 -m pip install --upgrade pip --user

# Install NumPy to user location
python3 -m pip install numpy --user

# If NumPy fails to build (Cython error):
python3 -m pip install --user Cython
python3 -m pip install --user numpy --no-cache-dir
```

**Note:** Use `python3 -m pip` instead of `pip3` to avoid wrapper issues.

### 3. Verify Python Environment

```bash
python3 --version  # Should be Python 3.6+
pip3 list          # Show installed packages

# Test CartPole import
python3 -c "import cartpole; print('Success!')"

# Run simple training
python3 train_ai.py
```

## CUDA and PyTorch

### 1. Verify CUDA Installation

CUDA comes pre-installed with JetPack:

```bash
# Check CUDA
nvcc --version

# Check cuDNN
dpkg -l | grep cudnn

# Test CUDA
nvidia-smi  # May not work on Jetson, this is normal
```

### 2. Install PyTorch for Jetson

PyTorch requires special builds for Jetson:

```bash
# For JetPack 4.6 (CUDA 10.2)
# Download from NVIDIA's forum
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Install PyTorch
python3 -m pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl --user

# Install torchvision (optional, for vision tasks)
sudo apt-get install -y libopenblas-base libopenmpi-dev
python3 -m pip install torchvision --user
```

**Note:** PyTorch versions vary by JetPack version. Check NVIDIA forums for the correct wheel:
- https://forums.developer.nvidia.com/t/pytorch-for-jetson

### 3. Verify PyTorch with CUDA

```bash
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Current device: {torch.cuda.current_device()}")
EOF
```

Expected output:
```
PyTorch version: 1.10.0
CUDA available: True
CUDA version: 10.2
Device: Xavier
Current device: 0
```

## Running Training

### 1. Simple Agent Training

```bash
# Run basic training with simple policy
python3 train_ai.py
```

### 2. Deep Q-Learning (DQN) with CUDA

```bash
# Run GPU-accelerated DQN training
python3 train_dqn.py
```

Monitor GPU usage:
```bash
# In another terminal
sudo tegrastats
```

### 3. Monitor System Resources

```bash
# Install htop for CPU monitoring
sudo apt-get install htop
htop

# Monitor GPU and memory
sudo tegrastats
```

### 4. Optimize Performance

For better performance:

```bash
# Set maximum power mode
sudo nvpmodel -m 0

# Set maximum clock speeds
sudo jetson_clocks
```

## Performance Tips

### 1. Memory Management

Jetson Nano has limited RAM (2GB or 4GB):

```python
# In your training script, use smaller batch sizes
agent.batch_size = 32  # Instead of 64

# Reduce replay buffer size
agent.memory = ReplayBuffer(capacity=5000)  # Instead of 10000
```

### 2. Swap Space

Add swap for better stability:

```bash
# Create 4GB swap file
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 3. Training Parameters

Optimize for Jetson:

```python
# Use mixed precision (if supported)
from torch.cuda.amp import autocast, GradScaler

# Reduce model size
hidden_size = 64  # Instead of 128

# Use CPU for some operations if GPU memory is tight
```

## Project Structure on Jetson

```
~/cartpole/
├── src/                      # Rust source code
├── target/release/           # Compiled binaries
│   ├── cartpole             # Game executable (needs display)
│   └── libcartpole.so       # Python module
├── cartpole.so              # Symlink to Python module
├── train_ai.py              # Simple training script
├── train_dqn.py             # DQN training with GPU
├── build_jetson.sh          # Build script
└── README.md                # Documentation
```

## Troubleshooting

### Build Errors

**Problem:** Rust build fails with memory error
```bash
# Solution: Add swap and build with single thread
export CARGO_BUILD_JOBS=1
cargo build --release
```

**Problem:** Python.h not found
```bash
# Solution: Install Python dev headers
sudo apt-get install python3-dev
```

### Runtime Errors

**Problem:** "cannot open shared object file"
```bash
# Solution: Update library path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```

**Problem:** CUDA out of memory
```python
# Solution: Reduce batch size and model size in train_dqn.py
agent.batch_size = 16
hidden_size = 32
```

**Problem:** PyTorch not using CUDA
```bash
# Check CUDA is accessible
python3 -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch for correct JetPack version
```

**Problem:** pip wrapper warning
```
WARNING: pip is being invoked by an old script wrapper...
```
```bash
# Solution: Use python3 -m pip instead of pip3
python3 -m pip install numpy --user
python3 -m pip list
```

### Performance Issues

**Problem:** Training is slow
```bash
# Enable max performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Monitor with tegrastats
sudo tegrastats
```

**Problem:** System freezes
```bash
# Add swap space (see Memory Management above)
# Reduce batch size in training scripts
```

## Running Headless (No Display)

The visual game requires a display, but training works headless:

```bash
# SSH into Jetson
ssh username@jetson-ip

# Run training
cd ~/cartpole
python3 train_dqn.py

# Monitor progress
tail -f training.log  # If you redirect output
```

## Benchmarking

Compare CPU vs GPU performance:

```python
# test_performance.py
import time
import torch
import cartpole

env = cartpole.PyCartPole()

# Benchmark environment speed
start = time.time()
for _ in range(10000):
    env.reset()
    for _ in range(100):
        env.step(1)
elapsed = time.time() - start
print(f"10000 episodes: {elapsed:.2f}s ({10000/elapsed:.0f} eps/s)")

# Benchmark GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.randn(1000, 1000).to(device)
    start = time.time()
    for _ in range(100):
        y = torch.mm(x, x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"GPU benchmark: {elapsed:.2f}s")
```

## Next Steps

1. **Experiment with hyperparameters** in `train_dqn.py`
2. **Implement more RL algorithms** (A3C, PPO, etc.)
3. **Create visualization** of training progress
4. **Deploy trained models** for real-time control
5. **Integrate with robotics projects** on Jetson

## Useful Resources

- NVIDIA Jetson Nano Developer Kit: https://developer.nvidia.com/embedded/jetson-nano-developer-kit
- JetPack Documentation: https://docs.nvidia.com/jetson/jetpack/
- PyTorch for Jetson: https://forums.developer.nvidia.com/t/pytorch-for-jetson
- Rust on ARM: https://rust-lang.github.io/rustup/installation/other.html

## Support

For issues:
1. Check the Troubleshooting section above
2. Review NVIDIA Jetson forums
3. Check Rust and PyTorch documentation
4. Open an issue in the project repository

Happy training on Jetson Nano! 🚀

