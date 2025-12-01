# Complete Guide: CartPole with CUDA & AI Training on Jetson Nano

A comprehensive guide to building, running, and training AI agents with CartPole on NVIDIA Jetson Nano using GPU acceleration.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [Building CartPole](#building-cartpole)
4. [Installing PyTorch with CUDA](#installing-pytorch-with-cuda)
5. [Running the Visual Game](#running-the-visual-game)
6. [Training AI Agents](#training-ai-agents)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Topics](#advanced-topics)

---

## Prerequisites

### Hardware
- **NVIDIA Jetson Nano** (4GB recommended)
- **MicroSD card** (64GB+ recommended for packages)
- **5V 4A power supply** (barrel jack for max performance)
- **Cooling fan** (highly recommended for AI training)
- **Internet connection**

### Software
- **JetPack 4.6** (Ubuntu 18.04, Python 3.6.9, CUDA 10.2)
- **Stock configuration** (no Python upgrade needed)
- **Rust**

### What You'll Get
- Real-time CartPole physics simulation (Rust)
- Python bindings for AI training (PyO3 0.15.2)
- GPU-accelerated deep learning (PyTorch 1.10.0 + CUDA)
- Interactive game with AI agent support

---

## Initial Setup

### 1. Connect to Jetson Nano

```bash
# Via SSH (recommended)
ssh student@<jetson-ip>

# Or direct terminal access
# Connect monitor, keyboard, mouse
```

### 2. Create Project Directory

```bash
# Create directory on microSD (saves eMMC space)
sudo mkdir -p /mnt/microsd/projects
sudo chown $USER:$USER /mnt/microsd/projects
cd /mnt/microsd/projects

# Transfer project files here
# Use scp, git, or USB drive
```

### 3. Navigate to Project

```bash
cd /mnt/microsd/projects/jetson.cartpole
```

---

## Building CartPole

### 1. Install Rust (if not already installed)

### 2. Install System Dependencies

```bash
# Install build tools and libraries
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    python3-dev \
    python3-numpy \
    libopenblas-base \
    libopenmpi-dev \
    pkg-config
```

### 3. Build the Project

```bash
# Run the complete build script
./build_all.sh
```

**Build Process:**
1. **Rust compilation** (~15-20 minutes first time)
   - Visual game executable
   - Python bindings library
2. **Python setup** (interactive)
   - NumPy installation
   - Environment verification

**Expected Output:**
```
✓ Visual game built successfully
  Run with: ./target/release/cartpole

✓ Python module built successfully
✓ Python module copied to: cartpole.so
```

### 4. Verify Build

```bash
# Check binary exists
ls -lh target/release/cartpole

# Check Python module
ls -lh cartpole.so

# Test Python import
python3.6 -c "import cartpole; print('✓ CartPole module loaded')"
```

---

## Installing PyTorch with CUDA

### 1. Check CUDA Installation

```bash
# Verify CUDA is available
nvcc --version
# Expected: CUDA 10.2 for JetPack 4.6

# Check GPU info
sudo tegrastats
# Should show GPU usage stats
```

### 2. Download PyTorch Wheel

```bash
# Navigate to download directory
cd /mnt/microsd

# Download PyTorch for JetPack 4.6 + Python 3.6
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl \
    -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Verify download (should be ~470MB)
ls -lh torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```

**Alternative:** Visit [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson) for latest wheels.

### 3. Install PyTorch

```bash
# Install to user directory (microSD)
python3.6 -m pip install --user torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# Installation takes ~5-10 minutes
# PyTorch will be installed to: /mnt/microsd/python-packages/
```

### 4. Verify PyTorch with CUDA

```bash
# Test PyTorch installation
python3.6 << 'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    print("Device capability:", torch.cuda.get_device_capability(0))
EOF
```

**Expected Output:**
```
PyTorch version: 1.10.0
CUDA available: True
CUDA version: 10.2
Device count: 1
Device name: NVIDIA Tegra X1
Device capability: (5, 3)
```

### 5. Optional: Install TorchVision

```bash
# If you need computer vision utilities
sudo apt-get install -y libopenblas-base libopenmpi-dev
python3.6 -m pip install --user torchvision
```

---

## Running the Visual Game

### 1. Run the Standalone Game

```bash
# Navigate to project
cd /mnt/microsd/projects/jetson.cartpole

# Run the game (requires display)
./target/release/cartpole
```

**Controls:**
- `←/→` - Move cart left/right
- `SPACE` - Toggle AI mode (if trained model loaded)
- `R` - Reset episode
- `ESC` - Quit

### 2. Headless Mode (No Display)

If you're running via SSH without display:

```bash
# Run performance test
python3.6 test_setup.py
```

This tests the CartPole environment without graphics and shows:
- Episodes per second
- Module functionality
- System performance

---

## Training AI Agents

### 1. Simple Random Agent (No GPU)

Start with a basic agent to verify everything works:

```bash
# Train simple random agent
python3.6 train_ai.py
```

**What happens:**
- Runs 500 episodes
- Random action selection
- Tracks average rewards
- ~30-60 seconds training time

**Expected Output:**
```
Episode 100/500 - Avg Reward: 22.5
Episode 200/500 - Avg Reward: 23.1
...
Training complete!
Average reward: ~22-25 steps
```

### 2. Deep Q-Network (DQN) with GPU

Train an intelligent agent using deep reinforcement learning:

```bash
# Enable maximum performance mode (recommended)
sudo nvpmodel -m 0
sudo jetson_clocks

# Train DQN agent with CUDA acceleration
python3.6 train_dqn.py
```

**Training Process:**

**Phase 1: Exploration (Episodes 1-200)**
- Random actions to explore environment
- Epsilon: 1.0 → 0.1 (less random over time)
- Building replay memory
- Early rewards: 15-30 steps

**Phase 2: Learning (Episodes 200-400)**
- Network learns from experiences
- Loss decreases
- Epsilon continues decreasing
- Rewards increase: 50-150 steps

**Phase 3: Exploitation (Episodes 400+)**
- Mostly optimal actions
- Low epsilon (0.01)
- Consistent high rewards: 200+ steps
- **Solved when avg reward ≥ 195 over 100 episodes**

**Expected Timeline:**
- Total episodes: 500-1000
- Training time: 5-15 minutes (GPU)
- Convergence: ~300-500 episodes

### 3. Monitor GPU Usage

Open a second terminal:

```bash
# Watch GPU/CPU/Memory in real-time
sudo tegrastats
```

**What to look for:**
- **GPU usage:** Should be 40-90% during training
- **RAM usage:** Watch for memory warnings
- **Temperature:** Keep below 80°C (fan recommended)

### 4. Training Parameters

You can customize training in `train_dqn.py`:

```python
# Network architecture
hidden_size = 128        # Neurons per layer (reduce if OOM)
learning_rate = 0.001    # How fast network learns

# Training hyperparameters
batch_size = 64          # Samples per update (reduce if OOM)
gamma = 0.99            # Future reward discount
epsilon_start = 1.0     # Initial exploration
epsilon_end = 0.01      # Final exploration
epsilon_decay = 0.995   # Exploration decay rate

# Memory
replay_buffer_size = 10000  # Past experiences stored
```

### 5. Compare Training Results

After training both agents, compare their performance:

```bash
# Compare simple vs DQN agents
python3.6 compare_training.py
```

**Expected Output:**
```
============================================================
Comparing AI Performance
============================================================

Testing simple baseline agent...
Simple Agent: 178.4 average reward

Testing trained DQN agent...
DQN Agent:    201.5 average reward

============================================================
✓ DQN is 12.9% better than baseline!
============================================================
```

**What This Means:**
- **Simple agent (~170-180):** Uses basic heuristic (lean opposite to pole angle)
- **Good DQN (195+):** Learned optimal policy, solves the task
- **Bad DQN (<100):** Not learning, needs more training or tuning

**Example Results (After Improvements):**
- Simple agent: **170.8** ✓ Good baseline
- DQN training: **107.9** (with exploration)
- DQN testing: **173.0** ✓ Competitive!
- DQN best: **292.0** ✓ Excellent!

**What this means:** The DQN is learning well! Testing at 173 is close to "solved" (195+).
Train for 500 episodes to reach the goal.

### 6. Save and Load Models

```bash
# Models are automatically saved after training
ls -lh cartpole_dqn.pth

# Load and test a trained model
python3.6 << 'EOF'
import torch
import cartpole

# Load model
model = torch.load('dqn_cartpole.pth')
model.eval()

# Test in environment
env = cartpole.PyCartPole()
# ... test code ...
EOF
```

---

## Performance Optimization

### 1. Enable Maximum Performance

```bash
# Set power mode to maximum (0 = 10W mode)
sudo nvpmodel -m 0

# Enable all CPU cores and max clocks
sudo jetson_clocks

# Verify current mode
sudo nvpmodel -q
```

**Power Modes:**
- **Mode 0:** 10W (all cores, max clocks) - **Use for training**
- **Mode 1:** 5W (2 cores, reduced clocks) - Use for inference

### 2. Monitor System

```bash
# Real-time stats
sudo tegrastats

# Detailed monitoring
sudo jtop  # Install with: sudo -H pip install -U jetson-stats
```

**Key Metrics:**
- **CPU:** 4 cores should show activity
- **GPU:** 40-90% utilization during DQN training
- **RAM:** Watch for swap usage (indicates OOM)
- **Temp:** Keep under 80°C

### 3. Cooling

**Critical for AI training!**

```bash
# Check temperature
cat /sys/devices/virtual/thermal/thermal_zone*/temp

# All temps should be < 80000 (80°C)
```

**Cooling options:**
- Active fan (required for extended training)
- Heatsink (included with kit)
- Good airflow around device

### 4. Reduce Memory Usage (if needed)

If you get Out of Memory (OOM) errors:

**In `train_dqn.py`, reduce:**
```python
batch_size = 32              # Was 64
replay_buffer_size = 5000    # Was 10000
hidden_size = 64             # Was 128
```

**Or add swap space:**
```bash
# Create 4GB swap file
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Verify
free -h
```

### 5. Training Performance Expectations

**CartPole DQN Training Speed:**
- **With GPU:** 100-200 episodes/minute
- **Without GPU (CPU only):** 10-30 episodes/minute
- **Speedup:** 5-10x with CUDA

**Environment Performance:**
- **Rust CartPole:** 1000+ steps/second
- **Python overhead:** ~100-200 episodes/second
- **PyTorch inference:** 50-100 predictions/second (GPU)

---

## Troubleshooting

### CUDA Not Available

**Problem:** `torch.cuda.is_available()` returns `False`

**Solutions:**
```bash
# 1. Check CUDA installation
nvcc --version

# 2. Verify PyTorch is correct wheel
python3.6 -m pip show torch | grep Location
# Should NOT be in /usr/lib (system package)

# 3. Check PyTorch wheel matches your JetPack
unzip -l torch-*.whl | grep _C.cpython
# Should show: cpython-36m (NOT cpython-38!)

# 4. Reinstall PyTorch
python3.6 -m pip uninstall torch
python3.6 -m pip install --user torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# 5. Verify GPU access
ls -l /dev/nvhost-* /dev/nvmap
# Should show accessible device files
```

### Out of Memory During Training

**Problem:** Process killed or CUDA OOM error

**Solutions:**
```bash
# 1. Close other applications
pkill -f chromium
pkill -f firefox

# 2. Reduce training parameters (in train_dqn.py)
# batch_size = 32
# replay_buffer_size = 5000
# hidden_size = 64

# 3. Add swap space (see Performance section)

# 4. Monitor memory
watch -n 1 free -h
```

### Slow Training

**Problem:** Training takes very long

**Solutions:**
```bash
# 1. Enable max performance
sudo nvpmodel -m 0
sudo jetson_clocks

# 2. Verify GPU is being used
sudo tegrastats
# GPU should show >0% usage

# 3. Check Python is not using CPU-only PyTorch
python3.6 -c "import torch; print(torch.cuda.is_available())"

# 4. Reduce unnecessary logging
# Comment out print statements in training loop
```

### Build Failures

**Problem:** `cargo build` fails

**Solutions:**
```bash
# 1. Check disk space
df -h
# Need at least 2GB free

# 2. Verify Rust installation
rustc --version
cargo --version

# 3. Clean and rebuild
cargo clean
export PYO3_PYTHON=/usr/bin/python3.6
cargo build --release --features python

# 4. Check Python dev headers
python3-config --includes
# Should show: -I/usr/include/python3.6m
```

### Import Errors

**Problem:** `import cartpole` fails

**Solutions:**
```bash
# 1. Verify .so file exists
ls -lh cartpole.so

# 2. Check it's the correct file
file cartpole.so
# Should show: ELF 64-bit LSB shared object, ARM aarch64

# 3. Verify Python path
python3.6 -c "import sys; print(sys.path)"
# Current directory '.' should be in path

# 4. Rebuild Python module
cargo build --release --features python
cp target/release/libcartpole_lib.so cartpole.so

# 5. Test import
python3.6 -c "import cartpole; print('OK')"
```

### High Temperature

**Problem:** Jetson overheating (>80°C)

**Solutions:**
```bash
# 1. Check temperature
cat /sys/devices/virtual/thermal/thermal_zone*/temp

# 2. Install active cooling fan (required!)

# 3. Reduce power mode temporarily
sudo nvpmodel -m 1  # 5W mode

# 4. Improve airflow
# - Remove any obstructions
# - Elevate device
# - Use external fan

# 5. Throttle training
# Add delays in training loop if needed
```

---

## Advanced Topics

### 1. Custom Neural Network Architectures

Edit `train_dqn.py` to experiment with different architectures:

```python
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        
        # Original: 2 layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Try 3 layers for more capacity
        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Or use dropout for regularization
        # self.dropout = nn.Dropout(0.2)
```

### 2. Hyperparameter Tuning

Experiment with different values:

```python
# Learning rates to try
learning_rates = [0.0001, 0.001, 0.01]

# Gamma (discount factor)
gammas = [0.95, 0.99, 0.999]

# Network sizes
hidden_sizes = [64, 128, 256]

# Exploration strategies
epsilon_decays = [0.990, 0.995, 0.999]
```

### 3. Alternative Algorithms

**Implement other RL algorithms:**

- **Policy Gradient (REINFORCE)**
  - Direct policy optimization
  - Better for continuous actions
  
- **Actor-Critic (A2C, A3C)**
  - Combines value and policy
  - More stable than pure policy gradient
  
- **Proximal Policy Optimization (PPO)**
  - State-of-the-art RL algorithm
  - Better sample efficiency

### 4. Distributed Training

Use multiple Jetson devices:

```python
# Use PyTorch Distributed
import torch.distributed as dist

# Multi-GPU training (if you have multiple Jetsons)
torch.nn.parallel.DistributedDataParallel
```

### 5. Model Export for Production

```python
# Export to TorchScript for deployment
import torch

model = torch.load('dqn_cartpole.pth')
model.eval()

# Trace the model
example_input = torch.rand(1, 4)
traced_model = torch.jit.trace(model, example_input)

# Save traced model
traced_model.save('dqn_cartpole_traced.pt')

# Use in C++ or other languages
```

### 6. TensorRT Optimization

For even faster inference:

```bash
# Install TensorRT (comes with JetPack)
# Convert PyTorch model to TensorRT engine
# Achieves 2-5x speedup for inference
```

### 7. Real-time Visualization

Add live plotting during training:

```python
import matplotlib.pyplot as plt
plt.ion()  # Interactive mode

# In training loop
plt.plot(episode_rewards)
plt.pause(0.001)
```

### 8. Curriculum Learning

Train on progressively harder tasks:

```python
# Start with easier physics
env.set_params(gravity=5.0)  # Reduced gravity

# Gradually increase difficulty
if avg_reward > 100:
    env.set_params(gravity=9.8)  # Normal gravity
```

---

## Quick Reference

### Essential Commands

```bash
# Setup
source quick_setup.sh

# Build
./build_all.sh

# Test
python3.6 test_setup.py

# Train
python3.6 train_ai.py         # Simple agent
python3.6 train_dqn.py        # DQN with GPU

# Performance
sudo nvpmodel -m 0            # Max performance
sudo jetson_clocks            # Max clocks
sudo tegrastats               # Monitor

# PyTorch CUDA verification (most important!)
python3.6 -c "import torch; print('CUDA:', torch.cuda.is_available())"
# If True, GPU acceleration works! You're ready to train.

# Optional: CUDA compiler check (not needed for training)
nvcc --version || echo "nvcc not in PATH (that's OK)"
```

### File Structure

```
jetson.cartpole/
├── src/                      # Rust source code
│   ├── main.rs              # Visual game
│   ├── cartpole.rs          # Physics engine
│   └── python_bindings.rs   # PyO3 bindings
├── target/release/
│   └── cartpole             # Compiled game
├── cartpole.so              # Python module
├── train_ai.py              # Simple training
├── train_dqn.py             # DQN training
├── test_setup.py            # Performance test
└── dqn_cartpole.pth         # Saved model
```

### Configuration Summary

| Component | Version | Location |
|-----------|---------|----------|
| Python | 3.6.9 | `/usr/bin/python3.6` |
| PyO3 | 0.15.2 | Cargo.toml |
| PyTorch | 1.10.0 | `/mnt/microsd/python-packages/` |
| CUDA | 10.2 | `/usr/local/cuda` |
| Rust | 1.70+ | `/mnt/microsd/cargo/` |
| NumPy | 1.13+ | `/mnt/microsd/python-packages/` |

---

## Performance Benchmarks

### CartPole Environment

| Metric | Value |
|--------|-------|
| Physics steps/sec | 10,000+ |
| Episodes/sec (Python) | 100-200 |
| Episode length (random) | 20-30 steps |
| Episode length (trained) | 200+ steps |

### DQN Training (GPU)

| Phase | Episodes | Time | GPU Usage |
|-------|----------|------|-----------|
| Exploration | 0-200 | 2-4 min | 40-60% |
| Learning | 200-400 | 3-6 min | 60-80% |
| Convergence | 400-500 | 2-3 min | 50-70% |
| **Total** | **500** | **~10 min** | **Average 60%** |

### Resource Usage

| Resource | Training | Inference |
|----------|----------|-----------|
| CPU | 200-300% (3-4 cores) | 50-100% |
| GPU | 40-90% | 10-30% |
| RAM | 1.5-2.5 GB | 1.0-1.5 GB |
| Power | 8-10W | 5-7W |
| Temp | 60-75°C (with fan) | 45-55°C |

---

## Success Criteria

### ✅ System is Working If:

1. **Build succeeds**
   - `cargo build --release --features python` completes
   - `cartpole.so` file exists
   - No errors importing: `python3.6 -c "import cartpole"`

2. **CUDA is enabled**
   - `torch.cuda.is_available()` returns `True`
   - `nvcc --version` shows CUDA 10.2
   - `sudo tegrastats` shows GPU activity during training

3. **Training converges**
   - DQN reaches avg reward > 195 in 100 episodes
   - Training completes in 5-15 minutes
   - Loss decreases over time
   - Epsilon decays to ~0.01

4. **Performance is good**
   - Episodes/second > 50 during training
   - GPU utilization > 40% during training
   - Temperature < 80°C
   - No OOM errors

---

## Support & Resources

### Official Documentation
- [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson)
- [JetPack SDK](https://developer.nvidia.com/embedded/jetpack)
- [PyO3 Guide](https://pyo3.rs/)

### Community
- [Jetson Community Forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems)
- [PyTorch Forums](https://discuss.pytorch.org/)

### This Project
- `JETSON_NANO_GUIDE.md` - Detailed setup guide
- `CHECKLIST.md` - Quick setup checklist
- `QUICKSTART.md` - 5-minute quick start
- `README.md` - Project overview

---

## Summary

You now have a complete CartPole environment with:
- ✅ High-performance Rust physics engine
- ✅ Python bindings for AI development
- ✅ GPU-accelerated PyTorch training
- ✅ DQN agent that learns to balance the pole
- ✅ Real-time visualization and monitoring

**Typical workflow:**
1. `source quick_setup.sh` - Setup environment
2. `./build_all.sh` - Build project (first time)
3. `python3.6 train_dqn.py` - Train AI agent with GPU
4. Monitor with `sudo tegrastats` in another terminal
5. Test trained model and iterate

**Training takes ~10 minutes to solve CartPole with GPU acceleration!**

Happy training! 🚀🤖

---

*Last updated: 2025-12-01*
*CartPole v0.1.0 - Rust + Python + CUDA*

