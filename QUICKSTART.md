# Quick Start Guide

Get up and running with CartPole in 5 minutes!

## Windows (Development)

### 1. Build and Run the Game
```powershell
cargo run --release
```

### 2. Test the Python Module (requires Python)
```powershell
cargo build --release --features python
copy target\release\cartpole.pyd cartpole.pyd
python train_ai.py
```

**Controls in Game:**
- `←/→` Arrow Keys - Move cart left/right
- `SPACE` - Toggle AI mode
- `R` - Reset episode
- `ESC` - Quit

---

## Jetson Nano (AI Training)

### Quick Setup
```bash
# 1. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# 2. Clone/copy project
cd ~/cartpole

# 3. Run automated build
chmod +x build_jetson.sh
./build_jetson.sh
```

### Verify Installation
```bash
python3 test_setup.py
```

### Start Training
```bash
# Simple agent (no PyTorch needed)
python3 train_ai.py

# Deep Q-Learning with GPU (requires PyTorch)
python3 train_dqn.py
```

---

## Project Files

| File | Description |
|------|-------------|
| `cargo run --release` | Run visual game |
| `train_ai.py` | Simple AI training (no deep learning) |
| `train_dqn.py` | DQN training with GPU support |
| `test_setup.py` | Verify installation and benchmark |
| `build_jetson.sh` | Automated Jetson build script |
| `README.md` | Full documentation |
| `JETSON_NANO_GUIDE.md` | Detailed Jetson setup guide |

---

## Common Commands

### Build Commands
```bash
# Build game only (no Python)
cargo build --release

# Build with Python support
cargo build --release --features python

# Run tests
cargo test
```

### Python Module
```bash
# Linux/Jetson
cp target/release/libcartpole.so cartpole.so

# Windows
copy target\release\cartpole.pyd cartpole.pyd
```

### Training
```bash
# Test environment
python3 -c "import cartpole; env = cartpole.PyCartPole(); print('OK')"

# Simple training
python3 train_ai.py

# GPU-accelerated training
python3 train_dqn.py

# Performance test
python3 test_setup.py
```

---

## Jetson Nano Performance Mode

For maximum performance:
```bash
sudo nvpmodel -m 0       # Max power mode
sudo jetson_clocks       # Max clock speeds
sudo tegrastats          # Monitor resources
```

---

## Troubleshooting Quick Fixes

### Build fails on Jetson
```bash
# Add swap space
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Python module not found
```bash
# Linux/Jetson
ls -la target/release/*.so
cp target/release/libcartpole.so cartpole.so

# Windows
dir target\release\*.pyd
copy target\release\cartpole.pyd cartpole.pyd
```

### CUDA not detected
```bash
# Verify CUDA
nvcc --version

# Test PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Out of memory on Jetson
Edit `train_dqn.py`:
```python
# Reduce batch size
self.batch_size = 16  # Line ~88

# Reduce buffer size  
ReplayBuffer(capacity=5000)  # Line ~100

# Reduce network size
DQN(..., hidden_size=64)  # Line ~91
```

---

## Next Steps

1. ✅ **Build the project** - `cargo build --release`
2. ✅ **Run the game** - `cargo run --release`
3. ✅ **Test Python** - `python3 test_setup.py`
4. ✅ **Train simple AI** - `python3 train_ai.py`
5. ✅ **Train DQN** - `python3 train_dqn.py`
6. 🚀 **Experiment** - Modify hyperparameters and algorithms

---

## Resources

- **Full Documentation**: `README.md`
- **Jetson Setup**: `JETSON_NANO_GUIDE.md`
- **Source Code**: `src/` directory
- **Examples**: `train_ai.py`, `train_dqn.py`

---

## Support

Having issues? Check:
1. Run `python3 test_setup.py` to diagnose
2. Review `JETSON_NANO_GUIDE.md` troubleshooting section
3. Verify Python and Rust versions
4. Check NVIDIA Jetson forums for PyTorch issues

**Enjoy training your AI! 🤖🎮**

