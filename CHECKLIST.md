# Jetson Nano Setup Checklist

Use this checklist when setting up CartPole on your Jetson Nano.

## Pre-Transfer (On Your Windows Machine)

- [x] CartPole game works on Windows (`cargo run --release`)
- [x] All files are ready in `C:\Projects\IDE\RustRover\cartpole`
- [ ] Files transferred to Jetson Nano (via SCP, USB, or git)

## On Jetson Nano - Initial Setup

- [ ] Jetson Nano is powered on and accessible
- [ ] SSH or direct terminal access works
- [ ] Internet connection is available

### 1. Verify Python Version

```bash
cd /mnt/microsd/projects/jetson.cartpole
chmod +x check_python.sh
./check_python.sh

# Verify Python 3.8+
python3 --version
```

**Expected output:**
- ✅ Python 3.8+ found (required for PyO3 0.27.2)

**Action:** Always use `python3 -m pip` for package management!

### 2. Install Rust (if needed)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustc --version
```

**Expected:** Rust 1.70+ installed

### 3. Install System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev build-essential
```

### 4. Build CartPole

```bash
cd ~/cartpole
chmod +x build_jetson.sh
./build_jetson.sh
```

**Expected output:**
- ✓ Visual game built successfully
- ✓ Python module built successfully
- ✓ Python module copied to: cartpole.so

**If build fails:**
- Check: `rustc --version` (Rust installed?)
- Check: `python3-config --includes` (Python dev headers?)
- Check disk space: `df -h`

### 5. Install Python Dependencies

```bash
# Install system packages first (avoids building from source)
sudo apt-get install -y python3-numpy python3-dev libopenblas-base libopenmpi-dev

# Install to user location (microSD if PYTHONUSERBASE is set)
python3 -m pip install --user numpy
```

**If NumPy fails with Cython error:**
```bash
python3 -m pip install --user Cython
python3 -m pip install --user numpy --no-cache-dir
```

**Expected:** NumPy installed successfully

**Note:** Use `python3 -m pip` instead of `pip3` to avoid wrapper warnings.

### 6. Test CartPole Module

```bash
python3 -c "import cartpole; print('Success!')"
```

**Expected:** "Success!" (no errors)

**If import fails:**
- Check: `ls -l cartpole.so` (file exists?)
- Check: `python3 -c "import sys; print(sys.path)"` (current dir in path?)
- Rebuild: `cargo build --release --features python`

### 7. Run Performance Test

```bash
python3 test_setup.py
```

**Expected:**
- ✓ CartPole module works
- ✓ NumPy works
- ✓ Performance: 100+ episodes/second

## Optional: GPU/CUDA Setup (for Deep Learning)

### 8. Verify CUDA

```bash
nvcc --version
```

**Expected:** CUDA 10.2 (or your JetPack version)

**If CUDA not found:**
- JetPack might not be installed
- Check: `dpkg -l | grep nvidia`

### 9. Install PyTorch

Find the correct PyTorch wheel for your JetPack version:
- https://forums.developer.nvidia.com/t/pytorch-for-jetson

```bash
# Example for JetPack 4.6:
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch.whl
python3 -m pip install torch.whl --user
```

### 10. Verify PyTorch with CUDA

```bash
python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
EOF
```

**Expected:**
- PyTorch: 1.10.0 (or your version)
- CUDA: True

**If CUDA: False:**
- Wrong PyTorch version for your JetPack
- Download Jetson-specific wheel from NVIDIA forums

## Training Phase

### 11. Train Simple Agent (No GPU needed)

```bash
python3 train_ai.py
```

**Expected:**
- Training starts
- Episodes complete
- Average reward increases

**Duration:** ~30-60 seconds for 100 episodes

### 13. Train DQN Agent (GPU accelerated)

First, enable max performance:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

Then train:

```bash
python3 train_dqn.py
```

**Monitor in another terminal:**
```bash
sudo tegrastats
```

**Expected:**
- Training starts
- GPU usage visible in tegrastats
- Loss decreases over time
- Epsilon decreases (exploration → exploitation)
- Average reward increases
- Solves in 200-500 episodes

**Duration:** ~3-10 minutes depending on settings

## Troubleshooting Checklist

### Import Error: "No module named cartpole"

- [ ] Check file exists: `ls -l cartpole.so`
- [ ] Rebuild: `cargo build --release --features python`
- [ ] Copy: `cp target/release/libcartpole*.so cartpole.so`
- [ ] Verify: `file cartpole.so` (should say "shared object")

### CUDA Not Available

- [ ] Check CUDA: `nvcc --version`
- [ ] Check PyTorch: `pip3 list | grep torch`
- [ ] Verify correct wheel for your JetPack version
- [ ] Reinstall Jetson-specific PyTorch

### Out of Memory

- [ ] Reduce batch size in `train_dqn.py`: `batch_size = 32`
- [ ] Reduce buffer: `ReplayBuffer(capacity=5000)`
- [ ] Reduce network: `hidden_size = 64`
- [ ] Add swap space (see guide)
- [ ] Close other programs

### Slow Training

- [ ] Enable max performance: `sudo nvpmodel -m 0`
- [ ] Enable max clocks: `sudo jetson_clocks`
- [ ] Check GPU usage: `sudo tegrastats`
- [ ] Verify CUDA is being used: `torch.cuda.is_available()`

### Build Fails

- [ ] Check Rust installed: `rustc --version`
- [ ] Check Python dev: `python3-config --includes`
- [ ] Check disk space: `df -h`
- [ ] Update system: `sudo apt-get update && sudo apt-get upgrade`
- [ ] Install build tools: `sudo apt-get install build-essential`

### NumPy Installation Fails (Cython error)

- [ ] Install system NumPy first: `sudo apt-get install python3-numpy libopenblas-base`
- [ ] Install Cython: `python3 -m pip install --user Cython`
- [ ] Retry NumPy: `python3 -m pip install --user numpy --no-cache-dir`
- [ ] OR use system NumPy (already available via apt)

### Build Fails: "Python interpreter version is lower than PyO3's minimum"

- [ ] Check Python version: `python3 --version` (need 3.7+)
- [ ] PyO3 0.27.2 requires Python 3.7 or newer
- [ ] If Python 3.6, upgrade to Python 3.8+
- [ ] Verify: `python3 -c "import sys; print(sys.version_info)"`

## Success Criteria

You'll know everything is working when:

- ✅ `python3 test_setup.py` shows all green checkmarks
- ✅ `python3 train_ai.py` completes without errors
- ✅ `python3 train_dqn.py` shows CUDA: True and training progresses
- ✅ GPU usage visible in `sudo tegrastats`
- ✅ Average reward increases to 400+ over training
- ✅ Training converges (pole balances consistently)

## Final Notes

**Remember:**
- Always use `python3`, never `python`
- Always use `python3 -m pip`, not `pip3` or `pip` (avoids wrapper warnings)
- Enable max performance before training
- Monitor temperature during training
- Training should complete in 3-10 minutes
- Solved = average reward ≥ 475 for 100 episodes

**Documentation:**
- Setup guide: `JETSON_NANO_GUIDE.md`
- This checklist: `CHECKLIST.md`

**Good luck with your AI training!** 🚀🤖

