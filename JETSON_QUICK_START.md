# Jetson Nano Quick Start - CartPole AI Training

## Critical: Python Version
```bash
# ❌ WRONG - This is Python 2.7 (too old)
python --version
# Python 2.7.17

# ✅ CORRECT - Use Python 3
python3 --version
# Python 3.6.9 or higher
```

**Always use `python3` and `pip3` on Jetson Nano!**

---

## Step-by-Step Setup on Jetson Nano

### 1. Transfer Files to Jetson
```bash
# Option A: Using SCP from your Windows machine
scp -r C:\Projects\IDE\RustRover\cartpole username@jetson-ip:~/

# Option B: USB drive or git clone
```

### 2. Install Rust (if not already installed)
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
rustc --version
```

### 3. Build CartPole
```bash
cd ~/cartpole
chmod +x build_jetson.sh
./build_jetson.sh
```

This will:
- Build the visual game (`target/release/cartpole`)
- Build the Python module (`cartpole.so`)
- Verify CUDA is available

### 4. Install Python Dependencies
```bash
# Install pip3 and dev tools
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev

# Install NumPy
pip3 install numpy --user

# Verify CartPole module works
python3 -c "import cartpole; print('✓ CartPole module loaded!')"
```

### 5. Install PyTorch with CUDA Support
```bash
# For JetPack 4.6 (check your version first)
# Download PyTorch wheel from NVIDIA forums
# https://forums.developer.nvidia.com/t/pytorch-for-jetson

# Example for PyTorch 1.10 on JetPack 4.6:
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl

pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl --user

# Verify PyTorch installation
python3 << 'EOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
EOF
```

Expected output:
```
PyTorch: 1.10.0
CUDA available: True
CUDA version: 10.2
Device: Xavier
```

### 6. Run Performance Test
```bash
python3 test_setup.py
```

This verifies:
- CartPole module works
- PyTorch is installed
- CUDA is available
- Environment performance

### 7. Train Simple AI Agent
```bash
# Simple policy-based agent (no GPU needed)
python3 train_ai.py
```

### 8. Train Deep Q-Learning Agent with CUDA
```bash
# Enable max performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Run DQN training with GPU acceleration
python3 train_dqn.py

# In another terminal, monitor GPU usage
sudo tegrastats
```

---

## Understanding the Training

### What Happens During Training:

1. **Rust CartPole** (`cartpole.so`)
   - Fast physics simulation
   - Runs on CPU
   - Provides state: `[x, x_dot, theta, theta_dot]`

2. **PyTorch Neural Network** (runs on GPU with CUDA)
   - Takes 4 inputs (cart position, velocity, pole angle, angular velocity)
   - Hidden layers (128 neurons each)
   - Outputs 2-3 action values (left, right, no force)

3. **DQN Training Loop**:
   ```
   For each episode:
     Reset environment → state = [0, 0, 0.01, 0]
     
     While not done:
       AI predicts action → Neural net on GPU
       Execute action → Rust physics
       Observe result → state, reward, done
       Store experience → Replay buffer
       Train neural net → GPU backpropagation
   ```

4. **CUDA Acceleration**:
   - Neural network forward pass (predict action)
   - Neural network backward pass (learn from experience)
   - Batch operations on replay buffer

### Training Progress:

```
Episode   10 | Steps:  12 | Avg Reward:  11.5 | Epsilon: 0.95
Episode   20 | Steps:  15 | Avg Reward:  14.2 | Epsilon: 0.90
Episode   50 | Steps:  48 | Avg Reward:  45.3 | Epsilon: 0.78
Episode  100 | Steps: 125 | Avg Reward: 118.7 | Epsilon: 0.60
Episode  500 | Steps: 500 | Avg Reward: 500.0 | Epsilon: 0.01  ← Solved!
```

- **Steps**: How long the pole stayed balanced
- **Avg Reward**: Average of last 10 episodes
- **Epsilon**: Exploration rate (starts high, decreases)
- **Solved**: When average reward ≥ 475 for 100 consecutive episodes

---

## Monitoring Performance

### Check GPU Usage:
```bash
sudo tegrastats
```

Look for:
- `GR3D_FREQ` - GPU frequency (should be 100% during training)
- `RAM` - Memory usage (don't exceed 80%)
- `TEMP` - Temperature (keep under 70°C)

### System Performance:
```bash
# CPU usage
htop

# Disk usage
df -h

# Process info
ps aux | grep python3
```

---

## Troubleshooting

### Problem: "ImportError: No module named cartpole"
```bash
# Check if .so file exists
ls -l cartpole.so

# Rebuild Python module
cargo build --release --features python
cp target/release/libcartpole*.so cartpole.so

# Check Python can see it
python3 -c "import sys; print(sys.path)"
```

### Problem: "CUDA available: False"
```bash
# Check CUDA installation
nvcc --version

# Check if JetPack is installed properly
dpkg -l | grep nvidia

# Reinstall PyTorch for Jetson (see step 5)
```

### Problem: Training is slow
```bash
# Enable max performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Reduce batch size in train_dqn.py
# Edit: self.batch_size = 32  (instead of 64)

# Add swap space if running out of memory
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Problem: Out of memory
```python
# In train_dqn.py, reduce these values:
agent.batch_size = 32          # Smaller batches
agent.memory = ReplayBuffer(capacity=5000)  # Smaller buffer
hidden_size = 64               # Smaller network
```

---

## What You're Testing

This CartPole setup demonstrates:

1. **Rust ↔ Python Interop**: Fast simulation in Rust, AI in Python
2. **CUDA Acceleration**: GPU trains neural networks faster
3. **Reinforcement Learning**: AI learns from trial and error
4. **Jetson Nano Capabilities**: Edge AI device can train models

### Key Metrics:
- **Simulation Speed**: ~100-500 episodes/second (Rust efficiency)
- **Training Speed**: 2-10 seconds per 100 episodes (with CUDA)
- **Convergence**: Usually solves in 200-500 episodes
- **Memory Usage**: ~500MB-1GB (depends on buffer size)

---

## Next Steps After Training

1. **Save trained model**:
   ```python
   torch.save(agent.policy_net.state_dict(), 'cartpole_model.pth')
   ```

2. **Load and test**:
   ```python
   agent.policy_net.load_state_dict(torch.load('cartpole_model.pth'))
   # Run episodes with epsilon=0 (no exploration)
   ```

3. **Visualize training**:
   ```bash
   pip3 install matplotlib --user
   # Plot rewards over episodes
   ```

4. **Try different algorithms**:
   - A3C (Actor-Critic)
   - PPO (Proximal Policy Optimization)
   - Rainbow DQN

---

## Reference: All Commands in Order

```bash
# On Jetson Nano:
cd ~/cartpole
./build_jetson.sh
pip3 install numpy --user
python3 test_setup.py
python3 train_ai.py
python3 train_dqn.py  # Requires PyTorch
```

**That's it! Your AI should now be learning to balance the pole!** 🎮🤖

