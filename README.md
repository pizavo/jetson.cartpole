# CartPole Game - Rust Implementation with Python Bindings

A high-performance CartPole game implementation in Rust with Python bindings for AI training and testing on Jetson Nano with CUDA support.

## 🚀 Quick Start

- **Playing the game**: `cargo run --release` (Windows/Linux)
- **Jetson Nano setup**: See **[JETSON_NANO_GUIDE.md](JETSON_NANO_GUIDE.md)** ⭐

## ⚠️ Important: Python Version on Jetson Nano

**Jetson Nano has both Python 2.7 and Python 3 installed:**
- ❌ `python` → Python 2.7 (DON'T USE - too old)
- ✅ `python3` → Python 3.6 (USE THIS for all commands)

**Always use `python3` and `python3 -m pip` on Jetson Nano!**

**Note:** 
- Use `python3 -m pip` instead of `pip3` to avoid wrapper warnings
- This project uses PyO3 0.20.3 (compatible with Python 3.6)
- PyO3 0.21+ requires Python 3.7+ and won't work on Jetson Nano

## Features

- **Visual Game**: Interactive CartPole game using ggez
- **Python Bindings**: Train AI agents using Python with Rust backend
- **High Performance**: Written in Rust for optimal performance
- **AI Ready**: Compatible with AI training frameworks on Jetson Nano with CUDA
- **Cross-platform**: Works on Windows, Linux (including Jetson Nano), and macOS

## Building and Running

### Play the Visual Game (No Python needed)

```bash
cargo run --release
```

**Controls:**
- `LEFT/RIGHT` Arrow Keys: Manual control (push cart left/right)
- `SPACE`: Toggle auto mode (simple AI)
- `R`: Reset the episode
- `ESC`: Quit

### Build for Python AI Training

#### Windows

```bash
# Ensure Python is in your PATH
cargo build --release --features python

# Copy the Python module
copy target\release\cartpole.pyd cartpole.pyd

# Run the training script
python train_ai.py
```

#### Linux / Jetson Nano

```bash
# Install Python development headers
sudo apt-get update
sudo apt-get install python3-dev

# Build with Python support
cargo build --release --features python

# Copy the Python module
cp target/release/libcartpole.so cartpole.so

# Run the training script
python3 train_ai.py
```

## Project Structure

```
cartpole/
├── src/
│   ├── main.rs              # Visual game with ggez
│   ├── lib.rs               # Library exports
│   ├── cartpole.rs          # CartPole environment implementation
│   └── python_bindings.rs   # PyO3 Python bindings
├── train_ai.py              # Example Python training script
├── Cargo.toml               # Rust dependencies
└── README.md                # This file
```

## Using with Python

The Rust CartPole environment can be used from Python for AI training:

```python
import cartpole

# Create environment
env = cartpole.PyCartPole()

# Get environment info
obs_space = env.observation_space()
action_space = env.action_space()

# Training loop
state = env.reset()
done = False
total_reward = 0

while not done:
    action = select_action(state)  # Your AI policy
    state, reward, done = env.step(action)
    total_reward += reward

print(f"Total reward: {total_reward}")
```

### Environment Details

**State Space** (4 dimensions):
- `x`: Cart position (-4.8 to 4.8)
- `x_dot`: Cart velocity
- `theta`: Pole angle in radians (-0.418 to 0.418, ~24 degrees)
- `theta_dot`: Pole angular velocity

**Action Space** (2 discrete actions):
- `0`: Push cart left
- `1`: Push cart right

**Reward**:
- +1 for every step the pole remains balanced
- 0 when the episode ends

**Episode Termination**:
- Pole angle > ±24 degrees
- Cart position > ±2.4 units
- Episode length > 500 steps

## Integration with AI Frameworks

This environment can be easily integrated with popular AI/RL frameworks:

### PyTorch Example

```python
import torch
import torch.nn as nn
import cartpole

env = cartpole.PyCartPole()

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.fc(x)

# Training code...
```

### TensorFlow/Keras Example

```python
import tensorflow as tf
import cartpole

env = cartpole.PyCartPole()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(2, activation='softmax')
])

# Training code...
```

## Jetson Nano CUDA Support

The CartPole environment itself is CPU-based, but you can use it to train neural networks with CUDA:

```python
import torch
import cartpole

# Enable CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = cartpole.PyCartPole()
model = PolicyNetwork().to(device)

# Training loop with GPU acceleration
state = env.reset()
state_tensor = torch.FloatTensor(state).to(device)
# ... continue training on GPU
```

## Performance Notes

- Rust implementation provides ~10-100x faster environment steps compared to Python
- No GIL (Global Interpreter Lock) limitations
- Perfect for large-scale training with thousands of parallel environments
- Optimized physics calculations

## Development

### Running Tests

```bash
cargo test
```

### Building Without Python Support

If you just want the visual game:

```bash
cargo build --release
```

## Dependencies

### Rust Dependencies
- `ggez`: Game framework for visualization
- `rand`: Random number generation
- `pyo3`: Python bindings (optional)
- `serde`: Serialization support

### Python Dependencies (for training)
- `numpy`: Numerical computing
- PyTorch or TensorFlow (optional, for deep RL)

## License

This project is open source. Feel free to use it for learning and experimentation.

## Contributing

Contributions are welcome! Some ideas:
- Add more sophisticated AI examples
- Implement additional classic control environments
- Optimize physics calculations further
- Add reward shaping options
- Create visualization of training progress

## Troubleshooting

### Python module not found

Make sure you:
1. Built with `--features python`
2. Copied the `.pyd` (Windows) or `.so` (Linux) file correctly
3. Renamed `.so` file to `cartpole.so` on Linux
4. Are running Python from the same directory as the module

### CUDA not available on Jetson Nano

```bash
# Check CUDA installation
nvcc --version

# Install PyTorch with CUDA support
# Follow NVIDIA's guide for Jetson Nano
```

### Build errors on Jetson Nano

```bash
# Update Rust
rustup update

# Install build essentials
sudo apt-get install build-essential python3-dev
```

