# CartPole - Python 3.6 Configuration (Final)

## What Changed

**Reverted everything to Python 3.6** - the stock Python version on Jetson Nano with JetPack 4.6.

## Why Python 3.6?

1. **PyTorch wheels for JetPack 4.6 are compiled for Python 3.6**
2. **Stock configuration** - No need to upgrade Python
3. **Officially supported** - All NVIDIA examples use Python 3.6
4. **Works immediately** - No compilation needed

## Configuration

**Cargo.toml:**
```toml
pyo3 = { version = "0.16.6", features = ["extension-module"], optional = true }
```

**Python Command:**
```bash
python3.6  # Use this everywhere
```

**Environment:**
```bash
export PYO3_PYTHON=/usr/bin/python3.6
export PYTHONUSERBASE="/mnt/microsd/python-packages"
export PYTHONPATH="$PYTHONUSERBASE/lib/python3.6/site-packages:$PYTHONPATH"
```

## On Jetson Nano - Complete Setup

```bash
cd /mnt/microsd/projects/jetson.cartpole

# 1. Source environment
source setup_jetson_env.sh

# 2. Clean and rebuild
cargo clean
./build_all.sh

# 3. Install PyTorch
cd /mnt/microsd
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
python3.6 -m pip install --user torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# 4. Test everything
cd ~
python3.6 -c "import torch; print('PyTorch:', torch.__version__)"
python3.6 -c "import numpy; print('NumPy:', numpy.__version__)"

cd /mnt/microsd/projects/jetson.cartpole
python3.6 -c "import cartpole; print('CartPole: OK')"

# 5. Run training
python3.6 train_ai.py
python3.6 test_setup.py
```

## Files Updated

All scripts now use Python 3.6:
- ✅ Cargo.toml → PyO3 0.16.6
- ✅ setup_jetson_env.sh → Python 3.6 paths
- ✅ build_rust.sh → Python 3.6
- ✅ setup_python.sh → Python 3.6
- ✅ build_all.sh → Python 3.6
- ✅ README.md → Python 3.6 docs
- ✅ JETSON_NANO_GUIDE.md → Python 3.6 docs
- ✅ CHECKLIST.md → Python 3.6 troubleshooting

## Files Removed

Cleaned up all the Python 3.8 confusion:
- ❌ fix_pytorch_install.sh
- ❌ fix_pytorch_complete.sh
- ❌ fix_pytorch_python36.sh
- ❌ fix_python38.sh
- ❌ use_python38_clean.sh
- ❌ use_python38.sh
- ❌ PYTORCH_FIX.md
- ❌ PYTHON_VERSION_DECISION.md

(Use setup_jetson_env.sh for Python 3.6 environment instead)

## Clean Start

Everything is now consistent and simple:
- One Python version: **3.6**
- One PyO3 version: **0.16.6**
- One PyTorch wheel: **cp36**

No more confusion, no more mismatches. Just clean Python 3.6 throughout! 🎉

## Commands to Remember

```bash
# Always use python3.6
python3.6 train_ai.py
python3.6 train_dqn.py
python3.6 -m pip install --user package_name

# Source environment when needed
source setup_jetson_env.sh

# Rebuild if needed
cargo clean
export PYO3_PYTHON=/usr/bin/python3.6
cargo build --release --features python
```

That's it! Simple, clean, and it works. 🚀

