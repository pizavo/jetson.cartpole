# PyTorch Installation Issues - RESOLVED

## The Problem

You installed PyTorch 1.10.0 for **Python 3.8** (cp38 wheel), but:

1. `python` command points to Python 2.7
2. `python3` command points to Python 3.6  
3. Your `PYTHONPATH` was set globally to Python 3.8 packages
4. When you run `python` or `python3.6`, they try to load Python 3.8 packages → **SyntaxError**

## The Error Messages Explained

**Python 2.7 error:**
```
SyntaxError: invalid syntax (f-string syntax)
```
→ Python 2.7 doesn't support f-strings (Python 3.6+ feature)

**Python 3.6 error:**
```
SyntaxError: future feature annotations is not defined
```
→ Python 3.6 doesn't support `from __future__ import annotations` (Python 3.7+ feature)

**Python 3.8 error:**
```
Failed to load PyTorch C extensions
```
→ Running from `/mnt/microsd/projects/jetson.cartpole` which might have conflicting files

## The Solution

**Use Python 3.8 ONLY for CartPole and PyTorch work.**

### On Your Jetson Nano - Run This:

```bash
cd /mnt/microsd/projects/jetson.cartpole

# Run the fix script
chmod +x fix_pytorch_install.sh
./fix_pytorch_install.sh

# This will:
# 1. Remove global PYTHONPATH from ~/.bashrc
# 2. Create ~/use_python38.sh activation script
# 3. Test PyTorch with Python 3.8
```

### How to Use Going Forward:

```bash
# Every time you start a new terminal session:
source ~/use_python38.sh

# Then use python3.8 for everything:
python3.8 -c "import torch; print(torch.__version__)"
python3.8 -c "import cartpole; print('OK')"
python3.8 train_ai.py
python3.8 train_dqn.py
```

### Do NOT Use:
- ❌ `python` (Python 2.7 - too old)
- ❌ `python3` (Python 3.6 - incompatible with your packages)
- ❌ `python3.6` (incompatible with PyTorch 1.10.0 cp38 wheel)

### Only Use:
- ✅ `python3.8` (explicitly)

## Why This Happened

When you installed packages with:
```bash
pip3.8 install numpy
pip3.8 install torch-1.10.0-cp38-cp38-linux_aarch64.whl
```

They went to: `/mnt/microsd/python-packages/lib/python3.8/site-packages/`

But your `PYTHONPATH` was set globally in `~/.bashrc`, so ALL Python versions tried to use these Python 3.8-specific packages, causing syntax errors.

## The Fix

The `fix_pytorch_install.sh` script:
1. Removes global `PYTHONPATH` setting
2. Creates `~/use_python38.sh` that sets environment ONLY for Python 3.8
3. You source it when you want to use Python 3.8

This way:
- Python 2.7 and 3.6 work normally (if needed for other tools)
- Python 3.8 has access to your PyTorch/NumPy packages
- No conflicts!

## Test After Fix

```bash
# Activate Python 3.8 environment
source ~/use_python38.sh

# Go to home directory (away from any torch folders)
cd ~

# Test PyTorch
python3.8 -c "import torch; print('PyTorch:', torch.__version__)"

# Test CartPole
cd /mnt/microsd/projects/jetson.cartpole
python3.8 -c "import cartpole; env = cartpole.PyCartPole(); print('CartPole: OK')"

# Run training
python3.8 train_ai.py
```

## Optional: Remove Python 2.7

If you don't need Python 2.7:
```bash
sudo apt remove python2.7
# But be careful - some system tools might still use it
```

---

**TL;DR:** Always use `python3.8` explicitly, and source `~/use_python38.sh` before working with CartPole/PyTorch.

