# FINAL SETUP GUIDE - Python 3.6 Only

## Current Status

Your Jetson Nano has:
- ✅ `python` → Python 2.7.17 (ignore this, don't use it)
- ✅ `python3.6` → Python 3.6.9 (USE THIS!)
- ❌ Your `PYTHONPATH` has both python3.6 AND python3.8 (MUST FIX!)

## The Problem

Your `PYTHONPATH` is:
```
/mnt/microsd/python-packages/lib/python3.6/site-packages:/mnt/microsd/python-packages/lib/python3.8/site-packages
```

This causes Python 3.6 to try loading Python 3.8 packages, which fail.

## The Fix - Run These Commands

```bash
cd /mnt/microsd/projects/jetson.cartpole

# 1. Fix PYTHONPATH
chmod +x fix_pythonpath.sh
./fix_pythonpath.sh
# Choose 'y' to fix

# 2. Restart terminal or reload
source ~/.bashrc

# 3. Source correct environment
source setup_jetson_env.sh

# 4. Verify PYTHONPATH (should show ONLY python3.6)
echo $PYTHONPATH
# Expected: /mnt/microsd/python-packages/lib/python3.6/site-packages

# 6. Clean rebuild
cargo clean
export PYO3_PYTHON=/usr/bin/python3.6
cargo build --release --features python

# 7. Install PyTorch for Python 3.6
cd /mnt/microsd
wget https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
python3.6 -m pip install --user torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# 8. Test everything
cd ~
python3.6 -c "import numpy; print('NumPy:', numpy.__version__)"
python3.6 -c "import torch; print('PyTorch:', torch.__version__)"

cd /mnt/microsd/projects/jetson.cartpole
python3.6 -c "import cartpole; print('CartPole: OK')"

# 9. Train!
python3.6 train_ai.py
```

## Important Rules

### ✅ Always Use:
```bash
python3.6 train_ai.py
python3.6 train_dqn.py
python3.6 test_setup.py
python3.6 -m pip install --user package_name
```

### ❌ Never Use:
```bash
python train_ai.py      # Wrong - Python 2.7
python3 train_ai.py     # Ambiguous
python3.8 train_ai.py   # Wrong - we don't use Python 3.8
```

## Verification Checklist

After running the fix, verify:

```bash
# 1. PYTHONPATH should ONLY have python3.6
echo $PYTHONPATH
# Should show: /mnt/microsd/python-packages/lib/python3.6/site-packages
# Should NOT show: python3.8

# 2. PYO3_PYTHON should point to python3.6
echo $PYO3_PYTHON
# Should show: /usr/bin/python3.6

# 3. Python 3.6 should be available
python3.6 --version
# Should show: Python 3.6.9

# 4. PyTorch should be for Python 3.6
python3.6 -m pip show torch | grep Location
# Should show: .../python3.6/site-packages

# 5. Check .so file in torch directory
ls -la /mnt/microsd/python-packages/lib/python3.6/site-packages/torch/ | grep _C.cpython
# Should show: _C.cpython-36m-aarch64-linux-gnu.so (cp36m, not cp38!)
```

## If PyTorch Still Fails

If you get "Failed to load PyTorch C extensions":

```bash
# 1. Check what wheel you have
ls -la /mnt/microsd/*.whl

# 2. Verify it's the correct one (should be cp36, not cp38)
unzip -l /mnt/microsd/torch-*.whl | grep _C.cpython
# Should show: cpython-36m (NOT cpython-38)

# 3. If it shows cpython-38, you have the wrong wheel!
# Delete it and download the correct Python 3.6 wheel

# 4. Remove existing PyTorch
rm -rf /mnt/microsd/python-packages/lib/python3.6/site-packages/torch*

# 5. Reinstall with correct wheel
python3.6 -m pip install --user torch-1.10.0-cp36-cp36m-linux_aarch64.whl
```

## Summary

**Key Points:**
1. ✅ Use `python3.6` for everything
2. ✅ PYTHONPATH should ONLY have `python3.6/site-packages`
3. ✅ PyTorch wheel must be `cp36` (for Python 3.6)
4. ✅ Cargo.toml uses PyO3 0.16.6
5. ❌ No Python 3.8 anywhere!

**The `python` command showing Python 2.7.17 is fine - just never use it!**

Run `./fix_pythonpath.sh` now to clean up your environment! 🚀

