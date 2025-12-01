#!/usr/bin/env python3
"""
Performance test and CUDA verification script for Jetson Nano
"""

import sys
import time

print("=" * 70)
print("CartPole Performance Test and CUDA Verification")
print("=" * 70)
print()

# Test 1: Check CartPole module
print("1. Testing CartPole Module...")
try:
    import cartpole

    print("   ✓ CartPole module imported successfully")

    env = cartpole.PyCartPole()
    state = env.reset()
    print(f"   ✓ Environment initialized: state shape = {len(state)}")

    # Quick test
    for _ in range(10):
        state, reward, done = env.step(1)
        if done:
            break
    print("   ✓ Environment step function works")

except ImportError as e:
    print(f"   ✗ CartPole module not found: {e}")
    print("   Build with: cargo build --release --features python")
    print("   Copy: cp target/release/libcartpole.so cartpole.so")
    sys.exit(1)

print()

# Test 2: Check PyTorch
print("2. Testing PyTorch...")
try:
    import torch

    print(f"   ✓ PyTorch version: {torch.__version__}")

    # Test CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"   {'✓' if cuda_available else '✗'} CUDA available: {cuda_available}")

    if cuda_available:
        print(f"   ✓ CUDA version: {torch.version.cuda}")
        print(f"   ✓ Device name: {torch.cuda.get_device_name(0)}")
        print(f"   ✓ Device count: {torch.cuda.device_count()}")

        # Memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   ✓ GPU memory: {total_memory:.2f} GB")
    else:
        print("   ⚠ CUDA not available - training will use CPU only")

except ImportError:
    print("   ✗ PyTorch not installed")
    print("   Install from: https://forums.developer.nvidia.com/t/pytorch-for-jetson")

print()

# Test 3: Check NumPy
print("3. Testing NumPy...")
try:
    import numpy as np

    print(f"   ✓ NumPy version: {np.__version__}")
except ImportError:
    print("   ✗ NumPy not installed")
    print("   Install with: pip3 install numpy")

print()

# Test 4: Environment Benchmark
print("4. Benchmarking CartPole Environment...")
try:
    env = cartpole.PyCartPole()

    num_episodes = 1000
    start_time = time.time()

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            action = 0 if steps % 2 == 0 else 1  # Alternate actions
            state, reward, done = env.step(action)
            steps += 1

    elapsed = time.time() - start_time
    eps_per_sec = num_episodes / elapsed

    print(f"   ✓ Completed {num_episodes} episodes in {elapsed:.2f}s")
    print(f"   ✓ Performance: {eps_per_sec:.0f} episodes/second")

    if eps_per_sec > 100:
        print("   ✓ Excellent performance!")
    elif eps_per_sec > 50:
        print("   ✓ Good performance")
    else:
        print("   ⚠ Slow performance - consider optimizations")

except Exception as e:
    print(f"   ✗ Benchmark failed: {e}")

print()

# Test 5: GPU Benchmark (if available)
try:
    import torch

    if torch.cuda.is_available():
        print("5. Benchmarking GPU Performance...")

        device = torch.device("cuda")

        # Matrix multiplication benchmark
        size = 1000
        x = torch.randn(size, size, device=device)

        # Warmup
        for _ in range(10):
            y = torch.mm(x, x)
        torch.cuda.synchronize()

        # Actual benchmark
        num_ops = 100
        start_time = time.time()
        for _ in range(num_ops):
            y = torch.mm(x, x)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        ops_per_sec = num_ops / elapsed
        print(f"   ✓ Matrix multiplication: {ops_per_sec:.1f} ops/sec")

        # Memory bandwidth test
        size_mb = 100
        x = torch.randn(size_mb * 1024 * 256, device=device)  # ~100MB

        num_copies = 50
        start_time = time.time()
        for _ in range(num_copies):
            y = x.clone()
        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        bandwidth = (size_mb * num_copies) / elapsed
        print(f"   ✓ Memory bandwidth: {bandwidth:.1f} MB/s")

        print()
    else:
        print("5. GPU Benchmark...")
        print("   ⚠ Skipped (CUDA not available)")
        print()
except Exception as e:
    print(f"   ✗ GPU benchmark failed: {e}")
    print()

# Test 6: System Info
print("6. System Information...")
try:
    import platform

    print(f"   • Platform: {platform.system()} {platform.release()}")
    print(f"   • Architecture: {platform.machine()}")
    print(f"   • Python: {platform.python_version()}")

    # Try to get Jetson info
    try:
        with open('/etc/nv_tegra_release', 'r') as f:
            tegra_info = f.read().strip()
            print(f"   • Tegra: {tegra_info.split(',')[0]}")
    except:
        print("   • Tegra: Not detected (not running on Jetson)")

except Exception as e:
    print(f"   ⚠ Could not get system info: {e}")

print()
print("=" * 70)
print("Test Complete!")
print("=" * 70)
print()

# Summary
print("Summary:")
has_cartpole = 'cartpole' in sys.modules
has_pytorch = 'torch' in sys.modules
has_cuda = has_pytorch and torch.cuda.is_available()

if has_cartpole and has_pytorch and has_cuda:
    print("✓ All systems ready for GPU-accelerated training!")
    print("  Run: python3 train_dqn.py")
elif has_cartpole and has_pytorch:
    print("✓ Ready for CPU training")
    print("  Run: python3 train_dqn.py")
    print("  (Note: Training will be slower without GPU)")
elif has_cartpole:
    print("✓ CartPole ready, but PyTorch missing")
    print("  Run: python3 train_ai.py (simple agent)")
    print("  Install PyTorch for deep learning training")
else:
    print("✗ Setup incomplete")
    print("  Build CartPole: cargo build --release --features python")
    print("  Install PyTorch: see JETSON_NANO_GUIDE.md")

print()
