# CartPole Build Script for Windows
# Run this script to build everything

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "CartPole - Windows Build Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check for Rust
Write-Host "Checking Rust installation..." -ForegroundColor Yellow
if (Get-Command cargo -ErrorAction SilentlyContinue) {
    $rustVersion = cargo --version
    Write-Host "✓ Rust found: $rustVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Rust not found!" -ForegroundColor Red
    Write-Host "  Install from: https://rustup.rs/" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Build the visual game
Write-Host "Building visual game..." -ForegroundColor Yellow
cargo build --release

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Visual game built successfully" -ForegroundColor Green
    Write-Host "  Run with: .\target\release\cartpole.exe" -ForegroundColor Cyan
} else {
    Write-Host "✗ Build failed" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Check for Python
Write-Host "Checking Python installation..." -ForegroundColor Yellow
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonVersion = python --version
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green

    # Build Python module
    Write-Host ""
    Write-Host "Building Python module..." -ForegroundColor Yellow
    cargo build --release --features python

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Python module built successfully" -ForegroundColor Green

        # Copy the module
        if (Test-Path "target\release\cartpole.pyd") {
            Copy-Item "target\release\cartpole.pyd" -Destination "cartpole.pyd" -Force
            Write-Host "✓ Python module copied to: cartpole.pyd" -ForegroundColor Green
        } elseif (Test-Path "target\release\cartpole_lib.pyd") {
            Copy-Item "target\release\cartpole_lib.pyd" -Destination "cartpole.pyd" -Force
            Write-Host "✓ Python module copied to: cartpole.pyd" -ForegroundColor Green
        } else {
            Write-Host "⚠ Warning: Could not find .pyd file" -ForegroundColor Yellow
            Write-Host "  Check target\release\ directory" -ForegroundColor Yellow
        }
    } else {
        Write-Host "⚠ Python build failed (this is optional)" -ForegroundColor Yellow
    }

    # Test Python import
    Write-Host ""
    Write-Host "Testing Python module..." -ForegroundColor Yellow
    $testResult = python -c "import cartpole; print('Success!')" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Python module works!" -ForegroundColor Green
    } else {
        Write-Host "⚠ Python module test failed" -ForegroundColor Yellow
        Write-Host "  This is normal if dependencies are missing" -ForegroundColor Yellow
    }

} else {
    Write-Host "⚠ Python not found" -ForegroundColor Yellow
    Write-Host "  Python bindings will not be built" -ForegroundColor Yellow
    Write-Host "  Install from: https://www.python.org/" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Build Complete!" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Next steps:" -ForegroundColor Green
Write-Host "  1. Run visual game: .\target\release\cartpole.exe" -ForegroundColor White
Write-Host "  2. Test Python: python -c 'import cartpole; print(cartpole)'" -ForegroundColor White
Write-Host "  3. Train AI: python train_ai.py" -ForegroundColor White
Write-Host ""

Write-Host "Controls in game:" -ForegroundColor Green
Write-Host "  LEFT/RIGHT - Move cart" -ForegroundColor White
Write-Host "  SPACE      - Toggle AI mode" -ForegroundColor White
Write-Host "  R          - Reset" -ForegroundColor White
Write-Host "  ESC        - Quit" -ForegroundColor White
Write-Host ""

# Optional: Install Python dependencies
if (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host "Would you like to install Python dependencies? (y/n): " -ForegroundColor Yellow -NoNewline
    $response = Read-Host

    if ($response -eq 'y' -or $response -eq 'Y') {
        Write-Host ""
        Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
        python -m pip install --upgrade pip
        python -m pip install numpy

        Write-Host ""
        Write-Host "Would you like to install PyTorch? (y/n): " -ForegroundColor Yellow -NoNewline
        $pytorchResponse = Read-Host

        if ($pytorchResponse -eq 'y' -or $pytorchResponse -eq 'Y') {
            Write-Host ""
            Write-Host "Installing PyTorch..." -ForegroundColor Yellow
            Write-Host "Visit: https://pytorch.org/get-started/locally/" -ForegroundColor Cyan
            Write-Host "For CUDA support, select your CUDA version" -ForegroundColor Cyan
            Write-Host ""

            # Default CPU-only installation
            python -m pip install torch torchvision torchaudio
        }
    }
}

Write-Host ""
Write-Host "All done! Happy coding! 🚀" -ForegroundColor Green

