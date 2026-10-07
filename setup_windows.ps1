param(
    [switch]$Gpu,
    [switch]$CpuAi
)

$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    $venvCreated = $false

    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            & py -3.11 -c "import sys; assert sys.version_info[:2] == (3, 11)"
            if ($LASTEXITCODE -ne 0) { throw "Python 3.11 unavailable" }
            Write-Host "Creating Python 3.11 virtual environment..."
            & py -3.11 -m venv .venv
            $venvCreated = $true
        }
        catch {
            # Fall through to a plain `python` executable below.
        }
    }

    if (-not $venvCreated) {
        $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
        if (-not $pythonCmd) {
            throw "Python 3.11 was not found. Install Python 3.11, then re-run this script."
        }
        $version = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($version.Trim() -ne "3.11") {
            throw "Found Python $version, but this setup expects Python 3.11. Install Python 3.11 and re-run it."
        }
        Write-Host "Creating Python 3.11 virtual environment..."
        & python -m venv .venv
    }

    $python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    $pip = Join-Path $PSScriptRoot ".venv\Scripts\pip.exe"

    & $python -m pip install --upgrade pip setuptools wheel
    & $pip install -e $PSScriptRoot

    if ($Gpu) {
        Write-Host "Installing CUDA-enabled PyTorch for NVIDIA GPUs..."
        # Official PyTorch 2.12.1 Windows/Linux CUDA 13.0 wheels.
        & $pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130
        # Install LaMa without letting its older dependency metadata downgrade Torch/OpenCV/Pillow.
        & $pip install "fire>=0.5"
        & $pip install simple-lama-inpainting==0.1.2 --no-deps
    }
    elseif ($CpuAi) {
        Write-Host "Installing CPU PyTorch + LaMa..."
        & $pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu
        & $pip install "fire>=0.5"
        & $pip install simple-lama-inpainting==0.1.2 --no-deps
    }

    Write-Host ""
    Write-Host "Installed. Verify it with:"
    Write-Host "  .\wmclean.cmd doctor"
    Write-Host ""
    Write-Host "Example:"
    Write-Host "  .\wmclean.cmd clean .\pages -o .\cleaned --backend auto --save-masks"
}
finally {
    Pop-Location
}
