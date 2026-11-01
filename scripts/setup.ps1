# One-time setup for the Fairness-Aware Credit Scoring project (Windows PowerShell)
# Run from the project root:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

Write-Host "== 1/5 Checking Python ==" -ForegroundColor Cyan
$ver = & python --version 2>&1
Write-Host $ver
if ($ver -notmatch "Python 3\.(10|11|12|13)") {
    Write-Host "Need Python 3.10 to 3.13 (3.11 recommended). Install from python.org and re-run." -ForegroundColor Red
    exit 1
}

Write-Host "== 2/5 Creating virtual environment (.venv) ==" -ForegroundColor Cyan
if (-not (Test-Path ".venv\Scripts\python.exe")) { & python -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip

Write-Host "== 3/5 Installing PyTorch (CUDA 12.8 build for RTX 50-series) ==" -ForegroundColor Cyan
& $py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

Write-Host "== 4/5 Installing project requirements ==" -ForegroundColor Cyan
& $py -m pip install -r requirements.txt

Write-Host "== 5/5 Checking GPU ==" -ForegroundColor Cyan
& $py -c "import torch; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

if (-not (Test-Path "data\raw\application_train.csv")) {
    Write-Host ""
    Write-Host "Data file missing: data\raw\application_train.csv" -ForegroundColor Yellow
    Write-Host "Run:  powershell -ExecutionPolicy Bypass -File scripts\download_data.ps1" -ForegroundColor Yellow
} else {
    Write-Host "Data file found." -ForegroundColor Green
}
Write-Host ""
Write-Host "Setup finished. If 'CUDA available' is False, stop and fix that before starting Claude Code." -ForegroundColor Green
