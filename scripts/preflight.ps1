# Preflight checks before handing the project to Claude Code. Read-only: changes nothing.
Set-Location (Split-Path $PSScriptRoot -Parent)
function Row($name, $ok, $detail) {
    $mark = if ($ok) { "[ OK ]" } else { "[FAIL]" }
    $color = if ($ok) { "Green" } else { "Red" }
    Write-Host ("{0} {1,-28} {2}" -f $mark, $name, $detail) -ForegroundColor $color
}
$py = & python --version 2>&1
Row "Python 3.10-3.13" ($py -match "Python 3\.(10|11|12|13)") "$py"
$git = & git --version 2>&1
Row "git" ($git -match "git version") "$git"
Row "venv" (Test-Path ".venv\Scripts\python.exe") ".venv\Scripts\python.exe"
Row "application_train.csv" (Test-Path "data\raw\application_train.csv") "data\raw\application_train.csv"
$drive = (Get-Item .).PSDrive
$freeGB = [math]::Round($drive.Free / 1GB, 1)
Row "Free disk >= 10 GB" ($freeGB -ge 10) "$freeGB GB free on $($drive.Name):"
try { $smi = & nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1; Row "NVIDIA GPU / driver" ($LASTEXITCODE -eq 0) "$smi" } catch { Row "NVIDIA GPU / driver" $false "nvidia-smi not found" }
if (Test-Path ".venv\Scripts\python.exe") {
    $t = & .\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())" 2>&1
    Row "torch + CUDA" ($t -match "True") "$t"
}
Write-Host ""
Write-Host "Still to do by hand: plug in the laptop and set Sleep to Never (Settings > System > Power)." -ForegroundColor Yellow
