# Downloads application_train.csv from the Kaggle competition "home-credit-default-risk".
# BEFORE running, you must do two things yourself:
#   1. Log in at https://www.kaggle.com and open https://www.kaggle.com/c/home-credit-default-risk
#      then click "Join Competition" and accept the rules (otherwise you get a 403 error).
#   2. Create an API token: Kaggle > Settings > API > "Create New Token" (downloads kaggle.json).
#      Put kaggle.json in  C:\Users\<you>\.kaggle\kaggle.json
#      (or set the KAGGLE_USERNAME and KAGGLE_KEY environment variables).
# Run from the project root:
#   powershell -ExecutionPolicy Bypass -File scripts\download_data.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$kaggle = ".\.venv\Scripts\kaggle.exe"
if (-not (Test-Path $kaggle)) { Write-Host "Run scripts\setup.ps1 first." -ForegroundColor Red; exit 1 }

$tokenFile = Join-Path $env:USERPROFILE ".kaggle\kaggle.json"
if (-not (Test-Path $tokenFile) -and -not ($env:KAGGLE_USERNAME -and $env:KAGGLE_KEY)) {
    Write-Host "No Kaggle credentials found. Put kaggle.json in $tokenFile (see comments at top of this script)." -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Force -Path "data\raw" | Out-Null
& $kaggle competitions download -c home-credit-default-risk -f application_train.csv -p data\raw
if ($LASTEXITCODE -ne 0) {
    Write-Host "Download failed. Most common cause: you have not clicked 'Join Competition' and accepted the rules." -ForegroundColor Red
    exit 1
}
$zip = "data\raw\application_train.csv.zip"
if (Test-Path $zip) { Expand-Archive -Path $zip -DestinationPath "data\raw" -Force; Remove-Item $zip }
if (Test-Path "data\raw\application_train.csv") {
    $rows = (Get-Content "data\raw\application_train.csv" | Measure-Object -Line).Lines
    Write-Host "Done. application_train.csv present, lines: $rows (expect about 307512 including header)." -ForegroundColor Green
} else {
    Write-Host "File not found after download. Check the data\raw folder." -ForegroundColor Red
    exit 1
}
