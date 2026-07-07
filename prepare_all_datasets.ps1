# Prepare all public datasets (PowerShell)
$ErrorActionPreference = "Stop"
$Python = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
& $Python src/download_datasets.py --datasets ml-1m ml-20m goodbooks
Write-Host "[prepare_all.ps1] done."
