$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment Python not found at $pythonExe. Run scripts/start_all.ps1 first."
}

Set-Location $projectRoot
& $pythonExe -m uvicorn api.backend:app --host 127.0.0.1 --port 8000 --reload
