$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$venvPath = Join-Path $projectRoot ".venv"
$pythonExe = Join-Path $venvPath "Scripts\python.exe"
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
$envFile = Join-Path $projectRoot ".env"
$envExample = Join-Path $projectRoot ".env.example"
$requirementsFile = Join-Path $projectRoot "requirements.txt"
$frontendDir = Join-Path $projectRoot "frontend"
$frontendPackage = Join-Path $frontendDir "package.json"
$frontendNodeModules = Join-Path $frontendDir "node_modules"
$frontendBuildDir = Join-Path $frontendDir "dist"
$frontendBuildScript = Join-Path $projectRoot "scripts\start_frontend.ps1"
$backendScript = Join-Path $projectRoot "scripts\start_backend.ps1"
$frontendUrl = "http://127.0.0.1:8000"
$backendHost = "127.0.0.1"
$backendPort = "8000"
$pythonBootstrapStamp = Join-Path $venvPath ".requirements.installed"

function Test-VenvHealthy {
    param([string]$PathToPython, [string]$PathToActivate)

    return (Test-Path $PathToPython) -and (Test-Path $PathToActivate)
}

function Test-HttpReady {
    param([string]$Url)

    try {
        $response = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 2
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 500
    } catch {
        return $false
    }
}

function Import-DotEnv {
    param([string]$Path)

    if (-not (Test-Path $Path)) {
        return
    }

    Get-Content $Path | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            return
        }

        $parts = $line.Split("=", 2)
        $name = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        Set-Item -Path "Env:$name" -Value $value
    }
}

Set-Location $projectRoot

if (-not (Test-VenvHealthy -PathToPython $pythonExe -PathToActivate $activateScript)) {
    if (Test-Path $venvPath) {
        $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $backupPath = Join-Path $projectRoot ".venv-broken-$timestamp"
        Write-Host "Existing virtual environment looks incomplete. Moving it to $backupPath"
        Move-Item -LiteralPath $venvPath -Destination $backupPath
    }

    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

if (-not (Test-Path $pythonBootstrapStamp)) {
    Write-Host "Upgrading pip..."
    & $pythonExe -m ensurepip --upgrade
    & $pythonExe -m pip install --upgrade pip

    Write-Host "Installing Python dependencies..."
    & $pythonExe -m pip install -r $requirementsFile
    Set-Content -LiteralPath $pythonBootstrapStamp -Value (Get-Date -Format "o")
} else {
    Write-Host "Python environment already prepared. Skipping dependency reinstall."
}

if (Test-Path $frontendPackage) {
    if (-not (Test-Path $frontendNodeModules)) {
        Write-Host "Installing frontend dependencies..."
        Set-Location $frontendDir
        & npm.cmd install
        Set-Location $projectRoot
    }

    Write-Host "Building React frontend..."
    & powershell -NoProfile -ExecutionPolicy Bypass -File $frontendBuildScript
}

if (-not (Test-Path $envFile) -and (Test-Path $envExample)) {
    Write-Host "Creating .env from .env.example..."
    Copy-Item -LiteralPath $envExample -Destination $envFile
}

Import-DotEnv -Path $envFile
Write-Host "Loaded LLM_PROVIDER=$env:LLM_PROVIDER"
Write-Host "Loaded OLLAMA_BASE_URL=$env:OLLAMA_BASE_URL"
Write-Host "Loaded OLLAMA_MODEL=$env:OLLAMA_MODEL"

Write-Host "Seeding demo data and document index..."
& $pythonExe scripts/seed_data.py

if (Test-HttpReady -Url $frontendUrl) {
    Write-Host "Inventory Pilot AI is already running at $frontendUrl"
} else {
    Write-Host "Starting backend..."
    Start-Process `
        -FilePath "powershell" `
        -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $backendScript `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden

    Write-Host "Waiting for the application to become available..."
    for ($attempt = 0; $attempt -lt 40; $attempt++) {
        Start-Sleep -Seconds 1
        if (Test-HttpReady -Url $frontendUrl) {
            break
        }
        if ($attempt -eq 39) {
            Write-Warning "Application did not respond before timeout. Opening the page anyway."
        }
    }
}

Write-Host "Opening the app in your browser..."
Start-Process $frontendUrl

Write-Host ""
Write-Host "Application is starting."
Write-Host "Frontend: $frontendUrl"
Write-Host "Backend API:  http://$backendHost`:$backendPort"
