$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$frontendDir = Join-Path $projectRoot "frontend"
$nodeModulesDir = Join-Path $frontendDir "node_modules"

if (-not (Test-Path $frontendDir)) {
    throw "Frontend directory not found at $frontendDir."
}

Set-Location $frontendDir

if (-not (Test-Path $nodeModulesDir)) {
    Write-Host "Installing frontend dependencies..."
    & npm.cmd install
}

Write-Host "Building React frontend..."
& npm.cmd run build
