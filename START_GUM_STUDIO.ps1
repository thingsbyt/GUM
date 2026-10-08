$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$readyMarker = Join-Path $PSScriptRoot ".venv\gum-ready"

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Preparing GUM's private Python environment (first launch only)..." -ForegroundColor Cyan
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $launcher) {
        & py -3 -m venv .venv
    } else {
        $python = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $python) {
            throw "Python 3.10 or newer is required. Install Python from python.org, then launch again."
        }
        & python -m venv .venv
    }
}

if (-not (Test-Path -LiteralPath $readyMarker)) {
    Write-Host "Installing GUM's declared dependencies (first launch only)..." -ForegroundColor Cyan
    & $venvPython -m pip install --disable-pip-version-check -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }
    New-Item -ItemType File -Path $readyMarker -Force | Out-Null
}

Write-Host "Starting GUM Studio. Close this window to stop it." -ForegroundColor Green
& $venvPython -m gum --workspace .gum-workspace serve --release . --open-browser
