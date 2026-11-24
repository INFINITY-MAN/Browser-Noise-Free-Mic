<#
.SYNOPSIS
    One-command Windows PowerShell launcher for NEON Browser Noise-Free Mic.
.DESCRIPTION
    Creates/activates a local Python virtual environment (.venv), installs pinned
    dependencies from requirements.txt, starts the FastAPI/Uvicorn server on
    http://localhost:8000 with COOP/COEP headers, and opens the default browser.
#>

[CmdletBinding()]
param(
    [int]$Port = 8000,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  NEON — Browser Noise-Free Mic Launcher (Port $Port)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Verify Python installation
$PythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $PythonCmd) {
    $PythonCmd = Get-Command py -ErrorAction SilentlyContinue
}
if (-not $PythonCmd) {
    Write-Error "Python 3.10+ was not found on PATH. Install Python and retry."
    exit 1
}

# 2. Create virtual environment if missing
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Host "[1/4] Creating Python virtual environment at .venv ..." -ForegroundColor Yellow
    & $PythonCmd.Source -m venv $VenvDir
} else {
    Write-Host "[1/4] Using existing virtual environment at .venv" -ForegroundColor Green
}

# 3. Install dependencies
if (-not $SkipInstall) {
    Write-Host "[2/4] Installing dependencies from requirements.txt ..." -ForegroundColor Yellow
    & $VenvPython -m pip install --upgrade pip --quiet
    & $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
} else {
    Write-Host "[2/4] Skipping pip install (-SkipInstall specified)" -ForegroundColor DarkGray
}

# 4. Check for ONNX model artifacts
$ModelPaths = @(
    "exports\gtcrn_int8.onnx",
    "exports\gtcrn_fp32.onnx",
    "web\models\gtcrn_int8.onnx",
    "web\models\gtcrn_fp32.onnx"
)
$FoundModel = $ModelPaths | Where-Object { Test-Path (Join-Path $ProjectRoot $_) } | Select-Object -First 1
if ($FoundModel) {
    Write-Host "[3/4] Found ONNX model artifact: $FoundModel" -ForegroundColor Green
} else {
    Write-Warning "[3/4] No ONNX model found in exports\ or web\models\. UI will load, but inference requires exported ONNX weights."
}

# 5. Launch browser asynchronously once /api/health responds
$Url = "http://localhost:$Port"
Write-Host "[4/4] Starting Uvicorn at $Url (Ctrl+C to stop) ..." -ForegroundColor Green

Start-Job -ScriptBlock {
    param($TargetUrl)
    for ($i = 0; $i -lt 30; $i++) {
        try {
            $resp = Invoke-WebRequest -Uri "$TargetUrl/api/health" -UseBasicParsing -TimeoutSec 2
            if ($resp.StatusCode -eq 200) {
                Start-Process $TargetUrl
                break
            }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
} -ArgumentList $Url | Out-Null

& $VenvPython -m uvicorn backend.main:app --host 127.0.0.1 --port $Port
