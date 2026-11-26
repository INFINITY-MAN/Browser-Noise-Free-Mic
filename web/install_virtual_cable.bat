@echo off
setlocal EnableExtensions
title Browser Noise Free Mic - One-Click Virtual Cable Installer

:: Auto-elevate to Administrator (required to install VB-Audio Virtual Cable driver)
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting Administrator privileges to install VB-Audio Virtual Cable...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

echo ====================================================================
echo   Browser Noise Free Mic - One-Click VB-Audio Virtual Cable Setup
echo ====================================================================
echo.

set "WORK_DIR=%TEMP%\BNFM_VBCABLE_Install"
if exist "%WORK_DIR%" rmdir /s /q "%WORK_DIR%"
mkdir "%WORK_DIR%"

echo [1/3] Downloading official VB-Audio Virtual Cable driver from vb-audio.com...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'Stop';" ^
  "$work = '%WORK_DIR%';" ^
  "$zip = Join-Path $work 'VBCABLE.zip';" ^
  "$urls = @('https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack46.zip', 'https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip', 'https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack43.zip');" ^
  "try {" ^
  "  $html = (Invoke-WebRequest -Uri 'https://vb-audio.com/Cable/' -UseBasicParsing -TimeoutSec 15).Content;" ^
  "  if ($html -match 'https://download\.vb-audio\.com/Download_CABLE/VBCABLE_Driver_Pack\d+\.zip') { $urls = @($Matches[0]) + $urls }" ^
  "} catch {}" ^
  "$ok = $false;" ^
  "foreach ($u in ($urls | Select-Object -Unique)) {" ^
  "  try {" ^
  "    Write-Host ('  Trying: ' + $u);" ^
  "    Invoke-WebRequest -Uri $u -OutFile $zip -UseBasicParsing -TimeoutSec 30;" ^
  "    if ((Test-Path $zip) -and ((Get-Item $zip).Length -gt 200000)) { $ok = $true; break }" ^
  "  } catch {}" ^
  "}" ^
  "if (-not $ok) { throw 'Could not download VB-CABLE driver package. Please download manually from https://vb-audio.com/Cable/' }" ^
  "Write-Host '[2/3] Extracting driver package...';" ^
  "Expand-Archive -Path $zip -DestinationPath $work -Force;"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Download or extraction failed. Opening https://vb-audio.com/Cable/ in your browser...
    start "" "https://vb-audio.com/Cable/"
    pause
    exit /b 1
)

echo [3/3] Launching VB-Audio Virtual Cable Installer...
if defined ProgramFiles(x86) (
    set "SETUP_EXE=%WORK_DIR%\VBCABLE_Setup_x64.exe"
) else (
    set "SETUP_EXE=%WORK_DIR%\VBCABLE_Setup.exe"
)

if not exist "%SETUP_EXE%" (
    for /r "%WORK_DIR%" %%F in (VBCABLE_Setup_x64.exe VBCABLE_Setup.exe) do (
        if exist "%%F" set "SETUP_EXE=%%F"
    )
)

echo   Running: "%SETUP_EXE%"
"%SETUP_EXE%" -i -h

echo.
echo ====================================================================
echo   Done! Click "Refresh Devices" in Browser Noise Free Mic
echo   (or reboot Windows if CABLE Input does not appear yet).
echo   Then pick "CABLE Input (VB-Audio Virtual Cable)" as Output,
echo   and "CABLE Output" as your Mic in Zoom / Meet / Discord / OBS.
echo ====================================================================
echo.
pause
