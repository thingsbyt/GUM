@echo off
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0START_GUM_STUDIO.ps1"
if errorlevel 1 (
  echo.
  echo GUM Studio could not start. The message above explains what needs attention.
  pause
)
