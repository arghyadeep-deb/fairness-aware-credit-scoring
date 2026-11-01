@echo off
REM Double-click this file. It runs setup, then the preflight checks, and keeps the window open.
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
echo.
powershell -ExecutionPolicy Bypass -File scripts\preflight.ps1
echo.
pause
