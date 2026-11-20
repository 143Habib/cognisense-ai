@echo off
setlocal
cd /d "%~dp0"
title CogniSense AI - Publish to GitHub
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0PUSH_TO_GITHUB.ps1"
if errorlevel 1 (
  echo.
  echo Publishing did not complete. Read the message above.
) else (
  echo.
  echo CogniSense AI was pushed to GitHub.
)
echo.
pause
