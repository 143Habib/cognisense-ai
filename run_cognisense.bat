@echo off
setlocal
cd /d "%~dp0"
title CogniSense AI - Research Prototype

echo.
echo  ==============================================================
echo    C O G N I S E N S E   A I
echo    Cognitive Workload Observatory - MAUS Research Prototype
echo  ==============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating a local Python environment...
    py -3.11 -m venv .venv >nul 2>nul
    if errorlevel 1 (
        echo Python 3.11 launcher was not found. Trying default Python...
        python -m venv .venv
        if errorlevel 1 (
            echo.
            echo ERROR: Python was not found. Install Python 3.11 or 3.12 and try again.
            echo Download: https://www.python.org/downloads/
            pause
            exit /b 1
        )
    )
) else (
    echo [1/3] Existing local environment found.
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo ERROR: Could not activate the local Python environment.
    pause
    exit /b 1
)

echo.
echo [2/3] Checking required packages. First run may take a few minutes...
python -m pip install --disable-pip-version-check --upgrade pip
python -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Package installation failed. Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo [3/3] Starting CogniSense AI...
echo The dashboard should open automatically in your browser.
echo Keep this black window open while using the app.
echo To stop the app, return here and press CTRL+C.
echo.
python server.py
pause
