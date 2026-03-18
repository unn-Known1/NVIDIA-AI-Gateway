@echo off
echo === NVIDIA AI Gateway Installer ===
echo.
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    pause
    exit /b 1
)
python --version
echo.
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Installation complete!
echo.
echo To run the gateway:
echo   venv\Scripts\activate
echo   python -m gateway --port 8080
echo.
pause
