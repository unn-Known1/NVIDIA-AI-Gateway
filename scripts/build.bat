@echo off
echo === Building NVIDIA AI Gateway ===
echo.
pip install --upgrade pip -q
pip install pyinstaller -q
pyinstaller --onefile ^
    --name "nvidia-ai-gateway" ^
    ..\src\gateway\__main__.py --distpath ..\dist >nul 2>&1
echo.
echo Build complete! Executable: dist\nvidia-ai-gateway.exe
pause
