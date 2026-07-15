@echo off
cd /d "%~dp0"
set "PYTHONPATH=%CD%\src"
if not exist ".venv\Scripts\python.exe" (
  echo DISHA virtual environment was not found.
  echo Expected: %CD%\.venv\Scripts\python.exe
  pause
  exit /b 1
)
echo Starting DISHA. The first analysis can take about 30 seconds while models load...
".venv\Scripts\python.exe" demo_app.py
if errorlevel 1 pause
