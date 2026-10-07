@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 -m pip install -e .
  if errorlevel 1 exit /b %errorlevel%
  py -3 signal_dashboard.py
  exit /b %errorlevel%
)

python -m pip install -e .
if errorlevel 1 exit /b %errorlevel%
python signal_dashboard.py
endlocal
