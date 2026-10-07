@echo off
setlocal
cd /d "%~dp0"

if exist "SakuraSignalTerminal.exe" (
  start "" "SakuraSignalTerminal.exe"
  exit /b 0
)

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 signal_dashboard.py
  exit /b %errorlevel%
)

python signal_dashboard.py
endlocal
