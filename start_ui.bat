@echo off
setlocal
cd /d "%~dp0"
if exist "AdaptiveAITradingLab.exe" (
  start "Adaptive AI Trading Lab" /wait "%~dp0AdaptiveAITradingLab.exe"
  exit /b %errorlevel%
)
echo AdaptiveAITradingLab.exe not found. Falling back to Python launcher...
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 -m streamlit run app.py
  exit /b %errorlevel%
)
python -m streamlit run app.py
endlocal
