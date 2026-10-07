@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 -m pip install -e ".[dashboard]"
  if errorlevel 1 exit /b %errorlevel%
  py -3 -m streamlit run app.py
  exit /b %errorlevel%
)

python -m pip install -e ".[dashboard]"
if errorlevel 1 exit /b %errorlevel%
python -m streamlit run app.py
endlocal
