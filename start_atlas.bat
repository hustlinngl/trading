@echo off
setlocal
cd /d "%~dp0"

set "BOOTSTRAP_PY=python"
where py >nul 2>nul
if not errorlevel 1 set "BOOTSTRAP_PY=py -3"

set "ATLAS_PY=%~dp0.venv-atlas\Scripts\python.exe"
if not exist "%ATLAS_PY%" (
    echo Creating Sakura Atlas Python environment...
    %BOOTSTRAP_PY% -m venv ".venv-atlas"
    if errorlevel 1 goto failed
)

"%ATLAS_PY%" -c "import numpy, pandas, sklearn, joblib, yfinance" >nul 2>nul
if errorlevel 1 (
    echo Installing first-launch dependencies for Sakura Atlas...
    "%ATLAS_PY%" -m pip install -r "requirements-atlas.txt"
    if errorlevel 1 goto failed
)

"%ATLAS_PY%" "sakura_atlas.py"
if errorlevel 1 goto failed
exit /b 0

:failed
echo.
echo Sakura Atlas could not start. Review the error shown above.
pause
exit /b 1
