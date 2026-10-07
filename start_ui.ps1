Set-Location $PSScriptRoot

if (Test-Path ".\SakuraSignalTerminal.exe") {
    Start-Process ".\SakuraSignalTerminal.exe"
    exit 0
}

python signal_dashboard.py
