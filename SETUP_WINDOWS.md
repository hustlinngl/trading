# Adaptive AI Signal Terminal — Windows 0.9.18

## Packaged EXE

The preferred Windows experience is the packaged read-only terminal:

`SakuraSignalTerminal.exe`

Keep `config.yaml` beside the executable. Model/deployment bundles remain external so provenance and compatibility checks continue to work across model generations.

## Source mode

Install Python 3.11+ 64-bit:

```powershell
python -m pip install -e .
python signal_dashboard.py
```

Run validation with:

```powershell
python -m pytest -q
python -m ai_trading_lab.main demo
python -m ai_trading_lab.main doctor
```

For reproducible EXE builds see `BUILD_WINDOWS_EXE.md`.

The terminal is read-only and paper/sandbox-oriented by default. Never commit API keys; use environment variables or GitHub Actions Secrets.
