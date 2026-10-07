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

## Live data bootstrap

For the read-only dashboard, public market data does not require exchange API keys. After installing the project, verify the runtime and bootstrap a local closed-candle cache:

```powershell
python -m ai_trading_lab.main doctor
python -m ai_trading_lab.main bootstrap-live-data --symbols BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,XRP/USDT --live-bars 600
python signal_dashboard.py
```

The dashboard refreshes realtime quotes through the configured public exchange adapter. The `bootstrap-live-data` command creates `data/historical/<SYMBOL>_<TIMEFRAME>.csv`; live model inference still requires a compatible trained bundle under `models/assets/<SYMBOL>/`.

