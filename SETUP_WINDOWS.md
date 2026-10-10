# Adaptive AI Signal Terminal — Windows Setup (0.9.43)

## Packaged EXE

The preferred Windows experience is the packaged read-only terminal:

`SakuraSignalTerminal.exe`

Keep `config.yaml` beside the executable. Model/deployment bundles remain external so provenance and compatibility checks continue to work across model generations.

## Source mode

Install Python 3.11+ 64-bit:

```powershell
python -m pip install -r requirements.txt
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
python -m ai_trading_lab.main bootstrap-live-data --live-bars 1200 --workers 4
python signal_dashboard.py
```

The dashboard refreshes realtime quotes through the configured public exchange adapter. The `bootstrap-live-data` command creates `data/historical/<SYMBOL>_<TIMEFRAME>.csv`; live model inference still requires a compatible trained bundle under `models/assets/<SYMBOL>/`. Running the command without `--symbols` discovers all active spot/swap/future markets for the configured exchange. It resumes from recent CSVs, writes replacements atomically, and supports bounded parallel workers; it may take substantial time on a full universe. Downloading a market's candles does not create a signal model or make that market signal-eligible.

### Scope and safety

Use `--market-types spot,swap,future` to make the scope explicit or `--symbols BTC/USDT,ETH/USDT` to limit a run. The command defaults to the full active OHLCV universe, caps concurrency at eight workers, resumes sufficiently fresh CSVs and records per-symbol errors in `logs/bootstrap_live_data.json`. Downloading candles does not train a model or make an asset signal-eligible.

