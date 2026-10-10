# Architecture and runtime boundaries — release 0.9.45

The project has two related but distinct paths: **research/training** produces asset-specific, evidence-bound model bundles; the **read-only terminal** consumes only compatible bundles and publishes strict, canonical signals. A successful build is not evidence of a profitable strategy.

## Runtime path

`signal_dashboard.py` is the stable launcher and forwards to `src/ai_trading_lab/dashboard_terminal.py`. The dashboard exposes local read-only HTTP routes. `src/ai_trading_lab/live.py` discovers markets, refreshes closed-candle inputs, evaluates eligible assets and ranks at most five signals. `inference.py` and `deployment.py` enforce artifact identity, model/runtime semantics, provenance fingerprints and holdout evidence. `live_tracker.py` records published signals and resolves their subsequent market outcomes; those returns are estimates from OHLC data, not exchange-executed PnL.

## Research and training path

`main.py` is the canonical CLI entry point. `engine.py` constructs features, targets, regimes, analog memory and the meta-policy; `models.py` trains the base ensemble. `trade_window.py` trains the 3–24h event/direction specialist. `evaluation.py`, `backtest.py`, `execution_semantics.py` and `research_gates.py` provide execution-aligned evaluation, costs and readiness controls. `deployment.py` creates and checks the per-symbol bundle manifest.

## Data and artifact boundaries

- `data/historical/<SYMBOL>_<TIMEFRAME>.csv`: local closed-candle inputs and reproducible training cache.
- `models/assets/<SYMBOL>/`: per-symbol training artifacts, holdout reports and deployment evidence.
- `models/champion/`: compatibility bundle for the configured primary symbol; never use it as a substitute for another asset's model.
- `logs/`: mutable reports, signal journal, training diagnostics and runtime state.
- `config.yaml`: tracked non-secret defaults; API credentials belong in environment variables and must never be committed.

Data, trained bundles and logs are generated outputs and are ignored by Git. Keep the source tree focused on code, tests, configuration and reproducible documentation.

## Reliability rules

1. Model inputs are closed-candle based; the real-time ticker is display-only.
2. A missing, stale, incompatible or unvalidated bundle means `WAIT`, not a guessed signal.
3. Event outcomes that are ambiguous from OHLC must remain ambiguous with unknown return.
4. Include fees, slippage, impact, borrow and applicable funding assumptions in evaluation.
5. Do not lower readiness thresholds to force signals or make a package look successful.
6. The terminal has no order endpoint and does not submit exchange orders.
7. Training diagnostics and execution outcomes must distinguish observed values from estimates.

## Verification

From a Python 3.11+ environment:

```bash
python -m pip install -e ".[full]"
python -m compileall -q src signal_dashboard.py
python -m pytest -q
python -m ai_trading_lab.main doctor
python signal_dashboard.py --once
```

For a full per-symbol training run use `python -m ai_trading_lab.main train-complete-all --config config.yaml --holdout-frac 0.15`; this is compute-intensive and rewrites local generated bundles/reports. The canonical frozen Windows package is produced by `.github/workflows/build-windows-exe.yml` and must pass its EXE and packaged HTTP-surface smoke tests before distribution.
