# Adaptive AI Trading Lab

Release **0.9.34** — research-first platform with a read-only signal terminal.

This repository is a research system for real market data, purged walk-forward validation, pristine holdouts, cost-aware event-driven backtesting, autonomous research and guarded paper/signal workflows. A sophisticated research stack is not evidence of a profitable edge: deployment remains fail-closed until each asset bundle independently clears its promotion gates.

## What is included

- Real-data adapters for Binance Vision and bounded Kraken validation.
- Heterogeneous ML ensemble, regimes, analog memory, meta-policy and conformal uncertainty.
- Purged walk-forward research plus untouched final-holdout governance.
- Economic edge hurdles, impact-aware risk sizing and event-driven execution simulation.
- Autonomous research memory, experience graph, matched-event studies and prequential shadow learning.
- Strict paper/live signal gates that default to WAIT unless evidence, freshness and model-consensus requirements are satisfied.
- Deployment evidence bound to model, runtime/economic semantics, dataset fingerprint and executable artifact hashes; stale or mixed-generation evidence fails closed.
- Deterministic benchmark controls and CI/regression coverage.
- A single read-only `signal_dashboard.py` terminal with public market data and provenance-aware bundle checks.

## Research, inference and the public signal boundary

Research owns labels, fitting, tuning, economic validation and promotion evidence. Live and paper workflows may only consume compatible trained bundles; they must never use research-only targets as live features or weaken readiness gates to create directional outputs. Funding, borrow, correlation, regimes, analog memory, meta-model diagnostics, and validation statistics remain internal evidence, not dashboard fields.

The canonical user-facing output is exactly:

```json
{
  "symbol": "BTC/USDT",
  "signal": "LONG",
  "confidence": 0.87,
  "expected_return": 0.0062,
  "price": 100000.0,
  "horizon_bars": 8
}
```

`src/ai_trading_lab/signal_contract.py` is the allowlisted boundary for compiling live and paper decisions into those six fields. Timestamps, reasons, model diagnostics and provenance belong in private records, never in the direct-signal payload. The web server exposes no order endpoint and does not place trades.

The intended architecture is:

```text
research / training -> validated, versioned bundle -> inference -> signal compiler -> LONG / SHORT / FLAT
```

The boundary is in place, but there is still an architectural coupling to remove: `live.py` currently loads and calls `AdaptiveEngine`, which owns research-time fitting as well as bundle inference. The next refactor should move bundle loading, target-free feature building, prediction and signal compilation behind an inference-only API, then make live and paper both call that API. Keep the six-field contract and all promotion gates unchanged while doing so.

## Single signal terminal

Run the terminal with:

`python signal_dashboard.py`

It binds to localhost by default, opens the browser, discovers exchange markets across configured market types and ranks evaluated asset-specific bundle results. A market without a compatible validated model is not given a borrowed asset model or a directional signal. Live prices can be displayed without a model; LONG/SHORT requires a compatible bundle and every required gate. Headless check: `python signal_dashboard.py --once`.

Windows builds produce `SakuraSignalTerminal.exe`; source launchers also point to this terminal directly.

## Safe operating model

Live order placement is not implemented as an autonomous capability. The default configuration remains paper/sandbox-oriented, and research code never grants itself production-trading authority.

## Common commands

```bash
pip install -e .
# Optional research stack: pip install -e ".[full]"
python -m ai_trading_lab.main doctor
python -m ai_trading_lab.main demo
python -m ai_trading_lab.main benchmark --benchmark-bars 1200
python -m ai_trading_lab.main research
python -m ai_trading_lab.main paper-daemon --cycles 1
```

For Windows EXE packaging see **BUILD_WINDOWS_EXE.md** and **SETUP_WINDOWS.md**.
For long-history experiments see **REAL_DATA_PLAYBOOK.md** and **RESEARCH_STATUS.md**.

### Live dashboard data

Bootstrap closed OHLCV history and launch the read-only terminal:

```powershell
python -m ai_trading_lab.main doctor
python -m ai_trading_lab.main bootstrap-live-data --symbols BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,XRP/USDT --live-bars 600
python signal_dashboard.py
```

By default, `bootstrap-live-data` discovers the full active exchange universe eligible for OHLCV. Use `--symbols` for a smaller subset, or `--market-types spot,swap,future` to control market types. Dashboard quotes follow the exchange-discovered markets; model signals still require compatible asset-specific bundles.
