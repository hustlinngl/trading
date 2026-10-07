# Adaptive AI Trading Lab

Release **0.9.16** — pre-alpha read-only signal terminal built on the trained research stack.

This repository is a research-first adaptive trading platform for real market data, purged walk-forward validation, pristine holdouts, cost-aware event-driven backtesting, autonomous research and guarded paper/signal workflows.

## What is included

- Real-data adapters for Binance Vision and bounded Kraken validation.
- Heterogeneous ML ensemble, regimes, analog memory, meta-policy and conformal uncertainty.
- Purged walk-forward research plus untouched final-holdout governance.
- Economic edge hurdles, impact-aware risk sizing and event-driven execution simulation.
- Autonomous research memory, experience graph, matched-event studies and prequential shadow learning.
- Strict paper/live signal gates that default to WAIT unless evidence, freshness and model-consensus requirements are satisfied.
- Deployment evidence is bound to model, runtime/economic semantics, dataset fingerprint and executable artifact hashes; stale or mixed-generation evidence fails closed.
- Deterministic benchmark controls and CI checks for compile, tests, CLI startup and benchmark execution.
- Single-file `signal_dashboard.py` signal terminal with automatic public-data refresh, provenance-aware bundle checks and prequential signal journal.
- Alpha visual foundation with Sakura Tactical glow, anime cursor, animated navigation, click feedback, decision trace and lightweight ambient effects.

## Single signal terminal

The intended pre-alpha user experience is now one program:

`python signal_dashboard.py`

It binds to localhost only, opens the browser automatically, scans the configured trained assets, applies the same strict live/paper decision gates, records signals and resolves mature signals against later public candles. It has no order endpoint and does not place trades.

Headless check: `python signal_dashboard.py --once`

Windows and shell launchers now point to this terminal directly.

## Safe operating model

Live order placement is not implemented as an autonomous capability. The default configuration remains paper/sandbox-oriented, and research code never grants itself production-trading authority.

## Common commands

```bash
pip install -r requirements.txt
# Dashboard only: pip install -e ".[dashboard]"
python -m ai_trading_lab.main doctor
python -m ai_trading_lab.main demo
python -m ai_trading_lab.main benchmark --benchmark-bars 1200
python -m ai_trading_lab.main research
python -m ai_trading_lab.main paper-daemon --cycles 1
```

For long-history experiments see **REAL_DATA_PLAYBOOK.md** and **RESEARCH_STATUS.md**.
