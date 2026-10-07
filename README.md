# Adaptive AI Trading Lab

Release **0.9.13** — hardened research, strict paper/live inference and reproducible diagnostics.

This repository is a research-first adaptive trading platform for real market data, purged walk-forward validation, pristine holdouts, cost-aware event-driven backtesting, autonomous research and guarded paper/signal workflows.

## What is included

- Real-data adapters for Binance Vision and bounded Kraken validation.
- Heterogeneous ML ensemble, regimes, analog memory, meta-policy and conformal uncertainty.
- Purged walk-forward research plus untouched final-holdout governance.
- Economic edge hurdles, impact-aware risk sizing and event-driven execution simulation.
- Autonomous research memory, experience graph, matched-event studies and prequential shadow learning.
- Strict paper/live signal gates that default to WAIT unless evidence, freshness and model-consensus requirements are satisfied.
- Deterministic benchmark controls and CI checks for compile, tests, CLI startup and benchmark execution.

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
