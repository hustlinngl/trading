# Adaptive AI Trading Lab

Release **0.9.35** — pre-alpha read-only signal terminal built on the trained research stack.

This repository is a research-first adaptive trading platform for real market data, purged walk-forward validation, governed holdout access, cost-aware event-driven backtesting, autonomous research and guarded paper/signal workflows.

## What is included

- Real-data adapters for Binance Vision and bounded Kraken validation.
- Heterogeneous ML ensemble, regimes, analog memory, meta-policy and conformal uncertainty.
- Purged walk-forward research plus untouched final-holdout governance.
- Economic edge hurdles, impact-aware risk sizing and event-driven execution simulation.
- Autonomous research memory, experience graph, matched-event studies and prequential shadow learning.
- Strict paper/live signal gates that default to WAIT unless evidence, freshness and model-consensus requirements are satisfied.
- Deployment evidence is bound to model, runtime/economic semantics, dataset fingerprint and executable artifact hashes; stale or mixed-generation evidence fails closed.
- Deterministic benchmark controls and CI checks for compile, tests, CLI startup and benchmark execution.
- `signal_dashboard.py` thin read-only launcher backed by a dedicated `ai_trading_lab.dashboard_terminal` module; automatic public-data refresh, provenance-aware bundle checks and prequential signal journal.
- Alpha visual foundation with Sakura Tactical glow, anime operator, animated navigation, decision trace, signal timeline, decision inspector and lightweight ambient effects.

## Single signal terminal

The intended pre-alpha user experience is now one program:

`python signal_dashboard.py`

It binds to localhost only, opens the browser automatically, discovers active exchange markets across the configured market types, evaluates every deployment-eligible asset bundle before ranking the Top 5, applies the same strict live/paper decision gates, records signals and resolves mature signals against later public candles. The web request path stays responsive while a full-universe scan runs, while the Overview stays focused on direct signal results. Top 5 cards are directly inspectable from keyboard or mouse, and the market explorer is populated from the same canonical scan universe. A market without a compatible asset-specific model remains outside the Top 5 rather than borrowing another asset's model. It has no order endpoint and does not place trades.

Headless check: `python signal_dashboard.py --once`

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

The dashboard is a read-only market-data terminal. To make it immediately useful even before model training, bootstrap closed OHLCV history and launch it:

```powershell
python -m ai_trading_lab.main doctor
python -m ai_trading_lab.main bootstrap-live-data --symbols BTC/USDT,ETH/USDT,BNB/USDT,SOL/USDT,XRP/USDT --live-bars 600
python signal_dashboard.py
```

Signals are deliberately fail-closed: live prices can be displayed without a model, but LONG/SHORT results require compatible trained and validated model bundles.

By default, `bootstrap-live-data` now discovers the full active exchange universe instead of only BTC/ETH/SOL. Use `--symbols` for a smaller subset, or `--market-types spot,swap,future` to control the market types downloaded. The dashboard's live quote universe follows the exchange-discovered markets; model signals still require compatible bundles.

