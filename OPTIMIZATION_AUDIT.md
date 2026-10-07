# Adaptive AI Trading Lab 0.9.16 — autonomous optimization audit

> Historical note: the evidence and synthetic metrics below describe the prior research build. The 0.9.15 pass is primarily an engineering-completion and correctness pass; it does not convert the strategy into a proven profitable system.

## Scope

This pass continues the previous hardening work and focuses on **research validity, promotion governance and economic realism**. The objective was not to maximize the synthetic backtest headline return; it was to remove paths that could overstate a strategy's edge and to make failure visible.

## Corrections applied

1. **Champion promotion now requires an untouched final holdout.** `auto-update` first evaluates the challenger with purged walk-forward evidence, then evaluates the same challenger on the newest chronological holdout. Only when both gates pass is the model refit on all available history and persisted as champion.
2. **Master-tuning no-trade guard hardened.** Completed Optuna trials are identified explicitly, and tuning aborts when there is no completed candidate or when every completed candidate has zero executed trades.
3. **Master-tuning final statistical diagnostic fixed.** The canonical event-driven `run_backtest` is now imported and used for final holdout evidence, so a missing import cannot silently downgrade the report.
4. **Economic edge hurdle added.** Policy decisions now require expected edge to clear estimated round-trip fees + slippage + conservative market impact + a configurable buffer (`min_edge_after_cost_bps`).
5. **Risk sizing now budgets for conservative impact.** The per-trade risk budget includes fee/slippage and a maximum-participation impact approximation.
6. **Execution diagnostics improved.** Trade records now contain entry-equity snapshots and aggregate round-trip notional turnover; exposure is calculated from entry equity rather than exit-time equity.
7. **Regression coverage expanded.** Tests cover holdout-gated promotion, strict master-tuner safeguards, economic edge hurdles, impact-aware risk sizing, exposure accounting, and the existing leakage/execution invariants.

## Verification

- **87/87 pytest tests passed.**
- `compileall` passed.
- `main doctor` passed core readiness checks.
- CLI help loads successfully.
- Demo backtest runs successfully after the changes.
- Project version is synchronized to **0.9.11**.

## Synthetic backtest result

The included synthetic demo remains deliberately pessimistic:

- total return: **-12.10%**
- benchmark return: **+22.65%**
- excess return: **-34.75%**
- max drawdown: **-12.14%**
- trades: **114**
- win rate: **33.33%**
- profit factor: **0.465**
- fees: **379.09**
- slippage + impact: **273.80**
- round-trip turnover: **540,724** notional units

This is a research diagnostic, not evidence of a profitable trading strategy.

## Research conclusion

The current system is substantially harder to fool than before, but **there is still no evidence of a production-grade market edge without versioned real OHLCV and external state data**. The execution container still cannot directly reach Binance Vision. A bounded real-market validation was nevertheless completed through Kraken public data retrieval: 720 completed BTC/USD 4h candles from 2026-06-09 through 2026-10-06 UTC. This validates the real-data path and market-state assumptions, but is not a strategy-performance claim.

The next valid optimization target is therefore not “increase backtest return.” It is:

- real multi-asset historical data;
- purged walk-forward + pristine holdout;
- realistic fee/slippage/impact/borrow assumptions;
- regime-by-regime and asset-by-asset consistency;
- placebo / negative controls;
- parameter-plateau stability;
- sequential promotion logs and model lineage;
- only then, bounded hyperparameter search.

## 0.9.15 engineering completion

The repository-level completion pass fixed the master-tuner fold-cache subset bug, restored the benchmark command, aligned configured economic-edge loading, unified paper/live inference with the canonical engine, activated strict live/paper gates, repaired intensive-workflow shell continuation, made configured holdout cost-stress multipliers effective, added CI and regression coverage, and synchronized release metadata. Full execution of the new suite is delegated to the GitHub Actions CI runner; this environment could inspect and patch the repository but could not execute its complete dependency stack locally.

## Pre-alpha terminal freeze

The final pre-alpha pass keeps the research stack conservative and makes the read-only dashboard boundary explicit. The terminal has no order route, tolerates realtime ticker failure without blocking model signals, restores provenance/evidence rendering from each asset bundle, rejects non-finite JSON values, and avoids stale asynchronous history responses overwriting a newly selected asset.

## Operational state

Paper/sandbox remains the enforced default. A dataset or model that fails the evidence gates should remain **WAIT / research-only** rather than being promoted.

## 0.9.11 real-data layer

- Added public Kraken OHLC/ticker adapter with explicit 720-candle coverage limit.
- Added provenance, stable content fingerprinting and real-market CLI commands.
- Added real-data validation report with a 720-candle BTC/USD 4h window and live ticker checks.
- Binance Vision remains the canonical source for long-history training and reproducible multi-year research.
