# Research status — 0.9.13

## Current state

The repository contains the full research source tree plus the autonomous orchestration, data utilities, model bundles, UI controller, tests and GitHub Actions workflows needed to operate the project coherently.

The system is deliberately research-first. No result in this repository should be interpreted as proof of a durable trading edge, and the live/paper signal path defaults to conservative WAIT behavior.

## Research protocol

1. Download versioned real historical OHLCV.
2. Remove the live/incomplete candle.
3. Verify chronology, gaps, duplicates, OHLC consistency and provenance.
4. Train only on information available at the decision timestamp.
5. Use purged walk-forward out-of-sample evaluation.
6. Keep a chronological final holdout untouched by tuning.
7. Stress fees, slippage, impact and borrow assumptions.
8. Run bootstrap/statistical diagnostics, placebo controls and stability checks.
9. Compare against simple non-ML controls and cross-market validation.
10. Never auto-promote from a research runner without passing all promotion evidence gates.

## Engineering state

The 0.9.13 hardening pass restores the missing benchmark command, fixes master-tuning cache partitioning, repairs paper/live inference, wires strict signal safety gates, makes configured cost-stress multipliers effective, and adds CI/regression coverage.

The current code remains suitable for research and paper/signal operation. A production-grade market edge remains **unproven** until fresh multi-year, multi-asset real-data evidence passes the full protocol.

## Required deployment gate

A candidate must have:
- purged walk-forward evidence;
- a pristine final holdout;
- positive economic value after realistic costs;
- sufficient trade count;
- acceptable drawdown;
- placebo / negative-control sanity checks;
- parameter stability;
- consistent behavior across assets and regimes.

The repository keeps deployment authority separate from autonomous research so learning cannot silently become execution authority.
