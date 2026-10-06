# Research status — 0.9.12

## Current state

The repository is connected and writable. The full hardened 0.9.12 codebase is currently still maintained in the working release archive; this repository currently contains the bootstrap/control layer rather than the full source tree.

The research harness is designed to run on an internet-enabled GitHub Actions runner so it can retrieve real historical market data directly.

## Research protocol

1. Download real historical OHLCV from Binance Vision.
2. Remove the live/incomplete candle.
3. Verify chronology, gaps, duplicates, OHLC consistency and provenance.
4. Train only on the past.
5. Use purged walk-forward out-of-sample evaluation.
6. Keep a final chronological holdout untouched.
7. Stress costs at 1x/1.5x/2x/3x.
8. Run bootstrap/evidence diagnostics and benchmark controls.
9. Never auto-promote from this research runner.

## Current evidence

A bounded real-market cross-check was already performed using Kraken public BTC/USD 4h data: 720 completed candles spanning 2026-06-09 through 2026-10-06 UTC. This validates the real-data path but does not establish strategy profitability.

The synthetic benchmark remains negative by design and must not be optimized into a false success.

## Required final gate

A candidate can become a deployment champion only after:
- purged walk-forward evidence;
- pristine final holdout;
- positive economic value after realistic costs;
- minimum trade count;
- acceptable drawdown;
- placebo / negative-control sanity checks;
- stability across assets and regimes.
