# Research and validation status — release 0.9.43

## What is established

The repository contains a real-data ingestion path, temporal training/validation, a heterogeneous base model, an event/direction specialist, an event-driven backtester and a strict read-only live signal surface. Current engineering checks include compilation, regression tests, integrity checks, CLI smoke tests and benchmark smoke tests. A successful CI or package build demonstrates software correctness for the covered checks; it does **not** prove a profitable trading edge.

## Required standard for a model to publish signals

1. Use closed candles with validated chronology, coverage, OHLC consistency and provenance.
2. Fit only on information available at each decision timestamp, with purging around forward-looking labels.
3. Keep a chronological holdout outside tuning and report its exact time range and data fingerprint.
4. Evaluate the actual deployable policy rather than model probabilities in isolation.
5. Include realistic fees, slippage, impact, short borrow and funding where applicable.
6. Require sufficient sample support, non-negative net economics, acceptable drawdown and robust statistical utility.
7. Bind model, dataset, execution semantics and artifact hashes in deployment evidence.
8. Publish no LONG/SHORT signal unless the current bundle passes every required gate.

The intended behavior under weak evidence is abstention. An empty signal list is a valid—and safer—result than forcing a trade.

## User-facing interpretation

The system is signal-only. Journal returns are estimates resolved from later OHLC bars and configured transaction-cost assumptions; they are not actual fills, account returns or proof of strategy profitability. Historical reports in the repository retain the version and data window they originally evaluated and should not be read as current validation.

## Reproducibility

```bash
python -m compileall -q src signal_dashboard.py
python -m pytest -q
python -m ai_trading_lab.main doctor
python signal_dashboard.py --once
```

For the current runtime map, user setup and Windows packaging contract, see `docs/PROJECT_MAP.md`, `SETUP_WINDOWS.md` and `BUILD_WINDOWS_EXE.md`.
