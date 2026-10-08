# Adaptive AI Trading Lab — architecture (0.9.34)

## Product boundary

The product is a quantitative research system whose deployment-facing output is deliberately small. Research complexity must improve evidence quality, not leak into the user interface or gain execution authority.

```text
RESEARCH / TRAINING
  OHLCV + verified auxiliary data
  -> quality checks and target/feature construction
  -> labels, costs, fitting, walk-forward, holdout, DSR/PBO
  -> promotion decision
                 |
                 v
VERSIONED, VALIDATED MODEL BUNDLE
                 |
                 v
INFERENCE
  current closed market data -> target-free features -> model/policy -> signal compiler
                 |
                 v
PUBLIC DIRECT SIGNAL
  symbol | signal | confidence | expected_return | price | horizon_bars
```

## Responsibilities

### Research and training

- Own labels, triple-barrier/event definitions, fitting, optimization, backtesting and validation.
- Treat funding, borrow, correlation, regime, analog-memory and meta-model signals as internal model/economic evidence.
- Preserve purge/embargo, pristine holdout, cost stress and independent promotion gates.
- Write artifacts with enough model, data, semantics and feature-schema provenance to detect incompatible or stale bundles.
- Never promote from a favorable metric alone. A bundle that fails any required gate remains unavailable to live inference.

### Inference

- Load a bundle only after deployment compatibility and readiness checks.
- Build current features from closed market observations only; labels and future returns are not inference inputs.
- Run prediction and safety policy without training, optimization, research persistence, or exchange-order responsibilities.
- Convert the internal decision once through the canonical signal compiler.

### Live and paper adapters

- Fetch or read market observations, verify data quality/freshness, invoke inference, and retain private operational records.
- Live and paper must call the same inference and signal-compilation path so their decisions cannot drift.
- The dashboard gets direct signal fields only. Internal reason codes and diagnostics remain in private records.
- The server is read-only. There is no order endpoint, and autonomous research cannot grant itself trading authority.

## Canonical public schema

The allowlist is exactly six fields:

```text
symbol: str
signal: LONG | SHORT | FLAT
confidence: finite float in [0, 1]
expected_return: finite float
price: finite positive float
horizon_bars: positive integer
```

Do not add timestamps, reason codes, scores, regime, analog/meta/funding/borrow data, validation statistics, or model provenance to the public signal. Such evidence may remain in internal records and logs. Realtime ticker values are display-only; inference must use closed candles.

## Current implementation and remaining coupling

The six-field compiler lives in `src/ai_trading_lab/signal_contract.py`; live and paper outputs use it, and the dashboard serializes `LiveAssessment.to_dict()` rather than its private assessment record. Data and deployment gates fail closed before a directional signal can be published.

One structural gap remains: `src/ai_trading_lab/live.py` still imports `AdaptiveEngine` directly. `AdaptiveEngine` includes fit-time labels, research feature audits, meta-learning, and bundle prediction. Encapsulate inference in a dedicated module that owns bundle load/compatibility, inference feature construction, prediction and compilation; then switch both live and paper to that API. Keep the training code and bundle compatibility behavior unchanged during that extraction, and verify feature parity on fixtures before retiring the live dependency on `AdaptiveEngine`.

## Non-negotiable validation

- A missing, stale, incompatible, or unready bundle produces FLAT/WAIT, never a success-shaped fallback.
- A model must pass all configured out-of-sample, holdout, trade-support, cost and risk gates before promotion.
- Changes to feature formulas or execution semantics require an intentional schema/semantics version change and retraining; old artifacts must fail compatibility rather than inherit new meaning.
- Tests cover target-free inference, identical live/paper decisions, exact public field sets, stale/corrupt bundle rejection, and no order routes.

The research stack may remain complex. The deployment boundary must stay boring, explicit and fail-closed.
