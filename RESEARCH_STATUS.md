# Research status — 0.9.34

## Product objective

Help a person do something they cannot do reliably by hand: scan a broad market universe, combine many independent data and research checks, and surface only a small set of asset-specific, evidence-qualified decisions. The system's useful output is not a more impressive score; it is a trustworthy `LONG`, `SHORT`, or `FLAT` decision with confidence, expected return, price, and horizon. A correct `FLAT` is useful when evidence is weak, stale, incompatible, or uneconomic.

This is a research and decision-support system, not a claim of durable alpha or an autonomous execution bot. No model, backtest, or recent training run should be described as proof of profitability. The dashboard must make the decision legible and the boundary boring: no internal diagnostics in the public signal, no order endpoint, and no bypass around validation.

## Current engineering state

The repository has purged walk-forward evaluation, chronological holdouts, cost-aware backtesting, model/deployment provenance, guarded paper/live inference, a six-field direct-signal contract, and a read-only dashboard. Research retains labels, fitting, tuning, cost modeling, regimes, analog memory, meta-models, and validation evidence. Deployment-facing behavior is intended to consume compatible bundles and collapse internal results to the direct-signal contract.

One structural gap remains: `live.py` still imports and constructs `AdaptiveEngine`, which owns both research-time fitting and inference-time bundle prediction. The next architecture step is to extract bundle loading, target-free feature generation, prediction, and signal compilation behind an inference-only API, and route both live and paper through that API. This should preserve model semantics and promotion gates; it is an isolation refactor, not a strategy change.

## Five-asset training result

The latest recorded complete training run covered BNB, BTC, ETH, SOL and XRP, with 35,040 15-minute rows per asset. Training completed for all five, but **none passed production readiness**. Every base-model final holdout had zero executed trades and therefore failed holdout trade support, positive-return and profit-factor checks. The duration specialist also failed readiness for all five:

| Asset | Specialist holdout signals | Precision | Ready |
| --- | ---: | ---: | --- |
| BNB/USDT | 10 | 0.800 | No |
| BTC/USDT | 31 | 0.645 | No |
| ETH/USDT | 49 | 0.612 | No |
| SOL/USDT | 44 | 0.455 | No |
| XRP/USDT | 39 | 0.513 | No |

BNB's precision result is below the 12-signal minimum support. No bundle was promoted. Do not loosen gates or present these artifacts as deployment-ready to make the interface appear active; an unready/missing model must remain unavailable for directional inference.

## Research and promotion protocol

1. Download versioned historical data and verify source/provenance.
2. Exclude incomplete candles; validate chronology, gaps, duplicates and OHLC consistency.
3. Build features using only information available at the decision timestamp.
4. Use purged walk-forward out-of-sample evaluation and preserve a chronological final holdout from tuning.
5. Include realistic fees, spread, slippage, impact, borrow and funding where applicable.
6. Compare against simple controls; check placebo/negative controls, stability, drawdown, adequate trade support, and cross-asset/regime behavior.
7. Bind promotion evidence to exact model artifacts, feature schema, dataset and runtime/economic semantics.
8. Promote only when every configured gate passes; autonomous research has no deployment or order authority.

A production-grade market edge remains **unproven** until fresh independent multi-year, multi-asset evidence passes this protocol.