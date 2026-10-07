# Changelog

## 0.9.15
- Deployment readiness now binds model, runtime/economic policy, dataset provenance and executable artifact hashes.
- Incomplete bundles and stale legacy trade-window fallbacks fail closed before inference.
- Global champion promotion removes stale managed artifacts instead of mixing model generations.
- Auto-update cache invalidates on model or deployment semantics drift, not just data changes.
- Meta-policy training uses the full cost hurdle, including impact and directional short borrow.
- Short-side live/tuning conformal gates consistently use the adverse upper bound.

## 0.9.14
- Triple-barrier same-bar TP/SL collisions are now treated as ambiguous rather than directionally bearish.
- OOS feature stitching rejects conflicting overlapping bars instead of silently choosing one source.
- OHLCV fetching/cache paths reject conflicting duplicates and enforce closed-bar semantics consistently.
- Deployment manifests are now cryptographically bound to model semantics and dataset fingerprints.
- Auto-promotion refreshes model provenance and re-trains the duration verifier before a bundle can become signal-eligible.
- Added regression coverage for stale deployment evidence and provenance-bound deployment readiness.

## 0.9.13
- Restored the missing deterministic benchmark module behind the `benchmark` CLI.
- Added regression tests for data timing, risk sizing, execution semantics, configuration and strict signal gating.
- Added a CI workflow covering compile, pytest, CLI startup and benchmark smoke execution.
- Fixed master-tuning fold-cache key handling so subsetted fold caches are deterministic.
- Fixed paper/live inference to use canonical feature construction and the full engine prediction stack.
- Activated strict live/paper quality gates for probability, robust expected return, meta confidence, score, model disagreement and analog-memory support.
- Fixed the configured economic-edge hurdle being silently dropped while loading YAML settings.
- Fixed the intensive research workflow shell continuation and made requested holdout cost multipliers actually run.
- Made challenger promotion bootstrap evidence comparable against the incumbent score.
- Clarified doctor dependency readiness and synchronized the release metadata.

- Directional conformal uncertainty corrected for short-side decisions.
- Promotion no longer uses an invalid frozen-champion historical walk-forward comparison; incumbent comparison is now made on the current untouched holdout.
- Adaptive duration specialist holdout training is properly purged and economically gated.
- Short borrow is included consistently in signal hurdles, risk sizing and cost-stress backtests.
- Historical/archive ingestion rejects conflicting duplicates and malformed OHLCV rows while preserving useful microstructure fields.
- Autonomous research routing is deterministic and external intelligence clients are resilient to missing credentials.
- Model bundles carry symbol, timeframe and training-semantics provenance; live/paper resolution is asset-local and fail-closed.
- Paper/live ledgers are idempotent per completed candle.
- Convenience optimization now preserves a final untouched holdout.
- UI launchers and diagnostics now expose truthful dependency/readiness state.

## 0.9.11
- Real-data Binance Vision and bounded Kraken adapters.
- Strong dataset fingerprints and provenance.
- Purged walk-forward and untouched final-holdout governance.
- Cost-aware execution, impact-aware risk sizing and promotion gates.
- Adversarial integrity checks, placebo controls, DSR/PBO diagnostics.
- Autonomous research, persistent experience graph, causal-style matching and prequential shadow learning.
- Paper/sandbox remains the default; no production edge is claimed.
