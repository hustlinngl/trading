# Repository map and maintainer guide

This is the canonical index for people maintaining the Adaptive AI Signal Terminal. Use it before making architectural changes: the project contains substantial research/compatibility code, but only one supported user-facing terminal.

## Product contract

- **Product:** local, read-only crypto signal terminal. It does not submit exchange orders.
- **Main entry point:** `python signal_dashboard.py`; the Windows equivalent is `SakuraSignalTerminal.exe`.
- **Public signal contract:** a signal is eligible only when data quality/freshness, symbol-specific model provenance, deployment evidence, the base policy and the required 3–24h specialist all agree. Otherwise return `WAIT` / no public signal.
- **Market-data split:** closed OHLCV bars drive model decisions. Realtime ticker/quotes are display-only.
- **Realtime quotes:** a bounded eight-second cache reduces repeated ticker requests; a valid bid/ask midpoint is preferred for display while the last trade remains separately available. Quote caching never participates in model decisions.
- **Outcome reporting:** signal journal returns are OHLC-derived estimates after configured costs, not actual exchange fills or account PnL. Intrabar ambiguity must stay `AMBIGUOUS` with unknown return.
- **Research claim:** software tests and packaged startup success do not establish profitable market edge. Never relax readiness gates just to populate the Top 5.

## Source-of-truth map

| Concern | Canonical files | Responsibility |
|---|---|---|
| User-facing application | `signal_dashboard.py`, `src/ai_trading_lab/dashboard_terminal.py` | Launcher, local read-only routes, HTML/CSS/JS and presentation. Dashboard markup is intentionally centralized in one module for the frozen build. |
| Live universe and signal ranking | `src/ai_trading_lab/live.py` | Market discovery, data-quality checks, per-symbol assessment, strict gate calls, Top 5 and scan telemetry. |
| Model serving boundary | `src/ai_trading_lab/inference.py`, `deployment.py` | Asset-specific bundle loading; model/runtime semantic fingerprints; artifact and holdout provenance checks. |
| Training CLI and asset lifecycle | `src/ai_trading_lab/main.py` | Canonical CLI dispatch, base training, temporal holdout, specialist fitting and per-symbol bundle management. |
| Base model | `engine.py`, `models.py`, `features.py`, `labels.py`, `policy.py`, `meta.py`, `memory.py`, `regimes.py` | Feature construction, forward targets, calibrated model ensemble, memory/regime/meta-policy and candidate actions. |
| Duration/event specialist | `trade_window.py` | 3–24h event/direction prediction, class-prior correction, holdout readiness. Required for public signals by default. |
| Execution/economics | `backtest.py`, `risk.py`, `execution_semantics.py`, `evaluation.py`, `objectives.py` | Entry/exit simulation, sizing, fees/impact/borrow/funding, holdout economics and alignment checks. |
| Journal outcomes | `live_tracker.py`, `live.py` | Persist signals, align history timestamps, resolve later OHLC outcomes and report estimated costs/returns. |
| Data inputs | `data.py`, `dataset.py`, `data_quality.py`, `binance_vision.py`, `kraken_data.py` | Public market data, historical archive windows, normalized datasets and safety/provenance checks. |
| Deployment authorization | `deployment.py`, `promotion.py`, `research_gates.py`, `statistical_evidence.py` | Evidence checks and model eligibility. Training must never grant itself order-placement authority. |
| Automated checks | `tests/`, `.github/workflows/ci.yml` | Regression, integrity, CLI and benchmark smoke checks. |
| Windows release | `.github/workflows/build-windows-exe.yml` | Rebuild historical data, train 15 configured spot assets, build the frozen app, exercise packaged inference/API and publish the portable artifact. |

The tiny wrappers in `src/ai_trading_lab/` and the top-level `run_*.py` scripts are not automatically dead code. Keep them until a reference/compatibility audit proves they can be removed safely.

## Generated files and boundaries

These directories are runtime outputs and must not be committed:

- `data/`: downloaded/derived data and historical OHLCV caches.
- `models/`: trained models, manifests and reports, separated per symbol.
- `logs/`: training diagnostics, mutable application state and the signal journal.
- `build/`, `dist/`, `release/`: packaging outputs.

The repository `.gitignore` excludes these paths. Do not place secrets or personal data in the source tree. Public, unauthenticated data is sufficient for normal terminal display; API keys are not needed for read-only market quotes.

### Model bundle layout

`models/assets/<SYMBOL>/` is the authoritative per-symbol location. Do not let the BTC/global champion bundle substitute for an ETH or other symbol model. A successful training call is not equivalent to `production_ready=true`; deployment manifests must match the data fingerprint, model/runtime semantics, held-out evidence and model artifact hashes.

## Development loop

Use Python 3.11+ in a clean virtual environment:

```bash
python -m pip install -e ".[full]"
python -m compileall -q src signal_dashboard.py
python -m pytest -q
python -m ai_trading_lab.main doctor
python signal_dashboard.py --once
```

Use `python -m ai_trading_lab.main demo` for a deterministic smoke run. The full multi-asset training path is compute-intensive:

```bash
python -m ai_trading_lab.main train-complete-all --config config.yaml --holdout-frac 0.15
```

The Windows workflow intentionally validates that the frozen application starts, serves the required local HTTP endpoints, discovers all 15 asset bundles and emits only signals whose symbols are deployment-ready. A research-only build is allowed when none passes the economic/readiness gates; it must contain zero public LONG/SHORT signals.

## Release checklist

1. Make the smallest isolated change; add a failing regression test for the original defect.
2. Preserve strict no-lookahead, closed-candle, symbol-isolation and signal-only invariants.
3. Run compile/tests, adversarial integrity, CLI and benchmark checks in CI.
4. Keep `VERSION`, `pyproject.toml`, `src/ai_trading_lab/__init__.py`, README and CHANGELOG aligned on release changes.
5. Check the Windows build for the exact intended commit. Do not describe an EXE as ready until packaging, frozen startup and packaged dashboard contract tests pass.
6. Read `MODEL_READINESS.txt`; distinguish “software package succeeded” from “models validated for signal publication.”
7. Revisit generated artifact and log paths before committing so no trained model or market cache enters source control accidentally.

## Documentation index

### Current operating references
- [README](../README.md) — product overview and user-facing signal contract.
- [START_HERE.txt](../START_HERE.txt) — short quick start.
- [Windows setup](../SETUP_WINDOWS.md) — install, launch and data bootstrap.
- [Architecture](../ARCHITECTURE.md) — runtime and model boundaries.
- [Windows build contract](../BUILD_WINDOWS_EXE.md) — package contents, readiness distinction and release verification.
- [Real-data playbook](../REAL_DATA_PLAYBOOK.md) — research data-source hierarchy.

### Design references
- [Alpha UI blueprint](../ALPHA_UI_BLUEPRINT.md) — visual and interaction requirements; not proof that every blueprint idea is implemented.

### Historical research snapshots
- [Optimization audit](../OPTIMIZATION_AUDIT.md) — describes a historical code/evaluation state and historical metrics.
- [Real-data validation](../REAL_DATA_VALIDATION.md) — reports a bounded historical data slice, not current model performance.
- [Research status](../RESEARCH_STATUS.md) — current interpretation/rules; individual archived reports still retain their original dates and datasets.
- [3–24h training note](../TRAINING_3_4H.md) — compact specialist-specific reminder.

Historical numerical results must be read with their original version, asset, window and assumptions. Do not present them as current production evidence.
