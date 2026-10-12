# Sakura Atlas — SPY research contract

The shared ChatGPT conversation described a separate Sakura Atlas desktop prototype,
but no related SPY source, workspace_jobs.py, or SPY_RESEARCH_BRIEF.md was found
on the repository's default branch on 2026-10-12. This is a clean, auditable
first SPY implementation, not a claim to reproduce the uncommitted prototype.

## Operator experience

- Windows: double-click start_atlas.bat. Linux: run bash start_atlas.sh.
- A native desktop Tk application opens; no browser or broker connection.
- Analyze SPY downloads public adjusted daily data from Yahoo, builds a
  strictly chronological research-only candidate, evaluates it after costs,
  and saves a separate immutable local study and model file.
- Previous studies appear in the same desktop window. The fixed theme has
  no personalization controls. Old crypto research code is unaffected.

## Research assumptions and boundaries

- Instrument: SPY ETF, daily regular market bars; price adjustment includes
  dividends and stock splits (Yahoo auto_adjust=True).
- Features known after close(t) only, hypothetical entry at open(t+1) and
  exit at open(t+2). Entry/exit each cost 7 bps by default.
- Temporal 60% train / 20% validation / 20% test. No random splits,
  and no use of holdout labels for fitting. The logistic-regression
  probability threshold is pre-registered at 0.57, not optimized on test.
- The buy-and-hold benchmark uses adjusted prices and one round-trip cost,
  unlike daily entries that pay two fees per trade.
- Each study saves a model, full adjusted CSV snapshot, SHA-256 fingerprints,
  date boundaries, economic report and failure-to-qualify disclosure under
  models/spy_research/ (gitignored).
- Validation/test from existing historical data is **not** independent,
  pristine forward evidence. No study can set production_ready=true.
  Repeating the same experiment does not accumulate learning or establish
  a tradeable edge. No broker APIs or order placement are included.
- Yahoo may fail or adjust past history; a failed retrieval blocks analysis
  and leaves earlier studies intact.

## Work still needed before trusted paper signals

1. Audit data continuity/corporate-action adjustments against independent
   sources; store vendor/version and timestamps for reproducibility.
2. Use a purged forward-chaining study with market-regime tests and independent
   final holdout selection. Do not re-use a viewed final holdout as pristine.
3. Compare stronger but parsimonious approaches to cash, buy-and-hold and
   transactional baselines, including uncertainty and stress costs.
4. Add model registry promotion gates and forward-only paper journaling.
   Do not turn on live signal publication without independent evidence.
5. Consolidate useful modules from the earlier external Sakura Atlas archive
   if and when its code is made available.

## Verification

Run python -m pytest -q tests/test_spy_research.py and
python sakura_atlas.py --check in a Python environment with the
requirements-atlas.txt dependencies installed. Real Yahoo availability is
external and is not implied by these offline tests.
