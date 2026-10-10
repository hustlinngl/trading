# Changelog

## 0.9.43
- Added a visible Market cockpit universe freshness label (`LIVE`, `CACHED` or `LOCAL`) and used last-known market count for navigation coverage while keeping scan/eligible counts separate.
- Persisted the last authoritative market universe for offline market-selector navigation, with a 30-day expiry, atomic writes and explicit stale/source metadata; saved markets never widen signal eligibility.
- Added an 8-second bounded realtime quote cache to reduce repeated public ticker calls while keeping signal inference on closed candles.
- Realtime display price now prefers a valid bid/ask midpoint and retains the last trade separately; malformed/crossed books fall back safely.
- Corrected short-signal net-return math to use simple PnL over entry notional, matching the event-driven backtester instead of overstating returns via an inverse-price formula.
- Added short win/loss/timeout regression tests and a maintainer-facing source/data map.
- Tightened Git exclusions for generated market data, trained bundles, runtime logs and Python tooling caches.
- Added flushed JSON progress events for each asset and long-running training stage, including elapsed durations, readiness status and immediate per-asset failures.
- Added a regression test for the multi-stage training progress contract.

## 0.9.42
- Fixed signal-history timestamp handoff so historical outcomes resolve against the original market-data candle, including legacy journal entries.
- Corrected historical signal net returns to include configured round-trip market impact and elapsed short-borrow costs.
- Marked outcomes as ambiguous with an unknown return when OHLC data cannot determine barrier ordering; invalid cost inputs fail closed.
- Labeled the dashboard history return as an estimate rather than a realized exchange return, with a regression guard.
- Added regression tests for execution-cost accounting and ambiguous candle outcomes.

## 0.9.41
- Aligned live signal outcome tracking with executable opening-gap, minimum-holding and time-stop semantics.
- Corrected class-balanced trade-window scores for the training class prior before treating them as event probabilities; legacy model bundles are invalidated by the semantic fingerprint.
- Added regression coverage for outcome execution, class-prior correction and release-version consistency.
- Made the Windows package's displayed version derive from VERSION instead of a hard-coded release number and removed a stale version label from the Windows setup guide.

## 0.9.40
- Expanded the model/data defaults to 30k training bars, 1.2k live bars and 15 configured live assets.
- Hardened dashboard presentation around the direct six-field signal contract and removed malformed CSS boundaries.
- Added duplicate-universe and dashboard-markup regression guards.

## 0.9.34
- Hardened full-universe live scanning with bounded parallel assessment workers and an explicit `live_scan_workers` setting.
- Made explicit live symbol scopes authoritative even when exchange-wide discovery is available.
- Fixed realtime canonical pricing to prefer the validated bid/ask midpoint while retaining last-trade price separately.
- Routed Binance public streams to the correct spot, USD-M and COIN-M endpoints; unsupported option symbols now fail cleanly instead of reconnecting forever.
- Rejected out-of-order book updates so stale market events cannot overwrite newer quotes; stale-event drops are observable in tracker state.
- Aligned package, project and release versions at 0.9.34.

## 0.9.33
- Extended live market discovery/quote display from the small configured symbol list to the full active exchange universe.
- `bootstrap-live-data` now discovers all active spot/swap/future markets by default when no explicit `--symbols` list is supplied.
- Added resumable historical bootstrapping: fresh existing CSVs are reused instead of redownloaded.
- Added `--all-symbols` and `--market-types` controls for explicit full-universe history generation.

## 0.9.32
- Fixed the anime cursor click feedback so the click animation can never overwrite the pointer-position transform and jump to the top-left corner.
- Exposed realtime market quotes in `/api/state` even when no model-backed signal is available, so the dashboard still shows useful live market data.
- History loading now prefers fresh local cache, refreshes stale cache from the live exchange, and falls back to bundled history only when the network is unavailable.
- Added `bootstrap-live-data` to populate `data/historical/` with closed OHLCV bars for the configured live symbols.

## 0.9.31
- Final dashboard polish: the primary Top 5 surface is result-first, with restrained card entrance motion, compact confidence visualization and cleaner empty/loading states.
- Removed stale primary-state and legacy metric CSS so the visible dashboard stays compact and quiet.


## 0.9.30
- Fixed the packaged dashboard smoke test to stage `config.yaml` and bundled historical CSVs beside the exact frozen EXE before exercising its localhost history routes.
- Kept packaging validation on the same user-facing one-file binary from build through functional verification.

## 0.9.29
- Simplified the Windows release validation to test the exact user-facing windowed EXE instead of rebuilding a second console twin.
- Functional packaging smoke test now validates the final EXE directly over localhost, including the Top 5 result surface, confidence meter and bundled history for the five packaged markets.

## 0.9.28
- Refined the Overview into a cleaner direct-results surface: compact `Top 5 signals` header, no redundant explanatory copy, and no loading/error status text in the primary feed.
- Added a restrained confidence meter to each signal card using the signal's own probability value; no new status badges or synthetic scoring scales were introduced.
- Added regression assertions for the result-only card presentation and updated packaged release metadata.

## 0.9.27
- Fixed a packaging-blocking dashboard indentation regression.
- Replaced primary scan-progress copy with lightweight skeleton results so the main screen stays signal-focused while data is loading.


## 0.9.26
- Reworked the primary dashboard as a signal-only results surface: no visible system metrics, connection status, universe coverage or deployment badges.
- Simplified the decision inspector to show only signal outputs, supporting metrics and machine reasons.
- Removed redundant diagnostic UI from the main DOM instead of merely hiding it.

## 0.9.25
- Removed redundant Top 5/history capsules and inspector trace pills to reduce dashboard visual noise.
- Replaced document click ripples with bounded primary-pointer ripples, eliminating synthetic `(0, 0)` keyboard-click effects in the top-left corner.
- Added a low-volume Sakura lo-fi ambient soundtrack generated locally with Web Audio, with a visible toggle and persisted preference.

## 0.9.24
- Made offline Market state explicit so stale realtime badges cannot look live after a failed state refresh.
- Added the initial Overview navigation state to the page semantics and regression coverage for truthful offline presentation.

## 0.9.23
- Hardened frontend refresh scheduling to prevent overlapping state requests.
- Made realtime quote updates latest-request-wins when the selected asset changes quickly.
- Added active-navigation semantics and scroll offsets that respect the sticky section bar.

## 0.9.22
- Fixed Radar/Intelligence row activation by using the canonical market list instead of the input element's nonexistent options collection.
- Added Enter/Space activation and focus treatment for interactive market rows.

## 0.9.21
- Fixed the decision inspector lifecycle so it is actually wired, visible on open and correctly hidden on close.
- Added dialog semantics, focus return and keyboard focus trapping for the inspector.
- Redrew Top 5 mini-charts on resize with a frame-coalesced handler to keep the overview crisp.

## 0.9.20
- Refined the Top 5 visual hierarchy with a stronger primary pick, tighter card density and clearer signal accents.
- Made Top 5 cards keyboard-accessible and directly connected them to the decision inspector and market history.
- Added explicit connection status and refresh-busy feedback to make the terminal state legible at a glance.
- Added live scan progress presentation with evaluated/total coverage and a lightweight progress bar.
- Added regression coverage for the polished interaction and status states.
- Restored a functional sticky navigation with an explicit Overview entry; secondary sections reveal on demand instead of remaining decorative/hidden.
- Added restrained section-open transitions and connection-state feedback, with reduced-motion fallbacks.

## 0.9.19
- Full-universe discovery now treats active exchange metadata as authoritative and avoids inventing coverage from stale configured/local symbols when the exchange map is available.
- Signal ranking is normalized to the active probability, robust-edge, score and trade-window gates rather than fixed unrelated scales.
- Dashboard scans no longer hold the state lock across network/model work; repeated state requests remain responsive during a broad scan.
- Market explorer/history routes now use the discovered active market map, with explicit exchange/local-fallback provenance in the state payload.
- Added regressions for non-blocking scans and exchange-authoritative universe coverage.
- Fixed refresh lifecycle wiring so the Top 5 and secondary Market/Journal/Evidence surfaces are rendered from the same completed state.
- Shared the read-only exchange client with outcome tracking to avoid redundant market-client initialization during dashboard scans.
- Market explorer now consumes the canonical scanner universe, including offline/local fallback symbols.
- Added live scan-progress reporting in the UI and a regression test for incremental coverage updates.

## 0.9.18
- Full-universe live scanner: discovers active spot/margin/swap/future/option markets, checks deployment compatibility and ranks the best five model-backed signals.
- Added closed-candle assessment reuse and non-blocking dashboard scans so broad coverage does not freeze the UI.
- Hardened the packaged dashboard with release-time markup validation so malformed critical HTML fails the frozen smoke test instead of reaching users.
- Fixed frozen-mode journal persistence so signal outcomes are read from the same writable state root used to store them.
- Added regression coverage for dashboard markup integrity and frozen journal restoration.

## 0.9.17
- Expanded Alpha art direction with a dedicated anime operator stage, visual signal timeline, and non-blocking decision inspector drawer.
- Upgraded the market cockpit with volume context, subtle close-path bloom, regime HUD labeling and DOM crosshair overlays while preserving closed-candle model semantics.
- Added keyboard-accessible timeline/inspector interactions and maintained reduced-motion fallbacks; no new frontend dependencies were introduced.

## 0.9.16
- Alpha visual foundation added on top of the frozen read-only terminal.
- Expanded the visual layer with ambient Sakura particles, animated decision-state bloom and richer navigation/interaction feedback.
- Added Sakura Tactical presentation layer: anime operator/cursor motif, pink neon glow language, animated HUD sweep and ambient grid.
- Added 60fps-oriented interaction motion: requestAnimationFrame cursor tracking, click ripples, button/menu feedback, signal glow and reduced-motion fallback.
- Added navigable Market / Intelligence / Journal / Evidence menu and click-through asset selection from radar/detail rows.
- Final pre-alpha freeze: restored bundle provenance/evidence into the dashboard payload and kept compatibility checks fail-closed.
- Hardened dashboard JSON serialization, browser escaping and asynchronous asset switching.
- Optimized realtime ticker collection with bulk-fetch fallback while keeping the selected-asset ticker display-only.
- Kept the model decision clock on closed candles and the realtime visual clock independent.
- Added `signal_dashboard.py`, a single local read-only signal terminal driven by the trained model bundles.
- Added fast realtime ticker polling for the selected asset while preserving closed-candle semantics for model signals.
- Added a smooth canvas market cockpit with selectable 120/240/480-bar history, OHLC visualization, realtime price line and historical signal markers.
- Added a pre-alpha signal journal that resolves mature signals against later public candles without placing orders.
- Added dedicated regression tests for terminal state, read-only routing, realtime quotes and historical market data.
- Made the primary launchers start the single signal terminal directly.

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
