# Adaptive AI Trading Lab — V6 architecture (0.9.35)

The project is a research system that can grow itself without self-authorizing live trading.

The trained event target and runtime execution geometry are locked together: triple-barrier horizon/ATR geometry defines both model supervision and configured stop/target/time-stop behavior. Any mismatch fails closed rather than silently running a different trade definition.

Research multiple-testing exposure is persistent across runs through a ledger of experiment trials and final-holdout accesses. A repeated holdout is diagnostic-only and cannot regain promotion eligibility; DSR/PSR use the cumulative trial count.

Derivative evaluation prices signed funding flows and configured short-borrow costs. Live Top-5 selection applies a pairwise return-correlation cap so model-backed signals are treated as one portfolio rather than five independent bets.

Core flow: market observations → data-quality gate → feature/regime/external state → heterogeneous ML → memory/meta-policy → risk engine → paper decision → counterfactual ledger → realized outcome → growth registry → autonomous research → walk-forward validation → champion/challenger governance.

Deep-evolution components include experience graph, matched-event memory, prequential shadow learning, drift monitoring and an information-per-cost research router.

Idempotence and anti-leakage rules are enforced: no future returns are backfilled as zeros, repeated cycles do not re-assimilate identical timestamps, event controls exclude nearby windows, and shadow learners have no promotion authority.

The 3–24h specialist is an additional signal verifier, not a substitute for the base engine. Live deployment remains paper/sandbox-only until independent evidence gates pass.

The research runner, paper runner and live radar share canonical feature construction and strict signal gating; diagnostic benchmarks are controls, never promotion objectives.

The pre-alpha presentation boundary is the single local `signal_dashboard.py` launcher backed by `ai_trading_lab.dashboard_terminal`: `/api/state` exposes model decisions and provenance, `/api/history` exposes closed-candle OHLC, and `/api/quote` is a display-only realtime ticker path. These routes are intentionally read-only so the alpha UI can evolve without changing model semantics or execution authority.
