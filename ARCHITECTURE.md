# Adaptive AI Trading Lab — V6 architecture (0.9.13)

The project is a research system that can grow itself without self-authorizing live trading.

Core flow: market observations → data-quality gate → feature/regime/external state → heterogeneous ML → memory/meta-policy → risk engine → paper decision → counterfactual ledger → realized outcome → growth registry → autonomous research → walk-forward validation → champion/challenger governance.

Deep-evolution components include experience graph, matched-event memory, prequential shadow learning, drift monitoring and an information-per-cost research router.

Idempotence and anti-leakage rules are enforced: no future returns are backfilled as zeros, repeated cycles do not re-assimilate identical timestamps, event controls exclude nearby windows, and shadow learners have no promotion authority.

The 3–24h specialist is an additional signal verifier, not a substitute for the base engine. Live deployment remains paper/sandbox-only until independent evidence gates pass.

The research runner, paper runner and live radar share canonical feature construction and strict signal gating; diagnostic benchmarks are controls, never promotion objectives.
