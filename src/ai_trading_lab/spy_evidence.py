"""Evidence gate for the SPY research study, deliberately separate from deployment.

This compares each model's *paired* daily test returns against the same day's
adjusted SPY open-to-open benchmark. A moving-block bootstrap better acknowledges
serial dependence than resampling individual market days. Random-position
placebos detect apparent edges that come merely from trading in rising markets.

This is an exploratory diagnostic, not a proof of profitability or a license to
publish actionable signals. Repeated inspection invalidates pristine holdouts.
"""
from __future__ import annotations

import numpy as np

MIN_TEST_TRADES = 30
BOOTSTRAP_REPS = 500
BLOCK_DAYS = 20


def _array(name: str, values) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite, nonempty 1-D array")
    return a


def study_evidence(
    probabilities,
    future_returns,
    *,
    probability_gate: float = 0.57,
    cost_bps_each_side: float = 7.0,
    trials: int = BOOTSTRAP_REPS,
    block_days: int = BLOCK_DAYS,
) -> dict:
    """Return a stable and *non-authorizing* verdict for a held-out interval.

    No random split, no tuning based on these results, no promotion.
    The buy-and-hold benchmark has one opening and one closing fee, amortized
    here for a paired daily *mean* comparison (not an equity-path forecast).
    """
    p = _array("probabilities", probabilities)
    r = _array("future_returns", future_returns)
    if p.shape != r.shape or not (0.0 < probability_gate < 1.0):
        raise ValueError("Probability and return arrays must align with a valid gate")
    if (p < 0).any() or (p > 1).any() or (r <= -1).any():
        raise ValueError("Invalid probabilities or future returns")
    if cost_bps_each_side < 0 or not np.isfinite(cost_bps_each_side):
        raise ValueError("Costs must be finite and nonnegative")
    if trials < 100 or block_days < 1:
        raise ValueError("Evidence simulations need >=100 trials and positive block length")

    n = len(r)
    selections = p >= probability_gate
    traded = int(np.count_nonzero(selections))
    side_cost = cost_bps_each_side / 10000.0
    # Match the existing study simulator's fixed bps cost approximation.
    candidate = np.where(selections, r - 2.0 * side_cost, 0.0)
    # One round trip for buy & hold, rather than daily round trips.
    benchmark = r - (2.0 * side_cost / n)
    paired_excess = candidate - benchmark
    mean_excess = float(paired_excess.mean())

    rng = np.random.default_rng(240415)
    b = min(block_days, n)
    draws_per_trial = (n + b - 1) // b
    block_starts = rng.integers(0, n - b + 1, size=(trials, draws_per_trial))
    offsets = np.arange(b)
    idx = (block_starts[..., None] + offsets).reshape(trials, -1)[:, :n]
    bootstrap_means = paired_excess[idx].mean(axis=1)
    lower, upper = (float(v) for v in np.quantile(bootstrap_means, [0.05, 0.95]))

    # Placebo with the SAME count of positions and SAME costs; only their
    # timestamps are randomized. It never influences model fitting.
    random_daily = np.empty(trials, dtype=float)
    for k in range(trials):
        mask = np.zeros(n, dtype=bool)
        if traded:
            mask[rng.choice(n, size=traded, replace=False)] = True
        random_daily[k] = float(np.where(mask, r - 2.0 * side_cost, 0.0).mean())
    placebo_p90 = float(np.quantile(random_daily, 0.90))
    observed_daily = float(candidate.mean())
    beats_placebo = observed_daily > placebo_p90
    candidate_compounded = float(np.expm1(np.log1p(candidate).sum()))

    if traded == 0:
        verdict = "NO_TRADES"
        explanation = "The model found no entry that met its pre-registered probability threshold."
    elif traded < MIN_TEST_TRADES:
        verdict = "TOO_FEW_TRADES"
        explanation = f"Only {traded} evaluated trades; at least {MIN_TEST_TRADES} are needed even for preliminary evidence."
    elif lower > 0.0 and beats_placebo and candidate_compounded > 0.0:
        verdict = "PROMISING_BUT_UNCONFIRMED"
        explanation = "Historical excess appears positive, but fresh forward paper evidence is still required."
    elif upper < 0.0:
        verdict = "UNDERPERFORMED_BENCHMARK"
        explanation = "In this historical test, adjusted buy-and-hold consistently outperformed the candidate."
    else:
        verdict = "INCONCLUSIVE"
        explanation = "After costs and dependent observations, there is no convincing advantage over alternatives."

    return {
        "verdict": verdict,
        "explanation": explanation,
        "production_ready": False,
        "requires_forward_paper_validation": True,
        "trades": traded,
        "test_sessions": n,
        "min_trades_required": MIN_TEST_TRADES,
        "mean_daily_excess_return_vs_buy_hold": mean_excess,
        "mean_daily_excess_90pct_block_bootstrap": [lower, upper],
        "block_days": b,
        "bootstrap_trials": int(trials),
        "placebo_random_entries_trials": int(trials),
        "model_mean_daily_return_after_cost": observed_daily,
        "random_timing_p90_mean_daily_return_after_cost": placebo_p90,
        "beats_random_timing_placebo": bool(beats_placebo),
        "caveat": (
            "Exploratory block-bootstrap and shuffled-entry checks on an already viewed "
            "historical test set. They are NOT causal proof, untouched evidence, "
            "actual account returns, or authorization to publish trading signals."
        ),
    }
