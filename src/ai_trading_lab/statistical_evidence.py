from __future__ import annotations

from itertools import combinations
import math
import numpy as np
from scipy.stats import norm


def _clean_returns(returns) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    return values[np.isfinite(values)]


def _period_sharpe(returns) -> float:
    """Unannualized per-observation Sharpe; its uncertainty is also per observation."""
    values = _clean_returns(returns)
    if len(values) < 2:
        return 0.0
    sd = float(np.std(values, ddof=1))
    return 0.0 if sd <= 1e-12 else float(np.mean(values) / sd)


def _sr_standard_error(sr: float, skew: float, kurtosis: float, n: int) -> float:
    """Approximate standard error for the per-period Sharpe estimate."""
    if n <= 1:
        return float("inf")
    variance = (
        1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr * sr
    ) / max(n - 1, 1)
    return math.sqrt(max(1e-12, variance))


def _distribution_moments(values: np.ndarray) -> tuple[float, float] | None:
    centered = values - float(np.mean(values))
    m2 = float(np.mean(centered ** 2))
    if not np.isfinite(m2) or m2 <= 1e-18:
        return None
    skew = float(np.mean(centered ** 3) / (m2 ** 1.5))
    kurtosis = float(np.mean(centered ** 4) / (m2 ** 2))
    if not np.isfinite(skew) or not np.isfinite(kurtosis):
        return None
    return skew, kurtosis


def _period_benchmark(benchmark_sr: float, periods_per_year: float) -> float:
    """Convert the public API's annualized benchmark Sharpe to per-period units."""
    annualizer = math.sqrt(max(float(periods_per_year), 1.0))
    return float(benchmark_sr) / annualizer


def probabilistic_sharpe_ratio(
    returns,
    periods_per_year: float = 365.0,
    benchmark_sr: float = 0.0,
) -> float:
    """PSR computed entirely in per-period units.

    periods_per_year is used only to convert an annualized benchmark to the
    matching per-period scale. Changing bar frequency cannot inflate the
    probability by multiplying observed Sharpe by sqrt(periods).
    """
    values = _clean_returns(returns)
    n = len(values)
    if n < 30:
        return 0.5
    moments = _distribution_moments(values)
    if moments is None:
        return 0.5
    sr = _period_sharpe(values)
    skew, kurtosis = moments
    se = _sr_standard_error(sr, skew, kurtosis, n)
    if not np.isfinite(se) or se <= 0.0:
        return 0.5
    benchmark = _period_benchmark(benchmark_sr, periods_per_year)
    return float(np.clip(norm.cdf((sr - benchmark) / se), 0.0, 1.0))


def expected_max_sharpe_under_null(
    trials: int, *, euler_gamma: float = 0.5772156649015329
) -> float:
    """Expected maximum of standard-normal draws (dimensionless)."""
    m = max(1, int(trials))
    if m == 1:
        return 0.0
    q1 = norm.ppf(max(1e-12, 1.0 - 1.0 / m))
    q2 = norm.ppf(max(1e-12, 1.0 - 1.0 / (m * math.e)))
    return float((1.0 - euler_gamma) * q1 + euler_gamma * q2)


def deflated_sharpe_ratio(
    returns,
    trials: int,
    periods_per_year: float = 365.0,
    benchmark_sr: float = 0.0,
) -> float:
    """Deflated Sharpe probability with selection adjustment in consistent units.

    The expected maximum under the null is a standardized normal quantile. It
    must be multiplied by the estimated standard error; adding it directly to
    an annualized Sharpe mixes incompatible units and greatly overstates evidence.
    """
    values = _clean_returns(returns)
    n = len(values)
    if n < 30:
        return 0.5
    moments = _distribution_moments(values)
    if moments is None:
        return 0.5
    sr = _period_sharpe(values)
    skew, kurtosis = moments
    se = _sr_standard_error(sr, skew, kurtosis, n)
    if not np.isfinite(se) or se <= 0.0:
        return 0.5
    benchmark = _period_benchmark(benchmark_sr, periods_per_year)
    selection_penalty = expected_max_sharpe_under_null(max(1, int(trials))) * se
    adjusted_benchmark = benchmark + selection_penalty
    return float(np.clip(norm.cdf((sr - adjusted_benchmark) / se), 0.0, 1.0))


def selection_adjusted_psr(
    returns,
    trials: int,
    periods_per_year: float = 365.0,
    benchmark_sr: float = 0.0,
) -> float:
    return deflated_sharpe_ratio(returns, trials, periods_per_year, benchmark_sr)


def _unavailable_pbo(
    reason: str, *, rows: int, candidates: int, effective_candidates: int = 0
) -> dict:
    return {
        "available": False,
        "reason": reason,
        "pbo": None,
        "combinations": 0,
        "mean_logit_rank": None,
        "rows": int(rows),
        "candidates": int(candidates),
        "effective_candidates": int(effective_candidates),
    }


def combinatorial_pbo(
    score_matrix,
    partitions: int = 8,
    seed: int = 42,
    max_combinations: int = 2000,
) -> dict:
    """Estimate PBO from fold-by-candidate scores; degenerate evidence is unavailable.

    Candidate ranks are computed against candidate-level out-of-sample means,
    not against the flattened row-by-candidate matrix. Ties use midranks. If
    candidates are identical (common when no trial trades), PBO is undefined and
    must not report 0.0 as a successful result.
    """
    x = np.asarray(score_matrix, dtype=float)
    if x.ndim != 2:
        raise ValueError("score_matrix must be 2-D")
    x = np.where(np.isfinite(x), x, np.nan)
    rows, candidates = x.shape
    parts = max(4, int(partitions))
    if rows < max(parts, 6) or candidates < 2:
        return _unavailable_pbo(
            "insufficient_blocks_or_candidates", rows=rows, candidates=candidates
        )

    min_observations = max(3, int(math.ceil(rows * 0.5)))
    coverage = np.sum(np.isfinite(x), axis=0)
    usable = coverage >= min_observations
    x = x[:, usable]
    original_candidates = candidates
    candidates = x.shape[1]
    if candidates < 2:
        return _unavailable_pbo(
            "insufficient_candidate_coverage",
            rows=rows,
            candidates=original_candidates,
            effective_candidates=candidates,
        )

    unique_columns: list[np.ndarray] = []
    for j in range(candidates):
        col = x[:, j]
        duplicate = False
        for prior in unique_columns:
            both = np.isfinite(col) & np.isfinite(prior)
            if (
                int(both.sum()) >= min_observations
                and np.array_equal(np.isfinite(col), np.isfinite(prior))
                and np.allclose(col[both], prior[both], rtol=1e-10, atol=1e-12)
            ):
                duplicate = True
                break
        if not duplicate:
            unique_columns.append(col)
    if len(unique_columns) < 2:
        return _unavailable_pbo(
            "degenerate_identical_candidates",
            rows=rows,
            candidates=original_candidates,
            effective_candidates=len(unique_columns),
        )
    x = np.column_stack(unique_columns)
    candidates = x.shape[1]
    finite_values = x[np.isfinite(x)]
    if not len(finite_values) or float(np.nanstd(x)) <= 1e-12:
        return _unavailable_pbo(
            "degenerate_constant_scores",
            rows=rows,
            candidates=original_candidates,
            effective_candidates=candidates,
        )

    block_ids = np.array_split(np.arange(rows), parts)
    combos = list(combinations(range(parts), parts // 2))
    if len(combos) > int(max_combinations):
        rng = np.random.default_rng(int(seed))
        picks = rng.choice(len(combos), size=int(max_combinations), replace=False)
        combos = [combos[int(i)] for i in np.sort(picks)]

    logits: list[float] = []
    ranks: list[float] = []
    all_blocks = set(range(parts))
    for left_blocks in combos:
        left = set(left_blocks)
        right = all_blocks - left
        train_idx = np.concatenate([block_ids[i] for i in sorted(left)])
        test_idx = np.concatenate([block_ids[i] for i in sorted(right)])
        train = x[train_idx, :]
        test = x[test_idx, :]
        train_counts = np.sum(np.isfinite(train), axis=0)
        test_counts = np.sum(np.isfinite(test), axis=0)
        min_split_obs = max(2, int(math.ceil(min(len(train_idx), len(test_idx)) * 0.5)))
        eligible = (train_counts >= min_split_obs) & (test_counts >= min_split_obs)
        if int(eligible.sum()) < 2:
            continue
        train_scores = np.full(candidates, np.nan)
        test_scores = np.full(candidates, np.nan)
        for j in np.flatnonzero(eligible):
            train_scores[j] = float(np.nanmean(train[:, j]))
            test_scores[j] = float(np.nanmean(test[:, j]))
        valid_candidates = np.flatnonzero(np.isfinite(train_scores) & np.isfinite(test_scores))
        if len(valid_candidates) < 2:
            continue
        selected = int(valid_candidates[np.argmax(train_scores[valid_candidates])])
        selected_oos = float(test_scores[selected])
        candidate_oos = test_scores[valid_candidates]
        ties = np.isclose(candidate_oos, selected_oos, rtol=1e-10, atol=1e-12)
        less = int(np.sum(candidate_oos < selected_oos - 1e-12))
        tie_count = int(np.sum(ties))
        percentile = (less + 0.5 * tie_count) / float(len(candidate_oos))
        percentile = min(1.0 - 1e-9, max(1e-9, percentile))
        logits.append(math.log(percentile / (1.0 - percentile)))
        ranks.append(percentile)

    if not logits:
        return _unavailable_pbo(
            "no_valid_partitions",
            rows=rows,
            candidates=original_candidates,
            effective_candidates=candidates,
        )
    arr = np.asarray(logits, dtype=float)
    pbo = float(np.mean(arr < 0.0))
    return {
        "available": True,
        "reason": None,
        "pbo": pbo,
        "combinations": int(len(arr)),
        "mean_logit_rank": float(np.mean(arr)),
        "median_percentile_rank": float(np.median(ranks)),
        "selected_below_test_median_rate": pbo,
        "rows": int(rows),
        "candidates": int(original_candidates),
        "effective_candidates": int(candidates),
    }


def bootstrap_max_drawdown(
    returns, reps: int = 1000, block: int = 8, seed: int = 42
) -> dict:
    values = _clean_returns(returns)
    if len(values) < 10:
        return {
            "available": False,
            "reason": "insufficient_returns",
            "median_drawdown": None,
            "p95_drawdown": None,
            "prob_drawdown_below_20pct": None,
        }
    if int(reps) < 100:
        return {
            "available": False,
            "reason": "insufficient_bootstrap_replicates",
            "median_drawdown": None,
            "p95_drawdown": None,
            "prob_drawdown_below_20pct": None,
        }
    rng = np.random.default_rng(seed)
    block = max(1, int(block))
    n_blocks = int(math.ceil(len(values) / block))
    drawdowns = np.empty(int(reps), dtype=float)
    for j in range(int(reps)):
        starts = rng.integers(0, len(values), size=n_blocks)
        idx = np.concatenate([(s + np.arange(block)) % len(values) for s in starts])[:len(values)]
        path = np.cumprod(1.0 + values[idx])
        dd = path / np.maximum.accumulate(path) - 1.0
        drawdowns[j] = float(dd.min())
    return {
        "available": True,
        "reason": None,
        "median_drawdown": float(np.quantile(drawdowns, 0.50)),
        "p95_drawdown": float(np.quantile(drawdowns, 0.05)),
        "prob_drawdown_below_20pct": float(np.mean(drawdowns < -0.20)),
    }
