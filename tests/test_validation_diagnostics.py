from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import pytest

from ai_trading_lab.evaluation import directional_validation_diagnostics
from ai_trading_lab.validation import compare_equity, _block_bootstrap_samples


def test_directional_validation_compares_predictions_to_realized_returns():
    result = directional_validation_diagnostics(
        probabilities=[0.9, 0.8, 0.2, 0.1, 0.7, np.nan, 0.5],
        realized_returns=[0.01, -0.01, -0.02, 0.02, 0.0, 0.03, np.inf],
    )

    assert result["observations"] == 5
    assert result["true_positive"] == 1
    assert result["false_positive"] == 2
    assert result["true_negative"] == 1
    assert result["false_negative"] == 1
    assert result["directional_accuracy"] == pytest.approx(0.4)
    assert result["balanced_directional_accuracy"] == pytest.approx(5 / 12)
    assert result["majority_class_baseline_accuracy"] == pytest.approx(0.6)
    assert result["probability_brier_score"] == pytest.approx(1.99 / 5)
    assert result["long_precision"] == pytest.approx(1 / 3)
    assert result["short_precision"] == pytest.approx(1 / 2)


def test_directional_validation_rejects_misaligned_arrays():
    with pytest.raises(ValueError, match="directional_validation_length_mismatch"):
        directional_validation_diagnostics([0.6, 0.4], [0.01])


def test_directional_validation_handles_no_usable_holdout_observations():
    result = directional_validation_diagnostics(
        probabilities=[np.nan, 1.2, -0.1],
        realized_returns=[0.01, -0.01, 0.02],
    )

    assert result == {
        "observations": 0,
        "reason": "no_finite_oos_observations",
    }


def test_compare_equity_uses_one_bootstrap_distribution_for_ci_and_probability():
    index = pd.date_range("2026-01-01", periods=30, freq="15min", tz="UTC")
    returns_a = np.asarray([0.001, -0.002, 0.003, 0.0, -0.001] * 6)
    returns_b = np.asarray([0.0, -0.001, 0.001, 0.0, -0.002] * 6)
    equity_a = pd.Series(100.0 * np.cumprod(1.0 + returns_a), index=index)
    equity_b = pd.Series(100.0 * np.cumprod(1.0 + returns_b), index=index)
    diff = (
        equity_a.pct_change().fillna(0).to_numpy()
        - equity_b.pct_change().fillna(0).to_numpy()
    )
    samples = _block_bootstrap_samples(
        diff, np.random.default_rng(17), reps=101, block=3
    )

    result = compare_equity(equity_a, equity_b, seed=17, reps=101, block=3)

    assert result["mean_return_diff"] == pytest.approx(float(diff.mean()))
    assert result["ci_low"] == pytest.approx(float(np.quantile(samples, 0.025)))
    assert result["ci_high"] == pytest.approx(float(np.quantile(samples, 0.975)))
    assert result["bootstrap_probability_mean_diff_gt_0"] == pytest.approx(float(np.mean(samples > 0)))
