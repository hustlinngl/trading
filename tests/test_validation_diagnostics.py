from __future__ import annotations

import numpy as np
import pytest

from ai_trading_lab.evaluation import directional_validation_diagnostics


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
