from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ai_trading_lab.meta import MetaPolicy
from ai_trading_lab.models import SignalModel, _finite_sample_conformal_quantile


class _FakeClassifier:
    def __init__(self, offset: float, fit_lengths: list[int]):
        self.offset = offset
        self.fit_lengths = fit_lengths

    def fit(self, X, y):
        self.fit_lengths.append(len(X))
        return self

    def predict_proba(self, X):
        p = np.clip(0.5 + 0.2 * np.tanh(X.iloc[:, 0].to_numpy(float) + self.offset), 0.01, 0.99)
        return np.column_stack([1.0 - p, p])


class _FakeRegressor:
    def __init__(self, offset: float, fit_lengths: list[int]):
        self.offset = offset
        self.fit_lengths = fit_lengths

    def fit(self, X, y):
        self.fit_lengths.append(len(X))
        return self

    def predict(self, X):
        return 0.002 * X.iloc[:, 0].to_numpy(float) + self.offset


def test_signal_model_keeps_calibrated_core_learners_as_the_deployed_model():
    n = 800
    index = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    x = pd.DataFrame({"feature": np.linspace(-2.0, 2.0, n)}, index=index)
    y_cls = pd.Series((np.arange(n) % 3 != 0).astype(float), index=index)
    y_ret = pd.Series(0.001 * np.sin(np.arange(n) / 11.0), index=index)

    clf_fit_lengths: list[int] = []
    reg_fit_lengths: list[int] = []
    model = SignalModel()
    model.clfs = [_FakeClassifier(shift, clf_fit_lengths) for shift in (-0.15, 0.0, 0.15)]
    model.regs = [_FakeRegressor(shift, reg_fit_lengths) for shift in (-0.0002, 0.0, 0.0002)]

    model.fit(x, y_cls, y_ret, purge_bars=12)

    assert model.calibration_oos_ is not None
    # A later calibration tail is reserved for meta training after a purge gap;
    # its labels are not used to fit ensemble weights, the calibrator, or intervals.
    assert model.calibration_oos_.index.equals(index[-36:])
    assert {"p_up", "p_up_raw"} <= set(model.calibration_oos_.columns)
    expected_core_rows = n - max(48, int(n * 0.15)) - 12
    assert model.fit_rows_ == expected_core_rows
    assert clf_fit_lengths == [expected_core_rows] * len(model.clfs)
    assert reg_fit_lengths == [expected_core_rows] * len(model.regs)
    assert np.allclose(model.weights_cls, np.full(len(model.clfs), 1 / len(model.clfs)))
    assert np.allclose(model.weights_reg, np.full(len(model.regs), 1 / len(model.regs)))

    predictions = model.predict(x.tail(4))
    assert {"p_up_raw", "p_up"} <= set(predictions.columns)


def test_isotonic_calibration_uses_platt_fallback_outside_oos_score_range():
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    raw = np.linspace(0.2, 0.8, 120)
    target = (raw >= 0.5).astype(int)
    model = SignalModel()
    model.calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw, target)
    model.calibrator_fallback = LogisticRegression(C=1.0, solver="lbfgs").fit(
        pd.DataFrame({"p_up_raw": raw}), target
    )

    probes = np.asarray([0.05, 0.10, 0.90, 0.95])
    calibrated = model._calibrated_probabilities(probes)
    expected = model.calibrator_fallback.predict_proba(
        pd.DataFrame({"p_up_raw": probes})
    )[:, 1]

    assert np.all(np.isfinite(calibrated))
    assert np.all((calibrated > 0.0) & (calibrated < 1.0))
    assert np.all(np.diff(calibrated) > 0.0)
    assert np.allclose(calibrated, expected)



def test_meta_policy_uses_oos_probability_and_training_time_return_features():
    index = pd.date_range("2026-01-01", periods=1, freq="15min", tz="UTC")
    pred = pd.DataFrame(
        {
            # The execution policy consumes the calibrated probability, while the
            # meta policy was trained with the raw, out-of-sample ensemble score.
            "p_up_raw": [0.31],
            "p_up": [0.91],
            "expected_return": [0.004],
            # Conformal bounds exist at inference but were not available in the
            # meta-training OOS frame; they must not create a train/serve mismatch.
            "expected_return_lcb": [-0.03],
            "expected_return_ucb": [0.04],
            "model_disagreement": [0.02],
            "return_disagreement": [0.01],
        },
        index=index,
    )
    features = pd.DataFrame(index=index)
    regimes = pd.Series(["trend_up"], index=index)
    analog = pd.DataFrame(
        {"edge": [0.002], "agreement": [0.8], "dispersion": [0.01]},
        index=index,
    )

    frame = MetaPolicy.frame(pred, features, regimes, analog)

    assert np.isclose(frame.loc[index[0], "p_up"], 0.31)
    assert np.isclose(frame.loc[index[0], "expected_return_lcb"], 0.004)
    assert np.isclose(frame.loc[index[0], "expected_return_ucb"], 0.004)


def test_nan_conformal_scale_uses_fitted_imputer_and_keeps_bounds_finite():
    model = SignalModel()
    model.clfs = [_FakeClassifier(0.0, [])]
    model.regs = [_FakeRegressor(0.0, [])]
    model.feature_cols = ["feature", "atr_pct"]
    model.fill_values = pd.Series({"feature": 0.0, "atr_pct": 0.02})
    model.weights_cls = np.array([1.0])
    model.weights_reg = np.array([1.0])
    model.calibrator = None
    model.calibrator_fallback = None
    model.conformal_scale_col = "atr_pct"
    model.conformal_scaled_error_ = 2.0
    model.conformal_abs_error_ = 0.0
    model.ready = True
    frame = pd.DataFrame({"feature": [0.5], "atr_pct": [np.nan]})

    predictions = model.predict(frame)

    assert np.isfinite(
        predictions[["expected_return_lcb", "expected_return_ucb"]].to_numpy()
    ).all()
    assert (
        predictions.loc[frame.index[0], "expected_return_ucb"]
        - predictions.loc[frame.index[0], "expected_return_lcb"]
    ) == pytest.approx(0.08)


def test_conformal_quantile_uses_finite_sample_order_statistic():
    residuals = np.arange(1.0, 11.0)

    # ceil((n+1)*0.8) = 9, so the order statistic is the ninth value.
    assert _finite_sample_conformal_quantile(residuals, 0.80) == pytest.approx(9.0)
    # ceil((4+1)*0.9) = 5 is beyond the available sample: fail conservatively.
    assert np.isinf(_finite_sample_conformal_quantile([1.0, 2.0, 3.0, 4.0], 0.90))



def test_cpu_only_xgboost_runtime_keeps_standard_classifier_api():
    from ai_trading_lab import models

    assert models.HAVE_XGB
    assert callable(models.XGBClassifier)
    assert callable(models.XGBRegressor)
