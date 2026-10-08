from __future__ import annotations

import numpy as np
import pandas as pd

from ai_trading_lab.models import SignalModel


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


def test_signal_model_refits_production_learners_on_all_rows_after_oos_calibration():
    n = 320
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
    assert model.fit_rows_ == n
    assert clf_fit_lengths[-1] == n
    assert reg_fit_lengths[-1] == n
    assert clf_fit_lengths.count(n) == len(model.clfs)
    assert reg_fit_lengths.count(n) == len(model.regs)
