from __future__ import annotations

import numpy as np
import pandas as pd

from ai_trading_lab.engine import directional_target_from_returns, execution_aligned_targets
from ai_trading_lab.features import make_features


def test_directional_target_uses_realized_return_sign_not_barrier_class():
    # Label 0 means no barrier was hit; its realized return can still be positive
    # or negative. It must not be collapsed into the short/non-positive class by
    # the training pipeline.
    barrier_labels = pd.Series([0, 0, -1, 1, 0, 0])
    realized = pd.Series([0.003, -0.002, -0.004, 0.010, 0.0, np.nan])

    target = directional_target_from_returns(realized)

    assert target.iloc[:5].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]
    assert np.isnan(target.iloc[5])
    # A positive realized return with neutral barrier label is explicitly LONG-positive.
    assert barrier_labels.iloc[0] == 0
    assert target.iloc[0] == 1.0


def test_directional_target_masks_nonfinite_returns():
    target = directional_target_from_returns(pd.Series([np.inf, -np.inf, np.nan, 0.001, -0.001]))

    assert target.iloc[:3].isna().all()
    assert target.iloc[3:].tolist() == [1.0, 0.0]


def _flat_ohlcv(rows=64):
    index = pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": np.full(rows, 100.0),
            "high": np.full(rows, 101.0),
            "low": np.full(rows, 99.0),
            "close": np.full(rows, 100.0),
            "volume": np.full(rows, 1000.0),
        },
        index=index,
    )


def test_execution_target_uses_barrier_exit_when_time_stop_has_opposite_sign():
    frame = _flat_ohlcv()
    decision = frame.index[20]
    # Entry is open[21]. The lower barrier is hit at bar 22, but the time-stop
    # open[29] later recovers sharply, so open-to-open and executable returns
    # have opposite signs.
    frame.loc[frame.index[22], ["open", "high", "low", "close"]] = [
        100.0, 101.0, 97.0, 100.0
    ]
    frame.loc[frame.index[29], ["open", "high", "low", "close"]] = [
        110.0, 111.0, 109.0, 110.0
    ]

    _, raw_direction, raw_return = make_features(frame, horizon=8)
    barrier_labels, target_returns, target = execution_aligned_targets(
        frame, horizon_bars=8, pt_atr=1.6, sl_atr=1.0
    )

    assert raw_return.loc[decision] > 0.0
    assert raw_direction.loc[decision] == 1.0
    assert barrier_labels.loc[decision, "tb_label"] == -1.0
    assert target_returns.loc[decision] < 0.0
    assert target.loc[decision] == 0.0


def test_execution_target_excludes_ambiguous_intrabar_barrier_collision():
    frame = _flat_ohlcv()
    decision = frame.index[20]
    frame.loc[frame.index[22], ["open", "high", "low", "close"]] = [
        100.0, 104.0, 97.0, 100.0
    ]

    labels, returns, target = execution_aligned_targets(
        frame, horizon_bars=8, pt_atr=1.6, sl_atr=1.0
    )

    assert np.isnan(labels.loc[decision, "tb_label"])
    assert np.isnan(returns.loc[decision])
    assert np.isnan(target.loc[decision])


def test_execution_target_time_stop_matches_open_to_open_when_no_barrier_hits():
    frame = _flat_ohlcv()
    decision = frame.index[20]
    frame.loc[frame.index[29], ["open", "high", "low", "close"]] = [
        100.5, 101.0, 100.0, 100.5
    ]

    labels, returns, target = execution_aligned_targets(
        frame, horizon_bars=8, pt_atr=1.6, sl_atr=1.0
    )

    assert labels.loc[decision, "tb_label"] == 0.0
    assert np.isclose(returns.loc[decision], 0.005)
    assert target.loc[decision] == 1.0
