from __future__ import annotations

import numpy as np
import pandas as pd

from ai_trading_lab.engine import directional_target_from_returns


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
