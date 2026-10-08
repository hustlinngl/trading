from __future__ import annotations

import numpy as np
import pandas as pd

from ai_trading_lab.memory import AnalogMemory


def test_analog_memory_exclusion_handles_fewer_than_k_neighbors():
    index = pd.date_range("2026-01-01", periods=20, freq="15min", tz="UTC")
    features = pd.DataFrame(
        {
            "x1": np.linspace(0.0, 1.0, 20),
            "x2": np.sin(np.arange(20)),
        },
        index=index,
    )
    future_ret = pd.Series(np.linspace(-0.02, 0.02, 20), index=index)

    memory = AnalogMemory(k=8, exclusion_bars=9)
    memory.fit(features, future_ret)

    result = memory.query_many(
        features.iloc[[10]],
        exclude_self=True,
        exclusion_bars=9,
    )

    assert len(result) == 1
    assert int(result.iloc[0]["n"]) == 1
    assert np.isfinite(float(result.iloc[0]["edge"]))
