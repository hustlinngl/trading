from __future__ import annotations

import numpy as np
import pandas as pd

from ai_trading_lab.autolearn import _final_holdout_report_payload
from ai_trading_lab.config import load_settings
from ai_trading_lab.fingerprint import strong_dataset_fingerprint


def test_auto_update_holdout_report_preserves_timeframe_and_exact_split():
    settings = load_settings("config.yaml")
    frame = pd.DataFrame(
        {
            "open": np.arange(12, dtype=float) + 100.0,
            "high": np.arange(12, dtype=float) + 101.0,
            "low": np.arange(12, dtype=float) + 99.0,
            "close": np.arange(12, dtype=float) + 100.5,
            "volume": np.full(12, 1000.0),
        },
        index=pd.date_range("2026-01-01", periods=12, freq="15min", tz="UTC"),
    )
    holdout = {
        "holdout_rows": 3,
        "stats": {"trades": 24, "total_return": 0.02},
        "utility": 0.3,
    }

    payload = _final_holdout_report_payload(
        frame, settings, holdout, "model-sem", "deployment-sem"
    )

    assert payload["timeframe"] == settings.timeframe
    assert payload["rows"] == 12
    assert payload["train_rows"] == 9
    assert payload["holdout_rows"] == 3
    assert payload["validation_train_data_fingerprint"] == strong_dataset_fingerprint(frame.iloc[:9])
    assert payload["validation_holdout_data_fingerprint"] == strong_dataset_fingerprint(frame.iloc[9:])
    assert payload["validation_holdout_start"] == str(frame.index[9])
    assert payload["validation_holdout_end"] == str(frame.index[-1])
    assert payload["validation_holdout_frac"] == 0.25
