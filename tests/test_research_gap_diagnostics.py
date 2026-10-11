from __future__ import annotations

import pandas as pd

from ai_trading_lab.data_quality import audit_market_data
from run_intensive_real_research import _data_quality_diagnostics


def _ohlcv(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
        },
        index=index,
    )


def test_data_quality_diagnostics_reports_exact_gap_boundaries() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    index = start + pd.to_timedelta(
        [0, 1, 2, 3, 12, 13, 14, 15], unit="m"
    ) * 15
    frame = _ohlcv(pd.DatetimeIndex(index))
    quality = audit_market_data(frame, "15m")

    report = _data_quality_diagnostics(frame, quality, "15m")

    assert "long_data_gap" in report["reasons"]
    assert report["max_gap_bars"] == 8
    assert report["first_gap_details"] == [
        {
            "after": "2026-01-01T00:45:00+00:00",
            "before": "2026-01-01T03:00:00+00:00",
            "missing_bars_estimate": 8,
            "gap_minutes": 135,
        }
    ]


def test_data_quality_diagnostics_keeps_gap_list_empty_for_contiguous_data() -> None:
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    index = pd.date_range(start, periods=120, freq="15min")
    frame = _ohlcv(index)
    quality = audit_market_data(frame, "15m")

    report = _data_quality_diagnostics(frame, quality, "15m")

    assert report["first_gap_details"] == []
    assert report["gap_rows"] == 0
