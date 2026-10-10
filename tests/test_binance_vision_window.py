from __future__ import annotations

import pandas as pd
import pytest

from ai_trading_lab.binance_vision import rolling_closed_history_window


def test_rolling_window_uses_monthly_archives_and_latest_closed_daily_range():
    window = rolling_closed_history_window(
        now=pd.Timestamp("2026-10-10T17:20:00Z"),
        months=24,
    )

    assert window == {
        "start": "2024-10-01",
        "monthly_end": "2026-09-30",
        "recent_start": "2026-10-01",
        "end": "2026-10-09",
    }


def test_rolling_window_on_first_day_does_not_request_an_unclosed_day():
    window = rolling_closed_history_window(
        now=pd.Timestamp("2026-11-01T00:05:00Z"),
        months=24,
    )

    assert window["start"] == "2024-11-01"
    assert window["monthly_end"] == "2026-10-31"
    assert window["recent_start"] == "2026-11-01"
    assert window["end"] == "2026-10-31"
    assert window["recent_start"] > window["end"]


def test_rolling_window_normalizes_naive_clock_as_utc_and_validates_months():
    window = rolling_closed_history_window(
        now=pd.Timestamp("2026-10-10 00:10:00"),
        months=12,
    )
    assert window["start"] == "2025-10-01"
    assert window["end"] == "2026-10-09"

    with pytest.raises(ValueError, match="months must be >= 1"):
        rolling_closed_history_window(now=pd.Timestamp("2026-10-10T00:00:00Z"), months=0)
