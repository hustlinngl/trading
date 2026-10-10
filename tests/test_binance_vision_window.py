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



def test_windows_build_uses_rolling_history_and_recent_daily_archives():
    from pathlib import Path

    workflow = Path(".github/workflows/build-windows-exe.yml").read_text(
        encoding="utf-8"
    )

    assert 'rolling_closed_history_window(months=24)' in workflow
    assert 'if recent_start <= end:' in workflow
    assert 'exact=True' in workflow
    assert 'logs/bundled_history_window.json' in workflow
    assert 'Historical OHLCV window (UTC)' in workflow
    assert 'start = "2024-10-01"' not in workflow
    assert 'end = "2026-10-01"' not in workflow



def test_download_range_exact_uses_daily_archives_without_invalid_timestamp_kwargs(monkeypatch, tmp_path):
    import ai_trading_lab.binance_vision as vision

    requested_days = []

    def fake_daily(symbol, interval, day, market, out_dir, timeout, verify_checksum):
        requested_days.append(day)
        return f"{symbol}-{interval}-{day.isoformat()}.zip"

    def unexpected_monthly(*args, **kwargs):
        raise AssertionError("partial-month request must use daily archives")

    monkeypatch.setattr(vision, "download_daily", fake_daily)
    monkeypatch.setattr(vision, "download_month", unexpected_monthly)

    paths = vision.download_range(
        "BTC/USDT",
        "15m",
        "2026-10-01",
        "2026-10-03",
        out_dir=tmp_path,
        verify_checksum=False,
        exact=True,
    )

    assert requested_days == [
        pd.Timestamp("2026-10-01").date(),
        pd.Timestamp("2026-10-02").date(),
        pd.Timestamp("2026-10-03").date(),
    ]
    assert len(paths) == 3
