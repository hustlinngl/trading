from __future__ import annotations

import pandas as pd

import ai_trading_lab.binance_vision as vision


def test_exact_range_recovers_missing_monthly_archive_from_daily_klines(monkeypatch, tmp_path):
    monthly_calls = []
    daily_calls = []

    def fake_month(symbol, interval, year, month, market, out_dir, timeout, verify_checksum):
        monthly_calls.append((year, month))
        if (year, month) == (2026, 4):
            raise FileNotFoundError("monthly archive missing")
        return f"{symbol}-{interval}-{year}-{month:02d}.zip"

    def fake_daily(symbol, interval, day, market, out_dir, timeout, verify_checksum):
        daily_calls.append(day)
        return f"{symbol}-{interval}-{day.isoformat()}.zip"

    monkeypatch.setattr(vision, "download_month", fake_month)
    monkeypatch.setattr(vision, "download_daily", fake_daily)

    paths = vision.download_range(
        "BTC/USDT",
        "15m",
        "2026-03-01",
        "2026-05-31",
        out_dir=tmp_path,
        verify_checksum=False,
        exact=True,
    )

    assert monthly_calls == [(2026, 3), (2026, 4), (2026, 5)]
    assert daily_calls == list(pd.date_range("2026-04-01", "2026-04-30", freq="D").date)
    assert len(paths) == 32


def test_non_exact_range_preserves_skip_behavior_for_missing_monthly_archive(monkeypatch, tmp_path):
    daily_calls = []

    def missing_month(*args, **kwargs):
        raise FileNotFoundError("monthly archive missing")

    def fake_daily(*args, **kwargs):
        daily_calls.append(args[2])
        return "unexpected-daily.zip"

    monkeypatch.setattr(vision, "download_month", missing_month)
    monkeypatch.setattr(vision, "download_daily", fake_daily)

    paths = vision.download_range(
        "BTC/USDT",
        "15m",
        "2026-03-01",
        "2026-05-31",
        out_dir=tmp_path,
        verify_checksum=False,
        exact=False,
    )

    assert paths == []
    assert daily_calls == []
