"""Offline causal tests for Sakura Atlas's one-session-at-a-time shadow journal."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ai_trading_lab.spy_research import analyze_spy, only_closed_sessions
from ai_trading_lab.spy_shadow import read_shadow_journal, track_spy


def history(rows=900):
    rng = np.random.default_rng(22)
    idx = pd.bdate_range("2017-01-02", periods=rows, tz="America/New_York")
    opens = 110 * np.exp(np.cumsum(rng.normal(0.0002, 0.003, rows)))
    closes = opens * np.exp(rng.normal(0, 0.002, rows))
    return pd.DataFrame({
        "open": opens, "close": closes,
        "high": np.maximum(opens, closes) * 1.004,
        "low": np.minimum(opens, closes) * 0.996,
        "volume": np.full(rows, 1_000_000.0),
    }, index=idx)


def after_close(frame):
    day = frame.index[-1].tz_convert("America/New_York").date()
    return pd.Timestamp(f"{day} 18:00", tz="America/New_York")


def extend(frame, days=3):
    latest = frame.index[-1]
    index = pd.bdate_range(latest + pd.Timedelta(days=1), periods=days, tz="America/New_York")
    o = float(frame["close"].iloc[-1])
    new = pd.DataFrame({
        "open": [o * (1.01 + 0.01 * k) for k in range(days)],
        "close": [o * (1.01 + 0.01 * k) for k in range(days)],
        "high": [o * (1.015 + 0.01 * k) for k in range(days)],
        "low": [o * (1.005 + 0.01 * k) for k in range(days)],
        "volume": [1_000_000.0] * days,
    }, index=index)
    return pd.concat([frame, new])


def test_partial_daily_candle_never_enters_research_data():
    frame = history()
    last = frame.index[-1].date()
    midday = pd.Timestamp(f"{last} 12:00", tz="America/New_York")
    evening = pd.Timestamp(f"{last} 18:00", tz="America/New_York")
    assert len(only_closed_sessions(frame, as_of=midday)) == len(frame) - 1
    assert len(only_closed_sessions(frame, as_of=evening)) == len(frame)
    with pytest.raises(ValueError, match="future-dated"):
        only_closed_sessions(frame, as_of=pd.Timestamp("2010-01-01", tz="UTC"))


def test_shadow_forecast_is_forward_only_deduplicated_and_settles_later(tmp_path):
    frame = history()
    report = analyze_spy(tmp_path, adjusted_history=frame)
    first = track_spy(tmp_path, adjusted_history=frame, as_of=after_close(frame))
    assert first["status"] == "SHADOW_ONLY"
    assert first["recorded_new_observation"] is True
    assert first["production_ready"] is False
    assert first["observations"] == 1
    record = read_shadow_journal(tmp_path)[0]
    assert record["model_run_id"] == report["run_id"]
    assert record["paper_only"]
    assert record["outcome"] is None
    second = track_spy(tmp_path, adjusted_history=frame, as_of=after_close(frame))
    assert second["recorded_new_observation"] is False
    assert len(read_shadow_journal(tmp_path)) == 1

    extended = extend(frame, days=3)
    later = track_spy(tmp_path, adjusted_history=extended, as_of=after_close(extended))
    assert later["new_outcomes_resolved"] == 1
    journal = read_shadow_journal(tmp_path)
    assert len(journal) == 2  # latest day prospectively recorded
    first_outcome = journal[0]["outcome"]
    assert first_outcome is not None
    expected = extended["open"].iloc[-1] / extended["open"].iloc[-2] - 1
    assert first_outcome["market_open_to_open_return"] == pytest.approx(expected)
    assert first_outcome["paper_only"]


def test_fingerprint_blocks_modified_serialized_models_before_deserialization(tmp_path):
    frame = history()
    report = analyze_spy(tmp_path, adjusted_history=frame)
    file = tmp_path / "models" / "spy_research" / report["run_id"] / "research_model.joblib"
    assert hashlib.sha256(file.read_bytes()).hexdigest() == report["model_sha256"]
    file.write_bytes(b"forged bytes")
    with pytest.raises(RuntimeError, match="fingerprint"):
        track_spy(tmp_path, adjusted_history=frame, as_of=after_close(frame))
    assert read_shadow_journal(tmp_path) == []


def test_no_hindsight_after_next_session_has_already_opened(tmp_path):
    frame = history()
    analyze_spy(tmp_path, adjusted_history=frame)
    next_day = pd.bdate_range(
        frame.index[-1] + pd.Timedelta(days=1), periods=1, tz="America/New_York"
    )[0].date()
    when = pd.Timestamp(f"{next_day} 11:00", tz="America/New_York")
    with pytest.raises(RuntimeError, match="retroactive"):
        track_spy(tmp_path, adjusted_history=frame, as_of=when)
    assert read_shadow_journal(tmp_path) == []


def test_corrupted_journal_is_never_silently_overwritten(tmp_path):
    path = tmp_path / "logs" / "spy_shadow_observations.json"
    path.parent.mkdir(parents=True)
    path.write_text("{corrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupted"):
        read_shadow_journal(tmp_path)
    assert path.read_text(encoding="utf-8") == "{corrupted"
