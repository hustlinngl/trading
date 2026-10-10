from __future__ import annotations

import numpy as np
import pandas as pd

from ai_trading_lab.live_tracker import _resolve_result


def _market(rows: int = 48) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": np.full(rows, 100.0),
            "high": np.full(rows, 100.1),
            "low": np.full(rows, 99.9),
            "close": np.full(rows, 100.0),
            "volume": np.full(rows, 1_000.0),
        },
        index=index,
    )


def _resolve(
    frame: pd.DataFrame, *, max_bars: int, min_bars: int = 1, side: str = "LONG",
    impact_bps_per_sqrt: float = 0.0, max_participation_pct: float = 0.10,
    short_borrow_bps_per_bar: float = 0.0,
):
    return _resolve_result(
        frame,
        {
            "data_timestamp": frame.index[14].isoformat(),
            "signal": side,
            "_min_bars": min_bars,
        },
        timeframe="15m",
        pt_atr=1.25,
        sl_atr=0.90,
        fee_bps=1.0,
        slippage_bps=2.0,
        max_bars=max_bars,
        impact_bps_per_sqrt=impact_bps_per_sqrt,
        max_participation_pct=max_participation_pct,
        short_borrow_bps_per_bar=short_borrow_bps_per_bar,
    )


def test_outcome_tracker_prioritizes_opening_gap_over_intrabar_collision():
    frame = _market()
    # ATR is 0.2, so upper/lower barriers are 100.25/99.82. The candle opens
    # beyond the upper barrier and later crosses both levels: the open comes first.
    frame.loc[frame.index[16], ["open", "high", "low", "close"]] = [
        100.4, 100.5, 99.5, 100.0
    ]

    result = _resolve(frame, max_bars=8)

    assert result is not None
    assert result["outcome"] == "WIN"
    assert np.isclose(result["realized_return"], 0.004 - 0.0006)
    assert np.isclose(result["holding_hours"], 0.25)


def test_outcome_tracker_time_stop_uses_next_open_and_ignores_exit_candle_range():
    frame = _market()
    # For max_bars=3, entry is open[15] and liquidation is open[18]. The
    # time-stop candle's extreme must not be mistaken for a target event.
    frame.loc[frame.index[18], ["open", "high", "low", "close"]] = [
        100.2, 110.0, 99.9, 105.0
    ]

    result = _resolve(frame, max_bars=3)

    assert result is not None
    assert result["outcome"] == "TIMEOUT"
    assert np.isclose(result["realized_return"], 0.002 - 0.0006)
    assert np.isclose(result["holding_hours"], 0.75)


def test_outcome_tracker_counts_minimum_holding_intervals_from_entry_open():
    early = _market()
    early.loc[early.index[26], ["open", "high", "low", "close"]] = [
        100.0, 100.7, 99.9, 100.5
    ]

    early_result = _resolve(early, max_bars=16, min_bars=12)
    assert early_result is not None
    assert early_result["outcome"] == "EARLY"
    assert np.isclose(early_result["holding_hours"], 2.75)

    on_time = _market()
    on_time.loc[on_time.index[27], ["open", "high", "low", "close"]] = [
        100.0, 100.7, 99.9, 100.5
    ]

    on_time_result = _resolve(on_time, max_bars=16, min_bars=12)
    assert on_time_result is not None
    assert on_time_result["outcome"] == "WIN"
    assert np.isclose(on_time_result["holding_hours"], 3.0)


def test_outcome_tracker_fails_closed_when_time_stop_open_is_not_available():
    frame = _market(rows=18)
    # Timestamp at 14 -> entry at 15, and max_bars=3 requires open[18].
    result = _resolve(frame, max_bars=3)
    assert result is None


def test_outcome_tracker_resolves_an_early_barrier_before_full_horizon_is_available():
    frame = _market(rows=17)
    # A new outcome is already determined at bar 16, even though the time-stop
    # open for an eight-bar horizon is not yet present in this partial history.
    frame.loc[frame.index[16], ["open", "high", "low", "close"]] = [
        100.4, 100.5, 99.5, 100.0
    ]

    result = _resolve(frame, max_bars=8)

    assert result is not None
    assert result["outcome"] == "WIN"



def test_outcome_tracker_subtracts_configured_round_trip_impact_cost():
    frame = _market()
    frame.loc[frame.index[16], ["open", "high", "low", "close"]] = [
        100.4, 100.5, 99.5, 100.0
    ]

    result = _resolve(
        frame, max_bars=8, impact_bps_per_sqrt=10.0, max_participation_pct=0.25
    )

    assert result is not None
    assert result["outcome"] == "WIN"
    # 2*(1 fee + 2 slippage + 10*sqrt(0.25) impact) = 16 bps.
    assert np.isclose(result["estimated_cost_bps"], 16.0)
    assert np.isclose(result["realized_return"], 0.004 - 0.0016)


def test_outcome_tracker_subtracts_short_borrow_for_elapsed_bars():
    frame = _market()
    # The lower barrier is gapped through at the second holding candle (one full
    # elapsed interval after entry); shorts win when price falls through lower.
    frame.loc[frame.index[16], ["open", "high", "low", "close"]] = [
        99.6, 100.6, 99.5, 100.0
    ]

    result = _resolve(
        frame, max_bars=8, side="SHORT", short_borrow_bps_per_bar=4.0
    )

    assert result is not None
    assert result["outcome"] == "WIN"
    assert np.isclose(result["estimated_cost_bps"], 10.0)
    assert np.isclose(
        result["realized_return"], ((100.0 - 99.6) / 100.0) - 0.0010
    )


def test_short_timeout_return_uses_entry_not_exit_denominator():
    frame = _market()
    # Keep the time-stop open inside both barriers so the short closes by timeout.
    frame.loc[frame.index[18], ["open", "high", "low", "close"]] = [
        99.9, 100.0, 99.85, 99.9
    ]

    result = _resolve(frame, max_bars=3, side="SHORT")

    assert result is not None
    assert result["outcome"] == "TIMEOUT"
    # Simple short return is (entry - exit) / entry, less 6 bps of costs.
    assert np.isclose(result["realized_return"], ((100.0 - 99.9) / 100.0) - 0.0006)


def test_short_stop_loss_return_uses_entry_not_exit_denominator():
    frame = _market()
    frame.loc[frame.index[16], ["open", "high", "low", "close"]] = [
        100.4, 100.5, 100.3, 100.4
    ]

    result = _resolve(frame, max_bars=8, side="SHORT")

    assert result is not None
    assert result["outcome"] == "LOSS"
    assert np.isclose(result["realized_return"], ((100.0 - 100.4) / 100.0) - 0.0006)


def test_outcome_tracker_does_not_report_fake_zero_return_for_ambiguous_barrier():
    frame = _market()
    frame.loc[frame.index[16], ["open", "high", "low", "close"]] = [
        100.0, 100.6, 99.5, 100.0
    ]

    result = _resolve(frame, max_bars=8)

    assert result is not None
    assert result["outcome"] == "AMBIGUOUS"
    assert result["realized_return"] is None
    assert np.isclose(result["estimated_cost_bps"], 6.0)


def test_outcome_tracker_rejects_non_finite_cost_assumptions():
    frame = _market()
    result = _resolve(frame, max_bars=3, impact_bps_per_sqrt=float("nan"))
    assert result is None


def test_outcome_tracker_accepts_legacy_signal_history_timestamp_field():
    frame = _market()
    frame.loc[frame.index[16], ["open", "high", "low", "close"]] = [
        100.4, 100.5, 99.5, 100.0
    ]
    legacy_record = {
        "timestamp": frame.index[14].isoformat(),
        "signal": "LONG",
        "_min_bars": 1,
    }

    result = _resolve_result(
        frame, legacy_record, "15m", 1.25, 0.90, 1.0, 2.0, 8
    )

    assert result is not None
    assert result["outcome"] == "WIN"


def test_outcome_tracker_rejects_missing_or_invalid_timestamp_without_raising():
    frame = _market()
    for record in (
        {"signal": "LONG"},
        {"timestamp": "", "signal": "LONG"},
        {"timestamp": "not-a-timestamp", "signal": "LONG"},
    ):
        assert _resolve_result(frame, record, "15m", 1.25, 0.90, 1.0, 2.0, 8) is None



def test_outcome_tracker_updates_legacy_journal_and_entry_metadata(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    import ai_trading_lab.live_tracker as tracker

    frame = _market()
    frame.loc[frame.index[16], ["open", "high", "low", "close"]] = [
        100.4, 100.5, 99.5, 100.0
    ]
    history_dir = tmp_path / "logs"
    history_dir.mkdir()
    history_path = history_dir / tracker.HISTORY_NAME
    legacy_record = {
        "symbol": "BTC/USDT",
        "timestamp": frame.index[14].isoformat(),
        "status": "SIGNAL",
        "signal": "LONG",
        "confidence": 0.9,
        "expected_return": 0.01,
    }
    history_path.write_text(json.dumps(legacy_record) + "\n", encoding="utf-8")

    monkeypatch.setattr(tracker, "fetch_ohlcv", lambda *args, **kwargs: frame)
    settings = SimpleNamespace(
        exchange="binance",
        timeframe="15m",
        live_lookback_bars=600,
        trade_window_min_hours=0.25,
        trade_window_max_hours=24.0,
        trade_window_pt_atr=1.25,
        trade_window_sl_atr=0.90,
        fee_bps=1.0,
        slippage_bps=2.0,
        impact_bps_per_sqrt=0.0,
        max_participation_pct=0.10,
        short_borrow_bps_per_bar=0.0,
    )

    result = tracker.update_live_signal_outcomes(settings, tmp_path, exchange=object())
    persisted = json.loads(history_path.read_text(encoding="utf-8").splitlines()[0])

    assert result == {"updated": 1, "open": 0, "closed": 1}
    assert persisted["outcome"] == "WIN"
    assert persisted["entry_timestamp"] == str(frame.index[15])
    assert persisted["entry_price"] == 100.0
    assert "tracking_error" not in persisted
