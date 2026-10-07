from __future__ import annotations

import time

import pandas as pd

from ai_trading_lab.config import load_settings
from ai_trading_lab.efficiency import (
    ExecutionEfficiencyTracker,
    execution_efficiency_snapshot,
)
from ai_trading_lab.live_tracker import LiveTracker
from ai_trading_lab.state_fusion import fuse_live_dashboard_state
from ai_trading_lab.streaming import StreamEvent


def test_live_tracker_async_queue_merges_and_exposes_state():
    tracker = LiveTracker(symbols=(), max_queue=128)
    tracker.start()
    try:
        timestamp = pd.Timestamp("2026-10-07T16:00:00+00:00")
        event = StreamEvent(
            received_at=timestamp.isoformat(),
            stream="btcusdt@bookTicker",
            event_time_ms=int(timestamp.timestamp() * 1000),
            symbol="BTC/USDT",
            event_type="bookTicker",
            payload={"s": "BTCUSDT", "b": "100.0", "a": "100.2", "E": int(timestamp.timestamp() * 1000)},
        )
        assert tracker.submit_update(event) is True

        deadline = time.monotonic() + 1.0
        state = {}
        while time.monotonic() < deadline:
            state = dict(tracker.get_current_state("BTC/USDT"))
            if state.get("bid") == 100.0:
                break
            time.sleep(0.01)

        assert state["symbol"] == "BTC/USDT"
        assert state["price"] == 100.0
        assert state["ask"] == 100.2
        assert state["mid"] == 100.1
        assert state["sequence"] == 1
    finally:
        tracker.stop()


def test_state_fusion_normalizes_regime_risk_and_execution():
    telemetry = ExecutionEfficiencyTracker()
    telemetry.record(
        expected_price=100.0,
        execution_price=100.05,
        side="BUY",
        sent_at=10.0,
        filled_at=10.020,
    )
    payload = fuse_live_dashboard_state(
        symbol="BTC/USDT",
        price=100.0,
        market_snapshot={"price": 100.0, "latency_ms": 8.0},
        regime_context={
            "raw_label": "high_vol_down",
            "confidence": 0.81,
        },
        risk_context={
            "equity": 9000.0,
            "peak_equity": 10000.0,
            "active_notional": 18000.0,
        },
        execution_telemetry=telemetry,
        cognition={
            "lines": ["Policy gate → SHORT"],
            "signature": "x",
        },
    )
    assert payload["regime"]["label"] == "High Volatility"
    assert payload["risk"]["drawdown"] == -0.1
    assert payload["risk"]["leverage"] == 2.0
    assert payload["efficiency"]["execution_available"] is True
    assert payload["efficiency"]["slippage_bps"] > 0


def test_efficiency_is_explicit_when_execution_telemetry_is_missing():
    payload = execution_efficiency_snapshot(
        None,
        market_data_latency_ms=12.5,
        configured_slippage_bps=5.0,
    )
    assert payload["slippage_bps"] is None
    assert payload["latency_ms"] == 12.5
    assert payload["execution_available"] is False
    assert payload["configured_slippage_bps"] == 5.0


def test_dashboard_contains_delta_kpis_cognition_and_live_endpoint():
    import signal_dashboard as terminal_mod

    html = terminal_mod.HTML
    assert 'id="kpiPrice"' in html
    assert 'id="kpiRegime"' in html
    assert 'id="kpiRisk"' in html
    assert 'id="kpiEfficiency"' in html
    assert "Live AI Cognition" in html
    assert "function applyLiveDelta" in html
    assert "/api/live?symbol=" in html


def test_settings_still_construct():
    settings = load_settings("config.yaml")
    assert settings.symbol