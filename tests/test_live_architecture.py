from __future__ import annotations

import threading
import time

import pandas as pd

from ai_trading_lab.config import load_settings
from ai_trading_lab.efficiency import (
    ExecutionEfficiencyTracker,
    execution_efficiency_snapshot,
)
from ai_trading_lab.live import LiveAssessment
from ai_trading_lab.live_tracker import LiveTracker
from ai_trading_lab.state_fusion import fuse_live_dashboard_state
from ai_trading_lab.streaming import BinancePublicStream, StreamEvent


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
        assert state["price"] == 100.1
        assert state["ask"] == 100.2
        assert state["mid"] == 100.1
        assert state["sequence"] == 1
    finally:
        tracker.stop()


def test_live_tracker_realtime_price_prefers_quote_mid_over_last_trade():
    tracker = LiveTracker(symbols=())
    timestamp = pd.Timestamp("2026-10-07T16:00:00+00:00")
    book = StreamEvent(
        received_at=timestamp.isoformat(),
        stream="btcusdt@bookTicker",
        event_time_ms=int(timestamp.timestamp() * 1000),
        symbol="BTC/USDT",
        event_type="bookTicker",
        payload={"s": "BTCUSDT", "b": "100.0", "a": "100.2", "E": int(timestamp.timestamp() * 1000)},
    )
    trade = StreamEvent(
        received_at=timestamp.isoformat(),
        stream="btcusdt@aggTrade",
        event_time_ms=int(timestamp.timestamp() * 1000) + 10,
        symbol="BTC/USDT",
        event_type="aggTrade",
        payload={"s": "BTCUSDT", "p": "100.4", "q": "2.0", "m": True, "E": int(timestamp.timestamp() * 1000) + 10},
    )
    tracker._merge_event(book)
    tracker._merge_event(trade)

    state = dict(tracker.get_current_state("BTC/USDT"))
    assert state["price"] == 100.1
    assert state["mid"] == 100.1
    assert state["last_trade_price"] == 100.4
    assert state["trade_side"] == "SELL"


def test_binance_public_stream_routes_market_types_to_matching_endpoints():
    spot = BinancePublicStream("BTC/USDT")
    usdm = BinancePublicStream("ETH/USDT:USDT")
    coinm = BinancePublicStream("BTC/USD:BTC")
    dated_coinm = BinancePublicStream("BTC/USD:BTC-251226")

    assert "stream.binance.com:9443/stream" in spot.url()
    assert "fstream.binance.com/public/stream" in usdm.url()
    assert "dstream.binance.com/stream" in coinm.url()
    assert "btcusd_251226@bookticker" in dated_coinm.url()
    with pytest.raises(ValueError):
        BinancePublicStream("BTC/USDT:USDT-251226-100000-C").url()


def test_live_universe_honors_explicit_online_scope(monkeypatch, tmp_path):
    import ai_trading_lab.live as live_mod

    settings = load_settings("config.yaml")

    class FakeExchange:
        markets = {
            "BTC/USDT": {"symbol": "BTC/USDT", "type": "spot", "active": True},
            "ETH/USDT": {"symbol": "ETH/USDT", "type": "spot", "active": True},
            "SOL/USDT": {"symbol": "SOL/USDT", "type": "spot", "active": True},
        }

    monkeypatch.setattr(
        live_mod,
        "resolve_signal_bundle",
        lambda settings, root, symbol: tmp_path / symbol.replace("/", "_"),
    )
    monkeypatch.setattr(
        live_mod,
        "bundle_compatibility",
        lambda settings, bundle, symbol: (True, "ok"),
    )
    for symbol in ("BTC/USDT", "ETH/USDT"):
        bundle = tmp_path / symbol.replace("/", "_")
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "signal_model.joblib").write_text("stub", encoding="utf-8")

    meta = live_mod.discover_live_universe(
        settings,
        tmp_path,
        FakeExchange(),
        symbols=["ETH/USDT"],
    )
    assert meta["universe_symbols"] == ["ETH/USDT"]
    assert meta["symbols"] == ["ETH/USDT"]

    
def test_live_scan_uses_bounded_parallel_workers(monkeypatch, tmp_path):
    import ai_trading_lab.live as live_mod

    settings = load_settings("config.yaml")
    settings.live_scan_workers = 4

    monkeypatch.setattr(
        live_mod,
        "discover_live_universe",
        lambda *args, **kwargs: {
            "symbols": ["A/USDT", "B/USDT", "C/USDT", "D/USDT"],
            "discovered_markets": 4,
            "model_backed_markets": 4,
            "model_eligible_markets": 4,
            "market_counts": {"spot": 4},
        },
    )

    lock = threading.Lock()
    active = 0
    max_active = 0

    def fake_assess(settings, root, symbol, exchange=None, skip_network=False):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return LiveAssessment(
            symbol,
            "2026-10-07T00:00:00+00:00",
            "WAIT",
            "FLAT",
            0.0,
            0.0,
            1.0,
            ["gate"],
            "fp",
        )

    monkeypatch.setattr(live_mod, "assess_symbol", fake_assess)
    picks, meta = live_mod.scan_top5(
        settings,
        tmp_path,
        exchange=object(),
        return_meta=True,
    )

    assert picks == []
    assert max_active >= 2
    assert meta["scan_workers"] == 4
    assert meta["scan_completed"] == 4


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
    assert abs(payload["risk"]["drawdown"] + 0.1) < 1e-9
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
    assert 1 <= settings.live_scan_workers <= 8
