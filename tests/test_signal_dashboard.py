from __future__ import annotations

import json

import pandas as pd

from ai_trading_lab.config import load_settings
from ai_trading_lab.live import LiveAssessment


def test_signal_terminal_builds_read_only_state(tmp_path, monkeypatch):
    import signal_dashboard as terminal_mod

    settings = load_settings("config.yaml")
    settings.live_symbols = ("BTC/USDT",)
    assessment = LiveAssessment(
        "BTC/USDT",
        "2026-10-07T00:00:00+00:00",
        "SIGNAL",
        "LONG",
        0.91,
        0.006,
        100000.0,
        [],
        "data-fp",
        {
            "p_up": 0.91,
            "expected_return_lcb": 0.004,
            "expected_return_ucb": 0.008,
            "score": 0.41,
            "meta_success": 0.84,
            "model_disagreement": 0.01,
            "regime": "trend_up",
            "analog_n": 32,
            "analog_agreement": 0.88,
            "trade_window_ready": True,
            "trade_window_direction": "LONG",
            "trade_window_confidence": 0.92,
            "data_age_minutes": 1.0,
        },
    )
    monkeypatch.setattr(terminal_mod, "scan_top5", lambda *args, **kwargs: [assessment])
    monkeypatch.setattr(terminal_mod, "update_live_signal_outcomes", lambda *args, **kwargs: {"updated": 0, "open": 1, "closed": 0})

    terminal = terminal_mod.SignalTerminal(settings, tmp_path, refresh_seconds=30)
    state = terminal._terminal_state(force=True)

    assert state["ok"] is True
    assert state["summary"]["assets_scanned"] == 1
    assert state["summary"]["active_signals"] == 1
    assert state["signals"][0]["signal"] == "LONG"
    assert state["signals"][0]["decision"]["score"] == 0.41
    assert state["notes"][0].startswith("Signals are read-only")


def test_signal_terminal_serves_only_read_routes(tmp_path, monkeypatch):
    import signal_dashboard as terminal_mod

    settings = load_settings("config.yaml")
    monkeypatch.setattr(terminal_mod, "scan_top5", lambda *args, **kwargs: [])
    terminal = terminal_mod.SignalTerminal(settings, tmp_path, refresh_seconds=30)
    handler = terminal_mod.make_handler(terminal)
    server = terminal_mod.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    try:
        assert server.server_address[0] == "127.0.0.1"
        assert not hasattr(handler, "do_POST")
    finally:
        server.server_close()


def test_signal_terminal_json_safe_handles_nan_and_nested_values():
    import signal_dashboard as terminal_mod

    value = terminal_mod._json_safe({"x": float("nan"), "nested": [1, float("nan")]})
    assert json.dumps(value, allow_nan=False)
