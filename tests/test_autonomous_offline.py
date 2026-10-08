from __future__ import annotations

from dataclasses import replace

from ai_trading_lab.autonomous import _offline_synthetic_data, autonomous_cycle
from ai_trading_lab.config import load_settings
from ai_trading_lab.data_quality import audit_market_data


def test_offline_synthetic_data_is_closed_and_quality_valid():
    settings = replace(load_settings("config.yaml"), timeframe="15m", seed=17)
    frame = _offline_synthetic_data(settings, n=600)

    assert len(frame) == 600
    assert frame.index.is_monotonic_increasing
    assert frame.index.tz is not None
    assert frame.index[-1] <= frame.index[-1].floor("15min")
    assert audit_market_data(frame, settings.timeframe).passed


def test_autonomous_cycle_offline_never_constructs_exchange(monkeypatch, tmp_path):
    import ai_trading_lab.autonomous as mod

    settings = replace(
        load_settings("config.yaml"),
        timeframe="15m",
        deep_evolution_enabled=False,
        seed=19,
    )

    class FakeCognition:
        def __init__(self, settings, root):
            self.registry = type("Registry", (), {"add_market_state": lambda *args, **kwargs: None})()

        def run_research(self, df, query):
            return {"strategy_candidates": [], "hypotheses": []}

        def run_growth(self, df, query, strategy_candidates):
            return {"external": {"provider_health": {}, "event_summary": {}}}

    def fail_exchange(*args, **kwargs):
        raise AssertionError("offline autonomous cycle attempted exchange access")

    monkeypatch.setenv("AUTONOMOUS_OFFLINE", "true")
    monkeypatch.setattr(mod, "exchange_client", fail_exchange)
    monkeypatch.setattr(mod, "CognitionEngine", FakeCognition)
    monkeypatch.setattr(mod, "auto_update", lambda *args, **kwargs: {"skipped": True})
    result = autonomous_cycle(settings, "offline smoke test", tmp_path)

    assert result["growth"]["market_state"] == {
        "mode": "offline",
        "exchange_access": False,
    }
    assert result["promotion"]["skipped"] is True
