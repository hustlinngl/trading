from __future__ import annotations

import pandas as pd

from ai_trading_lab.config import load_settings


def test_paper_rejects_derivative_when_funding_data_is_missing(tmp_path, monkeypatch):
    import ai_trading_lab.paper as paper_mod
    import ai_trading_lab.live as live_mod

    settings = load_settings("config.yaml")
    settings.symbol = "ETH/USDT:USDT"

    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000.0, 1000.0],
        },
        index=pd.date_range("2026-10-08T15:00:00Z", periods=2, freq="15min"),
    )

    class Quality:
        passed = True
        reasons = []

        def to_dict(self):
            return {"passed": True}

    exchange = object()
    monkeypatch.setattr(paper_mod, "exchange_client", lambda *args, **kwargs: exchange)
    monkeypatch.setattr(paper_mod, "fetch_ohlcv", lambda *args, **kwargs: frame)
    monkeypatch.setattr(paper_mod, "audit_market_data", lambda *args, **kwargs: Quality())
    monkeypatch.setattr(
        live_mod,
        "_funding_snapshot",
        lambda *args, **kwargs: {
            "market_type": "swap",
            "funding_rate": None,
            "funding_cost_return": 0.0,
            "funding_data_missing": True,
            "funding_error": "feed_unavailable",
        },
    )
    monkeypatch.setattr(
        paper_mod.InferenceBundle,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("paper inference must not run without derivative funding data")
        ),
    )

    model_path = tmp_path / "models" / "assets" / "ETH_USDT_USDT" / "signal_model.joblib"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        paper_mod,
        "resolve_signal_bundle",
        lambda *args, **kwargs: model_path.parent,
    )

    result = paper_mod.one_iteration(settings, tmp_path)

    assert result["status"] == "WAIT"
    assert result["signal"] == "FLAT"
    assert result["reason"] == ["funding_data_missing"]
    assert result["market_type"] == "swap"
