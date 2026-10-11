import pytest
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ai_trading_lab.config import load_settings
from ai_trading_lab.policy import live_signal_gate


def row(**overrides):
    base = {
        "p_up": 0.90,
        "probability_baseline": 0.39,
        "expected_return": 0.006,
        "expected_return_lcb": 0.004,
        "score": 0.40,
        "meta_success": 0.80,
        "model_disagreement": 0.01,
        "analog_n": 32,
        "analog_agreement": 0.85,
        "action": "LONG",
        "trade_window_available": True,
        "trade_window_ready": True,
        "trade_window_direction": "LONG",
        "trade_window_confidence": 0.90,
    }
    base.update(overrides)
    return pd.Series(base)


def test_strict_signal_gate_accepts_high_quality_signal():
    settings = replace(load_settings("config.yaml"), signal_only_mode=True)
    action, reasons = live_signal_gate(row(), settings)
    assert action == "LONG"
    assert reasons == []


def test_strict_signal_gate_rejects_low_confidence():
    settings = replace(load_settings("config.yaml"), signal_only_mode=True)
    action, reasons = live_signal_gate(row(p_up=0.61, expected_return_lcb=0.0001, score=0.10), settings)
    assert action == "FLAT"
    assert "signal_probability" in reasons


def test_strict_signal_gate_rejects_insufficient_memory_support():
    settings = replace(load_settings("config.yaml"), signal_only_mode=True)
    action, reasons = live_signal_gate(row(analog_n=2, analog_agreement=0.90), settings)
    assert action == "FLAT"
    assert "memory_neighbors" in reasons


def test_short_signal_uses_its_own_probability_and_return_head():
    settings = replace(
        load_settings("config.yaml"),
        signal_only_mode=True,
        require_short_borrow_cost=True,
        short_signals_enabled=True,
        short_borrow_bps_per_bar=1.0,
    )
    # Bearish p_up alone cannot stand in for the probability of a profitable short.
    missing_short_head = row(
        action="SHORT", trade_window_direction="SHORT", market_type="swap",
        p_up=0.10, expected_return=-0.006, expected_return_lcb=-0.012,
        expected_return_ucb=-0.004,
    )
    action, reasons = live_signal_gate(missing_short_head, settings)
    assert action == "FLAT"
    assert any("p_short" in reason for reason in reasons)

    risky = row(
        action="SHORT", trade_window_direction="SHORT", market_type="swap",
        p_up=0.10, expected_return=-0.006, expected_return_lcb=-0.012,
        expected_return_ucb=-0.004, p_short=0.90,
        short_probability_baseline=0.39, short_expected_return=0.002,
        short_expected_return_lcb=0.001, short_expected_return_ucb=0.004,
        short_meta_success=0.80, short_score=0.40,
        short_model_disagreement=0.01, short_analog_n=32,
        short_analog_agreement=0.85,
    )
    action, reasons = live_signal_gate(risky, settings)
    assert action == "FLAT"
    assert "signal_expected_return" in reasons

    robust = row(
        action="SHORT", trade_window_direction="SHORT", market_type="swap",
        p_up=0.10, expected_return=-0.006, expected_return_lcb=-0.012,
        expected_return_ucb=-0.004, p_short=0.90,
        short_probability_baseline=0.39, short_expected_return=0.006,
        short_expected_return_lcb=0.004, short_expected_return_ucb=0.008,
        short_meta_success=0.80, short_score=0.40,
        short_model_disagreement=0.01, short_analog_n=32,
        short_analog_agreement=0.85,
    )
    action, reasons = live_signal_gate(robust, settings)
    assert action == "SHORT"
    assert reasons == []


def test_short_economic_hurdle_includes_borrow():
    from ai_trading_lab.policy import decide_actions
    settings = replace(
        load_settings("config.yaml"),
        short_borrow_bps_per_bar=20.0,
        max_holding_bars=96,
    )
    idx = pd.date_range("2026-01-01", periods=1, freq="15min", tz="UTC")
    pred = pd.DataFrame({
        "p_up": [0.08],
        "expected_return": [-0.015],
        "expected_return_lcb": [-0.020],
        "expected_return_ucb": [-0.010],
        "model_disagreement": [0.0],
        "return_disagreement": [0.0],
    }, index=idx)
    analog = pd.DataFrame({"agreement":[0.9],"edge":[-0.01],"dispersion":[0.0],"n":[32]}, index=idx)
    regime = pd.Series(["high_vol_down"], index=idx)
    action, score = decide_actions(pred, regime, analog, np.array([0.9]), settings)
    assert action.iloc[0] in {"SHORT","FLAT"}
    assert score.attrs["economic_hurdle_bps"] >= 20.0 * 96


def test_short_is_disabled_by_default_even_for_bearish_predictions():
    settings = replace(
        load_settings("config.yaml"),
        signal_only_mode=True,
        short_signals_enabled=False,
        require_short_borrow_cost=False,
    )
    action, reasons = live_signal_gate(
        row(action="SHORT", trade_window_direction="SHORT", market_type="swap",
            p_up=0.10, expected_return=-0.006, expected_return_lcb=-0.012,
            expected_return_ucb=-0.004),
        settings,
    )

    assert action == "FLAT"
    assert "short_signals_disabled" in reasons


def test_short_is_rejected_for_spot_even_when_short_capability_is_enabled():
    settings = replace(
        load_settings("config.yaml"),
        signal_only_mode=True,
        short_signals_enabled=True,
        require_short_borrow_cost=True,
        short_borrow_bps_per_bar=1.0,
    )
    action, reasons = live_signal_gate(
        row(action="SHORT", trade_window_direction="SHORT", market_type="spot",
            p_up=0.10, expected_return=-0.006, expected_return_lcb=-0.012,
            expected_return_ucb=-0.004),
        settings,
    )

    assert action == "FLAT"
    assert "short_instrument_not_supported" in reasons


def test_live_assessment_empty_data_fails_closed(monkeypatch):
    import ai_trading_lab.live as live_mod
    from ai_trading_lab.live import assess_symbol
    settings = replace(load_settings("config.yaml"), symbol="BTC/USDT")
    monkeypatch.setattr(live_mod, "exchange_client", lambda *args, **kwargs: object())
    monkeypatch.setattr(live_mod, "fetch_ohlcv", lambda *args, **kwargs: pd.DataFrame())
    result = assess_symbol(settings, root=".")
    assert result.status == "WAIT"
    assert "empty_data" in result.reason_codes


def test_live_assessment_fetch_failure_fails_closed(monkeypatch):
    import ai_trading_lab.live as live_mod
    from ai_trading_lab.live import assess_symbol
    settings = replace(load_settings("config.yaml"), symbol="BTC/USDT")
    def boom(*args, **kwargs):
        raise RuntimeError("exchange unavailable")
    monkeypatch.setattr(live_mod, "exchange_client", lambda *args, **kwargs: object())
    monkeypatch.setattr(live_mod, "fetch_ohlcv", boom)
    result = assess_symbol(settings, root=".")
    assert result.status == "WAIT"
    assert any(x.startswith("data_fetch:RuntimeError") for x in result.reason_codes)

def test_research_profile_logs_looser_candidates_without_changing_strict_gate():
    settings = replace(
        load_settings("config.yaml"),
        signal_only_mode=True,
        trade_window_required_for_signal=True,
        research_trade_window_required_for_signal=False,
        short_signals_enabled=False,
    )
    candidate = row(
        p_up=0.65,
        probability_baseline=0.39,
        expected_return=0.003,
        expected_return_lcb=0.002,
        expected_return_ucb=0.004,
        score=0.10,
        meta_success=0.56,
        model_disagreement=0.10,
        analog_n=10,
        analog_agreement=0.60,
        trade_window_confidence=0.90,
    )
    strict_signal, strict_reasons = live_signal_gate(candidate, settings)
    research_signal, research_reasons = live_signal_gate(
        candidate, settings, profile="research"
    )

    assert strict_signal == "FLAT"
    assert "signal_probability" in strict_reasons
    assert research_signal == "LONG"
    assert research_reasons == []


def test_research_profile_cannot_bypass_short_or_funding_safety():
    settings = replace(
        load_settings("config.yaml"),
        signal_only_mode=True,
        short_signals_enabled=False,
        require_short_borrow_cost=True,
    )
    candidate = row(
        action="SHORT",
        market_type="swap",
        funding_data_missing=True,
        p_short=0.95,
        short_probability_baseline=0.39,
        short_expected_return=0.01,
        short_expected_return_lcb=0.008,
        short_expected_return_ucb=0.012,
        short_meta_success=0.95,
        short_score=0.90,
        short_model_disagreement=0.0,
        short_analog_n=32,
        short_analog_agreement=0.95,
    )
    signal, reasons = live_signal_gate(candidate, settings, profile="research")

    assert signal == "FLAT"
    assert "short_signals_disabled" in reasons
    assert "funding_data_missing" in reasons


def test_unknown_live_signal_profile_is_rejected():
    settings = replace(load_settings("config.yaml"), signal_only_mode=True)
    with pytest.raises(ValueError, match="unknown_signal_profile"):
        live_signal_gate(row(), settings, profile="experimental")

