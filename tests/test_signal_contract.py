from __future__ import annotations

import math

import pytest

from ai_trading_lab.signal_contract import (
    PUBLIC_SIGNAL_FIELDS,
    DirectSignal,
    SignalContractError,
    compile_direct_signal,
)


def test_long_compiles_to_the_exact_six_field_public_contract():
    signal = compile_direct_signal(
        {"action": "LONG", "p_up": 0.91, "expected_return": 0.0062, "score": 999},
        symbol="BTC/USDT",
        timestamp="2026-10-08T00:00:00+00:00",
        price=100000.0,
        horizon_bars=8,
    )
    assert signal.to_dict() == {
        "symbol": "BTC/USDT",
        "signal": "LONG",
        "confidence": 0.91,
        "expected_return": 0.0062,
        "price": 100000.0,
        "horizon_bars": 8,
    }
    assert set(signal.to_dict()) == PUBLIC_SIGNAL_FIELDS


def test_short_confidence_is_directional():
    signal = compile_direct_signal(
        {"action": "SHORT", "p_up": 0.17, "expected_return": -0.004},
        symbol="ETH/USDT",
        timestamp="2026-10-08T00:00:00+00:00",
        price=4000.0,
        horizon_bars=8,
    )
    assert signal.signal == "SHORT"
    assert signal.confidence == pytest.approx(0.83)


def test_flat_has_zero_confidence_without_actionability_telemetry():
    signal = compile_direct_signal(
        {"action": "FLAT", "p_up": 0.54, "expected_return": 0.0001},
        symbol="SOL/USDT",
        timestamp="2026-10-08T00:00:00+00:00",
        price=200.0,
        horizon_bars=8,
    )
    assert signal.signal == "FLAT"
    assert signal.confidence == 0.0
    assert set(signal.to_dict()) == PUBLIC_SIGNAL_FIELDS


@pytest.mark.parametrize(
    "row",
    [
        {"action": "LONG", "p_up": math.nan, "expected_return": 0.01},
        {"action": "LONG", "p_up": 1.2, "expected_return": 0.01},
        {"action": "LONG", "p_up": 0.8, "expected_return": math.nan},
    ],
)
def test_non_finite_or_invalid_model_output_fails_closed(row):
    with pytest.raises(SignalContractError):
        compile_direct_signal(
            row,
            symbol="BTC/USDT",
            timestamp="2026-10-08T00:00:00+00:00",
            price=100000.0,
            horizon_bars=8,
        )


def test_public_contract_does_not_leak_internal_fields():
    signal = compile_direct_signal(
        {
            "action": "LONG",
            "p_up": 0.9,
            "expected_return": 0.005,
            "regime": "trend_up",
            "analog_edge": 0.4,
            "meta_success": 0.8,
            "selection_score": 0.7,
            "reason": "private_policy_reason",
        },
        symbol="BTC/USDT",
        timestamp="2026-10-08T00:00:00+00:00",
        price=100000.0,
        horizon_bars=8,
    )
    public = signal.to_dict()
    assert set(public) == PUBLIC_SIGNAL_FIELDS
    assert "regime" not in public
    assert "analog_edge" not in public
    assert "meta_success" not in public
    assert "reason" not in public
    assert "timestamp" not in public
    assert "actionable" not in public


def test_direct_signal_rejects_invalid_price_and_horizon():
    with pytest.raises(SignalContractError, match="invalid_price"):
        DirectSignal("BTC/USDT", "LONG", 0.9, 0.01, 0.0, 8)
    with pytest.raises(SignalContractError, match="invalid_horizon"):
        DirectSignal("BTC/USDT", "LONG", 0.9, 0.01, 100.0, 0)


def test_engine_rejects_a_missing_required_feature_before_prediction():
    import pandas as pd
    from types import SimpleNamespace

    from ai_trading_lab.engine import AdaptiveEngine

    engine = AdaptiveEngine.__new__(AdaptiveEngine)
    engine.model = SimpleNamespace(feature_cols=["feature_a", "feature_b"])
    features = pd.DataFrame({"feature_a": [1.0]})

    with pytest.raises(ValueError, match=r"missing_live_features:feature_b"):
        engine.predict_frame(features, strict=True)
