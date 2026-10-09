from __future__ import annotations

import pytest

from ai_trading_lab.live import LiveAssessment
from ai_trading_lab.signal_contract import (
    PUBLIC_SIGNAL_FIELDS,
    DirectSignal,
    SignalContractError,
    compile_direct_signal,
)


def test_direct_signal_is_exactly_six_fields():
    signal = DirectSignal(
        symbol="BTC/USDT",
        signal="LONG",
        confidence=0.91,
        expected_return=0.006,
        price=100000.0,
        horizon_bars=8,
    )
    payload = signal.to_dict()

    assert set(payload) == set(PUBLIC_SIGNAL_FIELDS)
    assert payload == {
        "symbol": "BTC/USDT",
        "signal": "LONG",
        "confidence": 0.91,
        "expected_return": 0.006,
        "price": 100000.0,
        "horizon_bars": 8,
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"signal": "BUY"},
        {"confidence": 1.1},
        {"price": 0.0},
        {"horizon_bars": 0},
    ],
)
def test_direct_signal_rejects_invalid_public_values(kwargs):
    base = {
        "symbol": "BTC/USDT",
        "signal": "LONG",
        "confidence": 0.91,
        "expected_return": 0.006,
        "price": 100000.0,
        "horizon_bars": 8,
    }
    base.update(kwargs)
    with pytest.raises(SignalContractError):
        DirectSignal(**base)


def test_live_assessment_public_payload_never_leaks_internal_state():
    assessment = LiveAssessment(
        "BTC/USDT",
        "2026-10-08T11:30:00+00:00",
        "SIGNAL",
        "SHORT",
        0.88,
        -0.004,
        100000.0,
        ["internal_gate", "bundle_ok"],
        "private-fingerprint",
        {"score": 0.42, "regime": "trend_down"},
    )

    payload = assessment.to_dict()

    assert set(payload) == set(PUBLIC_SIGNAL_FIELDS)
    assert payload["signal"] == "SHORT"
    assert payload["confidence"] == 0.88
    # Internal expected_return is the LONG-side forecast; public SHORT
    # payloads express return in the direction of the emitted signal.
    assert payload["expected_return"] == 0.004
    assert "reason_codes" not in payload
    assert "data_fingerprint" not in payload
    assert "details" not in payload



def test_compiled_signal_expected_return_is_directional():
    common = {
        "symbol": "BTC/USDT",
        "timestamp": "2026-10-09T12:00:00+00:00",
        "price": 100000.0,
        "horizon_bars": 8,
    }
    long = compile_direct_signal(
        {"action": "LONG", "p_up": 0.8, "expected_return": 0.006},
        **common,
    )
    short = compile_direct_signal(
        {"action": "SHORT", "p_up": 0.2, "expected_return": -0.006},
        **common,
    )
    flat = compile_direct_signal(
        {"action": "FLAT", "p_up": 0.2, "expected_return": -0.006},
        **common,
    )

    assert long.expected_return == 0.006
    assert short.expected_return == 0.006
    assert flat.expected_return == 0.006


def test_compiled_signal_rejects_action_direction_mismatch():
    with pytest.raises(SignalContractError, match="action_probability_mismatch"):
        compile_direct_signal(
            {"action": "SHORT", "p_up": 0.8, "expected_return": -0.006},
            symbol="BTC/USDT",
            timestamp="2026-10-09T12:00:00+00:00",
            price=100000.0,
            horizon_bars=8,
        )
