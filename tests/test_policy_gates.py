from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from ai_trading_lab.config import load_settings
from ai_trading_lab.policy import live_signal_gate


def row(**overrides):
    base = {
        "p_up": 0.90,
        "expected_return": 0.006,
        "expected_return_lcb": 0.004,
        "score": 0.40,
        "meta_success": 0.80,
        "model_disagreement": 0.01,
        "analog_n": 32,
        "analog_agreement": 0.85,
        "action": "LONG",
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


def test_short_signal_uses_upper_bound_for_directional_uncertainty():
    settings = replace(load_settings("config.yaml"), signal_only_mode=True)
    # A bearish forecast is only robust if the upper confidence bound remains below zero.
    risky = row(p_up=0.10, expected_return=-0.004, expected_return_lcb=-0.010, expected_return_ucb=0.002)
    action, reasons = live_signal_gate(risky, settings)
    assert action == "FLAT"
    assert "signal_expected_return" in reasons

    robust = row(p_up=0.10, expected_return=-0.006, expected_return_lcb=-0.012, expected_return_ucb=-0.002)
    action, reasons = live_signal_gate(robust, settings)
    assert action == "SHORT"
    assert reasons == []
