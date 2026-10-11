"""Adversarial, offline checks for the evidence engine."""
import numpy as np
import pytest

from ai_trading_lab.spy_evidence import study_evidence


def test_no_trades_does_not_claim_edge_even_on_a_rising_market():
    r = study_evidence(np.full(160, 0.4), np.full(160, 0.01), trials=120)
    assert r["verdict"] == "NO_TRADES"
    assert r["trades"] == 0
    assert r["production_ready"] is False
    assert r["mean_daily_excess_return_vs_buy_hold"] < 0


def test_random_entry_placebo_prevents_claims_from_always_long():
    r = study_evidence(np.full(180, 0.8), np.full(180, 0.01), trials=120)
    assert r["trades"] == 180
    assert not r["beats_random_timing_placebo"]
    assert r["verdict"] != "PROMISING_BUT_UNCONFIRMED"


def test_clear_chronological_selectivity_is_promising_but_not_ready():
    p = np.tile([0.85, 0.45], 120)
    ret = np.tile([0.025, -0.025], 120)
    result = study_evidence(p, ret, block_days=10, trials=120)
    assert result["verdict"] == "PROMISING_BUT_UNCONFIRMED"
    assert result["production_ready"] is False
    assert result["requires_forward_paper_validation"]
    assert result["mean_daily_excess_90pct_block_bootstrap"][0] > 0.0
    assert result["beats_random_timing_placebo"]


def test_support_gate_never_promotes_tiny_subset():
    p = np.r_[np.full(10, 0.9), np.full(200, 0.1)]
    ret = np.r_[np.full(10, 0.15), np.full(200, -0.01)]
    r = study_evidence(p, ret, trials=120)
    assert r["verdict"] == "TOO_FEW_TRADES"
    assert r["production_ready"] is False


def test_evidence_is_deterministic_and_checks_inputs():
    p = np.linspace(0.2, 0.9, 180)
    ret = np.tile([0.03, -0.02], 90)
    one = study_evidence(p, ret, trials=120)
    two = study_evidence(p, ret, trials=120)
    assert one == two
    with pytest.raises(ValueError):
        study_evidence([0.8, 0.4], [0.1], trials=120)
    with pytest.raises(ValueError):
        study_evidence([1.2], [0.1], trials=120)
    with pytest.raises(ValueError):
        study_evidence([0.8], [np.nan], trials=120)
    with pytest.raises(ValueError):
        study_evidence([0.8], [0.1], trials=50)
    with pytest.raises(ValueError):
        study_evidence([0.8], [0.1], cost_bps_each_side=-1)


def test_very_negative_benchmark_does_not_grant_profitability():
    p = np.full(200, 0.9)
    r = np.full(200, -0.02)
    result = study_evidence(p, r, trials=120)
    assert result["verdict"] != "PROMISING_BUT_UNCONFIRMED"
    assert result["production_ready"] is False
