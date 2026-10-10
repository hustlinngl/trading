from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from ai_trading_lab.master_tuner import _placebo_test, _stability_test, _suggest
from ai_trading_lab.research_ledger import register_holdout_access, load_ledger
from ai_trading_lab.statistical_evidence import (
    bootstrap_max_drawdown,
    combinatorial_pbo,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)


def test_psr_and_dsr_are_invariant_to_annualization_when_benchmark_is_zero():
    values = np.random.default_rng(20261010).normal(0.0, 0.01, 1200)
    psr_daily = probabilistic_sharpe_ratio(values, periods_per_year=365.0)
    psr_intraday = probabilistic_sharpe_ratio(values, periods_per_year=35040.0)
    dsr_daily = deflated_sharpe_ratio(values, trials=100, periods_per_year=365.0)
    dsr_intraday = deflated_sharpe_ratio(values, trials=100, periods_per_year=35040.0)

    assert np.isclose(psr_daily, psr_intraday, atol=1e-12)
    assert np.isclose(dsr_daily, dsr_intraday, atol=1e-12)


def test_dsr_null_returns_do_not_get_annualization_boost():
    rng = np.random.default_rng(42)
    strategies = [rng.normal(0.0, 0.01, 2500) for _ in range(40)]
    scores = [deflated_sharpe_ratio(r, trials=40, periods_per_year=35040.0) for r in strategies]

    # Under the null, 40 strategies should not yield 1/3 of trials above a
    # high-confidence gate solely because observations are intraday bars.
    assert sum(score >= 0.80 for score in scores) <= 1


def test_dsr_increases_selection_penalty_for_more_trials():
    values = np.random.default_rng(21).normal(0.0002, 0.01, 1000)
    assert deflated_sharpe_ratio(values, trials=1000) <= deflated_sharpe_ratio(values, trials=10)


def test_pbo_identical_candidates_are_unavailable_not_perfect():
    base = np.linspace(-0.1, 0.1, 8)
    result = combinatorial_pbo(np.column_stack([base, base, base, base]), partitions=8)

    assert result["available"] is False
    assert result["pbo"] is None
    assert result["reason"] == "degenerate_identical_candidates"


def test_pbo_zero_trade_constant_scores_are_unavailable():
    result = combinatorial_pbo(np.zeros((8, 20)), partitions=8)

    assert result["available"] is False
    assert result["pbo"] is None
    assert "degenerate" in result["reason"]


def test_pbo_non_degenerate_candidate_matrix_produces_usable_result():
    matrix = np.random.default_rng(123).normal(0.0, 0.02, (8, 12))
    result = combinatorial_pbo(matrix, partitions=8, seed=7)

    assert result["available"] is True
    assert 0.0 <= result["pbo"] <= 1.0
    assert result["effective_candidates"] == 12
    assert result["combinations"] > 0


def test_no_trade_stability_cannot_pass_vacuously(monkeypatch):
    import ai_trading_lab.master_tuner as tuner

    rows = pd.DataFrame({
        "fold": list(range(6)),
        "cost_multiplier": [1.0] * 6,
        "trades": [0] * 6,
        "robust_score": [0.0] * 6,
    })
    monkeypatch.setattr(tuner, "_evaluate_params", lambda *args, **kwargs: {"objective": 0.0, "rows": rows})
    result = _stability_test(folds=list(range(6)), settings=object(), best={}, n=8)

    assert result["available"] is False
    assert result["stable"] is False
    assert result["reason"] == "insufficient_trading_activity"


def test_no_trade_placebo_cannot_pass_vacuously(monkeypatch):
    import ai_trading_lab.master_tuner as tuner

    rows = pd.DataFrame({
        "fold": list(range(6)),
        "cost_multiplier": [1.0] * 6,
        "trades": [0] * 6,
        "robust_score": [0.0] * 6,
    })
    monkeypatch.setattr(tuner, "_evaluate_params", lambda *args, **kwargs: {"objective": 0.0, "rows": rows})
    result = _placebo_test(folds=list(range(6)), settings=object(), params={})

    assert result["available"] is False
    assert result["passes_95pct_negative_control"] is False
    assert result["reason"] == "insufficient_real_trading_activity"


def test_daily_shifted_overlapping_holdout_is_not_pristine_when_dataset_hash_changes(tmp_path):
    path = tmp_path / "research_ledger.json"
    first = register_holdout_access(
        path, dataset_fingerprint="fingerprint-day-1",
        holdout_start="2026-06-01T00:00:00+00:00",
        holdout_end="2026-06-30T23:45:00+00:00",
        holdout_frac=0.15, purpose="nightly-run",
        scope="BTC/USDT|15m|adaptive", run_id="run-1",
    )
    second = register_holdout_access(
        path, dataset_fingerprint="fingerprint-day-2",
        holdout_start="2026-06-02T00:00:00+00:00",
        holdout_end="2026-07-01T23:45:00+00:00",
        holdout_frac=0.15, purpose="nightly-run",
        scope="BTC/USDT|15m|adaptive", run_id="run-2",
    )

    assert first["pristine"] is True
    assert second["pristine"] is False
    assert second["overlapping_prior_runs"] == ["run-1"]


def test_unrelated_asset_holdout_does_not_poison_scoped_pristine_status(tmp_path):
    path = tmp_path / "research_ledger.json"
    register_holdout_access(
        path, dataset_fingerprint="btc",
        holdout_start="2026-06-01T00:00:00+00:00",
        holdout_end="2026-06-30T23:45:00+00:00",
        holdout_frac=0.15, purpose="run",
        scope="BTC/USDT|15m|adaptive", run_id="btc-run",
    )
    second = register_holdout_access(
        path, dataset_fingerprint="eth",
        holdout_start="2026-06-02T00:00:00+00:00",
        holdout_end="2026-07-01T23:45:00+00:00",
        holdout_frac=0.15, purpose="run",
        scope="ETH/USDT|15m|adaptive", run_id="eth-run",
    )

    assert second["pristine"] is True


def test_corrupt_existing_ledger_fails_closed_instead_of_resetting_trial_count(tmp_path):
    path = tmp_path / "research_ledger.json"
    path.write_text("{corrupt", encoding="utf-8")

    try:
        load_ledger(path)
    except RuntimeError as exc:
        assert "research_ledger_unreadable" in str(exc)
    else:
        raise AssertionError("Corrupt persisted trial history must not reset to a pristine ledger")


def test_master_tuner_uses_configured_execution_geometry():
    class FakeTrial:
        def suggest_float(self, name, low, high, **kwargs):
            return float((low + high) / 2.0)

    geometry = {"stop_atr_mult": 1.8, "take_profit_rr": 2.2, "max_holding_bars": 96}
    suggested = _suggest(FakeTrial(), settings=object(), aligned=geometry)

    assert suggested["stop_atr_mult"] == 1.8
    assert suggested["take_profit_rr"] == 2.2
    assert suggested["max_holding_bars"] == 96


def test_bootstrap_drawdown_does_not_return_zero_risk_for_insufficient_samples():
    result = bootstrap_max_drawdown(np.zeros(5), reps=1000)

    assert result["available"] is False
    assert result["median_drawdown"] is None
    assert result["prob_drawdown_below_20pct"] is None


def test_live_inference_rejects_missing_feature_columns_before_imputation():
    from ai_trading_lab.engine import AdaptiveEngine

    engine = AdaptiveEngine.__new__(AdaptiveEngine)
    engine.settings = SimpleNamespace(live_max_missing_feature_fraction=0.0)
    engine.model = SimpleNamespace(feature_cols=["required_a", "required_b"])
    frame = pd.DataFrame({"required_a": [1.0]}, index=pd.date_range("2026-01-01", periods=1, tz="UTC"))

    try:
        engine.predict_frame(frame, strict=True)
    except ValueError as exc:
        assert "missing_live_feature_columns:required_b" in str(exc)
    else:
        raise AssertionError("Missing trained features must not be filled silently")


def test_live_inference_rejects_non_finite_feature_fraction_before_imputation():
    from ai_trading_lab.engine import AdaptiveEngine

    engine = AdaptiveEngine.__new__(AdaptiveEngine)
    engine.settings = SimpleNamespace(live_max_missing_feature_fraction=0.0)
    engine.model = SimpleNamespace(feature_cols=["required_a", "required_b"])
    frame = pd.DataFrame(
        {"required_a": [1.0], "required_b": [np.nan]},
        index=pd.date_range("2026-01-01", periods=1, tz="UTC"),
    )

    try:
        engine.predict_frame(frame, strict=True)
    except ValueError as exc:
        assert "live_feature_missing_fraction" in str(exc)
        assert "required_b" in str(exc)
    else:
        raise AssertionError("A non-finite newest-bar feature must fail closed")


def test_intensive_research_workflow_restores_and_persists_governance_ledger():
    from pathlib import Path

    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "workflows"
        / "intensive-real-research.yml"
    ).read_text(encoding="utf-8")

    assert "contents: write" in workflow
    assert "Restore persistent research ledger" in workflow
    assert "Persist research ledger across runs" in workflow
    assert "refs/heads/research-ledger" in workflow
    assert "if: always()" in workflow
