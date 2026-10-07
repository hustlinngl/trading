from __future__ import annotations

import json

from ai_trading_lab.research_ledger import (
    register_holdout_access,
    register_trials,
    total_trials,
)
from ai_trading_lab.signal_contract import direct_signal


def test_direct_signal_contract_contains_only_decision_fields():
    signal = direct_signal(
        symbol="BTC/USDT",
        timestamp="2026-10-08T00:00:00+00:00",
        signal="LONG",
        confidence=0.91,
        expected_return=0.006,
        price=100000.0,
        horizon_bars=8,
    )
    payload = signal.to_dict()
    assert payload["signal"] == "LONG"
    assert payload["confidence"] == 0.91
    assert payload["actionable"] is True
    assert payload["horizon_bars"] == 8
    assert "regime" not in payload
    assert "analog_n" not in payload
    assert "meta_success" not in payload


def test_research_ledger_accumulates_trials_and_holdout_governance(tmp_path):
    path = tmp_path / "ledger.json"
    assert total_trials(path) == 0
    first = register_trials(path, 250, kind="strategy_evolution")
    second = register_trials(path, 40, kind="master_tune")
    assert first["cumulative_trials_after"] == 250
    assert second["cumulative_trials_after"] == 290
    assert total_trials(path) == 290

    first_access = register_holdout_access(
        path,
        dataset_fingerprint="fp",
        holdout_start="2026-01-01",
        holdout_end="2026-02-01",
        holdout_frac=0.15,
        purpose="master_tune_final_holdout",
        run_id="run-1",
    )
    same_run = register_holdout_access(
        path,
        dataset_fingerprint="fp",
        holdout_start="2026-01-01",
        holdout_end="2026-02-01",
        holdout_frac=0.15,
        purpose="master_tune_final_holdout",
        run_id="run-1",
    )
    next_run = register_holdout_access(
        path,
        dataset_fingerprint="fp",
        holdout_start="2026-01-01",
        holdout_end="2026-02-01",
        holdout_frac=0.15,
        purpose="master_tune_final_holdout",
        run_id="run-2",
    )
    assert first_access["pristine"] is True
    assert same_run["pristine"] is True
    assert next_run["pristine"] is False
    assert json.loads(path.read_text())["total_trials"] == 290
