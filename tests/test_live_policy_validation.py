from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ai_trading_lab import live_policy_validation as validation


def _settings():
    return SimpleNamespace(
        symbol="BTC/USDT",
        timeframe="15m",
        seed=11,
        trade_window_required_for_signal=True,
        backtest_market_type="spot",
        signal_only_mode=True,
        signal_confidence_threshold=0.82,
        signal_probability_threshold=0.72,
        signal_min_expected_return=0.003,
        signal_meta_threshold=0.62,
        signal_min_score=0.22,
        signal_max_disagreement=0.05,
        signal_memory_min_neighbors=16,
        signal_memory_min_agreement=0.70,
        short_signals_enabled=False,
        require_short_borrow_cost=True,
        short_borrow_bps_per_bar=0.0,
        base_min_holdout_trades=20,
        master_optimization_folds=6,
        master_trials=3,
        initial_cash=10000.0,
        fee_bps=7.0,
        slippage_bps=5.0,
    )


def _market(rows=120):
    idx = pd.date_range("2025-01-01", periods=rows, freq="15min", tz="UTC")
    close = 100.0 + np.linspace(0.0, 3.0, rows)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(rows, 10000.0),
            "atr_14": np.full(rows, 1.0),
        },
        index=idx,
    )


def test_combined_gate_rejects_missing_live_fields():
    idx = pd.date_range("2025-01-01", periods=2, freq="15min", tz="UTC")
    predictions = pd.DataFrame({"action": ["LONG", "LONG"]}, index=idx)
    specialist = pd.DataFrame(
        {
            "trade_window_available": [True, True],
            "trade_window_ready": [True, True],
            "trade_window_direction": ["LONG", "LONG"],
            "trade_window_confidence": [0.99, 0.99],
        },
        index=idx,
    )
    actions = validation._actions_for_gate(predictions, specialist, _settings())
    assert actions.tolist() == ["FLAT", "FLAT"]


def test_holdout_builder_persists_combined_gate_provenance(monkeypatch, tmp_path):
    settings = _settings()
    settings.research_ledger_path = str(tmp_path / "research_ledger.json")
    full = _market()
    train = full.iloc[:90].copy()
    holdout = full.iloc[90:].copy()
    predictions = pd.DataFrame(
        {
            "p_up": 0.95,
            "expected_return": 0.02,
            "expected_return_lcb": 0.01,
            "expected_return_ucb": 0.03,
            "meta_success": 0.95,
            "score": 0.95,
            "model_disagreement": 0.0,
            "analog_n": 32,
            "analog_agreement": 0.95,
            "action": "LONG",
        },
        index=holdout.index,
    )
    specialist = pd.DataFrame(
        {
            "trade_window_available": True,
            "trade_window_ready": True,
            "trade_window_direction": "LONG",
            "trade_window_confidence": 0.99,
            "trade_window_event_probability": 0.99,
            "trade_window_reason": "ok",
        },
        index=full.index,
    )
    monkeypatch.setattr(
        "ai_trading_lab.trade_window.score_trade_window_frame",
        lambda *args, **kwargs: specialist,
    )

    trade_pnl = np.linspace(-1.0, 2.0, 30)
    fake_trades = pd.DataFrame(
        {"pnl_net": trade_pnl, "entry_equity": np.full(30, 10000.0)}
    )

    def fake_backtest(market, actions, settings):
        return SimpleNamespace(
            stats={
                "trades": 30,
                "total_return": 0.10,
                "net_compounded_return": 0.10,
                "profit_factor": 1.5,
                "max_drawdown": -0.05,
                "sharpe_like": 1.0,
                "sortino_like": 1.0,
                "top_trade_share": 0.1,
            },
            trades=fake_trades,
        )

    monkeypatch.setattr(validation, "run_configured_backtest", fake_backtest)
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    (asset_dir / "base_holdout_report.json").write_text(
        json.dumps(
            {
                "validation_holdout_data_fingerprint": validation.strong_dataset_fingerprint(holdout.drop(columns=["atr_14"], errors="ignore")),
                "validation_holdout_start": str(holdout.index.min()),
                "validation_holdout_end": str(holdout.index.max()),
            }
        ),
        encoding="utf-8",
    )
    output = asset_dir / "live_policy_holdout_report.json"
    report = validation.build_live_policy_holdout_report(
        full_df=full,
        train_df=train,
        holdout_df=holdout,
        holdout_predictions=predictions,
        settings=settings,
        specialist_path=asset_dir / "specialist.joblib",
        output_path=output,
    )
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["live_signal_gate_applied"] is True
    assert report["specialist_gate_applied"] is True
    assert report["holdout_governance"]["pristine"] is True
    assert persisted["validation_holdout_data_fingerprint"] == validation.strong_dataset_fingerprint(holdout.drop(columns=["atr_14"], errors="ignore"))
    assert persisted["holdout_start"] == str(holdout.index.min())
    assert persisted["statistical_evidence"]["deflated_sharpe_ratio_available"] is True
    assert persisted["statistical_evidence"]["deflated_sharpe_ratio"] is not None
    assert len(persisted["variant_diagnostics"]) == 3
