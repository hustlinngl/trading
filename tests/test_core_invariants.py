from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from ai_trading_lab.config import load_settings
from ai_trading_lab.data import asof_join, drop_unclosed_tail, timeframe_offset, load_cached
from ai_trading_lab.master_tuner import _fold_cache_key
from ai_trading_lab.risk import RiskEngine
from ai_trading_lab.backtest import run_backtest
from ai_trading_lab.engine import AdaptiveEngine
from ai_trading_lab.features import make_oos_features
from ai_trading_lab.labels import triple_barrier_labels


def market_frame(n: int = 240, freq: str = "15min") -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    close = np.linspace(100.0, 112.0, n)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n, 100_000.0),
        },
        index=idx,
    )


def test_timeframe_and_unclosed_tail():
    df = market_frame(4)
    now = df.index[-1] + pd.Timedelta(minutes=7)
    out = drop_unclosed_tail(df, "15m", now=now)
    assert len(out) == 3
    assert timeframe_offset("15m") == pd.Timedelta(minutes=15)


def test_asof_join_respects_lag_and_never_looks_forward():
    base = pd.DataFrame(index=pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC"))
    source = pd.DataFrame(
        {"value": [10.0, 20.0, 30.0]},
        index=pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC"),
    )
    out = asof_join(base, source, lag=pd.Timedelta(minutes=15))
    assert np.isnan(out.iloc[0]["value"])
    assert out.iloc[1]["value"] == 10.0
    assert out.iloc[2]["value"] == 20.0
    assert out.iloc[3]["value"] == 30.0


def test_risk_sizing_respects_position_cap_and_economic_costs():
    risk = RiskEngine(
        initial_cash=10_000.0,
        risk_per_trade=0.01,
        max_position_pct=0.10,
        max_daily_loss_pct=0.02,
        stop_atr_mult=2.0,
        rr=2.0,
        fee_bps=7.0,
        slippage_bps=5.0,
        max_participation_pct=0.10,
        impact_bps_per_sqrt=1.5,
    )
    decision = risk.size(10_000.0, 100.0, 1.0, 1)
    assert decision.allowed
    assert decision.qty * 100.0 <= 1_000.0 + 1e-9
    assert decision.take_profit > 100.0 > decision.stop


def test_backtest_executes_on_next_bar_not_same_bar():
    df = market_frame(40)
    actions = pd.Series("FLAT", index=df.index)
    actions.iloc[4] = "LONG"
    risk = RiskEngine(10_000.0, 0.01, 0.25, 0.20, 1.0, 2.0, 0.0, 0.0, 1.0, 0.0)
    result = run_backtest(
        df.assign(atr_14=1.0),
        actions,
        risk,
        10_000.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        max_holding_bars=10,
        intrabar_barriers=False,
    )
    assert not result.trades.empty
    assert pd.Timestamp(result.trades.iloc[0]["entry_timestamp"]) == df.index[5]


def test_master_tuner_cache_key_accepts_fold_subset():
    settings = load_settings(Path("config.yaml"))
    df = market_frame(100)
    key_all = _fold_cache_key(df, settings)
    key_subset = _fold_cache_key(df, settings, {1, 3})
    assert key_all != key_subset
    assert len(key_all) == 24
    assert len(key_subset) == 24


def test_engine_exposes_live_inference_compatibility_api():
    settings = replace(load_settings(Path("config.yaml")), min_train_rows=100)
    engine = AdaptiveEngine(settings)
    features = engine.features(market_frame(120))
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    assert callable(engine.load)


def test_import_normalization_preserves_microstructure_and_rejects_conflicts():
    from ai_trading_lab.dataset import normalize_ohlcv
    idx = pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC")
    raw = pd.DataFrame({
        "timestamp": list(idx),
        "open": [100.0, 101.0, 102.0],
        "high": [101.0, 102.0, 103.0],
        "low": [99.0, 100.0, 101.0],
        "close": [100.5, 101.5, 102.5],
        "volume": [1000.0, 1100.0, 1200.0],
        "quote_volume": [100500.0, 111650.0, 123000.0],
        "trades": [10, 11, 12],
    })
    normalized = normalize_ohlcv(raw)
    assert {"quote_volume", "trades"}.issubset(normalized.columns)

    conflicting = pd.concat([raw, raw.iloc[[1]].assign(close=999.0)], ignore_index=True)
    try:
        normalize_ohlcv(conflicting)
    except ValueError as exc:
        assert "Conflicting duplicate timestamps" in str(exc)
    else:
        raise AssertionError("conflicting duplicate data must be rejected")


def test_short_borrow_reduces_short_risk_size():
    from ai_trading_lab.risk import RiskEngine
    base = dict(
        initial_cash=10_000.0,
        risk_per_trade=0.01,
        max_position_pct=1.0,
        max_daily_loss_pct=0.20,
        stop_atr_mult=2.0,
        rr=2.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        max_participation_pct=1.0,
        impact_bps_per_sqrt=0.0,
        max_holding_bars=96,
    )
    no_borrow = RiskEngine(**base, short_borrow_bps_per_bar=0.0).size(10_000.0, 100.0, 1.0, -1)
    with_borrow = RiskEngine(**base, short_borrow_bps_per_bar=10.0).size(10_000.0, 100.0, 1.0, -1)
    assert no_borrow.allowed and with_borrow.allowed
    assert with_borrow.qty < no_borrow.qty


def test_triple_barrier_same_bar_collision_is_ambiguous():
    idx = pd.date_range("2026-01-01", periods=20, freq="15min", tz="UTC")
    close = np.full(20, 100.0)
    df = pd.DataFrame({
        "open": close,
        "high": np.full(20, 101.0),
        "low": np.full(20, 99.0),
        "close": close,
        "volume": np.full(20, 10_000.0),
    }, index=idx)
    # Seed a non-zero ATR, then force both barriers inside the same executable bar.
    df.loc[idx[14], "high"] = 102.0
    df.loc[idx[14], "low"] = 98.0
    out = triple_barrier_labels(df, horizon=2, pt_atr=0.5, sl_atr=0.5)
    assert np.isnan(out.loc[idx[13], "tb_label"])
    assert np.isnan(out.loc[idx[13], "tb_return"])


def test_oos_feature_stitch_rejects_conflicting_overlap():
    history = market_frame(80)
    future = market_frame(10)
    future.index = pd.date_range(history.index[-1] - pd.Timedelta(minutes=15), periods=10, freq="15min", tz="UTC")
    future.iloc[0, future.columns.get_loc("close")] = 999.0
    try:
        make_oos_features(history, future, horizon=4)
    except ValueError as exc:
        assert "Conflicting overlapping OOS data" in str(exc)
    else:
        raise AssertionError("conflicting overlap must be rejected")


def test_oos_feature_stitch_allows_identical_overlap():
    history = market_frame(80)
    future = history.iloc[-1:].copy()
    out = make_oos_features(history, future, horizon=4)
    assert list(out.index) == list(future.index)


def test_cached_duplicate_microstructure_conflict_is_rejected():
    from ai_trading_lab.data import _deduplicate_ohlcv
    idx = pd.to_datetime(["2026-01-01 00:00:00+00:00"] * 2)
    df = pd.DataFrame({
        "open":[100.0,100.0],"high":[101.0,101.0],"low":[99.0,99.0],
        "close":[100.0,100.0],"volume":[1000.0,1000.0],"trades":[10.0,11.0],
    }, index=idx)
    try:
        _deduplicate_ohlcv(df)
    except ValueError as exc:
        assert "Conflicting duplicate market timestamp" in str(exc)
    else:
        raise AssertionError("conflicting microstructure duplicate must be rejected")


def test_master_tuner_rebuilds_both_conformal_bounds(monkeypatch):
    from ai_trading_lab.master_tuner import FoldSnapshot, _actions
    import ai_trading_lab.master_tuner as tuner_mod
    settings = load_settings(Path("config.yaml"))
    idx = pd.date_range("2026-01-01", periods=1, freq="15min", tz="UTC")
    pred = pd.DataFrame({
        "p_up":[0.10],
        "expected_return":[-0.004],
        "expected_return_lcb":[-0.004],
        "expected_return_ucb":[-0.004],
        "model_disagreement":[0.0],
        "return_disagreement":[0.0],
    }, index=idx)
    features = pd.DataFrame({"atr_pct":[0.01]}, index=idx)
    snap = FoldSnapshot(
        0, market_frame(1), features, pred,
        pd.Series(["high_vol_down"], index=idx),
        pd.DataFrame({"agreement":[0.9],"edge":[-0.01],"dispersion":[0.0],"n":[32]}, index=idx),
        np.array([0.9]), None, None, None, np.array([0.5]*20)
    )
    captured = {}
    def fake_decide(pred_in, *args, **kwargs):
        captured["pred"] = pred_in.copy()
        return pd.Series(["FLAT"], index=idx), pd.Series([0.0], index=idx)
    monkeypatch.setattr(tuner_mod, "decide_actions", fake_decide)
    _actions(snap, settings, {"conformal_level":0.90})
    rebuilt = captured["pred"]
    assert np.isclose(float(rebuilt["expected_return_lcb"].iloc[0]), -0.009)
    assert np.isclose(float(rebuilt["expected_return_ucb"].iloc[0]), 0.001)


def test_meta_target_accepts_directional_borrow_hurdle():
    from ai_trading_lab.meta import cost_aware_meta_target
    returns = np.array([-0.004, -0.020])
    p_up = np.array([0.10, 0.10])
    hurdle = np.array([0.005, 0.005])
    out = cost_aware_meta_target(returns, p_up, hurdle)
    assert out.tolist() == [0, 1]


def test_meta_learning_core_mask_purges_calibration_boundary():
    from ai_trading_lab.engine import _pre_calibration_core_mask
    idx = pd.date_range("2026-01-01", periods=20, freq="15min", tz="UTC")
    mask = _pre_calibration_core_mask(idx, idx[15], 3)
    assert mask.iloc[:12].all()
    assert not mask.iloc[12:].any()


def test_cross_asset_snapshot_is_bounded_by_anchor_time():
    from ai_trading_lab.cross_asset import cross_asset_snapshot
    idx = pd.date_range("2026-01-01", periods=12, freq="15min", tz="UTC")
    anchor = pd.DataFrame({"open":100.0,"high":101.0,"low":99.0,"close":np.arange(10)+100.0,"volume":1000.0}, index=idx[:10])
    other = pd.DataFrame({"open":100.0,"high":101.0,"low":99.0,"close":np.arange(12)+100.0,"volume":1000.0}, index=idx)
    out = cross_asset_snapshot({"BTC/USDT":anchor,"ETH/USDT":other},"BTC/USDT")
    assert "cross_asset_mean_return" in out


def test_auto_update_cache_hits_only_when_artifact_hash_matches(tmp_path, monkeypatch):
    import json
    from ai_trading_lab.autolearn import auto_update
    from ai_trading_lab.deployment import bundle_artifact_fingerprint, deployment_semantics_fingerprint, model_semantics_fingerprint
    settings = load_settings("config.yaml")
    bundle = tmp_path / "models"
    bundle.mkdir()
    artifact = bundle / "signal_model.joblib"
    artifact.write_bytes(b"stable")
    fp = "data-fp"
    state = {
        "data_fingerprint": fp,
        "model_semantics_fingerprint": model_semantics_fingerprint(settings),
        "deployment_semantics_fingerprint": deployment_semantics_fingerprint(settings),
        "bundle_artifact_fingerprint": bundle_artifact_fingerprint(bundle),
        "score": 0.1,
    }
    (bundle / "promotion_state.json").write_text(json.dumps(state), encoding="utf-8")
    import ai_trading_lab.autolearn as al
    monkeypatch.setattr(al, "strong_dataset_fingerprint", lambda df: fp)
    result = auto_update(pd.DataFrame(index=pd.date_range("2026-01-01", periods=2, freq="15min", tz="UTC")), settings, model_dir=bundle)
    assert result["skipped"] is True

    artifact.write_bytes(b"tampered")
    def should_not_skip(*args, **kwargs):
        raise AssertionError("artifact tampering must invalidate cache")
    monkeypatch.setattr(al, "evaluate_engine", should_not_skip)
    try:
        auto_update(pd.DataFrame(index=pd.date_range("2026-01-01", periods=2, freq="15min", tz="UTC")), settings, model_dir=bundle)
    except AssertionError:
        pass
    else:
        raise AssertionError("tampered artifact unexpectedly hit cache")
