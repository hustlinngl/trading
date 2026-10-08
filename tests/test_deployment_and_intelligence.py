from __future__ import annotations

from pathlib import Path

from ai_trading_lab.config import load_settings
from ai_trading_lab.deployment import asset_bundle_dir, resolve_signal_bundle, bundle_compatibility, bundle_artifact_fingerprint, ENGINE_ARTIFACTS
from ai_trading_lab.external_intelligence import ExaClient, TavilyClient, extract_event_terms


def test_asset_bundle_resolution_is_symbol_safe(tmp_path):
    settings = load_settings("config.yaml")
    settings.symbol = "BTC/USDT"
    btc = asset_bundle_dir(tmp_path, "BTC/USDT")
    eth = asset_bundle_dir(tmp_path, "ETH/USDT")
    btc.mkdir(parents=True)
    (btc / "signal_model.joblib").write_bytes(b"sentinel")

    assert resolve_signal_bundle(settings, tmp_path, "BTC/USDT") == btc
    assert resolve_signal_bundle(settings, tmp_path, "ETH/USDT") == eth


def test_bundle_compatibility_rejects_wrong_identity(tmp_path):
    settings = load_settings("config.yaml")
    settings.symbol = "BTC/USDT"
    settings.timeframe = "15m"
    bundle = asset_bundle_dir(tmp_path, "ETH/USDT")
    bundle.mkdir(parents=True)
    (bundle / "base_training_meta.json").write_text(
        '{"symbol":"ETH/USDT","timeframe":"15m"}',
        encoding="utf-8",
    )
    ok, reason = bundle_compatibility(settings, bundle, "BTC/USDT")
    assert not ok
    assert reason == "model_symbol_mismatch"


def test_event_features_are_deterministic():
    features = extract_event_terms("Fed FOMC hawkish interest rate hike with ETF outflow and exploit risk")
    assert features["event_rates_hits"] >= 2
    assert features["event_flows_hits"] >= 1
    assert features["event_security_hits"] >= 1
    assert features["event_risk_balance"] < 0


def test_external_clients_fail_closed_without_credentials(monkeypatch):
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert ExaClient().search("test", num_results=2) == []
    assert TavilyClient().search("test", num_results=2) == []


def test_strict_signal_mode_rejects_unlabelled_model(tmp_path):
    settings = load_settings("config.yaml")
    bundle = asset_bundle_dir(tmp_path, "BTC/USDT")
    bundle.mkdir(parents=True)
    (bundle / "signal_model.joblib").write_bytes(b"sentinel")
    ok, reason = bundle_compatibility(settings, bundle, "BTC/USDT")
    assert not ok
    assert reason == "model_metadata_missing"


def test_research_router_is_deterministic():
    from ai_trading_lab.experiment_router import ResearchRouter, ResearchTask
    tasks = [
        ResearchTask("b","B", "h", 0.0, 0.8, 0.2, 1.0, 0.5),
        ResearchTask("a","A", "h", 0.0, 0.8, 0.2, 1.0, 0.5),
        ResearchTask("c","C", "h", 0.0, 0.2, 0.8, 2.0, 0.9),
    ]
    router1 = ResearchRouter(seed=7)
    router2 = ResearchRouter(seed=7)
    assert [x.task_id for x in router1.rank(tasks, budget=10.0, top_k=3)] == [x.task_id for x in router2.rank(tasks, budget=10.0, top_k=3)]


def test_bundle_compatibility_rejects_training_semantics_mismatch(tmp_path):
    from ai_trading_lab.deployment import model_semantics_fingerprint, deployment_semantics_fingerprint, bundle_artifact_fingerprint
    settings = load_settings("config.yaml")
    settings.symbol = "BTC/USDT"
    settings.timeframe = "15m"
    bundle = asset_bundle_dir(tmp_path, "BTC/USDT")
    bundle.mkdir(parents=True)
    (bundle / "base_training_meta.json").write_text(
        '{"symbol":"BTC/USDT","timeframe":"15m","model_semantics_fingerprint":"incompatible"}',
        encoding="utf-8",
    )
    ok, reason = bundle_compatibility(settings, bundle, "BTC/USDT")
    assert not ok
    assert reason == "model_semantics_mismatch"
    assert model_semantics_fingerprint(settings)


def test_bundle_compatibility_rejects_stale_manifest_provenance(tmp_path):
    import json
    from ai_trading_lab.deployment import model_semantics_fingerprint, deployment_semantics_fingerprint
    settings = load_settings("config.yaml")
    bundle = asset_bundle_dir(tmp_path, "BTC/USDT")
    bundle.mkdir(parents=True)
    fp = "dataset-current"
    sem = model_semantics_fingerprint(settings)
    dep_sem = deployment_semantics_fingerprint(settings)
    for name in ENGINE_ARTIFACTS:
        (bundle / name).write_bytes(name.encode())
    (bundle / "trade_window_specialist.joblib").write_bytes(b"window")
    (bundle / "base_training_meta.json").write_text(
        json.dumps({"symbol":"BTC/USDT","timeframe":"15m","data_fingerprint":fp,"model_semantics_fingerprint":sem,"deployment_semantics_fingerprint":dep_sem}),
        encoding="utf-8",
    )
    (bundle / "deployment_manifest.json").write_text(
        json.dumps({"ready":True,"symbol":"BTC/USDT","timeframe":"15m","data_fingerprint":"dataset-old","model_semantics_fingerprint":sem,"deployment_semantics_fingerprint":dep_sem,"bundle_artifact_fingerprint":bundle_artifact_fingerprint(bundle)}),
        encoding="utf-8",
    )
    ok, reason = bundle_compatibility(settings, bundle, "BTC/USDT")
    assert not ok
    assert reason == "deployment_manifest_data_mismatch"


def test_refresh_deployment_manifest_requires_matching_evidence(tmp_path):
    import json
    from ai_trading_lab.deployment import model_semantics_fingerprint, deployment_semantics_fingerprint, refresh_deployment_manifest
    settings = load_settings("config.yaml")
    bundle = asset_bundle_dir(tmp_path, "BTC/USDT")
    bundle.mkdir(parents=True)
    fp = "dataset-current"
    sem = model_semantics_fingerprint(settings)
    dep_sem = deployment_semantics_fingerprint(settings)
    for name in ENGINE_ARTIFACTS:
        (bundle / name).write_bytes(name.encode())
    (bundle / "trade_window_specialist.joblib").write_bytes(b"window")
    (bundle / "base_training_meta.json").write_text(
        json.dumps({"symbol":"BTC/USDT","timeframe":"15m","data_fingerprint":fp,"model_semantics_fingerprint":sem,"deployment_semantics_fingerprint":dep_sem}),
        encoding="utf-8",
    )
    base_stats = {
        "total_return":0.05,"benchmark_return":0.0,"sharpe_like":1.0,"sortino_like":1.0,
        "trades":25,"profit_factor":1.4,"max_drawdown":-0.10,"top_trade_share":0.10,
    }
    duration_stats = {**base_stats,"trades":15}
    (bundle / "base_holdout_report.json").write_text(
        json.dumps({"symbol":"BTC/USDT","timeframe":"15m","data_fingerprint":fp,"validation_train_data_fingerprint":"train-fp","validation_holdout_data_fingerprint":"holdout-fp","validation_holdout_start":"2026-01-01T00:00:00+00:00","validation_holdout_end":"2026-03-01T00:00:00+00:00","validation_holdout_frac":0.15,"model_semantics_fingerprint":sem,"deployment_semantics_fingerprint":dep_sem,"holdout":base_stats}),
        encoding="utf-8",
    )
    (bundle / "trade_window_training_report.json").write_text(
        json.dumps({"production_ready":True,"symbol":"BTC/USDT","timeframe":"15m","data_fingerprint":fp,"model_semantics_fingerprint":sem,"deployment_semantics_fingerprint":dep_sem,"holdout":{"backtest":duration_stats}}),
        encoding="utf-8",
    )
    manifest = refresh_deployment_manifest(settings, tmp_path)
    assert manifest["ready"] is True
    assert manifest["data_fingerprint"] == fp
    assert manifest["model_semantics_fingerprint"] == sem
    assert manifest["bundle_artifact_fingerprint"] == bundle_artifact_fingerprint(bundle)


def test_incomplete_bundle_fails_cleanly_before_manifest_evaluation(tmp_path):
    import json
    settings = load_settings("config.yaml")
    settings.signal_only_mode = False
    settings.require_deployment_manifest_for_signal = True
    bundle = asset_bundle_dir(tmp_path, "BTC/USDT")
    bundle.mkdir(parents=True)
    (bundle / "deployment_manifest.json").write_text(
        json.dumps({"ready":True,"symbol":"BTC/USDT","timeframe":"15m"}),
        encoding="utf-8",
    )
    ok, reason = bundle_compatibility(settings, bundle, "BTC/USDT")
    assert not ok
    assert reason == "deployment_bundle_incomplete"


def test_asset_deployment_manifest_respects_root(tmp_path):
    from ai_trading_lab.main import asset_deployment_manifest
    settings = load_settings("config.yaml")
    settings.symbol = "BTC/USDT"
    assert asset_deployment_manifest(settings, tmp_path) == asset_bundle_dir(tmp_path, "BTC/USDT") / "deployment_manifest.json"


def test_bundle_compatibility_rejects_runtime_policy_mismatch(tmp_path):
    import json
    from ai_trading_lab.deployment import model_semantics_fingerprint, deployment_semantics_fingerprint
    settings = load_settings("config.yaml")
    bundle = asset_bundle_dir(tmp_path, "BTC/USDT")
    bundle.mkdir(parents=True)
    fp = "dataset-current"
    sem = model_semantics_fingerprint(settings)
    dep_sem = deployment_semantics_fingerprint(settings)
    (bundle / "base_training_meta.json").write_text(
        json.dumps({"symbol":"BTC/USDT","timeframe":"15m","data_fingerprint":fp,"model_semantics_fingerprint":sem,"deployment_semantics_fingerprint":dep_sem}),
        encoding="utf-8",
    )
    settings.min_edge_after_cost_bps = float(settings.min_edge_after_cost_bps) + 1.0
    ok, reason = bundle_compatibility(settings, bundle, "BTC/USDT")
    assert not ok
    assert reason == "deployment_semantics_mismatch"


def test_bundle_compatibility_rejects_tampered_artifact(tmp_path):
    import json
    from ai_trading_lab.deployment import model_semantics_fingerprint, deployment_semantics_fingerprint, bundle_artifact_fingerprint
    settings = load_settings("config.yaml")
    bundle = asset_bundle_dir(tmp_path, "BTC/USDT")
    bundle.mkdir(parents=True)
    for name in ENGINE_ARTIFACTS:
        (bundle / name).write_bytes(b"original-" + name.encode())
    fp = "dataset-current"
    sem = model_semantics_fingerprint(settings)
    dep_sem = deployment_semantics_fingerprint(settings)
    artifact_fp = bundle_artifact_fingerprint(bundle)
    (bundle / "base_training_meta.json").write_text(
        json.dumps({"symbol":"BTC/USDT","timeframe":"15m","data_fingerprint":fp,"model_semantics_fingerprint":sem,"deployment_semantics_fingerprint":dep_sem}),
        encoding="utf-8",
    )
    (bundle / "deployment_manifest.json").write_text(
        json.dumps({"ready":True,"symbol":"BTC/USDT","timeframe":"15m","data_fingerprint":fp,"model_semantics_fingerprint":sem,"deployment_semantics_fingerprint":dep_sem,"bundle_artifact_fingerprint":artifact_fp}),
        encoding="utf-8",
    )
    (bundle / "trade_window_specialist.joblib").write_bytes(b"original-trade-window")
    (bundle / "signal_model.joblib").write_bytes(b"tampered-model")
    ok, reason = bundle_compatibility(settings, bundle, "BTC/USDT")
    assert not ok
    assert reason == "deployment_manifest_artifact_mismatch"


def test_global_promotion_removes_stale_managed_artifacts(tmp_path):
    from ai_trading_lab.main import _promote_asset_bundle
    src = tmp_path / "asset"
    dst = tmp_path / "champion"
    src.mkdir()
    dst.mkdir()
    (src / "signal_model.joblib").write_bytes(b"new-signal")
    (src / "base_training_meta.json").write_text("{}", encoding="utf-8")
    (dst / "signal_model.joblib").write_bytes(b"old-signal")
    (dst / "trade_window_specialist.joblib").write_bytes(b"stale-window")
    (dst / "deployment_manifest.json").write_text("{}", encoding="utf-8")
    _promote_asset_bundle(src, dst)
    assert (dst / "signal_model.joblib").read_bytes() == b"new-signal"
    assert (dst / "base_training_meta.json").exists()
    assert not (dst / "trade_window_specialist.joblib").exists()
    assert not (dst / "deployment_manifest.json").exists()


def test_train_complete_all_uses_local_history_without_exchange(monkeypatch, tmp_path):
    """Multi-asset training must not instantiate a live exchange for the default symbol."""
    import json
    import sys
    from types import SimpleNamespace
    import ai_trading_lab.main as main_module

    settings = SimpleNamespace(
        symbol="BTC/USDT",
        timeframe="15m",
    )
    monkeypatch.setattr(main_module, "load_settings", lambda _path: settings)
    monkeypatch.setattr(main_module, "validate_research_data", lambda _df, _settings: None)
    monkeypatch.setattr(
        main_module,
        "train_complete_asset",
        lambda df, ss, holdout_frac, promote_champion=True: {
            "symbol": ss.symbol,
            "status": "trained",
            "production_ready": True,
            "rows": len(df),
        },
    )
    monkeypatch.setattr(
        main_module,
        "exchange_client",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("train-complete-all touched the exchange before using local history")
        ),
    )
    monkeypatch.chdir(tmp_path)

    history = tmp_path / "data" / "historical"
    history.mkdir(parents=True)
    rows = []
    start = __import__("pandas").Timestamp("2026-01-01", tz="UTC")
    for i in range(32):
        ts = start + __import__("pandas").Timedelta(minutes=15 * i)
        close = 100.0 + i
        rows.append(
            f"{ts.isoformat()},{close - 0.5},{close + 1.0},{close - 1.0},{close},1000"
        )
    (history / "BTC_USDT_15m.csv").write_text(
        "timestamp,open,high,low,close,volume\n" + "\n".join(rows),
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["main.py", "train-complete-all", "--config", "config.yaml"])
    main_module.main()

    report = json.loads(
        (tmp_path / "logs" / "train_complete_all_report.json").read_text(encoding="utf-8")
    )
    assert report == [{
        "symbol": "BTC/USDT",
        "status": "trained",
        "production_ready": True,
        "rows": 32,
        "source": "historical",
    }]


def test_train_all_ignores_stale_registry_path_when_historical_asset_exists(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    import pandas as pd
    import ai_trading_lab.main as main_module
    import ai_trading_lab.dataset as dataset_module

    settings = SimpleNamespace(symbol="BTC/USDT", timeframe="15m")
    monkeypatch.setattr(main_module, "load_settings", lambda _path: settings)
    monkeypatch.setattr(main_module, "validate_research_data", lambda _df, _settings: None)
    monkeypatch.setattr(
        main_module,
        "train_base_asset",
        lambda df, ss: (tmp_path / "models" / "assets" / "BTC_USDT", None),
    )
    monkeypatch.setattr(
        dataset_module,
        "imported_registry",
        lambda _data_dir: {
            "BTC/USDT": {"path": str(tmp_path / "missing" / "BTC.csv")}
        },
    )
    monkeypatch.chdir(tmp_path)

    history = tmp_path / "data" / "historical"
    history.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.0] * 10,
            "volume": [1000.0] * 10,
        },
        index=pd.date_range("2026-01-01", periods=10, freq="15min", tz="UTC"),
    )
    frame.to_csv(history / "BTC_USDT_15m.csv", index_label="timestamp")

    monkeypatch.setattr(
        sys, "argv", ["main.py", "train-all", "--config", "config.yaml"]
    )
    main_module.main()

    import json
    report = json.loads(
        (tmp_path / "logs" / "train_all_report.json").read_text(encoding="utf-8")
    )
    assert report[0]["status"] == "trained"
    assert report[0]["source"] == "historical"


def test_train_complete_asset_can_skip_global_champion_promotion(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import ai_trading_lab.main as main_module

    settings = SimpleNamespace(symbol="ETH/USDT")
    calls = {}

    monkeypatch.setattr(
        main_module,
        "train_base_asset",
        lambda df, ss: (tmp_path / "bundle", object()),
    )
    monkeypatch.setattr(
        main_module,
        "test_base_holdout",
        lambda df, ss, holdout_frac: {"holdout": {}},
    )
    monkeypatch.setattr(
        main_module,
        "train_duration_asset",
        lambda df, ss, *, holdout_frac, copy_legacy: (
            calls.update(copy_legacy=copy_legacy) or (tmp_path / "duration.joblib", {"production_ready": True})
        ),
    )
    monkeypatch.setattr(
        main_module,
        "refresh_deployment_manifest",
        lambda ss: {"ready": True},
    )
    promoted = []
    monkeypatch.setattr(
        main_module,
        "_promote_asset_bundle",
        lambda path: promoted.append(path),
    )

    main_module.train_complete_asset(
        object(), settings, holdout_frac=0.15, promote_champion=False
    )

    assert calls["copy_legacy"] is False
    assert promoted == []
