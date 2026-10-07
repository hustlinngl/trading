from __future__ import annotations

from pathlib import Path

from ai_trading_lab.config import load_settings
from ai_trading_lab.deployment import asset_bundle_dir, resolve_signal_bundle, bundle_compatibility
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
    features = extract_event_terms("Fed hawkish rate hike with ETF outflow and exploit risk")
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
