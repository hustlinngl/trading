from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


def _settings():
    return SimpleNamespace(
        exchange="binance",
        symbol="BTC/USDT",
        timeframe="15m",
        live_lookback_bars=120,
        max_parallel_downloads=6,
    )


def _history_frame(rows=120):
    end = pd.Timestamp.now(tz="UTC").floor("15min") - pd.Timedelta(minutes=15)
    index = pd.date_range(end=end, periods=rows, freq="15min")
    close = pd.Series(range(rows), index=index, dtype=float) + 100.0
    return pd.DataFrame(
        {"open": close - 0.2, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1000.0},
        index=index,
    )



def test_main_keeps_kraken_ohlcv_adapter_separate_from_exchange_ohlcv():
    import inspect
    import ai_trading_lab.main as main_module

    source = inspect.getsource(main_module.main)
    assert "from .kraken_data import fetch_ohlcv as fetch_kraken_ohlcv" in source
    assert "df, provenance = fetch_kraken_ohlcv(" in source
    assert "from .kraken_data import fetch_ohlcv, fingerprint_frame, save_provenance" not in source

def test_bootstrap_live_data_uses_worker_clients_and_atomic_csvs(monkeypatch, tmp_path, capsys):
    import ai_trading_lab.main as main_module

    settings = _settings()
    discovery_client = SimpleNamespace(markets={}, has={"fetchOHLCV": True})
    worker_clients = []
    fetched = []

    def fake_client(*args, **kwargs):
        if not worker_clients:
            worker_clients.append(discovery_client)
            return discovery_client
        client = SimpleNamespace(markets={}, has={"fetchOHLCV": True})
        worker_clients.append(client)
        return client

    frame = _history_frame()

    def fake_fetch(client, symbol, timeframe, limit, include_unclosed):
        assert client is not discovery_client
        assert timeframe == "15m"
        assert limit == 120
        assert include_unclosed is False
        fetched.append((client, symbol))
        return frame.copy()

    monkeypatch.setattr(main_module, "load_settings", lambda _path: settings)
    monkeypatch.setattr(main_module, "exchange_client", fake_client)
    monkeypatch.setattr(main_module, "fetch_ohlcv", fake_fetch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py", "bootstrap-live-data", "--config", "config.yaml",
            "--symbols", "BTC/USDT,ETH/USDT", "--live-bars", "120", "--workers", "2",
        ],
    )
    monkeypatch.chdir(tmp_path)

    main_module.main()

    summary = json.loads((tmp_path / "logs" / "bootstrap_live_data.json").read_text(encoding="utf-8"))
    assert summary["requested_symbols"] == 2
    assert summary["completed"] == 2
    assert summary["failed"] == 0
    assert summary["workers"] == 2
    assert {symbol for _, symbol in fetched} == {"BTC/USDT", "ETH/USDT"}
    for symbol in ("BTC/USDT", "ETH/USDT"):
        slug = symbol.replace("/", "_") + "_15m.csv"
        saved = tmp_path / "data" / "historical" / slug
        assert saved.exists()
        assert len(pd.read_csv(saved)) == 120
        assert not saved.with_suffix(saved.suffix + ".tmp").exists()
    assert "ERROR" not in capsys.readouterr().out


def test_bootstrap_live_data_reuses_fresh_cache_without_exchange_requests(monkeypatch, tmp_path):
    import ai_trading_lab.main as main_module

    settings = _settings()
    client = SimpleNamespace(markets={}, has={"fetchOHLCV": True})
    client_calls = []

    monkeypatch.setattr(main_module, "load_settings", lambda _path: settings)
    monkeypatch.setattr(main_module, "exchange_client", lambda *args, **kwargs: client_calls.append(client) or client)
    monkeypatch.setattr(
        main_module,
        "fetch_ohlcv",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fresh cache should not hit the network")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py", "bootstrap-live-data", "--config", "config.yaml",
            "--symbols", "BTC/USDT", "--live-bars", "120", "--workers", "4",
        ],
    )
    monkeypatch.chdir(tmp_path)
    cached_path = tmp_path / "data" / "historical" / "BTC_USDT_15m.csv"
    cached_path.parent.mkdir(parents=True)
    _history_frame().to_csv(cached_path, index_label="timestamp")

    main_module.main()

    summary = json.loads((tmp_path / "logs" / "bootstrap_live_data.json").read_text(encoding="utf-8"))
    assert summary["completed"] == 1
    assert summary["failed"] == 0
    assert summary["results"][0]["source"] == "cache"
    # Only the discovery client should be constructed because the cache is sufficient.
    assert len(client_calls) == 1


def test_bootstrap_live_data_discovers_only_supported_active_market_types(monkeypatch, tmp_path):
    import ai_trading_lab.main as main_module

    settings = _settings()
    exchange = SimpleNamespace(
        markets={
            "BTC/USDT": {"symbol": "BTC/USDT", "type": "spot", "active": True},
            "ETH/USDT": {"symbol": "ETH/USDT", "type": "spot", "active": True},
            "SOL/USDT:USDT": {"symbol": "SOL/USDT:USDT", "type": "swap", "active": True, "contract": True},
            "BROKEN/USDT:USDT": {"symbol": "BROKEN/USDT:USDT", "type": "future", "active": True, "contract": False},
            "OLD/USDT": {"symbol": "OLD/USDT", "type": "spot", "active": False},
            "BTC/USDT-OPT": {"symbol": "BTC/USDT-OPT", "type": "option", "active": True},
        },
        has={"fetchOHLCV": True},
    )
    fetched = []

    monkeypatch.setattr(main_module, "load_settings", lambda _path: settings)
    monkeypatch.setattr(main_module, "exchange_client", lambda *args, **kwargs: exchange)
    monkeypatch.setattr(
        main_module,
        "fetch_ohlcv",
        lambda client, symbol, timeframe, limit, include_unclosed: (
            fetched.append(symbol) or _history_frame()
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py", "bootstrap-live-data", "--config", "config.yaml",
            "--all-symbols", "--market-types", "spot,swap", "--live-bars", "120", "--workers", "2",
        ],
    )
    monkeypatch.chdir(tmp_path)

    main_module.main()

    summary = json.loads((tmp_path / "logs" / "bootstrap_live_data.json").read_text(encoding="utf-8"))
    assert summary["requested_symbols"] == 3
    assert summary["completed"] == 3
    assert summary["failed"] == 0
    assert set(fetched) == {"BTC/USDT", "ETH/USDT", "SOL/USDT:USDT"}
