from __future__ import annotations

import json
import threading
import time

import pandas as pd

from ai_trading_lab.config import load_settings
from ai_trading_lab.live import LiveAssessment


def test_dashboard_responsive_ui_and_chart_edge_case():
    import signal_dashboard as terminal_mod

    html = terminal_mod.HTML
    assert "UI polish: restrained Sakura identity" in html
    assert "render(data);\n    renderFocus(data);" in html
    assert "@media(max-width:460px)" in html
    assert "bars.length===1?pad.l+cw/2" in html
    assert "table-wrap{border-radius" in html
    assert "focus-visible" in html
    assert "pick-primary" in html
    assert "bindFocusCards" in html
    assert 'role="button"' in html
    assert "connection-stamp" not in html
    assert 'aria-busy="true"' in html
    assert ".signal-skeleton-grid" in html
    assert 'data-target="focusDashboard"' in html
    assert 'class="nav" id="nav"' in html
    assert "const reveal=id=>" in html
    assert "section-reveal" in html
    assert "prefers-reduced-motion:reduce" in html
    assert 'id="sakuraMusic"' in html
    assert 'id="metrics"' not in html
    assert 'id="focusStatus"' not in html
    assert 'id="focusCoverage"' not in html
    assert 'id="deckBundle"' not in html
    assert 'function renderMetrics' not in html
    assert 'function pill(' not in html


def test_dashboard_market_explorer_ui():
    import signal_dashboard as terminal_mod

    html = terminal_mod.HTML
    assert 'id="marketSymbols"' in html
    assert 'id="loadAsset"' in html
    assert 'id="asset" list="marketSymbols"' in html
    assert "state.marketSymbols" in html
    assert "openSelectedAsset" in html


def test_signal_terminal_builds_read_only_state(tmp_path, monkeypatch):
    import signal_dashboard as terminal_mod

    settings = load_settings("config.yaml")
    settings.live_symbols = ("BTC/USDT",)
    assessment = LiveAssessment(
        "BTC/USDT",
        "2026-10-07T00:00:00+00:00",
        "SIGNAL",
        "LONG",
        0.91,
        0.006,
        100000.0,
        [],
        "data-fp",
        {
            "p_up": 0.91,
            "expected_return_lcb": 0.004,
            "expected_return_ucb": 0.008,
            "score": 0.41,
            "meta_success": 0.84,
            "model_disagreement": 0.01,
            "regime": "trend_up",
            "analog_n": 32,
            "analog_agreement": 0.88,
            "trade_window_ready": True,
            "trade_window_direction": "LONG",
            "trade_window_confidence": 0.92,
            "data_age_minutes": 1.0,
        },
    )
    monkeypatch.setattr(terminal_mod, "scan_top5", lambda *args, **kwargs: [assessment])
    monkeypatch.setattr(terminal_mod, "update_live_signal_outcomes", lambda *args, **kwargs: {"updated": 0, "open": 1, "closed": 0})

    terminal = terminal_mod.SignalTerminal(settings, tmp_path, refresh_seconds=30)
    state = terminal._terminal_state(force=True)

    assert state["ok"] is True
    assert state["summary"]["assets_scanned"] == 1
    assert state["summary"]["active_signals"] == 1
    assert state["signals"][0]["signal"] == "LONG"
    assert state["signals"][0]["decision"]["score"] == 0.41
    assert "bundle" in state["signals"][0]
    assert state["notes"][0].startswith("Sola lettura")



def test_dashboard_cursor_click_does_not_override_pointer_position():
    import signal_dashboard as terminal_mod

    html = terminal_mod.HTML
    assert '#anime-cursor.click{filter:drop-shadow(0 0 18px rgba(255,120,200,.95));animation:none}' in html


def test_signal_terminal_publishes_market_data_without_model_signals(tmp_path, monkeypatch):
    import signal_dashboard as terminal_mod

    settings = load_settings("config.yaml")
    settings.live_symbols = ("BTC/USDT", "ETH/USDT")

    class FakeExchange:
        markets = {
            "BTC/USDT": {"symbol": "BTC/USDT", "active": True, "type": "spot"},
            "ETH/USDT": {"symbol": "ETH/USDT", "active": True, "type": "spot"},
        }

        def fetch_tickers(self, symbols):
            return {
                "BTC/USDT": {
                    "last": 101.0, "bid": 100.9, "ask": 101.1,
                    "timestamp": 123, "quoteVolume": 1000.0,
                },
                "ETH/USDT": {
                    "last": 5.0, "bid": 4.9, "ask": 5.1,
                    "timestamp": 123, "quoteVolume": 2000.0,
                },
            }

    monkeypatch.setattr(
        terminal_mod, "exchange_client", lambda *args, **kwargs: FakeExchange()
    )
    monkeypatch.setattr(
        terminal_mod,
        "scan_top5",
        lambda *args, **kwargs: ([], {
            "universe_total": 2,
            "universe_model_backed": 0,
            "universe_model_eligible": 0,
            "universe_evaluated": 0,
            "universe_signals": 0,
            "universe_waits": 0,
            "assessment_failures": 0,
            "market_symbols": ["BTC/USDT", "ETH/USDT"],
            "market_counts": {"spot": 2},
        }),
    )
    monkeypatch.setattr(
        terminal_mod,
        "update_live_signal_outcomes",
        lambda *args, **kwargs: {"updated": 0, "open": 0, "closed": 0},
    )

    terminal = terminal_mod.SignalTerminal(settings, tmp_path, refresh_seconds=30)
    state = terminal._terminal_state(force=True)

    assert state["market_data"]["symbols"] == ["BTC/USDT", "ETH/USDT"]
    assert state["market_data"]["quotes"]["BTC/USDT"]["price"] == 101.0
    assert state["market_data"]["quotes"]["ETH/USDT"]["ask"] == 5.1



def test_signal_terminal_uses_discovered_market_universe_for_realtime_quotes(tmp_path, monkeypatch):
    import signal_dashboard as terminal_mod

    settings = load_settings("config.yaml")
    settings.live_symbols = ("BTC/USDT",)
    seen = {}

    class FakeExchange:
        def fetch_tickers(self):
            seen["called_without_symbols"] = True
            return {
                "BTC/USDT": {"last": 100.0, "bid": 99.9, "ask": 100.1, "timestamp": 1},
                "ETH/USDT": {"last": 5.0, "bid": 4.9, "ask": 5.1, "timestamp": 1},
                "SOL/USDT": {"last": 2.0, "bid": 1.9, "ask": 2.1, "timestamp": 1},
            }

        def fetch_ticker(self, symbol):
            raise AssertionError("exchange-wide ticker path should cover discovered universe")

    monkeypatch.setattr(
        terminal_mod, "exchange_client", lambda *args, **kwargs: FakeExchange()
    )
    monkeypatch.setattr(
        terminal_mod,
        "scan_top5",
        lambda *args, **kwargs: ([], {
            "universe_total": 150,
            "universe_model_backed": 0,
            "universe_model_eligible": 0,
            "universe_evaluated": 0,
            "universe_signals": 0,
            "universe_waits": 0,
            "assessment_failures": 0,
            "market_symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"] + [f"COIN{i}/USDT" for i in range(147)],
            "market_counts": {"spot": 150},
        }),
    )
    monkeypatch.setattr(
        terminal_mod,
        "update_live_signal_outcomes",
        lambda *args, **kwargs: {"updated": 0, "open": 0, "closed": 0},
    )

    terminal = terminal_mod.SignalTerminal(settings, tmp_path, refresh_seconds=30)
    state = terminal._terminal_state(force=True)

    assert seen["called_without_symbols"] is True
    assert len(state["market_data"]["symbols"]) == 150
    assert state["market_data"]["quotes"]["ETH/USDT"]["price"] == 5.0


def test_bootstrap_command_is_available():
    from ai_trading_lab import main as main_mod

    # argparse is built inside main(), so inspect the source-level command contract.
    import inspect
    source = inspect.getsource(main_mod.main)
    assert "bootstrap-live-data" in source
    assert "--all-symbols" in source
    assert "--market-types" in source


def test_signal_terminal_reuses_exchange_for_outcome_tracking(tmp_path, monkeypatch):
    import signal_dashboard as terminal_mod

    settings = load_settings("config.yaml")
    terminal = terminal_mod.SignalTerminal(settings, tmp_path, refresh_seconds=30)
    exchange = object()
    seen = {}

    terminal._get_exchange = lambda: exchange
    monkeypatch.setattr(terminal_mod, "scan_top5", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        terminal_mod,
        "update_live_signal_outcomes",
        lambda settings, root, exchange=None: seen.update(exchange=exchange) or {"updated": 0, "open": 0, "closed": 0},
    )

    terminal._terminal_state(force=True)

    assert seen["exchange"] is exchange


def test_signal_terminal_serves_only_read_routes(tmp_path, monkeypatch):
    import signal_dashboard as terminal_mod

    settings = load_settings("config.yaml")
    monkeypatch.setattr(terminal_mod, "scan_top5", lambda *args, **kwargs: [])
    terminal = terminal_mod.SignalTerminal(settings, tmp_path, refresh_seconds=30)
    handler = terminal_mod.make_handler(terminal)
    server = terminal_mod.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    try:
        assert server.server_address[0] == "127.0.0.1"
        assert not hasattr(handler, "do_POST")
    finally:
        server.server_close()


def test_signal_terminal_json_safe_handles_nan_and_nested_values():
    import signal_dashboard as terminal_mod

    value = terminal_mod._json_safe(
        {"x": float("nan"), "nested": [1, float("inf"), float("-inf")]}
    )
    assert value == {"x": None, "nested": [1, None, None]}
    assert json.dumps(value, allow_nan=False)


def test_signal_terminal_history_endpoint_uses_closed_market_bars(tmp_path, monkeypatch):
    import signal_dashboard as terminal_mod

    settings = load_settings("config.yaml")
    settings.exchange = "binance"
    settings.symbol = "BTC/USDT"
    frame_index = pd.date_range("2026-10-06", periods=4, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.5, 102.5, 103.5, 104.5],
            "low": [99.5, 100.5, 101.5, 102.5],
            "close": [101.0, 102.0, 103.0, 104.0],
            "volume": [1000.0, 1100.0, 1200.0, 1300.0],
        },
        index=frame_index,
    )
    monkeypatch.setattr(terminal_mod, "exchange_client", lambda *args, **kwargs: object())
    monkeypatch.setattr(terminal_mod, "fetch_ohlcv", lambda *args, **kwargs: frame)
    terminal = terminal_mod.SignalTerminal(settings, tmp_path, history_bars=4)
    result = terminal._history("BTC/USDT", 4)
    assert "error" not in result
    assert len(result["bars"]) == 4
    assert result["bars"][-1]["c"] == 104.0


def test_signal_terminal_realtime_quotes_are_best_effort(tmp_path, monkeypatch):
    import signal_dashboard as terminal_mod

    settings = load_settings("config.yaml")
    settings.live_symbols = ("BTC/USDT",)
    class FakeExchange:
        def fetch_ticker(self, symbol):
            return {"last": 123.45, "bid": 123.40, "ask": 123.50, "timestamp": 1234567890000, "quoteVolume": 99999.0}
    monkeypatch.setattr(terminal_mod, "exchange_client", lambda *args, **kwargs: FakeExchange())
    terminal = terminal_mod.SignalTerminal(settings, tmp_path)
    out = terminal._quotes(["BTC/USDT"])
    assert out["BTC/USDT"]["price"] == 123.45
    assert out["BTC/USDT"]["bid"] == 123.40
    assert out["BTC/USDT"]["ask"] == 123.50


def test_signal_terminal_bulk_ticker_path_and_browser_escape(tmp_path, monkeypatch):
    import signal_dashboard as terminal_mod

    settings = load_settings("config.yaml")
    settings.live_symbols = ("BTC/USDT",)
    class FakeExchange:
        def fetch_tickers(self, symbols):
            return {
                "BTC/USDT": {
                    "last": 321.0,
                    "bid": 320.9,
                    "ask": 321.1,
                    "timestamp": 123,
                    "quoteVolume": 1000.0,
                }
            }

        def fetch_ticker(self, symbol):
            raise AssertionError("bulk ticker path should avoid per-symbol fallback")

    monkeypatch.setattr(
        terminal_mod, "exchange_client", lambda *args, **kwargs: FakeExchange()
    )
    terminal = terminal_mod.SignalTerminal(settings, tmp_path)
    out = terminal._quotes(["BTC/USDT"])
    assert out["BTC/USDT"]["price"] == 321.0
    assert 'function esc(v){return String(v??"").replace(/[&<>"]/g,c=>c==="&"?"&amp;":c==="<"?"&lt;":c===">"?"&gt;":"&quot;");}' in terminal_mod.HTML


def test_signal_terminal_alpha_ui_keeps_visual_layer_separate_from_execution():
    import signal_dashboard as terminal_mod

    html = terminal_mod.HTML
    assert "anime-cursor" in html
    assert "click-ripple" in html
    assert "initAlphaMotion" in html
    assert "initNavigation" in html
    assert 'data-target="market"' in html
    assert 'data-target="detail"' in html
    assert 'data-target="journal"' in html
    assert 'data-target="evidencePanel"' in html
    assert 'id="detail"' in html
    assert "body.alpha-pointer" in html
    assert "prefers-reduced-motion:reduce" in html
    assert "do_POST" not in html


def test_signal_terminal_alpha_decision_deck_is_wired():
    import signal_dashboard as terminal_mod

    html = terminal_mod.HTML
    assert 'id="decisionDeck"' in html
    assert 'id="traceGrid"' in html
    assert "renderDecisionDeck" in html
    assert "Decision trace" in html
    assert "data-target=\"detail\"" in html


def test_signal_terminal_alpha_art_direction_layer_is_wired():
    import signal_dashboard as terminal_mod

    html = terminal_mod.HTML
    assert 'id="timeline"' in html
    assert 'id="signalTimeline"' in html
    assert 'id="inspectorDrawer"' in html
    assert 'id="chartCrosshair"' in html
    assert "renderTimeline" in html
    assert "openInspector" in html
    assert "Prediction" in html
    assert "Outcome" in html


def test_signal_terminal_alpha_ambient_fx_is_wired():
    import signal_dashboard as terminal_mod

    html = terminal_mod.HTML
    assert 'id="ambient-canvas"' in html
    assert "initAmbientFX" in html
    assert "requestAnimationFrame" in html
    assert "click-ripple" in html
    assert "signal-live-" in html
    assert "prefers-reduced-motion:reduce" in html


def test_signal_terminal_history_falls_back_to_bundled_cache(tmp_path, monkeypatch):
    import signal_dashboard as terminal_mod

    settings = load_settings("config.yaml")
    settings.symbol = "BTC/USDT"
    history_dir = tmp_path / "data" / "historical"
    history_dir.mkdir(parents=True)
    (history_dir / "BTC_USDT_15m.csv").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-10-06T00:00:00+00:00,100,101,99,100.5,1000\n"
        "2026-10-06T00:15:00+00:00,100.5,102,100,101.5,1100\n",
        encoding="utf-8",
    )

    def fail_network(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(terminal_mod, "exchange_client", fail_network)
    terminal = terminal_mod.SignalTerminal(settings, tmp_path, history_bars=240)

    result = terminal._history("BTC/USDT", 240)

    assert result["source"] == "bundled"
    assert len(result["bars"]) == 2
    assert result["bars"][1]["c"] == 101.5


def test_signal_terminal_focus_surface_is_only_verified_picks():
    import signal_dashboard as terminal_mod

    html = terminal_mod.HTML
    assert 'id="focusDashboard"' in html
    assert 'id="top5Grid"' in html
    assert 'Top 5 signals' in html
    assert 'pick-stat-meter' in html
    assert 'style="--meter:' in html
    assert 'Solo LONG e SHORT attivi' not in html
    assert 'legacy-hidden' in html
    assert 'class="legacy-hidden panel chart-panel" id="market"' in html
    assert 'class="legacy-hidden panel" id="detail"' in html
    assert 'class="legacy-hidden panel" id="journal"' in html
    assert 'class="legacy-hidden panel" id="timeline"' in html
    assert 'class="legacy-hidden panel" id="evidencePanel"' in html


def test_scan_top5_filters_wait_and_caps_verified_picks(monkeypatch, tmp_path):
    import ai_trading_lab.live as live_mod

    settings = load_settings("config.yaml")
    settings.live_symbols = tuple(f"ASSET{i}/USDT" for i in range(8))

    def fake_assess(settings, root, symbol, exchange=None, skip_network=False):
        idx = int(symbol.replace("ASSET", "").split("/")[0])
        if idx == 7:
            return LiveAssessment(symbol, "2026-10-07T00:00:00+00:00", "WAIT", "FLAT", 0.99, 0.9, 1.0, ["gate"], "fp")
        return LiveAssessment(
            symbol,
            "2026-10-07T00:00:00+00:00",
            "SIGNAL",
            "LONG" if idx % 2 == 0 else "SHORT",
            0.60 + idx / 100,
            0.001 + idx / 10000,
            100.0 + idx,
            [],
            "fp",
        )

    monkeypatch.setattr(live_mod, "exchange_client", lambda *args, **kwargs: object())
    monkeypatch.setattr(live_mod, "discover_live_universe", lambda *args, **kwargs: {
        "symbols": list(settings.live_symbols),
        "universe_symbols": list(settings.live_symbols),
        "discovered_markets": len(settings.live_symbols),
        "model_backed_markets": len(settings.live_symbols),
        "model_eligible_markets": len(settings.live_symbols),
        "market_counts": {"spot": len(settings.live_symbols)},
    })
    monkeypatch.setattr(live_mod, "assess_symbol", fake_assess)

    picks = live_mod.scan_top5(settings, tmp_path)

    assert len(picks) == 5
    assert all(p.signal in {"LONG", "SHORT"} and p.status == "SIGNAL" for p in picks)
    assert picks[0].confidence >= picks[-1].confidence


def test_signal_terminal_dashboard_markup_passes_release_smoke_check():
    import signal_dashboard as terminal_mod

    terminal_mod.validate_dashboard_markup()


def test_signal_terminal_journal_reads_frozen_state_root(tmp_path, monkeypatch):
    import signal_dashboard as terminal_mod

    settings = load_settings("config.yaml")
    terminal = terminal_mod.SignalTerminal(settings, tmp_path)
    state_root = tmp_path / "runtime-state"
    journal_path = state_root / "logs" / "live_signal_history.jsonl"
    journal_path.parent.mkdir(parents=True)
    journal_path.write_text(
        '{"symbol":"BTC/USDT","status":"SIGNAL","signal":"LONG","timestamp":"2026-10-07T00:00:00+00:00"}\n',
        encoding="utf-8",
    )
    terminal.state_root = state_root

    rows = terminal._journal()

    assert len(rows) == 1
    assert rows[0]["signal"] == "LONG"


def test_discover_live_universe_uses_all_active_market_types(monkeypatch, tmp_path):
    import ai_trading_lab.live as live_mod

    settings = load_settings("config.yaml")
    settings.live_symbols = ("BTC/USDT", "GHOST/USDT")
    settings.live_market_types = ("spot", "swap", "future")
    asset_root = tmp_path / "models" / "assets"
    asset_root.mkdir(parents=True)
    for symbol in ("BTC/USDT", "ETH/USDT:USDT", "XRP/USDT:USDT", "GHOST/USDT"):
        bundle = asset_root / symbol.replace("/", "_").replace(":", "_")
        bundle.mkdir()
        (bundle / "signal_model.joblib").write_text("stub", encoding="utf-8")

    class FakeExchange:
        markets = {
            "BTC/USDT": {"symbol": "BTC/USDT", "type": "spot", "active": True},
            "ETH/USDT:USDT": {"symbol": "ETH/USDT:USDT", "type": "swap", "contract": True, "active": True},
            "XRP/USDT:USDT": {"symbol": "XRP/USDT:USDT", "type": "future", "contract": True, "active": True},
            "DOGE/USDT": {"symbol": "DOGE/USDT", "type": "spot", "active": False},
            "EUR/USD": {"symbol": "EUR/USD", "type": "spot", "active": True},
        }

    monkeypatch.setattr(live_mod, "resolve_signal_bundle", lambda settings, root, symbol:
        asset_root / symbol.replace("/", "_").replace(":", "_")
    )
    monkeypatch.setattr(live_mod, "bundle_compatibility", lambda settings, bundle, symbol: (True, "ok"))

    meta = live_mod.discover_live_universe(settings, tmp_path, FakeExchange())

    assert set(meta["symbols"]) == {"BTC/USDT", "ETH/USDT:USDT", "XRP/USDT:USDT"}
    assert meta["discovered_markets"] == 4
    assert meta["model_backed_markets"] == 3
    assert meta["model_eligible_markets"] == 3
    assert meta["market_counts"]["spot"] == 2
    assert meta["market_counts"]["swap"] == 1
    assert meta["market_counts"]["future"] == 1
    assert meta["exchange_market_metadata"] is True
    assert "GHOST/USDT" not in meta["symbols"]


def test_scan_top5_return_meta_reports_universe_coverage(monkeypatch, tmp_path):
    import ai_trading_lab.live as live_mod

    settings = load_settings("config.yaml")
    settings.live_symbols = ("BTC/USDT",)

    class FakeExchange:
        markets = {
            "BTC/USDT": {"symbol": "BTC/USDT", "type": "spot", "active": True},
        }

    monkeypatch.setattr(live_mod, "exchange_client", lambda *args, **kwargs: FakeExchange())
    monkeypatch.setattr(live_mod, "discover_live_universe", lambda *args, **kwargs: {
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "discovered_markets": 12,
        "model_backed_markets": 2,
        "market_counts": {"spot": 8, "swap": 4},
    })
    monkeypatch.setattr(
        live_mod,
        "assess_symbol",
        lambda settings, root, symbol, exchange=None, skip_network=False:
            LiveAssessment(symbol, "2026-10-07T00:00:00+00:00", "SIGNAL", "LONG", 0.9, 0.01, 1.0, [], "fp"),
    )

    picks, meta = live_mod.scan_top5(settings, tmp_path, return_meta=True)

    assert len(picks) == 2
    assert meta["universe_total"] == 12
    assert meta["universe_model_backed"] == 2
    assert meta["universe_evaluated"] == 2
    assert meta["universe_mode"] == "all_active_markets"


def test_scan_top5_reuses_same_closed_candle_assessment(monkeypatch, tmp_path):
    import ai_trading_lab.live as live_mod
    import pandas as pd

    settings = load_settings("config.yaml")
    settings.live_symbols = ("BTC/USDT", "ETH/USDT")

    monkeypatch.setattr(
        live_mod,
        "discover_live_universe",
        lambda *args, **kwargs: {
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "discovered_markets": 2,
            "model_backed_markets": 2,
            "market_counts": {"spot": 2},
        },
    )
    calls = {"n": 0}
    stamp = pd.Timestamp.now(tz="UTC").floor("15min").isoformat()

    def fake_assess(settings, root, symbol, exchange=None, skip_network=False):
        calls["n"] += 1
        return LiveAssessment(symbol, stamp, "SIGNAL", "LONG", 0.9, 0.01, 1.0, [], "fp")

    monkeypatch.setattr(live_mod, "assess_symbol", fake_assess)
    cache = {}
    live_mod.scan_top5(settings, tmp_path, exchange=object(), cache=cache)
    live_mod.scan_top5(settings, tmp_path, exchange=object(), cache=cache)

    assert calls["n"] == 2


def test_signal_terminal_background_state_does_not_hold_lock_during_scan(tmp_path, monkeypatch):
    import signal_dashboard as terminal_mod

    settings = load_settings("config.yaml")
    started = threading.Event()
    release = threading.Event()

    def slow_scan(*args, **kwargs):
        started.set()
        assert release.wait(2.0)
        return []

    monkeypatch.setattr(terminal_mod, "scan_top5", slow_scan)
    terminal = terminal_mod.SignalTerminal(settings, tmp_path, refresh_seconds=30)
    terminal._get_exchange = lambda: None

    first = terminal._terminal_state(force=True, background=True)
    assert first["scan_in_progress"] is True
    scan_thread = terminal._scan_thread
    assert scan_thread is not None
    assert started.wait(1.0)

    second = {}
    done = threading.Event()

    def second_request():
        begin = time.monotonic()
        second["state"] = terminal._terminal_state(background=True)
        second["elapsed"] = time.monotonic() - begin
        done.set()

    request = threading.Thread(target=second_request, daemon=True)
    request.start()
    returned_before_release = done.wait(0.5)
    release.set()
    request.join(2.0)
    scan_thread.join(2.0)

    assert returned_before_release is True
    assert second["state"]["scan_in_progress"] is True
    assert second["elapsed"] < 0.45


def test_scan_top5_reports_incremental_progress(monkeypatch, tmp_path):
    import ai_trading_lab.live as live_mod

    settings = load_settings("config.yaml")
    monkeypatch.setattr(
        live_mod,
        "discover_live_universe",
        lambda *args, **kwargs: {
            "symbols": ["A/USDT", "B/USDT"],
            "discovered_markets": 2,
            "model_backed_markets": 2,
            "model_eligible_markets": 2,
            "market_counts": {"spot": 2},
        },
    )
    monkeypatch.setattr(
        live_mod,
        "assess_symbol",
        lambda settings, root, symbol, exchange=None, skip_network=False:
            LiveAssessment(symbol, "2026-10-07T00:00:00+00:00", "WAIT", "FLAT", 0.5, 0.0, 1.0, ["gate"], "fp"),
    )

    progress = []
    live_mod.scan_top5(
        settings,
        tmp_path,
        exchange=object(),
        progress_callback=progress.append,
    )

    assert progress[0]["evaluated"] == 0
    assert progress[-1]["evaluated"] == 2
    assert progress[-1]["total"] == 2
    assert progress[-1]["waits"] == 2


def test_scan_top5_prefers_stronger_robust_selection_score(monkeypatch, tmp_path):
    import ai_trading_lab.live as live_mod

    settings = load_settings("config.yaml")

    monkeypatch.setattr(
        live_mod,
        "discover_live_universe",
        lambda *args, **kwargs: {
            "symbols": ["A/USDT", "B/USDT"],
            "discovered_markets": 2,
            "model_backed_markets": 2,
            "market_counts": {"spot": 2},
        },
    )
    def fake_assess(settings, root, symbol, exchange=None, skip_network=False):
        score = 0.8 if symbol == "A/USDT" else 0.3
        return LiveAssessment(
            symbol, "2026-10-07T00:00:00+00:00", "SIGNAL", "LONG",
            0.85, 0.01, 1.0, [], "fp",
            {"selection_score": score, "robust_directional_edge": score / 20.0},
        )

    monkeypatch.setattr(live_mod, "assess_symbol", fake_assess)
    picks = live_mod.scan_top5(settings, tmp_path, exchange=object())

    assert [p.symbol for p in picks] == ["A/USDT", "B/USDT"]

def test_signal_terminal_inspector_is_accessible_and_lifecycle_wired():
    import signal_dashboard as terminal_mod

    html = terminal_mod.HTML
    assert 'role="dialog"' in html
    assert 'aria-modal="true"' in html
    assert 'aria-labelledby="inspectorTitle"' in html
    assert 'id="inspectorTitle"' in html
    assert 'initInspector();' in html
    assert 'drawer.classList.remove("legacy-hidden")' in html
    assert 'drawer.classList.add("legacy-hidden")' in html
    assert 'state.inspectorReturnFocus' in html
    assert 'e.key!=="Tab"' in html

def test_signal_terminal_interactive_rows_use_canonical_market_list():
    import signal_dashboard as terminal_mod

    html = terminal_mod.HTML
    assert 'input.options' not in html
    assert 'new Set(state.marketSymbols||[])' in html
    assert 'role="button" aria-label="Apri ' in html
    assert 'row.onkeydown=e=>' in html

def test_signal_terminal_frontend_races_are_latest_request_wins():
    import signal_dashboard as terminal_mod

    html = terminal_mod.HTML
    assert "let quoteRequest=0;" in html
    assert "request!==quoteRequest" in html
    assert "let refreshBusy=false;" in html
    assert "if(refreshBusy)return;" in html
    assert "clearTimeout(refreshTimer)" in html
    assert 'aria-current="page"' in html
    assert "scroll-margin-top:82px" in html

def test_signal_terminal_primary_surface_is_result_first():
    import signal_dashboard as terminal_mod

    html = terminal_mod.HTML
    assert 'id="focusDashboard"' in html
    assert 'id="top5Grid"' in html
    assert 'id="focusCoverage"' not in html
    assert 'id="focusStatus"' not in html
    assert 'id="stamp"' not in html
    assert 'id="deckBundle"' not in html
    assert 'history-badge' not in html
    assert 'function pill(' not in html
    assert 'Price' in html
    assert 'Confidence' in html
    assert 'Edge' in html
    assert 'Score' in html
    assert '.signal-skeleton-grid' in html
    assert '.signal-meter' in html
    assert 'Sto cercando i prossimi segnali' not in html


def test_signal_terminal_frontend_music_and_click_effect_are_bounded():
    import signal_dashboard as terminal_mod

    html = terminal_mod.HTML
    assert 'id="sakuraMusic"' in html
    assert 'aria-pressed="false"' in html
    assert 'let sakuraAudio=null;' in html
    assert 'sakuraMaster.gain.value=.035' in html
    assert 'document.addEventListener("pointerdown",e=>' in html
    assert 'e.isPrimary || e.button!==0 || e.clientX<0 || e.clientY<0' in html
    assert 'document.addEventListener("click",e=>' not in html
    assert 'history-badge' not in html
    assert 'class="focus-status"' not in html


def test_signal_terminal_persists_discovered_universe_for_restart(tmp_path):
    import signal_dashboard as terminal_mod

    settings = load_settings("config.yaml")
    terminal = terminal_mod.SignalTerminal(settings, tmp_path, refresh_seconds=30)
    meta = {
        "universe_total": 3,
        "universe_mode": "all_active_markets",
        "market_counts": {"spot": 2, "swap": 1},
    }
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    terminal._save_market_snapshot(symbols, meta)

    reloaded = terminal_mod.SignalTerminal(settings, tmp_path, refresh_seconds=30)
    snapshot = reloaded._load_market_snapshot()

    assert snapshot["symbols"] == symbols
    assert snapshot["universe_total"] == 3
    assert snapshot["market_counts"]["swap"] == 1
    assert set(symbols).issubset(reloaded._allowed_live_symbols())


def test_bootstrap_live_data_is_parallel_resumable_and_uses_canonical_timeframe_helper():
    from ai_trading_lab import main as main_mod
    import inspect

    source = inspect.getsource(main_mod.main)
    assert "ThreadPoolExecutor" in source
    assert "thread_name_prefix='bootstrap'" in source
    assert "timeframe_offset(" in source
    assert ".tmp" in source
    assert ".replace(path)" in source
    assert "--workers" in source
