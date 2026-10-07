#!/usr/bin/env python3
"""
Adaptive AI Signal Terminal.

Single executable-style local dashboard for the trained signal stack.
Read-only by design: no order endpoints, no exchange write calls and no
execution authority. It reads public market data, validates provenance,
serves signals and tracks their subsequent outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

def _runtime_root() -> Path:
    """Repository root in source mode; executable directory when frozen by PyInstaller."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


ROOT = _runtime_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _state_root(resource_root: Path) -> Path:
    """Return a writable runtime directory without making the app require an elevated install."""
    if not getattr(sys, "frozen", False):
        return resource_root
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData/Local")
    else:
        base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    return base / "SakuraSignalTerminal"

from ai_trading_lab import __version__
from ai_trading_lab.config import Settings, load_settings
from ai_trading_lab.data import exchange_client, fetch_ohlcv
from ai_trading_lab.deployment import (
    bundle_artifact_fingerprint,
    bundle_compatibility,
    resolve_signal_bundle,
)
from ai_trading_lab.live import (
    LiveAssessment,
    append_live_signal_history,
    scan_top5,
    write_live_snapshot,
)
from ai_trading_lab.live_tracker import update_live_signal_outcomes


APP_TITLE = "Adaptive AI Signal Terminal"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_REFRESH = 20
DEFAULT_HISTORY_BARS = 240


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _signal_rank(signal: str) -> int:
    return {"LONG": 3, "SHORT": 2, "WAIT": 1, "FLAT": 0}.get(signal, 0)


def _signal_class(signal: str) -> str:
    return {
        "LONG": "signal-long",
        "SHORT": "signal-short",
        "WAIT": "signal-wait",
        "FLAT": "signal-flat",
    }.get(signal, "signal-flat")


class SignalTerminal:
    def __init__(
        self,
        settings: Settings,
        root: str | Path = ROOT,
        refresh_seconds: int = DEFAULT_REFRESH,
        history_bars: int = DEFAULT_HISTORY_BARS,
    ):
        self.settings = settings
        self.root = Path(root).resolve()
        self.state_root = _state_root(self.root)
        self.refresh_seconds = max(10, int(refresh_seconds))
        self.history_bars = max(80, int(history_bars))
        self._lock = threading.Lock()
        self._cached_state: dict | None = None
        self._cached_at = 0.0
        self._exchange = None
        self._exchange_error = None
        self._assessment_cache = {}
        self._scan_thread = None
        self._scan_started_at = 0.0
        self._scan_progress: dict[str, object] = {}
        self.state_root.joinpath("logs").mkdir(parents=True, exist_ok=True)
        self.state_root.joinpath("data", "history").mkdir(parents=True, exist_ok=True)

    def _get_exchange(self):
        """Reuse one read-only CCXT client; offline failure falls back to bundled data."""
        if self._exchange is not None:
            return self._exchange
        try:
            self._exchange = exchange_client(
                getattr(self.settings, "exchange", "binance"), sandbox=False
            )
            self._exchange_error = None
        except Exception as exc:
            self._exchange = None
            self._exchange_error = f"{type(exc).__name__}:{exc}"
        return self._exchange

    def _bundle_snapshot(self, symbol: str) -> dict:
        bundle = resolve_signal_bundle(self.settings, self.root, symbol)
        info = {
            "symbol": symbol,
            "bundle": str(bundle),
            "compatible": False,
            "compatibility": "not_checked",
            "manifest_ready": False,
            "training_rows": None,
            "training_end": None,
            "data_fingerprint": None,
            "model_semantics_fingerprint": None,
            "deployment_semantics_fingerprint": None,
            "artifact_fingerprint": None,
            "holdout": {},
        }
        meta = _read_json(bundle / "base_training_meta.json")
        manifest = _read_json(bundle / "deployment_manifest.json")
        holdout = _read_json(bundle / "base_holdout_report.json")
        info.update(
            {
                "training_rows": meta.get("rows"),
                "training_end": meta.get("end"),
                "data_fingerprint": meta.get("data_fingerprint"),
                "model_semantics_fingerprint": meta.get("model_semantics_fingerprint"),
                "deployment_semantics_fingerprint": meta.get(
                    "deployment_semantics_fingerprint"
                ),
                "manifest_ready": bool(manifest.get("ready", False)),
                "holdout": holdout.get(
                    "holdout", holdout if isinstance(holdout, dict) else {}
                ),
            }
        )
        try:
            info["artifact_fingerprint"] = bundle_artifact_fingerprint(bundle)
        except Exception:
            pass
        try:
            compatible, reason = bundle_compatibility(
                self.settings, bundle, symbol
            )
            info["compatible"] = bool(compatible)
            info["compatibility"] = reason
        except Exception as exc:
            info["compatibility"] = f"{type(exc).__name__}:{exc}"
        return info

    @staticmethod
    def _ticker_row(symbol: str, ticker: dict) -> dict:
        def as_float(key: str):
            value = ticker.get(key)
            return float(value) if value is not None else None

        return {
            "symbol": symbol,
            "price": as_float("last"),
            "bid": as_float("bid"),
            "ask": as_float("ask"),
            "timestamp": ticker.get("timestamp"),
            "quote_volume": ticker.get("quoteVolume"),
        }

    def _quotes(self, symbols: list[str]) -> dict:
        """Best-effort realtime ticker overlay; failure never blocks signals."""
        result = {}
        try:
            exchange = self._get_exchange()
        except Exception as exc:
            return {"_error": f"{type(exc).__name__}:{exc}"}

        missing = list(dict.fromkeys(symbols))
        bulk = getattr(exchange, "fetch_tickers", None)
        if missing and callable(bulk):
            try:
                tickers = bulk(missing)
                if isinstance(tickers, dict):
                    for symbol in missing:
                        ticker = tickers.get(symbol)
                        if isinstance(ticker, dict):
                            result[symbol] = self._ticker_row(symbol, ticker)
                    missing = [symbol for symbol in missing if symbol not in result]
            except Exception:
                pass

        for symbol in missing:
            try:
                result[symbol] = self._ticker_row(symbol, exchange.fetch_ticker(symbol))
            except Exception as exc:
                result[symbol] = {"symbol": symbol, "error": f"{type(exc).__name__}:{exc}"}
        return result

    def _quote(self, symbol: str) -> dict:
        try:
            exchange = self._get_exchange()
            payload = self._ticker_row(symbol, exchange.fetch_ticker(symbol))
            payload["generated_at"] = datetime.now(timezone.utc).isoformat()
            return payload
        except Exception as exc:
            return {"symbol": symbol, "error": f"{type(exc).__name__}:{exc}"}

    @staticmethod
    def _history_slug(symbol: str, timeframe: str) -> str:
        return symbol.replace("/", "_").replace(":", "_") + "_" + timeframe

    def _history_from_csv(self, path: Path, bars: int, source: str) -> dict:
        if not path.exists():
            return {"symbol": "", "bars": [], "error": "history_cache_missing"}
        rows = []
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    ts_raw = row.get("timestamp") or row.get("datetime") or row.get(reader.fieldnames[0])
                    stamp = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
                    if stamp.tzinfo is None:
                        stamp = stamp.replace(tzinfo=timezone.utc)
                    rows.append({
                        "t": int(stamp.timestamp() * 1000),
                        "o": float(row["open"]),
                        "h": float(row["high"]),
                        "l": float(row["low"]),
                        "c": float(row["close"]),
                        "v": float(row["volume"]),
                    })
                except (KeyError, TypeError, ValueError, IndexError):
                    continue
        rows = rows[-bars:]
        return {
            "symbol": path.stem.rsplit("_", 1)[0].replace("_", "/"),
            "timeframe": self.settings.timeframe,
            "bars": rows,
            "source": source,
            "cache_path": str(path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _history(self, symbol: str, limit: int | None = None) -> dict:
        bars = max(80, min(4000, int(limit or self.history_bars)))
        slug = self._history_slug(symbol, self.settings.timeframe)
        bundled = self.root / "data" / "historical" / f"{slug}.csv"
        writable = self.state_root / "data" / "history" / f"{slug}.csv"

        # The dashboard must render reliably even when Binance REST is blocked/offline.
        # Prefer a recent writable cache, then the bundled verified snapshot.
        for path, source in ((writable, "local_cache"), (bundled, "bundled")):
            if path.exists():
                try:
                    result = self._history_from_csv(path, bars, source)
                    if result.get("bars"):
                        return result
                except (OSError, ValueError, KeyError):
                    pass

        try:
            exchange = self._get_exchange()
            frame = fetch_ohlcv(exchange, symbol, self.settings.timeframe, bars)
            if frame is not None and not frame.empty:
                writable.parent.mkdir(parents=True, exist_ok=True)
                frame.tail(bars).to_csv(writable, index_label="timestamp")
                candles = [
                    {
                        "t": int(ts.timestamp() * 1000),
                        "o": float(open_),
                        "h": float(high),
                        "l": float(low),
                        "c": float(close),
                        "v": float(volume),
                    }
                    for ts, open_, high, low, close, volume in frame.tail(bars)[
                        ["open", "high", "low", "close", "volume"]
                    ].itertuples(index=True, name=None)
                ]
                return {
                    "symbol": symbol,
                    "timeframe": self.settings.timeframe,
                    "bars": candles,
                    "source": "network",
                    "cache_path": str(writable),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
        except Exception as exc:
            network_error = f"{type(exc).__name__}:{exc}"
        else:
            network_error = "empty_data"

        return {
            "symbol": symbol,
            "timeframe": self.settings.timeframe,
            "bars": [],
            "source": "unavailable",
            "error": network_error,
            "hint": "No historical cache is available. Reconnect to the internet or rebuild the portable package.",
        }

    def _journal(self) -> list[dict]:
        # Frozen builds persist mutable state outside the executable directory.
        # Read from the same state root used by the writer so journal history survives restarts.
        path = self.state_root / "logs" / "live_signal_history.jsonl"
        if not path.exists():
            return []
        rows: list[dict] = []
        for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()[-60:]:
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except json.JSONDecodeError:
                continue
        return rows

    def _assessment_payload(self, assessment: LiveAssessment) -> dict:
        payload = _json_safe(assessment.to_dict())
        payload["signal_class"] = _signal_class(str(payload.get("signal", "WAIT")))
        return payload

    def _bootstrap_state(self) -> dict:
        return {
            "ok": True,
            "app": APP_TITLE,
            "version": __version__,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scan_seconds": 0.0,
            "refresh_seconds": self.refresh_seconds,
            "scan_in_progress": True,
            "scan_started_at": None,
            "scan_progress": dict(self._scan_progress),
            "config": {
                "exchange": self.settings.exchange,
                "timeframe": self.settings.timeframe,
                "primary_symbol": self.settings.symbol,
                "live_symbols": list(getattr(self.settings, "live_symbols", ()) or ()),
                "scan_scope": "all_active_markets",
                "signal_only_mode": bool(self.settings.signal_only_mode),
                "paper_only": bool(self.settings.paper_only),
                "sandbox": bool(self.settings.sandbox),
                "trade_window_required": bool(self.settings.trade_window_required_for_signal),
            },
            "summary": {
                "exchange_available": self._exchange is not None,
                "exchange_error": self._exchange_error,
                "assets_scanned": 0,
                "universe_total": 0,
                "model_backed_assets": 0,
                "active_signals": 0,
                "waits": 0,
                "compatible_bundles": 0,
                "fresh_data_assets": 0,
                "terminal_ready": False,
            },
            "signals": [],
            "journal": self._journal(),
            "outcome_update": {},
            "notes": ["Scansione completa dell'universo attivo in corso."],
        }

    def _terminal_state(self, force: bool = False, *, background: bool = False) -> dict:
        now = time.time()
        with self._lock:
            if (
                not force
                and self._cached_state is not None
                and now - self._cached_at < self.refresh_seconds
            ):
                return self._cached_state

            if background:
                if self._scan_thread is not None and self._scan_thread.is_alive():
                    base = dict(self._cached_state or self._bootstrap_state())
                    base["scan_in_progress"] = True
                    base["scan_started_at"] = (
                        datetime.fromtimestamp(self._scan_started_at, tz=timezone.utc).isoformat()
                        if self._scan_started_at else None
                    )
                    base["scan_seconds"] = round(max(0.0, now - self._scan_started_at), 1)
                    base["scan_progress"] = dict(self._scan_progress)
                    return base

                self._scan_started_at = now
                self._scan_progress = {"stage": "discovering", "evaluated": 0, "total": 0, "reused": 0, "refreshed": 0, "signals": 0, "waits": 0, "failures": 0}
                self._scan_thread = threading.Thread(
                    target=self._terminal_state,
                    kwargs={"force": True},
                    name="signal-terminal-scan",
                    daemon=True,
                )
                self._scan_thread.start()
                base = dict(self._cached_state or self._bootstrap_state())
                base["scan_in_progress"] = True
                base["scan_started_at"] = datetime.fromtimestamp(
                    self._scan_started_at, tz=timezone.utc
                ).isoformat()
                base["scan_seconds"] = 0.0
                return base

        # Never hold the request/cache lock while network, model or disk work runs.
        started = time.monotonic()
        try:
                configured_symbols = list(
                    dict.fromkeys(
                        getattr(self.settings, "live_symbols", ())
                        or (self.settings.symbol,)
                    )
                )
                def report_progress(progress):
                    if isinstance(progress, dict):
                        with self._lock:
                            self._scan_progress = dict(progress)

                scan_result = scan_top5(
                    self.settings,
                    str(self.root),
                    exchange=self._get_exchange(),
                    cache=self._assessment_cache,
                    return_meta=True,
                    progress_callback=report_progress,
                )
                if isinstance(scan_result, tuple):
                    assessments, universe_meta = scan_result
                else:
                    assessments = scan_result
                    universe_meta = {
                        "universe_total": len(assessments),
                        "universe_model_backed": len(assessments),
                        "universe_evaluated": len(assessments),
                        "universe_mode": "compatibility_fallback",
                    }
                # Tickers are presentation-only; never poll the entire scan universe.
                symbols = list(dict.fromkeys(
                    configured_symbols + [x.symbol for x in assessments]
                ))
                write_live_snapshot(assessments, str(self.state_root))
                append_live_signal_history(assessments, str(self.state_root))

                try:
                    outcome_update = update_live_signal_outcomes(
                        self.settings, str(self.state_root), exchange=self._get_exchange()
                    )
                except Exception as exc:
                    outcome_update = {
                        "updated": 0,
                        "open": None,
                        "closed": None,
                        "error": f"{type(exc).__name__}:{exc}",
                    }

                market_symbols = sorted({
                    str(symbol).strip()
                    for symbol in (universe_meta.get("market_symbols") or [])
                    if str(symbol).strip()
                })
                quotes = self._quotes(symbols)
                signals = []
                for assessment in assessments:
                    row = self._assessment_payload(assessment)
                    row["bundle"] = self._bundle_snapshot(assessment.symbol)
                    realtime = quotes.get(assessment.symbol, {})
                    if realtime.get("price") is not None:
                        row["realtime_price"] = realtime["price"]
                    row["quote"] = realtime

                    details = row.get("details") or {}
                    row["decision"] = {
                        "p_up": details.get("p_up"),
                        "expected_return": row.get("expected_return"),
                        "expected_return_lcb": details.get("expected_return_lcb"),
                        "expected_return_ucb": details.get("expected_return_ucb"),
                        "robust_directional_edge": details.get("robust_directional_edge"),
                        "selection_score": details.get("selection_score"),
                        "score": details.get("score"),
                        "meta_success": details.get("meta_success"),
                        "model_disagreement": details.get("model_disagreement"),
                        "return_disagreement": details.get("return_disagreement"),
                        "regime": details.get("regime"),
                        "analog_n": details.get("analog_n"),
                        "analog_agreement": details.get("analog_agreement"),
                        "trade_window_ready": details.get("trade_window_ready"),
                        "trade_window_direction": details.get(
                            "trade_window_direction"
                        ),
                        "trade_window_confidence": details.get(
                            "trade_window_confidence"
                        ),
                        "data_age_minutes": details.get("data_age_minutes"),
                    }
                    signals.append(row)

                signals.sort(
                    key=lambda x: (
                        _signal_rank(str(x.get("signal", "WAIT"))),
                        float((x.get("decision") or {}).get("selection_score", 0.0) or 0.0),
                        float((x.get("decision") or {}).get("robust_directional_edge", 0.0) or 0.0),
                        float(x.get("confidence", 0.0) or 0.0),
                    ),
                    reverse=True,
                )

                active = sum(
                    str(x.get("signal")) in {"LONG", "SHORT"} for x in signals
                )
                compatible = int(
                    universe_meta.get("universe_model_eligible", len(signals))
                )
                fresh = sum(
                    float((x.get("decision") or {}).get("data_age_minutes", 1e9))
                    <= float(
                        getattr(
                            self.settings, "live_max_data_age_minutes", 30.0
                        )
                    )
                    for x in signals
                    if (x.get("decision") or {}).get("data_age_minutes")
                    is not None
                )

                state = {
                    "ok": True,
                    "app": APP_TITLE,
                    "version": __version__,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "scan_seconds": round(
                        time.monotonic() - started, 3
                    ),
                    "refresh_seconds": self.refresh_seconds,
                    "scan_progress": {
                        "stage": "complete",
                        "evaluated": int(universe_meta.get("scan_completed", universe_meta.get("universe_evaluated", len(signals)))),
                        "total": int(universe_meta.get("scan_total", universe_meta.get("universe_evaluated", len(signals)))),
                        "reused": int(universe_meta.get("assessments_reused", 0)),
                        "refreshed": int(universe_meta.get("assessments_refreshed", 0)),
                        "signals": int(universe_meta.get("universe_signals", len(signals))),
                        "waits": int(universe_meta.get("universe_waits", 0)),
                        "failures": int(universe_meta.get("assessment_failures", 0)),
                    },
                    "config": {
                        "exchange": self.settings.exchange,
                        "timeframe": self.settings.timeframe,
                        "primary_symbol": self.settings.symbol,
                        "live_symbols": configured_symbols,
                        "market_symbols": market_symbols,
                        "market_counts": universe_meta.get("market_counts", {}),
                        "scan_scope": universe_meta.get("universe_mode", "all_active_markets"),
                        "scan_source": "exchange" if market_symbols else "local_fallback",
                        "signal_only_mode": bool(
                            self.settings.signal_only_mode
                        ),
                        "paper_only": bool(self.settings.paper_only),
                        "sandbox": bool(self.settings.sandbox),
                        "trade_window_required": bool(
                            self.settings.trade_window_required_for_signal
                        ),
                    },
                    "summary": {
                        "exchange_available": self._get_exchange() is not None,
                        "exchange_error": self._exchange_error,
                        "assets_scanned": int(universe_meta.get("universe_evaluated", len(signals))),
                        "universe_total": int(universe_meta.get("universe_total", len(signals))),
                        "model_backed_assets": int(universe_meta.get("universe_model_backed", len(signals))),
                        "model_eligible_assets": int(universe_meta.get("universe_model_eligible", len(signals))),
                        "ineligible_model_assets": int(universe_meta.get("universe_ineligible_model", 0)),
                        "uncovered_assets": int(universe_meta.get("universe_uncovered", 0)),
                        "active_signals": active,
                        "waits": int(universe_meta.get("universe_waits", max(0, universe_meta.get("universe_evaluated", len(signals)) - active))),
                        "compatible_bundles": compatible,
                        "fresh_data_assets": fresh,
                        "assessment_failures": int(universe_meta.get("assessment_failures", 0)),
                        "assessments_reused": int(universe_meta.get("assessments_reused", 0)),
                        "assessments_refreshed": int(universe_meta.get("assessments_refreshed", 0)),
                        "terminal_ready": compatible > 0,
                    },
                    "signals": signals,
                    "journal": self._journal(),
                    "outcome_update": _json_safe(outcome_update),
                    "notes": [
                        "Sola lettura: il terminale non espone API per ordini.",
                        "Ogni scan valuta l'universo attivo scoperto dall'exchange, limitandosi ai bundle verificati per la Top 5.",
                        "WAIT è l'esito predefinito quando dati, provenienza o evidenze non sono sufficienti.",
                    ],
                }
            except Exception as exc:
                state = {
                    "ok": False,
                    "app": APP_TITLE,
                    "version": __version__,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "scan_seconds": round(
                        time.monotonic() - started, 3
                    ),
                    "refresh_seconds": self.refresh_seconds,
                    "config": {
                        "exchange": self.settings.exchange,
                        "timeframe": self.settings.timeframe,
                        "primary_symbol": self.settings.symbol,
                        "live_symbols": list(
                            getattr(self.settings, "live_symbols", ())
                            or ()
                        ),
                    },
                    "summary": {
                        "assets_scanned": 0,
                        "active_signals": 0,
                        "waits": 0,
                        "compatible_bundles": 0,
                        "fresh_data_assets": 0,
                        "terminal_ready": False,
                    },
                    "signals": [],
                    "journal": self._journal(),
                    "outcome_update": {},
                    "error": f"{type(exc).__name__}:{exc}",
                    "notes": [
                        "Il terminale ha eseguito un fail-closed.",
                        "Controlla dati pubblici, bundle addestrati e compatibilità.",
                    ],
                }

            cached_state = _json_safe(state)
            with self._lock:
                self._cached_state = cached_state
                self._cached_at = time.time()
                self._scan_progress = dict(cached_state.get("scan_progress") or {})
                self._scan_thread = None
            return cached_state

    def health(self) -> dict:
        return {
            "ok": True,
            "version": __version__,
            "cached": self._cached_state is not None,
            "scan_in_progress": bool(self._scan_thread is not None and self._scan_thread.is_alive()),
            "scan_age_seconds": (
                round(time.time() - self._scan_started_at, 1)
                if self._scan_started_at
                else None
            ),
            "cache_age_seconds": (
                round(time.time() - self._cached_at, 1)
                if self._cached_state is not None
                else None
            ),
            "last_scan_ok": (
                bool(self._cached_state.get("ok"))
                if self._cached_state is not None
                else None
            ),
        }


HTML = r"""<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<meta http-equiv="X-Content-Type-Options" content="nosniff">
<title>Signal Monitor</title>
<style>
:root{
  --bg:#07050b;--panel:#100d18;--panel2:#151020;--line:#2a2035;
  --text:#f7f2fb;--muted:#948aa4;--green:#45e39a;--red:#ff6f88;
  --amber:#ffd166;--blue:#78b8ff;--pink:#ff78c8;--pink2:#c85cff;
  --cyan:#7de8ff;--shadow:0 24px 90px rgba(0,0,0,.34);
  --glow-pink:0 0 24px rgba(255,120,200,.18);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;min-height:100vh;overflow-x:hidden;background:
radial-gradient(900px 500px at 10% -10%,rgba(255,120,200,.12),transparent 62%),
radial-gradient(1000px 520px at 100% 0,rgba(125,232,255,.08),transparent 58%),
radial-gradient(700px 420px at 50% 110%,rgba(200,92,255,.08),transparent 62%),
var(--bg);color:var(--text);
font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}
body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.18;
background-image:linear-gradient(rgba(255,120,200,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(255,120,200,.045) 1px,transparent 1px);
background-size:44px 44px;mask-image:linear-gradient(to bottom,black,transparent 78%);
animation:gridDrift 22s linear infinite}
.wrap{position:relative;z-index:1;max-width:1480px;margin:auto;padding:24px 24px 56px}
.top{display:flex;justify-content:space-between;gap:24px;align-items:flex-end;margin-bottom:14px;position:relative}
.eyebrow{font-size:10px;text-transform:uppercase;letter-spacing:.2em;color:var(--muted)}
h1{font-size:36px;line-height:1;margin:6px 0 9px;letter-spacing:-.03em}
.sub{max-width:930px;color:var(--muted)}
.actions{display:flex;gap:9px;align-items:center;flex-wrap:wrap}
button,select{position:relative;overflow:hidden;border:1px solid var(--line);background:linear-gradient(180deg,rgba(28,19,38,.96),rgba(16,12,24,.98));color:var(--text);border-radius:12px;padding:9px 12px;transition:transform .12s ease,border-color .2s ease,box-shadow .2s ease,background .2s ease;box-shadow:0 6px 24px rgba(0,0,0,.16)}
button{cursor:pointer}
button::after,select::after{content:"";position:absolute;inset:-40% -15%;transform:translateX(-120%) rotate(10deg);background:linear-gradient(90deg,transparent,rgba(255,255,255,.12),transparent);transition:transform .45s ease;pointer-events:none}
button:hover::after,select:hover::after{transform:translateX(120%) rotate(10deg)}
button:hover,select:hover{border-color:rgba(255,120,200,.58);box-shadow:0 0 0 1px rgba(255,120,200,.08),0 0 28px rgba(255,120,200,.14);transform:translateY(-1px)}
button:active,select:active{transform:translateY(1px) scale(.985)}
button:focus-visible,select:focus-visible{outline:none;border-color:var(--pink);box-shadow:0 0 0 2px rgba(255,120,200,.17),0 0 28px rgba(255,120,200,.18)}
.nav{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:8px;margin:0 0 12px;border:1px solid rgba(255,120,200,.13);border-radius:14px;background:rgba(15,9,21,.56);backdrop-filter:blur(16px);box-shadow:var(--glow-pink)}
.nav-btn{font-size:10px;letter-spacing:.14em;text-transform:uppercase;padding:8px 11px;color:var(--muted);background:transparent;border-color:transparent;box-shadow:none}
.nav-btn:hover{color:var(--text);background:rgba(255,120,200,.07)}
.nav-btn.active{color:#fff;border-color:rgba(255,120,200,.34);background:linear-gradient(180deg,rgba(255,120,200,.12),rgba(255,120,200,.04));box-shadow:inset 0 0 18px rgba(255,120,200,.06),0 0 18px rgba(255,120,200,.13)}
.operator-badge{display:flex;align-items:center;gap:8px;padding:4px 8px 4px 5px;border-radius:999px;border:1px solid rgba(255,120,200,.2);background:rgba(255,120,200,.045);box-shadow:0 0 24px rgba(255,120,200,.08)}
.operator-art{width:34px;height:34px;border-radius:50%;display:grid;place-items:center;background:radial-gradient(circle at 30% 25%,rgba(255,255,255,.18),transparent 32%),linear-gradient(135deg,rgba(255,120,200,.18),rgba(200,92,255,.08));box-shadow:inset 0 0 16px rgba(255,120,200,.15)}
.operator-badge svg{width:30px;height:30px;filter:drop-shadow(0 0 8px rgba(255,120,200,.32))}
.operator-copy{display:flex;flex-direction:column;line-height:1}.operator-copy strong{font-size:9px;letter-spacing:.16em}.operator-copy span{font-size:8px;color:var(--muted);margin-top:4px;letter-spacing:.12em}
#stamp{font-size:12px;color:var(--muted);white-space:nowrap}
.metrics{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px}
.decision-deck{display:grid;grid-template-columns:1.05fr 1.45fr;gap:14px;margin-top:14px}
.decision-hero{position:relative;overflow:hidden;padding:18px;border-radius:18px;background:
radial-gradient(420px 180px at 0 0,rgba(255,120,200,.11),transparent 65%),
linear-gradient(135deg,rgba(29,14,34,.98),rgba(12,9,18,.99));border:1px solid rgba(255,120,200,.18);box-shadow:var(--shadow),0 0 40px rgba(255,120,200,.06)}
.decision-hero::after{content:"";position:absolute;inset:auto -15% -45% 30%;height:110px;background:radial-gradient(circle,rgba(255,120,200,.16),transparent 68%);filter:blur(20px);pointer-events:none}
.decision-top{display:flex;align-items:flex-start;justify-content:space-between;gap:12px}
.decision-signal{font-size:40px;font-weight:950;line-height:1;letter-spacing:.04em}
.decision-meta{margin-top:8px;color:var(--muted);font-size:11px}
.decision-stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:18px}
.mini-stat{padding:9px 10px;border:1px solid rgba(255,120,200,.12);background:rgba(255,255,255,.018);border-radius:11px}
.mini-stat .k{font-size:9px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.mini-stat .v{margin-top:4px;font-weight:800;font-size:15px}
.trace{padding:15px 16px;border-radius:18px;border:1px solid var(--line);background:linear-gradient(180deg,rgba(18,12,26,.96),rgba(10,8,16,.985));box-shadow:var(--shadow)}
.trace-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px}
.trace-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
.trace-node{position:relative;padding:10px;border-radius:11px;border:1px solid var(--line);background:rgba(255,255,255,.014)}
.trace-node strong{display:block;font-size:10px;letter-spacing:.08em;text-transform:uppercase}
.trace-node span{display:block;margin-top:4px;font-size:10px;color:var(--muted)}
.trace-node.ready{border-color:rgba(69,227,154,.24);box-shadow:inset 0 0 18px rgba(69,227,154,.035)}
.trace-node.wait{border-color:rgba(255,209,102,.2)}
.trace-node.fail{border-color:rgba(255,111,136,.24)}
.trace-dot{position:absolute;right:9px;top:10px;width:6px;height:6px;border-radius:50%;box-shadow:0 0 12px currentColor;background:currentColor}
.trace-node.ready .trace-dot{color:var(--green)}.trace-node.wait .trace-dot{color:var(--amber)}.trace-node.fail .trace-dot{color:var(--red)}
.reason-strip{display:flex;flex-wrap:wrap;gap:5px;margin-top:12px}
.reason-strip .reason{background:rgba(255,120,200,.045);border-color:rgba(255,120,200,.16);color:#d9c9df}
@media(max-width:900px){.decision-deck{grid-template-columns:1fr}.trace-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
.card,.panel{position:relative;background:linear-gradient(180deg,rgba(18,12,26,.96),rgba(10,8,16,.985));
border:1px solid var(--line);box-shadow:var(--shadow);isolation:isolate}
.card::before,.panel::before{content:"";position:absolute;inset:0;border-radius:inherit;pointer-events:none;background:radial-gradient(420px 120px at 20% 0,rgba(255,120,200,.065),transparent 68%);opacity:.7}
.card:hover,.panel:hover{border-color:rgba(255,120,200,.24);box-shadow:var(--shadow),0 0 32px rgba(255,120,200,.07);transform:translateY(-1px)}
.card,.panel{transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease}
.card{border-radius:15px;padding:15px;animation:panelIn .58s ease both}
.metrics .card:nth-child(2){animation-delay:.04s}.metrics .card:nth-child(3){animation-delay:.08s}.metrics .card:nth-child(4){animation-delay:.12s}.metrics .card:nth-child(5){animation-delay:.16s}.metrics .card:nth-child(6){animation-delay:.20s}
.metric-label,.label{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.13em}
.metric-value{margin-top:6px;font-size:24px;font-weight:800;letter-spacing:-.02em}
.good{color:var(--green);text-shadow:0 0 12px rgba(69,227,154,.18)}.bad{color:var(--red);text-shadow:0 0 12px rgba(255,111,136,.18)}.warn{color:var(--amber);text-shadow:0 0 12px rgba(255,209,102,.18)}
.panel{border-radius:18px;margin-top:14px;overflow:hidden;animation:panelIn .62s ease both}
#market{animation-delay:.08s}#detail{animation-delay:.13s}#journal{animation-delay:.18s}#evidencePanel{animation-delay:.23s}
.panel-head{display:flex;justify-content:space-between;gap:14px;align-items:center;padding:15px 17px;border-bottom:1px solid var(--line)}
.title{font-weight:750;font-size:15px}
.small{font-size:11px;color:var(--muted)}
.layout{display:grid;grid-template-columns:1.5fr .5fr;gap:14px}
.chart-panel{min-height:500px;box-shadow:var(--shadow),inset 0 0 40px rgba(255,120,200,.025)}
.chart-tools{display:flex;gap:8px;align-items:center}
#chart{display:block;width:100%;height:430px;filter:drop-shadow(0 0 14px rgba(120,184,255,.09))}
.chart-wrap{position:relative;padding:10px 12px 12px}
.chart-wrap::before{content:"";position:absolute;inset:10px 12px 12px;border:1px solid rgba(255,120,200,.07);border-radius:12px;pointer-events:none;box-shadow:inset 0 0 28px rgba(255,120,200,.025)}
.chart-wrap::after{content:"";position:absolute;left:12%;right:12%;top:10px;height:1px;pointer-events:none;background:linear-gradient(90deg,transparent,rgba(255,120,200,.25),transparent);animation:sweep 5.5s ease-in-out infinite}
.chart-empty{display:flex;align-items:center;justify-content:center;height:430px;color:var(--muted)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--muted);padding:0 12px 12px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}
.controls{display:flex;gap:8px;align-items:center}
table{border-collapse:collapse;width:100%}
th,td{padding:11px 12px;text-align:left;border-bottom:1px solid rgba(55,38,68,.62);vertical-align:top}
th{font-size:9px;text-transform:uppercase;letter-spacing:.11em;color:var(--muted);position:sticky;top:0;background:rgba(11,7,17,.96);backdrop-filter:blur(10px)}
tbody tr{transition:background .16s ease,transform .16s ease}
tbody tr:hover{background:rgba(255,120,200,.035)}
.interactive-row{cursor:pointer}
.interactive-row:active{transform:scale(.998)}
.table-wrap{overflow:auto}
.signal{font-weight:850;letter-spacing:.08em}
.signal-long{color:var(--green)}.signal-short{color:var(--red)}.signal-wait{color:var(--amber)}.signal-flat{color:var(--muted)}
.pill{display:inline-flex;align-items:center;gap:6px;padding:4px 8px;border-radius:999px;border:1px solid var(--line);font-size:10px;box-shadow:inset 0 0 12px rgba(255,120,200,.025)}
.dot{width:7px;height:7px;border-radius:50%;background:currentColor;box-shadow:0 0 10px currentColor;animation:dotPulse 1.8s ease-in-out infinite}
.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;padding:14px}
.info{border:1px solid var(--line);border-radius:12px;background:#0d141d;padding:13px}
.info .big{font-size:20px;font-weight:800;margin-top:4px}
.reasons{display:flex;gap:5px;flex-wrap:wrap;margin-top:6px}
.reason{padding:3px 6px;border:1px solid #283443;border-radius:7px;background:#171f29;color:#a9b4c2;font-size:10px}
.num{font-variant-numeric:tabular-nums}
.footer{margin-top:14px;color:var(--muted);font-size:11px}
#cursor{position:absolute;pointer-events:none;display:none;background:#101822;border:1px solid #2b3949;border-radius:9px;padding:7px 9px;font-size:10px;box-shadow:var(--shadow)}
#ambient-canvas{position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:0;opacity:.62}
#pointer-aura,#anime-cursor{position:fixed;left:0;top:0;pointer-events:none;z-index:9999;opacity:0;transform:translate3d(-100px,-100px,0);will-change:transform,opacity}
#pointer-aura{width:130px;height:130px;margin:-65px 0 0 -65px;border-radius:50%;background:radial-gradient(circle,rgba(255,120,200,.13),rgba(200,92,255,.045) 42%,transparent 72%);filter:blur(2px)}
#anime-cursor{width:46px;height:46px;margin:-7px 0 0 -7px;filter:drop-shadow(0 0 10px rgba(255,120,200,.45));transition:filter .16s ease}
#livePrice.good{animation:liveGlow 2.2s ease-in-out infinite}
#anime-cursor.click{filter:drop-shadow(0 0 18px rgba(255,120,200,.95));animation:cursorHit .16s ease}
.click-ripple{position:fixed;width:16px;height:16px;margin:-8px;border:1px solid rgba(255,120,200,.8);border-radius:50%;pointer-events:none;z-index:9998;animation:ripple .52s ease-out forwards;box-shadow:0 0 22px rgba(255,120,200,.34)}
.reveal{animation:reveal .46s ease both}
@keyframes panelIn{from{opacity:0;transform:translateY(10px) scale(.99)}to{opacity:1;transform:none}}
@keyframes reveal{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:none}}
@keyframes gridDrift{from{background-position:0 0,0 0}to{background-position:44px 44px,-44px 44px}}
@keyframes sweep{0%,100%{opacity:.05;transform:translateX(-28%)}50%{opacity:.8;transform:translateX(28%)}}
@keyframes dotPulse{0%,100%{opacity:.55;transform:scale(.86)}50%{opacity:1;transform:scale(1.15)}}
@keyframes ripple{from{opacity:.8;transform:scale(.4)}to{opacity:0;transform:scale(9)}}
@keyframes cursorHit{0%{transform:scale(1)}50%{transform:scale(.82) rotate(-4deg)}100%{transform:scale(1)}}
@keyframes liveGlow{0%,100%{box-shadow:0 0 0 rgba(69,227,154,0)}50%{box-shadow:0 0 24px rgba(69,227,154,.22)}}
.decision-flash{animation:decisionFlash .52s ease}
.signal-live-long{box-shadow:var(--shadow),0 0 44px rgba(69,227,154,.08)}
.signal-live-short{box-shadow:var(--shadow),0 0 44px rgba(255,111,136,.08)}
.signal-live-wait{box-shadow:var(--shadow),0 0 44px rgba(255,209,102,.07)}
@keyframes decisionFlash{0%{filter:brightness(1)}25%{filter:brightness(1.32)}100%{filter:brightness(1)}}

@keyframes ambientFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
@media(pointer:fine){body.alpha-pointer,body.alpha-pointer button,body.alpha-pointer select{cursor:none}}
@media(pointer:coarse){#pointer-aura,#anime-cursor{display:none}}
@media(prefers-reduced-motion:reduce){
  html{scroll-behavior:auto}
  *,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}
  body::before{display:none}
}
@media(max-width:1180px){.metrics{grid-template-columns:repeat(3,minmax(0,1fr))}.layout{grid-template-columns:1fr}}
@media(max-width:760px){.wrap{padding:18px 13px 40px}.top{align-items:flex-start;flex-direction:column}h1{font-size:29px}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.grid3{grid-template-columns:1fr}.chart-panel{min-height:430px}#chart{height:350px}.nav{overflow:auto;flex-wrap:nowrap}.nav-btn{white-space:nowrap}}

/* 0.9.18 art-direction layer */
.operator-stage{
  position:absolute;right:14px;bottom:10px;width:126px;height:126px;pointer-events:none;opacity:.9;
  animation:operatorFloat 4.8s ease-in-out infinite;transform-origin:50% 80%;
}
.operator-stage::before{
  content:"";position:absolute;inset:16% 8% 4%;border-radius:50%;
  background:radial-gradient(circle,rgba(255,120,200,.16),rgba(200,92,255,.06) 45%,transparent 72%);
  filter:blur(10px);transform:scale(1.08);
}
.operator-stage svg{position:relative;width:100%;height:100%;filter:drop-shadow(0 0 14px rgba(255,120,200,.32))}
.operator-stage .operator-scan{
  position:absolute;left:8%;right:8%;top:12%;height:1px;
  background:linear-gradient(90deg,transparent,rgba(125,232,255,.7),transparent);
  animation:operatorScan 3.7s ease-in-out infinite;
}
.operator-caption{
  position:absolute;right:5px;bottom:0;padding:3px 6px;border-radius:999px;
  background:rgba(9,6,14,.75);border:1px solid rgba(255,120,200,.18);
  font-size:7px;letter-spacing:.18em;color:var(--muted);backdrop-filter:blur(8px);
}
.decision-hero{min-height:208px}
.decision-hero .reason-strip{max-width:calc(100% - 118px);padding-right:4px}
.timeline-shell{padding:16px 16px 18px;overflow-x:auto}
.signal-timeline{min-width:760px;display:grid;gap:11px}
.timeline-entry{
  position:relative;padding:13px 14px 14px;border:1px solid var(--line);border-radius:14px;
  background:linear-gradient(180deg,rgba(21,13,29,.93),rgba(11,8,17,.98));
  transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease;
}
.timeline-entry:hover{transform:translateY(-2px);border-color:rgba(255,120,200,.3);box-shadow:0 0 28px rgba(255,120,200,.07)}
.timeline-entry::before{
  content:"";position:absolute;left:16px;right:16px;top:48px;height:1px;
  background:linear-gradient(90deg,rgba(255,120,200,.12),rgba(255,120,200,.34),rgba(125,232,255,.2));
}
.timeline-top{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:14px}
.timeline-title{display:flex;align-items:center;gap:8px}
.timeline-stages{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}
.timeline-stage{
  position:relative;min-height:64px;padding:11px 10px 9px 12px;border:1px solid var(--line);
  border-radius:11px;background:rgba(255,255,255,.016);z-index:1;
}
.timeline-stage .stage-dot{
  position:absolute;left:11px;top:-4px;width:8px;height:8px;border-radius:50%;
  background:var(--muted);box-shadow:0 0 0 3px rgba(11,8,17,.96),0 0 10px currentColor;
}
.timeline-stage strong{display:block;font-size:10px;letter-spacing:.07em;text-transform:uppercase}
.timeline-stage span{display:block;margin-top:4px;font-size:10px;color:var(--muted)}
.timeline-stage.ready{border-color:rgba(69,227,154,.2)}.timeline-stage.ready .stage-dot{color:var(--green);background:var(--green)}
.timeline-stage.pending{border-color:rgba(255,209,102,.16)}.timeline-stage.pending .stage-dot{color:var(--amber);background:var(--amber)}
.timeline-stage.final{border-color:rgba(125,232,255,.18)}.timeline-stage.final .stage-dot{color:var(--cyan);background:var(--cyan)}
.timeline-empty{padding:18px;border:1px dashed rgba(255,120,200,.16);border-radius:12px;color:var(--muted);text-align:center}
.inspector-backdrop{
  position:fixed;inset:0;background:rgba(4,2,8,.58);backdrop-filter:blur(2px);
  opacity:0;pointer-events:none;transition:opacity .22s ease;z-index:10000;
}
.inspector-drawer{
  position:fixed;top:16px;right:16px;bottom:16px;width:min(460px,calc(100vw - 32px));
  transform:translate3d(112%,0,0);transition:transform .3s cubic-bezier(.2,.8,.2,1);
  z-index:10001;border:1px solid rgba(255,120,200,.24);border-radius:20px;
  background:linear-gradient(180deg,rgba(18,10,25,.985),rgba(8,6,13,.995));
  box-shadow:0 30px 120px rgba(0,0,0,.52),0 0 48px rgba(255,120,200,.08);
  overflow:hidden;display:flex;flex-direction:column;
}
body.drawer-open .inspector-backdrop{opacity:1;pointer-events:auto}
body.drawer-open .inspector-drawer{transform:translate3d(0,0,0)}
.inspector-head{
  display:flex;align-items:center;justify-content:space-between;gap:12px;padding:15px 16px;
  border-bottom:1px solid var(--line);background:linear-gradient(180deg,rgba(255,120,200,.06),transparent);
}
.inspector-close{width:34px;height:34px;padding:0;display:grid;place-items:center;border-radius:10px}
.inspector-scroll{overflow:auto;padding:16px}
.inspector-hero{
  position:relative;overflow:hidden;padding:16px;border-radius:15px;border:1px solid rgba(255,120,200,.18);
  background:radial-gradient(360px 160px at 0 0,rgba(255,120,200,.1),transparent 62%),rgba(255,255,255,.012);
}
.inspector-hero .verdict{font-size:34px;line-height:1;font-weight:950;letter-spacing:.04em}
.inspector-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:9px;margin-top:11px}
.inspector-card{padding:12px;border:1px solid var(--line);border-radius:12px;background:rgba(255,255,255,.015)}
.inspector-card .k{font-size:9px;text-transform:uppercase;letter-spacing:.11em;color:var(--muted)}
.inspector-card .v{margin-top:5px;font-size:15px;font-weight:800}
.inspector-section{margin-top:13px}.inspector-section h3{margin:0 0 8px;font-size:10px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted)}
.inspector-reasons{display:flex;gap:6px;flex-wrap:wrap}
.inspector-trace{display:grid;gap:7px}
.inspector-trace-row{display:grid;grid-template-columns:86px 1fr auto;align-items:center;gap:8px;padding:8px 9px;border:1px solid var(--line);border-radius:10px;background:rgba(255,255,255,.012)}
.inspector-trace-row strong{font-size:10px;text-transform:uppercase;letter-spacing:.06em}.inspector-trace-row span{font-size:10px;color:var(--muted)}
.inspector-code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:9px;line-height:1.55;color:#bdb3c4;word-break:break-word}
.chart-crosshair{position:absolute;inset:10px 12px 12px;display:none;pointer-events:none;overflow:hidden;border-radius:12px}
.chart-crosshair .cx{position:absolute;top:0;bottom:0;width:1px;transform:translate3d(0,0,0);background:linear-gradient(to bottom,transparent,rgba(125,232,255,.55),transparent)}
.chart-crosshair .cy{position:absolute;left:0;right:0;height:1px;transform:translate3d(0,0,0);background:linear-gradient(90deg,transparent,rgba(255,120,200,.55),transparent)}
.chart-crosshair .badge{
  position:absolute;right:8px;top:8px;padding:5px 7px;border-radius:8px;
  background:rgba(8,5,13,.9);border:1px solid rgba(125,232,255,.2);font-size:9px;color:#d8d0df;
  box-shadow:0 0 18px rgba(125,232,255,.08);white-space:nowrap;
}
.chart-regime{
  position:absolute;left:22px;top:18px;padding:4px 7px;border-radius:999px;
  background:rgba(11,8,17,.82);border:1px solid rgba(125,232,255,.16);font-size:9px;
  color:var(--muted);letter-spacing:.08em;text-transform:uppercase;pointer-events:none;backdrop-filter:blur(8px);
}
@keyframes operatorFloat{0%,100%{transform:translateY(0) rotate(.2deg)}50%{transform:translateY(-5px) rotate(-.4deg)}}
@keyframes operatorScan{0%,100%{opacity:.08;transform:translateX(-18%)}50%{opacity:.7;transform:translateX(18%)}}
@media(max-width:760px){
  .decision-hero{min-height:220px}.decision-hero .reason-strip{max-width:100%;padding-right:0;padding-bottom:34px}
  .operator-stage{width:94px;height:94px;right:7px;bottom:6px}.operator-caption{display:none}
  .signal-timeline{min-width:0}.timeline-entry::before{display:none}.timeline-stages{grid-template-columns:1fr 1fr}
  .inspector-drawer{top:8px;right:8px;bottom:8px;width:calc(100vw - 16px)}
}
@media(prefers-reduced-motion:reduce){
  .operator-stage{animation:none}.operator-stage .operator-scan{animation:none}
}


/* Focus mode: only verified Top 5 picks and their historical charts. */
.focus-only .top5-head{display:flex;justify-content:space-between;gap:18px;align-items:flex-end;margin:10px 0 18px}
.focus-only .top5-title{font-size:28px;font-weight:800;letter-spacing:-.03em}
.focus-only .top5-sub{color:var(--muted);max-width:780px;margin-top:5px}
.top5-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}
.pick-card{position:relative;overflow:hidden;border:1px solid rgba(255,120,200,.14);border-radius:18px;background:linear-gradient(180deg,rgba(255,255,255,.035),rgba(255,255,255,.015)),var(--panel);box-shadow:var(--shadow);cursor:pointer;transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease}
.pick-card:hover{transform:translateY(-3px);border-color:rgba(255,120,200,.34);box-shadow:var(--shadow),0 0 34px rgba(255,120,200,.12)}
.pick-card.pick-long{--pick-glow:rgba(69,227,154,.18)}.pick-card.pick-short{--pick-glow:rgba(255,111,136,.18)}
.pick-head{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;padding:18px 18px 10px}
.pick-rank{font-size:11px;color:var(--muted);letter-spacing:.16em}.pick-symbol{font-size:20px;font-weight:800}
.pick-signal{font-size:13px;font-weight:800;letter-spacing:.12em;padding:7px 10px;border-radius:999px;border:1px solid currentColor;box-shadow:0 0 22px var(--pick-glow)}
.pick-stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;padding:0 18px 12px}.pick-stat{padding:8px 10px;border-radius:12px;background:rgba(255,255,255,.025)}
.pick-stat .k{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}.pick-stat .v{margin-top:3px;font-weight:700}.pick-chart{height:220px;padding:0 8px 8px}.pick-chart canvas{display:block;width:100%;height:100%}
.pick-foot{display:flex;justify-content:space-between;gap:10px;align-items:center;padding:10px 18px 15px;border-top:1px solid rgba(255,255,255,.05);font-size:11px;color:var(--muted)}
.history-badge{padding:4px 8px;border-radius:999px;background:rgba(125,232,255,.06);border:1px solid rgba(125,232,255,.12);color:#9eeeff}
.pick-card{cursor:pointer;position:relative;isolation:isolate}
.pick-card::after{content:"↗";position:absolute;right:14px;top:11px;font-size:12px;color:#596473;opacity:0;transform:translate(-2px,2px);transition:opacity .16s ease,transform .16s ease}
.pick-card:hover::after,.pick-card:focus-visible::after{opacity:1;transform:none}
.pick-card:focus-visible{outline:none;border-color:#586474;box-shadow:0 0 0 2px rgba(229,138,184,.12),0 10px 28px rgba(0,0,0,.22)}
.pick-primary{grid-column:span 2;background:linear-gradient(180deg,#121820,#0f1319);border-color:#2d3946}
.pick-primary .pick-symbol{font-size:20px;letter-spacing:-.02em}
.pick-primary .pick-chart{height:210px}
.pick-primary .pick-stat .v{font-size:13px}
.pick-card.pick-long{--signal-accent:var(--green)}
.pick-card.pick-short{--signal-accent:var(--red)}
.pick-card .pick-head{position:relative}
.pick-card .pick-signal{box-shadow:inset 0 0 0 1px rgba(255,255,255,.08)}
.scan-progress{height:3px;max-width:420px;margin:18px auto 0;background:rgba(255,255,255,.06);border-radius:999px;overflow:hidden}.scan-progress span{display:block;height:100%;background:linear-gradient(90deg,var(--pink),var(--cyan));box-shadow:0 0 12px rgba(255,120,200,.35);transition:width .24s ease}.top5-empty{padding:48px 24px;text-align:center;border:1px dashed rgba(255,120,200,.18);border-radius:18px;background:rgba(255,255,255,.015)}
.legacy-hidden{display:none!important}@media(max-width:980px){.top5-grid{grid-template-columns:1fr}.pick-stats{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:640px){.wrap{padding:16px 12px 40px}.focus-only .top5-title{font-size:23px}.pick-chart{height:190px}}@media(prefers-reduced-motion:reduce){.pick-card{transition:none}}
</style>
<style>
/* Product-clean pass */
:root{--bg:#090b0f;--panel:#10141a;--panel2:#141922;--line:#232a35;--text:#edf1f5;--muted:#7f8997;--green:#55d79a;--red:#f2768e;--amber:#e7bc62;--blue:#79b5f0;--pink:#e58ab8;--pink2:#aa73c4;--cyan:#80c9dd;--shadow:0 16px 42px rgba(0,0,0,.22)}
body{background:linear-gradient(180deg,#0a0c10 0%,#090b0f 100%)}
body::before{display:none}
.wrap{max-width:1380px;padding:28px 28px 48px}
.top{align-items:center;margin-bottom:22px}
.eyebrow{letter-spacing:.14em;color:#77818e}
h1{font-size:32px;letter-spacing:-.025em;margin:5px 0 7px;font-weight:760}
.sub{max-width:820px;font-size:13px;color:#858f9c}
.actions button{background:#121820;border-color:#28303b;border-radius:9px;box-shadow:none;padding:8px 12px}
.actions button:hover{border-color:#3c4654;box-shadow:0 4px 16px rgba(0,0,0,.18);transform:none}
.actions button::after{display:none}
.focus-only .top5-head{margin:4px 0 14px;padding-bottom:13px;border-bottom:1px solid var(--line)}
.focus-only .top5-title{font-size:22px;font-weight:700;letter-spacing:-.015em}
.focus-only .top5-sub{margin-top:4px;max-width:720px;font-size:12px;color:#737e8b}
.top5-grid{grid-template-columns:repeat(3,minmax(280px,1fr));gap:12px}
.pick-card{border:1px solid #222a34;border-radius:12px;background:#10141a;box-shadow:none;transition:border-color .16s ease,background .16s ease}
.pick-card::before{display:none}
.pick-card:hover{transform:none;border-color:#34404e;box-shadow:0 8px 26px rgba(0,0,0,.18)}
.pick-head{padding:14px 14px 10px}
.pick-rank{font-size:10px;letter-spacing:.12em;color:#6f7986}
.pick-symbol{font-size:17px;font-weight:720}
.pick-signal{font-size:11px;padding:5px 8px;border-radius:7px;box-shadow:none}
.pick-stats{grid-template-columns:repeat(4,1fr);gap:5px;padding:0 14px 10px}
.pick-stat{padding:7px 8px;border-radius:7px;background:#0d1116}
.pick-stat .k{font-size:8px;letter-spacing:.08em;color:#697481}
.pick-stat .v{margin-top:3px;font-size:12px;font-weight:670}
.pick-chart{height:165px;padding:0 4px 4px}
.pick-foot{padding:9px 14px 12px;border-top:1px solid #1b222c;font-size:10px;color:#697481}
.history-badge{padding:3px 6px;border-radius:5px;background:transparent;border:1px solid #252d38;color:#7e8996}
.top5-empty{padding:56px 20px;border:1px dashed #2a313b;border-radius:12px;background:#0d1116}
.top5-empty h2{font-size:18px;font-weight:680}
#ambient-canvas{opacity:.10}
#pointer-aura{opacity:.38}
#anime-cursor{filter:drop-shadow(0 0 6px rgba(229,138,184,.28));width:40px;height:40px}
.click-ripple{border-color:rgba(229,138,184,.38);box-shadow:none}
@media(max-width:1050px){.top5-grid{grid-template-columns:repeat(2,minmax(280px,1fr))}.pick-primary{grid-column:span 2}}
@media(max-width:700px){.wrap{padding:20px 14px 34px}.top5-grid{grid-template-columns:1fr}.pick-primary{grid-column:span 1}.pick-chart,.pick-primary .pick-chart{height:150px}.focus-head-meta{justify-content:flex-start}.focus-coverage{white-space:normal}}
<style>
.focus-head-meta{display:flex;align-items:center;gap:9px;flex-wrap:wrap;justify-content:flex-end}
.focus-coverage{font-size:10px;color:#66717f;font-variant-numeric:tabular-nums;white-space:nowrap}
</style>
</style>

<style>
/* UI polish: restrained Sakura identity + responsive safety */
.pick-card:hover{box-shadow:0 0 0 1px rgba(229,138,184,.10),0 0 34px rgba(229,138,184,.16),0 16px 40px rgba(0,0,0,.22)}.pick-card::after{content:"";position:absolute;inset:-1px;border-radius:inherit;pointer-events:none;background:radial-gradient(280px 90px at 50% 0,rgba(229,138,184,.10),transparent 72%);opacity:.55;animation:cardGlow 4s ease-in-out infinite}.pick-card.pick-long::after{background:radial-gradient(280px 90px at 50% 0,rgba(85,215,154,.11),transparent 72%)}.pick-card.pick-short::after{background:radial-gradient(280px 90px at 50% 0,rgba(242,118,142,.11),transparent 72%)}@keyframes cardGlow{0%,100%{opacity:.28;filter:blur(0)}50%{opacity:.72;filter:blur(2px)}}.actions button:hover,.asset-picker button:hover{box-shadow:0 0 0 1px rgba(229,138,184,.12),0 0 24px rgba(229,138,184,.18),0 8px 24px rgba(0,0,0,.2)}
.asset-picker{display:flex;align-items:center;gap:6px;min-width:220px;max-width:330px}.asset-picker input{width:100%;min-width:0;height:38px;padding:0 11px;border:1px solid var(--line);border-radius:10px;background:#0e1218;color:var(--text);font:inherit;outline:none}.asset-picker input:focus{border-color:rgba(229,138,184,.55);box-shadow:0 0 0 3px rgba(229,138,184,.08),0 0 22px rgba(229,138,184,.10)}.asset-picker button{height:38px;padding-inline:12px;flex:0 0 auto}@media(max-width:760px){.asset-picker{width:100%;max-width:none;order:2}.chart-tools{width:100%}}
:root{
  --ui-radius:14px;
  --ui-radius-sm:10px;
  --ui-border:rgba(255,255,255,.075);
  --ui-border-strong:rgba(255,138,194,.24);
}
html{background:#090b0f}
body{overflow-x:hidden;-webkit-font-smoothing:antialiased;text-rendering:optimizeLegibility}
.wrap{position:relative;z-index:2}
.top{gap:18px}
.top h1{max-width:820px}
.actions{min-width:0}
.actions button,.actions select{min-height:38px}
button:focus-visible,select:focus-visible,.nav-btn:focus-visible,.pick-card:focus-visible{
  outline:2px solid rgba(229,138,184,.72);
  outline-offset:2px;
}
.metrics{gap:10px}
.metric,.card,.panel{
  border-color:var(--ui-border);
  border-radius:var(--ui-radius);
}
.metric{
  transition:transform .18s ease,border-color .18s ease,background .18s ease;
}
.metric:hover{
  transform:translateY(-2px);
  border-color:rgba(229,138,184,.18);
}
.nav{
  position:sticky;
  top:10px;
  z-index:20;
  padding:5px;
  border:1px solid rgba(255,255,255,.055);
  border-radius:13px;
  background:rgba(9,11,15,.78);
  backdrop-filter:blur(14px);
  -webkit-backdrop-filter:blur(14px);
}
.nav-btn{border-radius:9px}
.nav-btn.active{box-shadow:inset 0 -2px 0 rgba(229,138,184,.8),0 4px 18px rgba(229,138,184,.08)}
.top5-grid{align-items:stretch}
.pick-card{
  min-width:0;
  display:flex;
  flex-direction:column;
  contain:layout paint;
}
.pick-card .pick-chart{margin-top:auto}
.pick-head,.pick-stats,.pick-foot{min-width:0}
.pick-symbol{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pick-stats{gap:8px}
.pick-stat{min-width:0}
.pick-stat .v{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.chart-panel{overflow:hidden}
.chart-wrap{min-width:0;overflow:hidden}
#chart{max-width:100%;image-rendering:auto}
.chart-tools{min-width:0;flex-wrap:wrap}
.chart-tools select,.chart-tools button{flex:0 0 auto}
.table-wrap{border-radius:var(--ui-radius-sm);-webkit-overflow-scrolling:touch;scrollbar-width:thin}
.table-wrap table{min-width:720px}
.panel-head{gap:12px;min-width:0}
.panel-head>div{min-width:0}
.small{line-height:1.5}
.pill{white-space:nowrap}
.top5-empty{min-height:190px;display:flex;flex-direction:column;align-items:center;justify-content:center}
#pointer-aura{mix-blend-mode:screen}
#anime-cursor{transform-origin:8px 8px}
@media(max-width:1180px){
  .wrap{padding-left:20px;padding-right:20px}
  .top5-grid{grid-template-columns:repeat(2,minmax(0,1fr))}
}
@media(max-width:760px){
  .wrap{padding:15px 12px 34px}
  .top{gap:12px}
  .actions{width:100%;justify-content:space-between}
  .actions button,.actions select{flex:1 1 auto}
  .nav{top:6px;margin-inline:-2px;overflow-x:auto;scrollbar-width:none}
  .nav::-webkit-scrollbar{display:none}
  .nav-btn{flex:0 0 auto}
  .metrics{gap:8px}
  .metric{padding:11px}
  .metric .value{font-size:17px}
  .decision-deck{gap:10px}
  .trace-grid{grid-template-columns:1fr 1fr}
  .top5-grid{grid-template-columns:1fr;gap:10px}
  .pick-head{padding:14px 14px 9px}
  .pick-foot{padding:9px 14px 12px}
  .pick-chart{height:190px}
  .chart-panel{min-height:420px}
  #chart{height:350px}
}
@media(max-width:460px){
  .actions{display:grid;grid-template-columns:1fr 1fr}
  .actions .stamp{grid-column:1/-1;order:-1}
  .trace-grid{grid-template-columns:1fr}
  .pick-stats{grid-template-columns:1fr 1fr}
  .focus-only .top5-title{font-size:22px}
  .chart-panel{min-height:390px}
  #chart{height:315px}
}
@media(prefers-reduced-motion:reduce){
  .metric,.pick-card{transition:none!important}
  .nav{scroll-behavior:auto}
}
</style>

</head>
<body>
<canvas id="ambient-canvas" aria-hidden="true"></canvas>
<div id="pointer-aura" aria-hidden="true"></div>
<div id="anime-cursor" aria-hidden="true">
  <svg viewBox="0 0 64 64" fill="none">
    <path d="M8 53L55 12" stroke="#ff78c8" stroke-width="2.4" stroke-linecap="round"/>
    <path d="M46 13l6 6" stroke="#7de8ff" stroke-width="1.8" stroke-linecap="round"/>
    <path d="M18 31c1-12 8-20 18-20 11 0 19 8 20 20-4-4-7-5-11-5-2 0-4 1-6 2-3-4-7-6-12-6-3 0-6 3-9 9Z" fill="#ff78c8"/>
    <circle cx="32" cy="35" r="14.5" fill="#f8d5cc"/>
    <path d="M20 34c2-5 4-7 8-7 3 0 5 2 7 4 3-3 5-4 8-4 4 0 7 2 10 7" fill="#ff9fda"/>
    <circle cx="27" cy="36" r="2.2" fill="#342536"/><circle cx="37" cy="36" r="2.2" fill="#342536"/>
    <path d="M29 42c2 2 4 2 6 0" stroke="#9f4f7f" stroke-width="1.3" stroke-linecap="round"/>
    <path d="M24 49c3 3 8 4 12 0l4 7H20l4-7Z" fill="#22202f"/>
    <path d="M21 50c-3 1-6 4-7 7" stroke="#ff78c8" stroke-width="2" stroke-linecap="round"/>
    <path d="M43 50c3 1 6 4 7 7" stroke="#ff78c8" stroke-width="2" stroke-linecap="round"/>
  </svg>
</div>
<div class="wrap">
  <div class="top">
    <div>
      <div class="eyebrow">Signal monitor · 15m</div>
      <h1>Signal Monitor</h1>
      <div class="sub">Closed-candle signals, live market quote e storico delle evidenze. Solo segnali che superano i controlli correnti.</div>
    </div>
    <div class="actions">
      <div class="operator-badge" aria-hidden="true" style="display:none">
        <div class="operator-art">
          <svg viewBox="0 0 64 64" fill="none">
            <path d="M8 52L54 13" stroke="#ff78c8" stroke-width="2.1" stroke-linecap="round"/>
            <path d="M47 15l5 5" stroke="#7de8ff" stroke-width="1.6" stroke-linecap="round"/>
            <circle cx="32" cy="34" r="16" fill="#f8d5cc"/>
            <path d="M16 34c0-13 7-23 18-23 10 0 16 7 18 17l-7-4-4 8-5-9-6 8-7-5-7 8v0Z" fill="#ff78c8"/>
            <path d="M19 42c2 7 7 11 13 11 7 0 12-4 14-11-4 2-9 3-14 3s-10-1-13-3Z" fill="#f7a8d7"/>
            <circle cx="26" cy="36" r="2" fill="#342536"/><circle cx="38" cy="36" r="2" fill="#342536"/>
            <path d="M29 43c2 2 4 2 6 0" stroke="#ad5b89" stroke-width="1.3" stroke-linecap="round"/>
            <path d="M17 28c3-7 9-11 16-11 9 0 15 5 18 13" stroke="#fff" stroke-opacity=".28" stroke-width="1.1" stroke-linecap="round"/>
          </svg>
        </div>
        <div class="operator-copy"><strong>SAKURA</strong><span>WATCHER // ONLINE</span></div>
      </div>
      <span id="stamp">Connessione…</span>
      <button id="refresh">Refresh</button>
    </div>
  </div>

  <nav class="nav legacy-hidden" id="nav" aria-label="Sezioni dashboard">
    <button class="nav-btn active" data-target="market">Market</button>
    <button class="nav-btn" data-target="detail">Intelligence</button>
    <button class="nav-btn" data-target="journal">Journal</button>
    <button class="nav-btn" data-target="timeline">Timeline</button>
    <button class="nav-btn" data-target="evidencePanel">Evidence</button>
  </nav>

  <div class="metrics legacy-hidden" id="metrics"></div>

  <section class="focus-only" id="focusDashboard" aria-live="polite">
    <div class="top5-head">
      <div><div class="eyebrow">Current signals</div><div class="top5-title">Top 5</div><div class="top5-sub">Solo LONG e SHORT attivi. I dati restano separati dalla decisione del modello.</div></div>
      <div class="focus-head-meta"><span id="focusCoverage" class="focus-coverage">Universe —</span><span id="focusStatus" class="pill warn">Scanning…</span></div>
    </div>
    <div id="top5Grid" class="top5-grid"></div>
  </section>

  <section class="legacy-hidden decision-deck" id="decisionDeck" aria-live="polite">
    <div class="decision-hero">
      <div class="decision-top">
        <div>
          <div class="eyebrow">Decision deck</div>
          <div id="deckSignal" class="decision-signal signal-wait">WAIT</div>
          <div id="deckMeta" class="decision-meta">Seleziona un asset per ispezionare il verdetto.</div>
        </div>
        <span id="deckBundle" class="pill warn"><span class="dot"></span>WAIT</span>
      </div>
      <div class="decision-stats">
        <div class="mini-stat"><div class="k">Prezzo</div><div class="v num" id="deckPrice">—</div></div>
        <div class="mini-stat"><div class="k">Confidence</div><div class="v" id="deckConfidence">—</div></div>
        <div class="mini-stat"><div class="k">Robust edge</div><div class="v" id="deckEdge">—</div></div>
      </div>
      <div class="reason-strip" id="deckReasons"></div>

      <div class="operator-stage" aria-hidden="true">
        <div class="operator-scan"></div>
        <svg viewBox="0 0 140 140" fill="none">
          <defs>
            <linearGradient id="opHair" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stop-color="#ffb0e0"/><stop offset=".55" stop-color="#ff78c8"/><stop offset="1" stop-color="#c85cff"/>
            </linearGradient>
            <linearGradient id="opSuit" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0" stop-color="#34283e"/><stop offset="1" stop-color="#17131e"/>
            </linearGradient>
          </defs>
          <circle cx="73" cy="67" r="45" fill="rgba(255,120,200,.035)" stroke="rgba(255,120,200,.18)" stroke-dasharray="2 6"/>
          <path d="M37 70c0-28 13-44 34-44 23 0 35 18 38 42-6-7-13-11-21-11-6 0-12 2-18 7-6-7-13-10-20-9-4 0-8 5-13 15Z" fill="url(#opHair)"/>
          <path d="M48 61c4-11 12-17 24-17 12 0 20 7 25 18l-4 24c-5 11-12 17-22 17-11 0-19-6-24-17l1-25Z" fill="#f7d2cb"/>
          <path d="M53 61c5-8 11-12 18-12 9 0 16 4 22 12-5-3-10-4-15-4-7 0-14 2-21 7l-4-3Z" fill="#ff8fd1"/>
          <ellipse cx="63" cy="72" rx="3.2" ry="4.2" fill="#37253a"/><ellipse cx="84" cy="72" rx="3.2" ry="4.2" fill="#37253a"/>
          <circle cx="64" cy="71" r="1.1" fill="#fff"/><circle cx="85" cy="71" r="1.1" fill="#fff"/>
          <path d="M69 83c3 2 7 2 10 0" stroke="#a95380" stroke-width="1.6" stroke-linecap="round"/>
          <path d="M45 90c7 10 18 14 30 14 11 0 22-5 30-14 4 3 8 9 11 17l8 23H32l7-23c2-7 4-13 6-17Z" fill="url(#opSuit)" stroke="rgba(255,120,200,.22)"/>
          <path d="M69 104l4 8 5-8 7 13-12 13-11-13 7-13Z" fill="#ff78c8" fill-opacity=".22"/>
          <path d="M31 126h78" stroke="#7de8ff" stroke-opacity=".35" stroke-width="1"/>
          <path d="M111 39l12 8-8 3 7 9-15-4 4-8-8-4 8-4Z" fill="#7de8ff" fill-opacity=".16" stroke="#7de8ff" stroke-opacity=".4"/>
        </svg>
        <span class="operator-caption">OPERATOR // WATCHING</span>
      </div>    </div>
    <div class="trace">
      <div class="trace-head"><div><div class="title">Decision trace</div><div class="small">Ogni blocco è una condizione osservabile del gate.</div></div><div class="small" id="deckAsset">—</div></div>
      <div class="trace-grid" id="traceGrid"></div>
    </div>
  </section>

  <div class="layout">
    <div class="legacy-hidden panel chart-panel" id="market">
      <div class="panel-head">
        <div>
          <div class="title">Market cockpit</div>
          <div class="small">Candele storiche + marker segnali, con ticker realtime separato dal close usato dal modello.</div>
        </div>
        <div class="chart-tools">
          <span id="livePrice" class="pill good">REALTIME —</span>
          <span id="historySource" class="pill">HISTORY —</span>
          <div class="asset-picker"><input id="asset" list="marketSymbols" inputmode="search" autocomplete="off" spellcheck="false" placeholder="Cerca simbolo..." aria-label="Cerca un simbolo di mercato"/><datalist id="marketSymbols"></datalist><button id="loadAsset" type="button">Apri</button></div>
          <select id="range" aria-label="Numero di candele"><option value="120">120</option><option value="240" selected>240</option><option value="480">480</option></select>
        </div>
      </div>
      <div class="chart-wrap" id="chartWrap">
        <canvas id="chart"></canvas>
        <div id="chartRegime" class="chart-regime">REGIME —</div>
        <div id="chartCrosshair" class="chart-crosshair" aria-hidden="true">
          <span class="cx" id="crossX"></span><span class="cy" id="crossY"></span>
          <span class="badge" id="crossBadge">—</span>
        </div>
        <div id="cursor"></div>
        <div id="chartEmpty" class="chart-empty">Caricamento storico…</div>
      </div>
      <div class="legend">
        <span><i style="background:var(--green)"></i>LONG</span>
        <span><i style="background:var(--red)"></i>SHORT</span>
        <span><i style="background:var(--blue)"></i>Prezzo</span>
        <span><i style="background:var(--amber)"></i>Realtime</span>
      </div>
    </div>

    <div class="panel">
      <div class="panel-head"><div><div class="title">Radar</div><div class="small">Segnale principale per asset</div></div></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Asset</th><th>Segnale</th><th>Conf.</th></tr></thead>
          <tbody id="radarRows"></tbody>
        </table>
      </div>
    </div>
  </div>


  <div class="legacy-hidden panel" id="timeline">
    <div class="panel-head">
      <div>
        <div class="title">Signal timeline</div>
        <div class="small">Prediction → observation → maturity → outcome. La timeline mostra cosa il modello ha dichiarato e cosa è già stato osservato.</div>
      </div>
      <div class="small" id="timelineStatus">—</div>
    </div>
    <div class="timeline-shell">
      <div id="signalTimeline" class="signal-timeline"></div>
    </div>
  </div>

  <div class="legacy-hidden panel" id="detail">
    <div class="panel-head">
      <div><div class="title">Signal detail</div><div class="small">Stessa decisione usata dal motore: robust edge, meta, memoria, regime e duration gate.</div></div>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>Asset</th><th>Segnale</th><th>Prezzo</th><th>Robust edge</th><th>Score</th><th>Meta</th><th>Memoria</th><th>Regime</th><th>Duration</th><th>Dati</th><th>Provenienza</th></tr>
        </thead>
        <tbody id="detailRows"></tbody>
      </table>
    </div>
  </div>

  <div class="legacy-hidden panel" id="journal">
    <div class="panel-head">
      <div><div class="title">Signal journal</div><div class="small">Ogni segnale resta osservabile fino alla maturazione del suo orizzonte; gli esiti sono risolti con candele successive.</div></div>
      <div class="small" id="journalStatus"></div>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Ora</th><th>Asset</th><th>Segnale</th><th>Conf.</th><th>Atteso</th><th>Esito</th><th>Realizzato</th><th>Holding</th></tr></thead>
        <tbody id="journalRows"></tbody>
      </table>
    </div>
  </div>

  <div class="legacy-hidden panel" id="evidencePanel">
    <div class="panel-head"><div><div class="title">Training / deployment evidence</div><div class="small">Un segnale è ammissibile solo con bundle compatibile e provenance coerente.</div></div></div>
    <div class="grid3" id="evidence"></div>
  </div>


<div id="inspectorBackdrop" class="legacy-hidden inspector-backdrop"></div>
<aside id="inspectorDrawer" class="legacy-hidden inspector-drawer" aria-label="Decision inspector" aria-hidden="true">
  <div class="inspector-head">
    <div><div class="title">Decision inspector</div><div class="small" id="inspectorSubtitle">Asset —</div></div>
    <button class="inspector-close" id="inspectorClose" aria-label="Chiudi inspector">×</button>
  </div>
  <div class="inspector-scroll" id="inspectorContent">
    <div class="timeline-empty">Seleziona un asset da Radar o Intelligence.</div>
  </div>
</aside>

  <div class="footer" id="footer"></div>
</div>

<script>
const $ = (id) => document.getElementById(id);
const state = { data:null, history:null, selected:null, historyRequest:0, focusRequest:0, focusHistories:{} };

function initAmbientFX(){
  const canvas=$("ambient-canvas");
  if(!canvas || (window.matchMedia&&window.matchMedia("(prefers-reduced-motion: reduce)").matches))return;
  const ctx=canvas.getContext("2d");
  const particles=[];
  const count=Math.min(30,Math.max(14,Math.round(window.innerWidth/55)));
  const resize=()=>{
    const dpr=Math.min(2,window.devicePixelRatio||1);
    canvas.width=Math.floor(window.innerWidth*dpr);
    canvas.height=Math.floor(window.innerHeight*dpr);
    canvas.style.width=window.innerWidth+"px";
    canvas.style.height=window.innerHeight+"px";
    ctx.setTransform(dpr,0,0,dpr,0,0);
  };
  resize();window.addEventListener("resize",resize,{passive:true});
  for(let i=0;i<count;i++){
    particles.push({
      x:Math.random()*window.innerWidth,y:Math.random()*window.innerHeight,
      r:1.2+Math.random()*2.3,vx:-.03+Math.random()*.06,vy:.10+Math.random()*.18,
      a:.10+Math.random()*.18,rot:Math.random()*6.28,vr:-.004+Math.random()*.008
    });
  }
  let last=performance.now(),raf=0;
  const frame=now=>{
    const dt=Math.min(32,now-last);last=now;
    ctx.clearRect(0,0,window.innerWidth,window.innerHeight);
    particles.forEach(p=>{
      p.x+=p.vx*dt;p.y+=p.vy*dt;p.rot+=p.vr*dt;
      if(p.y>window.innerHeight+20){p.y=-20;p.x=Math.random()*window.innerWidth}
      if(p.x<-20)p.x=window.innerWidth+20;if(p.x>window.innerWidth+20)p.x=-20;
      ctx.save();ctx.translate(p.x,p.y);ctx.rotate(p.rot);
      ctx.fillStyle="rgba(255,120,200,"+p.a.toFixed(3)+")";
      ctx.beginPath();
      ctx.moveTo(0,-p.r*1.8);ctx.quadraticCurveTo(p.r*1.9,-p.r*.2,0,p.r*1.8);
      ctx.quadraticCurveTo(-p.r*1.9,-p.r*.2,0,-p.r*1.8);ctx.fill();
      ctx.restore();
    });
    raf=requestAnimationFrame(frame);
  };
  raf=requestAnimationFrame(frame);
  window.addEventListener("pagehide",()=>cancelAnimationFrame(raf),{once:true});
}

function initAlphaMotion(){
  const fine=window.matchMedia&&window.matchMedia("(pointer:fine)").matches;
  const reduced=window.matchMedia&&window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if(!fine || reduced)return;
  document.body.classList.add("alpha-pointer");
  const cursor=$("anime-cursor"), aura=$("pointer-aura");
  let px=-100,py=-100,tx=px,ty=py,pending=false;
  const tick=()=>{
    pending=false;
    px+=(tx-px)*.34; py+=(ty-py)*.34;
    cursor.style.opacity="1"; aura.style.opacity="1";
    cursor.style.transform=`translate3d(${px}px,${py}px,0)`;
    aura.style.transform=`translate3d(${px}px,${py}px,0)`;
    if(Math.abs(tx-px)>.2||Math.abs(ty-py)>.2)requestAnimationFrame(tick);
  };
  document.addEventListener("pointermove",e=>{
    tx=e.clientX;ty=e.clientY;
    if(!pending){pending=true;requestAnimationFrame(tick)}
  },{passive:true});
  document.addEventListener("pointerleave",()=>{cursor.style.opacity="0";aura.style.opacity="0"});
  document.addEventListener("pointerenter",e=>{tx=e.clientX;ty=e.clientY;cursor.style.opacity="1";aura.style.opacity="1"});
  document.addEventListener("click",e=>{
    cursor.classList.remove("click");void cursor.offsetWidth;cursor.classList.add("click");
    const ripple=document.createElement("span");
    ripple.className="click-ripple";
    ripple.style.left=e.clientX+"px";ripple.style.top=e.clientY+"px";
    document.body.appendChild(ripple);setTimeout(()=>ripple.remove(),560);
  },{passive:true});
}

function initNavigation(){
  const buttons=Array.from(document.querySelectorAll(".nav-btn"));
  const ids=buttons.map(b=>b.dataset.target).filter(Boolean);
  const setActive=id=>buttons.forEach(b=>b.classList.toggle("active",b.dataset.target===id));
  buttons.forEach(b=>b.addEventListener("click",()=>{
    const el=$(b.dataset.target);if(!el)return;
    el.scrollIntoView({behavior:"smooth",block:"start"});setActive(b.dataset.target);
  }));
  if("IntersectionObserver" in window){
    const observer=new IntersectionObserver(entries=>{
      const visible=entries.filter(x=>x.isIntersecting).sort((a,b)=>b.intersectionRatio-a.intersectionRatio)[0];
      if(visible)setActive(visible.target.id);
    },{rootMargin:"-18% 0px -65% 0px",threshold:[.1,.35,.7]});
    ids.forEach(id=>{const el=$(id);if(el)observer.observe(el)});
  }
}

function esc(v){return String(v??"").replace(/[&<>"]/g,c=>c==="&"?"&amp;":c==="<"?"&lt;":c===">"?"&gt;":"&quot;");}
function pct(v,d=1){return v==null||Number.isNaN(Number(v))?"—":(Number(v)*100).toFixed(d)+"%";}
function num(v,d=3){return v==null||Number.isNaN(Number(v))?"—":Number(v).toFixed(d);}
function age(v){return v==null||Number.isNaN(Number(v))?"—":Number(v).toFixed(1)+"m";}
function cls(sig){return sig==="LONG"?"signal-long":sig==="SHORT"?"signal-short":sig==="WAIT"?"signal-wait":"signal-flat";}
function pill(ok,label){return '<span class="pill '+(ok?'good':'warn')+'"><span class="dot"></span>'+esc(label)+'</span>';}

function drawPickChart(canvas,history,signal){
  const bars=(history&&history.bars)||[];
  if(!canvas||bars.length<2)return;
  const rect=canvas.getBoundingClientRect(), dpr=Math.max(1,window.devicePixelRatio||1);
  canvas.width=Math.max(1,Math.floor(rect.width*dpr)); canvas.height=Math.max(1,Math.floor(rect.height*dpr));
  const ctx=canvas.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
  const W=rect.width,H=rect.height,pad={l:8,r:8,t:14,b:20},cw=W-pad.l-pad.r,ch=H-pad.t-pad.b;
  let lo=Math.min(...bars.map(b=>Number(b.l))),hi=Math.max(...bars.map(b=>Number(b.h)));
  const span=Math.max(hi-lo,1e-9);lo-=span*.06;hi+=span*.06;
  const xAt=i=>pad.l+(i/(bars.length-1))*cw,yAt=v=>pad.t+(1-(v-lo)/(hi-lo))*ch;
  ctx.clearRect(0,0,W,H);ctx.lineWidth=1;ctx.strokeStyle="rgba(255,255,255,.055)";
  for(let i=0;i<4;i++){const y=pad.t+(i/3)*ch;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();}
  const candleW=Math.max(1,cw/bars.length*.62);
  bars.forEach((b,i)=>{
    const x=xAt(i),yo=yAt(b.o),yc=yAt(b.c),yh=yAt(b.h),yl=yAt(b.l),up=b.c>=b.o;
    ctx.strokeStyle=up?"rgba(69,227,154,.52)":"rgba(255,111,136,.52)";
    ctx.fillStyle=up?"rgba(69,227,154,.34)":"rgba(255,111,136,.34)";
    ctx.beginPath();ctx.moveTo(x,yh);ctx.lineTo(x,yl);ctx.stroke();
    ctx.fillRect(x-candleW/2,Math.min(yo,yc),candleW,Math.max(1,Math.abs(yc-yo)));
  });
  ctx.beginPath();bars.forEach((b,i)=>{const x=xAt(i),y=yAt(b.c);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});
  ctx.lineTo(xAt(bars.length-1),pad.t+ch);ctx.lineTo(xAt(0),pad.t+ch);ctx.closePath();
  const grad=ctx.createLinearGradient(0,pad.t,0,pad.t+ch);grad.addColorStop(0,"rgba(120,184,255,.10)");grad.addColorStop(1,"rgba(120,184,255,0)");ctx.fillStyle=grad;ctx.fill();
  ctx.beginPath();bars.forEach((b,i)=>{const x=xAt(i),y=yAt(b.c);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});
  ctx.strokeStyle=signal==="LONG"?"#45e39a":"#ff6f88";ctx.lineWidth=1.8;ctx.stroke();
  ctx.fillStyle="rgba(148,138,164,.65)";ctx.font="10px system-ui";ctx.textAlign="left";ctx.fillText(new Date(bars[0].t).toLocaleDateString(),pad.l,H-5);
  ctx.textAlign="right";ctx.fillText(new Date(bars[bars.length-1].t).toLocaleDateString(),W-pad.r,H-5);
}

function renderFocus(data){
  state.data=data;
  const box=$("top5Grid"),status=$("focusStatus");
  if(!box)return;
  if(data.scan_in_progress){
    const summary=data.summary||{};
    const coverage=$("focusCoverage");
    if(coverage){
      const total=Number(summary.universe_total??0), backed=Number(summary.model_backed_assets??0), eligible=Number(summary.model_eligible_assets??backed), evaluated=Number(summary.assets_scanned??0), waits=Number(summary.waits??0);
      coverage.textContent=total ? "Universe "+total+" · models "+backed+" · eligible "+eligible+" · evaluated "+evaluated+" · wait "+waits : "Discovering markets…";
    }
    box.innerHTML='<div class="top5-empty"><div class="eyebrow">Scanning</div><h2 style="margin:8px 0 6px">Analisi dell\'universo in corso</h2><div class="small">Il motore sta valutando i mercati con bundle verificati. La superficie si aggiorna appena il ranking è pronto.</div></div>';
    const p=data.scan_progress||{};
    const done=Number(p.evaluated??0), total=Number(p.total??summary.assets_scanned??0), signals=Number(p.signals??0);
    const pctDone=total>0?Math.round(done/total*100):0;
    status.textContent=(total?done+"/"+total+" · ":"")+"scanning";status.className="pill warn";
    if(done>0){
      box.innerHTML='<div class="top5-empty"><div class="eyebrow">Scanning · '+pctDone+'%</div><h2 style="margin:8px 0 6px">Analisi dell\'universo in corso</h2><div class="small">Valutati '+done+' / '+total+' mercati · '+signals+' segnali candidati · '+Number(p.waits??0)+' WAIT.</div><div class="scan-progress"><span style="width:'+pctDone+'%"></span></div></div>';
    }
    return;
  }
  if(!data.ok){
    box.innerHTML='<div class="top5-empty"><div class="eyebrow">Offline</div><h2 style="margin:8px 0 6px">Market data unavailable</h2><div class="small">'+esc(data.error||"Il terminale locale non è disponibile.")+'</div></div>';
    const coverage=$("focusCoverage"); if(coverage) coverage.textContent="Universe unavailable";
    status.textContent="offline";status.className="pill warn";return;
  }
  const picks=(data.signals||[]).filter(x=>x.signal==="LONG"||x.signal==="SHORT").slice(0,5);
  if(!picks.length){
    box.innerHTML='<div class="top5-empty"><div class="eyebrow">No active signals</div><h2 style="margin:8px 0 6px">Nessun segnale</h2><div class="small">Il gate corrente non trova un LONG o SHORT abbastanza solido da mostrare.</div></div>';
    const summary=data.summary||{};
    const coverage=$("focusCoverage");
    if(coverage) coverage.textContent="Universe "+(summary.universe_total??0)+" · eligible "+(summary.model_eligible_assets??summary.model_backed_assets??0)+" · evaluated "+(summary.assets_scanned??0)+" · wait "+(summary.waits??0);
    status.textContent="0 signals";status.className="pill warn";return;
  }
  const summary=data.summary||{};
  const coverage=$("focusCoverage");
  if(coverage){
    const total=Number(summary.universe_total??0), eligible=Number(summary.model_eligible_assets??summary.model_backed_assets??0), evaluated=Number(summary.assets_scanned??0);
    coverage.textContent="Universe "+total+" · eligible "+eligible+" · evaluated "+evaluated;
  }
  status.textContent=picks.length+" signal"+(picks.length===1?"":"s");status.className="pill good";
  box.innerHTML=picks.map((r,i)=>{
    const d=r.decision||{};
    return '<article class="pick-card pick-'+(r.signal==="LONG"?"long":"short")+(i===0?' pick-primary':'')+'" data-focus-symbol="'+esc(r.symbol)+'" tabindex="0" role="button" aria-label="Apri '+esc(r.symbol)+' nel market inspector">'+
      '<div class="pick-head"><div><div class="pick-rank">#'+(i+1)+'</div><div class="pick-symbol">'+esc(r.symbol)+'</div></div><span class="pick-signal '+cls(r.signal)+'">'+esc(r.signal)+'</span></div>'+
      '<div class="pick-stats">'+
      '<div class="pick-stat"><div class="k">Price</div><div class="v">'+num(r.realtime_price??r.price,2)+'</div></div>'+
      '<div class="pick-stat"><div class="k">Confidence</div><div class="v">'+pct(r.confidence,1)+'</div></div>'+
      '<div class="pick-stat"><div class="k">Edge</div><div class="v">'+pct(d.robust_directional_edge,2)+'</div></div>'+
      '<div class="pick-stat"><div class="k">Score</div><div class="v">'+num(d.score,2)+'</div></div>'+
      '</div>'+
      '<div class="pick-chart"><canvas data-pick-chart="'+esc(r.symbol)+'"></canvas></div>'+
      '<div class="pick-foot"><span>15m · closed candles</span><span class="history-badge" data-history-badge="'+esc(r.symbol)+'">240 bars</span></div>'+
      '</article>';
  }).join("");
  bindFocusCards();
  loadFocusHistories(picks);
}

function bindFocusCards(){
  document.querySelectorAll(".pick-card[data-focus-symbol]").forEach(card=>{
    const open=()=>{
      const symbol=card.dataset.focusSymbol;
      if(!symbol)return;
      state.selected=symbol;
      const input=$("asset");
      if(input)input.value=symbol;
      openInspector(symbol);
      renderDecisionDeck((state.data&&state.data.signals)||[]);
      const market=$("market");
      if(market)market.scrollIntoView({behavior:"smooth",block:"start"});
      loadHistory(symbol);
    };
    card.onclick=open;
    card.onkeydown=e=>{
      if(e.key==="Enter"||e.key===" "){e.preventDefault();open();}
    };
  });
}

async function loadFocusHistories(picks){
  const request=++state.focusRequest;
  const results=await Promise.all(picks.map(async r=>{
    try{
      const res=await fetch("/api/history?symbol="+encodeURIComponent(r.symbol)+"&limit=240",{cache:"no-store"});
      return [r.symbol,await res.json()];
    }catch(e){
      return [r.symbol,{symbol:r.symbol,bars:[],source:"unavailable",error:String(e)}];
    }
  }));
  if(request!==state.focusRequest)return;
  state.focusHistories=Object.fromEntries(results);
  picks.forEach(r=>{
    const h=state.focusHistories[r.symbol]||{};
    const canvas=Array.from(document.querySelectorAll("[data-pick-chart]")).find(el=>el.dataset.pickChart===r.symbol);
    const badge=Array.from(document.querySelectorAll("[data-history-badge]")).find(el=>el.dataset.historyBadge===r.symbol);
    if(canvas)drawPickChart(canvas,h,r.signal);
    if(badge){
      const labels={network:"live",local_cache:"cache",bundled:"bundled",unavailable:"offline"};
      badge.textContent=labels[h.source]||"—";
    }
  });
}

function populateAssets(signals){
  const input=$("asset");
  const cfg=(state.data&&state.data.config)||{};
  const discovered=Array.isArray(cfg.market_symbols)?cfg.market_symbols:[];
  const active=signals.map(x=>x.symbol).filter(Boolean);
  state.marketSymbols=Array.from(new Set([...discovered,...active])).sort();
  const list=$("marketSymbols");
  if(list){
    list.innerHTML=state.marketSymbols.map(s=>'<option value="'+esc(s)+'"></option>').join("");
    list.setAttribute("aria-label",state.marketSymbols.length+" active markets");
  }
  if(!state.selected||!state.marketSymbols.includes(state.selected)){
    state.selected=active[0]||cfg.primary_symbol||state.marketSymbols[0]||null;
  }
  if(input)input.value=state.selected||"";
}

function renderMetrics(data){
  const s=data.summary||{};
  const q=data.config||{};
  $("metrics").innerHTML=[
    ["Terminal",data.ok?"READY":"WAIT",data.ok?"good":"bad"],
    ["Segnali",s.active_signals??0,s.active_signals>0?"good":""],
    ["WAIT",s.waits??0,s.waits>0?"warn":""],
    ["Universe",s.universe_total??0,s.universe_total>0?"":"warn"],
    ["Eligible models",s.model_eligible_assets??s.model_backed_assets??0,s.model_eligible_assets>0?"":"warn"],
    ["Dati freschi",s.fresh_data_assets??0,s.fresh_data_assets>0?"good":"warn"]
  ].map(x=>'<div class="card"><div class="metric-label">'+x[0]+'</div><div class="metric-value '+x[2]+'">'+esc(x[1])+'</div></div>').join("");
  const source=q.scan_source==="exchange"?"market map live":q.scan_source==="local_fallback"?"local fallback":"—";
  const types=Object.entries(q.market_counts||{}).sort((a,b)=>String(a[0]).localeCompare(String(b[0]))).slice(0,4).map(([k,v])=>k+" "+v).join(" · ");
  $("stamp").textContent=(data.generated_at?new Date(data.generated_at).toLocaleTimeString():"—")+" · "+(q.exchange||"—")+" · "+source+(types?" · "+types:"");
}

function renderRadar(signals){
  $("radarRows").innerHTML=signals.map(r=>{
    const price=r.realtime_price??r.price;
    return '<tr class="interactive-row" data-symbol="'+esc(r.symbol)+'"><td><strong>'+esc(r.symbol)+'</strong><div class="small">'+num(price,2)+'</div></td>'+
      '<td><span class="signal '+cls(r.signal)+'">'+esc(r.signal||"WAIT")+'</span></td>'+
      '<td class="num">'+pct(r.confidence,1)+'</td></tr>';
  }).join("")||'<tr><td colspan="3" class="small">Nessun asset disponibile.</td></tr>';
}

function renderDetail(signals){
  $("detailRows").innerHTML=signals.map(r=>{
    const d=r.decision||{}, b=r.bundle||{};
    const robust=(d.expected_return_lcb==null)?"—":num(d.expected_return_lcb,4)+" / "+num(d.expected_return_ucb,4);
    const duration=d.trade_window_direction?esc(d.trade_window_direction)+" · "+pct(d.trade_window_confidence,0):"—";
    const why=(r.reason_codes||[]).map(x=>'<span class="reason">'+esc(x)+'</span>').join("");
    return '<tr class="interactive-row" data-symbol="'+esc(r.symbol)+'">'+
      '<td><strong>'+esc(r.symbol)+'</strong></td>'+
      '<td><span class="signal '+cls(r.signal)+'">'+esc(r.signal||"WAIT")+'</span><div class="small">'+esc(r.status||"WAIT")+'</div></td>'+
      '<td class="num">'+num(r.realtime_price??r.price,2)+'</td>'+
      '<td class="num">'+robust+'</td>'+
      '<td class="num">'+num(d.score,3)+'</td>'+
      '<td class="num">'+pct(d.meta_success,0)+'</td>'+
      '<td class="num">'+(d.analog_n==null?"—":esc(d.analog_n))+" · "+pct(d.analog_agreement,0)+'</td>'+
      '<td>'+esc(d.regime||"—")+'</td>'+
      '<td>'+duration+" "+pill(!!d.trade_window_ready,d.trade_window_ready?"READY":"WAIT")+'</td>'+
      '<td class="num">'+age(d.data_age_minutes)+'</td>'+
      '<td>'+pill(!!b.compatible&&!!b.manifest_ready, b.compatible&&b.manifest_ready?"PROVEN":"WAIT")+why+'</td>'+
    '</tr>';
  }).join("")||'<tr><td colspan="11" class="small">Nessun dato.</td></tr>';
}

function renderJournal(data){
  const rows=data.journal||[];
  $("journalRows").innerHTML=rows.length?rows.slice().reverse().map(r=>{
    const o=r.outcome||"OPEN";
    const oc=o==="WIN"?"good":o==="LOSS"?"bad":"warn";
    return '<tr>'+
      '<td class="small">'+esc(r.data_timestamp?new Date(r.data_timestamp).toLocaleString():"—")+'</td>'+
      '<td><strong>'+esc(r.symbol||"—")+'</strong></td>'+
      '<td><span class="signal '+cls(r.signal||"WAIT")+'">'+esc(r.signal||"WAIT")+'</span></td>'+
      '<td class="num">'+pct(r.confidence,1)+'</td>'+
      '<td class="num">'+pct(r.expected_return,2)+'</td>'+
      '<td class="'+oc+'"><strong>'+esc(o)+'</strong></td>'+
      '<td class="num">'+pct(r.realized_return,2)+'</td>'+
      '<td class="num">'+(r.holding_hours==null?"—":num(r.holding_hours,1)+"h")+'</td>'+
    '</tr>';
  }).join(""):'<tr><td colspan="8" class="small">Nessun segnale storico ancora registrato.</td></tr>';
  const u=data.outcome_update||{};
  $("journalStatus").textContent=u.error?"feedback non disponibile":("aggiornati "+(u.updated??0)+" · aperti "+(u.open??0)+" · chiusi "+(u.closed??0));
}

function renderEvidence(signals){
  $("evidence").innerHTML=signals.map(r=>{
    const b=r.bundle||{}, h=b.holdout||{};
    return '<div class="info">'+
      '<div class="label">'+esc(r.symbol)+'</div>'+
      '<div style="margin:7px 0">'+pill(!!b.compatible&&!!b.manifest_ready,b.compatible&&b.manifest_ready?"PROVEN":"WAIT")+'</div>'+
      '<div class="small">training: '+esc(b.training_rows??"—")+' · holdout: '+pct(h.total_return??h.net_compounded_return,2)+'</div>'+
      '<div class="small">dataset: '+esc((b.data_fingerprint||"—").slice(0,12))+'</div>'+
      '<div class="small">artefatti: '+esc((b.artifact_fingerprint||"—").slice(0,12))+'</div>'+
      '<div class="small">compatibilità: '+esc(b.compatibility||"—")+'</div>'+
    '</div>';
  }).join("")||'<div class="info">Nessuna evidenza bundle disponibile.</div>';
}

function bindInteractiveRows(){
  document.querySelectorAll(".interactive-row").forEach(row=>{
    row.onclick=()=>{
      const symbol=row.dataset.symbol;
      if(!symbol)return;
      const select=$("asset");
      if(Array.from(select.options).some(o=>o.value===symbol)){
        state.selected=symbol;
        select.value=symbol;
        loadHistory(symbol);
        const market=$("market");
        if(market)market.scrollIntoView({behavior:"smooth",block:"start"});
        openInspector(symbol);
        renderTimeline((state.data&&state.data.signals)||[],(state.data&&state.data.journal)||[]);
      }
    };
  });
}

function renderDecisionDeck(signals){
  const active=signals.find(x=>x.symbol===state.selected)||signals[0];
  const deckSignal=$("deckSignal"), deckMeta=$("deckMeta"), deckBundle=$("deckBundle"), deck=$("decisionDeck");
  const trace=$("traceGrid"), reasons=$("deckReasons");
  if(!active){
    deckSignal.textContent="WAIT";deckSignal.className="decision-signal signal-wait";
    deckMeta.textContent="Nessun asset disponibile.";
    deckBundle.className="pill warn";deckBundle.innerHTML='<span class="dot"></span>WAIT';
    $("deckAsset").textContent="—";$("deckPrice").textContent="—";$("deckConfidence").textContent="—";$("deckEdge").textContent="—";
    trace.innerHTML="";reasons.innerHTML="";return;
  }
  const d=active.decision||{}, b=active.bundle||{};
  const sig=active.signal||"WAIT";
  deckSignal.textContent=sig;deckSignal.className="decision-signal "+cls(sig);
  deck.className="decision-deck signal-live-"+sig.toLowerCase();
  deck.classList.remove("decision-flash");void deck.offsetWidth;deck.classList.add("decision-flash");
  deckMeta.textContent=sig==="LONG"?"Il gate corrente ammette un bias LONG.":sig==="SHORT"?"Il gate corrente ammette un bias SHORT.":"Nessun edge sufficientemente robusto per un segnale attivo.";
  deckBundle.className="pill "+(b.compatible&&b.manifest_ready?"good":"warn");
  deckBundle.innerHTML='<span class="dot"></span>'+esc(b.compatible&&b.manifest_ready?"PROVEN":"WAIT");
  $("deckAsset").textContent=active.symbol||"—";
  $("deckPrice").textContent=num(active.realtime_price??active.price,2);
  $("deckConfidence").textContent=pct(active.confidence,1);
  $("deckEdge").textContent=d.expected_return_lcb==null?"—":num(d.expected_return_lcb,4)+" / "+num(d.expected_return_ucb,4);
  reasons.innerHTML=(active.reason_codes||[]).slice(0,8).map(x=>'<span class="reason">'+esc(x)+'</span>').join("");
  const nodes=[
    ["Data",d.data_age_minutes!=null && Number(d.data_age_minutes)<=30,"fresh · "+age(d.data_age_minutes)],
    ["Model",d.p_up!=null,"p(up) · "+pct(d.p_up,1)],
    ["Meta",d.meta_success!=null && Number(d.meta_success)>=.5,"success · "+pct(d.meta_success,0)],
    ["Memory",d.analog_n!=null && Number(d.analog_n)>0,"support · "+esc(d.analog_n??"—")],
    ["Duration",!!d.trade_window_ready,d.trade_window_ready?"ready":"wait"],
    ["Deployment",!!b.compatible&&!!b.manifest_ready,b.compatible&&b.manifest_ready?"compatible":"wait"]
  ];
  trace.innerHTML=nodes.map(n=>'<div class="trace-node '+(n[1]?"ready":"wait")+'"><i class="trace-dot"></i><strong>'+n[0]+'</strong><span>'+n[2]+'</span></div>').join("");
}


function timeLabel(v){
  if(!v)return "—";
  const d=new Date(v); return Number.isFinite(d.getTime())?d.toLocaleString():String(v);
}
function signalOutcomeClass(outcome){
  return outcome==="WIN"?"good":outcome==="LOSS"?"bad":outcome==="TIMEOUT"?"warn":"";
}
function traceState(ok,label){
  return '<span class="pill '+(ok?'good':'warn')+'"><span class="dot"></span>'+esc(label)+'</span>';
}

function renderTimeline(signals,journal){
  const box=$("signalTimeline");
  if(!box)return;
  const selected=state.selected || (signals[0]&&signals[0].symbol);
  const active=(signals||[]).find(x=>x.symbol===selected);
  const records=(journal||[])
    .filter(r=>r.symbol===selected)
    .slice()
    .sort((a,b)=>new Date(a.data_timestamp||a.timestamp||0)-new Date(b.data_timestamp||b.timestamp||0))
    .slice(-8)
    .reverse();
  if(active && !records.some(r=>String(r.data_timestamp||r.timestamp)===String(active.timestamp))){
    records.unshift({...active,data_timestamp:active.timestamp,outcome:"OPEN",_live:true});
  }
  if(!records.length){
    box.innerHTML='<div class="timeline-empty">Nessuna traccia prequentiale disponibile per l’asset selezionato.</div>';
    $("timelineStatus").textContent="nessun evento";
    return;
  }
  $("timelineStatus").textContent=records.length+" tracce · "+(selected||"—");
  box.innerHTML=records.map((r,i)=>{
    const outcome=r.outcome||"OPEN";
    const closed=outcome!=="OPEN" && outcome!=="EARLY";
    const mature=r.holding_hours!=null || closed;
    const signal=r.signal||"WAIT";
    const forecast=r.expected_return==null?"—":pct(r.expected_return,2);
    const support=r.confidence==null?"—":pct(r.confidence,1);
    const held=r.holding_hours==null?"in corso":num(r.holding_hours,1)+"h";
    const outcomeLabel=closed?outcome:(outcome==="EARLY"?"EARLY":"OPEN");
    return '<article class="timeline-entry interactive-row" data-symbol="'+esc(r.symbol||selected||"")+'" tabindex="0" role="button">'+
      '<div class="timeline-top">'+
        '<div class="timeline-title"><span class="signal '+cls(signal)+'">'+esc(signal)+'</span><strong>'+esc(r.symbol||selected||"—")+'</strong><span class="small">confidence '+support+'</span></div>'+
        '<div class="small">'+esc(timeLabel(r.data_timestamp||r.timestamp))+'</div>'+
      '</div>'+
      '<div class="timeline-stages">'+
        '<div class="timeline-stage ready"><i class="stage-dot"></i><strong>Prediction</strong><span>edge '+esc(forecast)+' · score '+num((r.details&&r.details.score)||r.score,3)+'</span></div>'+
        '<div class="timeline-stage '+(closed?'ready':'pending')+'"><i class="stage-dot"></i><strong>Observation</strong><span>'+esc(closed?"market observed":"waiting for later candles")+'</span></div>'+
        '<div class="timeline-stage '+(mature?'final':'pending')+'"><i class="stage-dot"></i><strong>Maturity</strong><span>'+esc(held)+'</span></div>'+
        '<div class="timeline-stage '+(closed?'ready':'pending')+'"><i class="stage-dot"></i><strong>Outcome</strong><span class="'+signalOutcomeClass(outcome)+'">'+esc(outcomeLabel)+'</span></div>'+
      '</div>'+
    '</article>';
  }).join("");
  bindTimelineRows();
}

function bindTimelineRows(){
  document.querySelectorAll("#signalTimeline .interactive-row").forEach(row=>{
    const open=()=>{const symbol=row.dataset.symbol;if(symbol)openInspector(symbol);};
    row.onclick=open;
    row.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();open();}};
  });
}

function openInspector(symbol){
  const active=((state.data&&state.data.signals)||[]).find(x=>x.symbol===symbol);
  const content=$("inspectorContent"), drawer=$("inspectorDrawer");
  if(!content||!drawer)return;
  $("inspectorSubtitle").textContent=(active?.symbol||symbol||"—")+" · decision trace";
  if(!active){
    content.innerHTML='<div class="timeline-empty">Nessun dato attivo disponibile per '+esc(symbol||"asset")+'.</div>';
  }else{
    const d=active.decision||{}, b=active.bundle||{}, h=b.holdout||{};
    const signal=active.signal||"WAIT";
    const trace=[
      ["Data",d.data_age_minutes!=null && Number(d.data_age_minutes)<=30,"fresh · "+age(d.data_age_minutes)],
      ["Model",d.p_up!=null,"p(up) · "+pct(d.p_up,1)],
      ["Meta",d.meta_success!=null && Number(d.meta_success)>=.5,"success · "+pct(d.meta_success,0)],
      ["Memory",d.analog_n!=null && Number(d.analog_n)>0,"support · "+esc(d.analog_n??"—")],
      ["Duration",!!d.trade_window_ready,d.trade_window_ready?"ready":"wait"],
      ["Deployment",!!b.compatible&&!!b.manifest_ready,b.compatible&&b.manifest_ready?"compatible":"wait"]
    ];
    content.innerHTML=
      '<div class="inspector-hero signal-live-'+signal.toLowerCase()+'">'+
        '<div class="eyebrow">Selected verdict</div>'+
        '<div class="verdict '+cls(signal)+'">'+esc(signal)+'</div>'+
        '<div class="decision-meta">'+esc(signal==="LONG"?"Bias LONG ammesso dal gate.":signal==="SHORT"?"Bias SHORT ammesso dal gate.":"Nessun edge robusto ammesso dal gate.")+'</div>'+
        '<div class="inspector-grid">'+
          '<div class="inspector-card"><div class="k">Prezzo</div><div class="v num">'+num(active.realtime_price??active.price,2)+'</div></div>'+
          '<div class="inspector-card"><div class="k">Confidence</div><div class="v">'+pct(active.confidence,1)+'</div></div>'+
          '<div class="inspector-card"><div class="k">Robust edge</div><div class="v num">'+(d.expected_return_lcb==null?"—":num(d.expected_return_lcb,4)+" / "+num(d.expected_return_ucb,4))+'</div></div>'+
          '<div class="inspector-card"><div class="k">Holdout</div><div class="v">'+pct(h.total_return??h.net_compounded_return,2)+'</div></div>'+
        '</div>'+
      '</div>'+
      '<div class="inspector-section"><h3>Machine reasons</h3><div class="inspector-reasons">'+
        (active.reason_codes||[]).slice(0,12).map(x=>'<span class="reason">'+esc(x)+'</span>').join("")+
      '</div></div>'+
      '<div class="inspector-section"><h3>Decision trace</h3><div class="inspector-trace">'+
        trace.map(t=>'<div class="inspector-trace-row"><strong>'+t[0]+'</strong><span>'+t[2]+'</span>'+traceState(t[1],t[1]?"READY":"WAIT")+'</div>').join("")+
      '</div></div>'+
      '<div class="inspector-section"><h3>Provenance</h3>'+
        '<div class="inspector-card"><div class="k">Training</div><div class="v">'+esc(b.training_rows??"—")+' rows</div><div class="small">'+esc(b.training_end??"—")+'</div></div>'+
        '<div class="inspector-card" style="margin-top:8px"><div class="k">Compatibility</div><div class="small" style="margin-top:5px">'+esc(b.compatibility||"—")+'</div></div>'+
        '<div class="inspector-code" style="margin-top:8px">dataset: '+esc(b.data_fingerprint||"—")+'<br>artifact: '+esc(b.artifact_fingerprint||"—")+'</div>'+
      '</div>';
  }
  drawer.setAttribute("aria-hidden","false");
  document.body.classList.add("drawer-open");
}

function closeInspector(){
  const drawer=$("inspectorDrawer"); if(!drawer)return;
  drawer.setAttribute("aria-hidden","true");
  document.body.classList.remove("drawer-open");
}

function initInspector(){
  $("inspectorClose")?.addEventListener("click",closeInspector);
  $("inspectorBackdrop")?.addEventListener("click",closeInspector);
  document.addEventListener("keydown",e=>{if(e.key==="Escape")closeInspector();});
}

function render(data){
  state.data=data;
  renderMetrics(data);
  if(!data.ok){
    $("radarRows").innerHTML='<tr><td colspan="3"><span class="signal signal-wait">WAIT</span></td></tr>';
    $("detailRows").innerHTML='<tr><td colspan="11" class="small">'+esc(data.error||"Terminale non disponibile")+'</td></tr>';
    renderJournal(data); renderEvidence([]); renderDecisionDeck([]);
    return;
  }
  const signals=data.signals||[];
  populateAssets(signals);
  renderRadar(signals); renderDetail(signals); renderJournal(data); renderEvidence(signals); renderDecisionDeck(signals); renderTimeline(signals,data.journal||[]); bindInteractiveRows();
  const notes=(data.notes||[]).join(" · ");
  $("footer").textContent=notes+" · refresh "+data.refresh_seconds+"s · scan "+data.scan_seconds+"s";
}

function drawChart(history, signals, journal, realtimePrice){
  const canvas=$("chart"), empty=$("chartEmpty");
  const bars=(history&&history.bars)||[];
  if(!bars.length){empty.style.display="flex";return;}
  empty.style.display="none";
  const rect=canvas.getBoundingClientRect(), dpr=Math.max(1,window.devicePixelRatio||1);
  canvas.width=Math.floor(rect.width*dpr); canvas.height=Math.floor(rect.height*dpr);
  const ctx=canvas.getContext("2d"); ctx.scale(dpr,dpr);
  const W=rect.width,H=rect.height;
  ctx.clearRect(0,0,W,H);
  const pad={l:58,r:18,t:18,b:46}, cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  const lows=bars.map(x=>x.l), highs=bars.map(x=>x.h);
  let lo=Math.min(...lows), hi=Math.max(...highs);
  const rp=Number(realtimePrice); if(Number.isFinite(rp)){lo=Math.min(lo,rp);hi=Math.max(hi,rp);}
  const span=Math.max(hi-lo,1e-9), extra=span*.08; lo-=extra; hi+=extra;
  const xAt=i=>bars.length===1?pad.l+cw/2:pad.l+(i/(bars.length-1))*cw;
  const yAt=v=>pad.t+(1-(v-lo)/(hi-lo))*ch;

  ctx.strokeStyle="rgba(132,146,164,.14)";ctx.lineWidth=1;
  for(let i=0;i<=5;i++){const y=pad.t+(i/5)*ch;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();}
  ctx.font="10px system-ui";ctx.fillStyle="#718096";ctx.textAlign="right";
  const volumeMax=Math.max(...bars.map(b=>Number(b.v)||0),1);
  const volBase=H-14, volHeight=24;
  bars.forEach((b,i)=>{
    const x=xAt(i),vw=Math.max(1,cw/bars.length*.62),vh=((Number(b.v)||0)/volumeMax)*volHeight;
    ctx.fillStyle=b.c>=b.o?"rgba(69,227,154,.12)":"rgba(255,111,136,.12)";
    ctx.fillRect(x-vw/2,volBase-vh,vw,vh);
  });
  ctx.fillStyle="rgba(148,138,164,.42)";ctx.textAlign="left";ctx.fillText("VOL",pad.l,H-4);
  
  for(let i=0;i<=5;i++){const v=hi-(i/5)*(hi-lo),y=pad.t+(i/5)*ch;ctx.fillText(v.toFixed(2),pad.l-8,y+3);}
  
  const candleW=Math.max(2,cw/bars.length*.62);
  bars.forEach((b,i)=>{
    const x=xAt(i),yo=yAt(b.o),yc=yAt(b.c),yh=yAt(b.h),yl=yAt(b.l);
    const up=b.c>=b.o;
    ctx.strokeStyle=up?"rgba(69,227,154,.62)":"rgba(255,111,136,.62)";
    ctx.fillStyle=up?"rgba(69,227,154,.28)":"rgba(255,111,136,.28)";
    ctx.beginPath();ctx.moveTo(x,yh);ctx.lineTo(x,yl);ctx.stroke();
    const top=Math.min(yo,yc),bh=Math.max(1,Math.abs(yc-yo));
    ctx.fillRect(x-candleW/2,top,candleW,bh);
  });

  const signalTimes=new Map();
  [...(journal||[]),...(signals||[])].forEach(r=>{
    if(r.symbol!==state.selected)return;
    const t=new Date(r.timestamp||r.data_timestamp||"").getTime();
    if(Number.isFinite(t))signalTimes.set(t,r.signal);
  });
  for(const [t,sig] of signalTimes){
    let nearest=0,best=Infinity;
    bars.forEach((b,i)=>{const dist=Math.abs(b.t-t);if(dist<best){best=dist;nearest=i;}});
    if(best>Math.max((bars[bars.length-1].t-bars[0].t)/bars.length*2,300000))continue;
    const x=xAt(nearest), y=yAt(bars[nearest].c);
    ctx.fillStyle=sig==="LONG"?"#45e39a":sig==="SHORT"?"#ff6f88":"#ffd166";
    ctx.beginPath();ctx.arc(x,y,4.2,0,Math.PI*2);ctx.fill();
    ctx.strokeStyle="rgba(255,255,255,.18)";ctx.beginPath();ctx.moveTo(x,y+6);ctx.lineTo(x,Math.min(H-pad.b,y+28));ctx.stroke();
  }

  if(Number.isFinite(rp)){
    const y=yAt(rp);
    ctx.strokeStyle="#ffd166";ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle="#ffd166";ctx.textAlign="left";ctx.fillText("REALTIME",W-pad.r-64,y-6);
  }

  const area=ctx.createLinearGradient(0,pad.t,0,pad.t+ch);
  area.addColorStop(0,"rgba(120,184,255,.10)");area.addColorStop(1,"rgba(120,184,255,0)");
  ctx.fillStyle=area;ctx.beginPath();
  bars.forEach((b,i)=>{const x=xAt(i),y=yAt(b.c);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});
  ctx.lineTo(xAt(bars.length-1),pad.t+ch);ctx.lineTo(xAt(0),pad.t+ch);ctx.closePath();ctx.fill();
  ctx.strokeStyle="#78b8ff";ctx.lineWidth=1.7;ctx.shadowColor="rgba(120,184,255,.24)";ctx.shadowBlur=7;ctx.beginPath();
  bars.forEach((b,i)=>{const x=xAt(i),y=yAt(b.c);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();ctx.shadowBlur=0;

  const first=new Date(bars[0].t), last=new Date(bars[bars.length-1].t);
  ctx.fillStyle="#718096";ctx.textAlign="left";ctx.fillText(first.toLocaleDateString(),pad.l,H-8);ctx.textAlign="right";ctx.fillText(last.toLocaleDateString(),W-pad.r,H-8);

  const regimeBadge=$("chartRegime");
  const activeSignal=(signals||[]).find(x=>x.symbol===state.selected);
  if(regimeBadge)regimeBadge.textContent="REGIME · "+esc((activeSignal&&activeSignal.decision&&activeSignal.decision.regime)||"—");
  canvas.onmousemove=(ev)=>{
    const r=canvas.getBoundingClientRect(), mx=ev.clientX-r.left;
    const idx=Math.max(0,Math.min(bars.length-1,Math.round((mx-pad.l)/cw*(bars.length-1))));
    const b=bars[idx], x=xAt(idx), y=yAt(b.c);
    const cur=$("cursor");cur.style.display="block";cur.style.left=Math.min(W-150,Math.max(8,x+10))+"px";cur.style.top="12px";
    cur.innerHTML=new Date(b.t).toLocaleString()+"<br>O "+num(b.o,2)+" · H "+num(b.h,2)+" · L "+num(b.l,2)+" · C "+num(b.c,2);
    const chrome=$("chartCrosshair"), cx=$("crossX"), cy=$("crossY"), badge=$("crossBadge");
    if(chrome){chrome.style.display="block";cx.style.transform=`translate3d(${x}px,0,0)`;cy.style.transform=`translate3d(0,${y}px,0)`;badge.textContent=new Date(b.t).toLocaleString()+" · "+num(b.c,2);}
  };
  canvas.onmouseleave=()=>{
    $("cursor").style.display="none";
    $("chartCrosshair").style.display="none";
  };
}

let quoteBusy=false;
async function loadQuote(symbol){
  if(!symbol || quoteBusy)return;
  quoteBusy=true;
  try{
    const res=await fetch("/api/quote?symbol="+encodeURIComponent(symbol),{cache:"no-store"});
    const q=await res.json();
    if(state.selected!==symbol)return;
    if(q.price!=null){
      $("livePrice").textContent="REALTIME "+num(q.price,2);
      $("livePrice").className="pill good";
      drawChart(state.history,(state.data&&state.data.signals)||[],(state.data&&state.data.journal)||[],q.price);
    }else{
      $("livePrice").textContent="REALTIME —";
      $("livePrice").className="pill warn";
    }
  }catch(e){
    if(state.selected===symbol){
      $("livePrice").textContent="REALTIME offline";
      $("livePrice").className="pill warn";
    }
  }finally{
    quoteBusy=false;
  }
}

async function loadHistory(symbol){
  if(!symbol)return;
  const request=++state.historyRequest;
  try{
    const limit=$("range").value;
    const res=await fetch("/api/history?symbol="+encodeURIComponent(symbol)+"&limit="+encodeURIComponent(limit),{cache:"no-store"});
    const history=await res.json();
    if(request!==state.historyRequest || state.selected!==symbol)return;
    state.history=history;
    const source=$("historySource");
    if(source){
      const label=history.source==="network"?"HISTORY · LIVE":history.source==="local_cache"?"HISTORY · CACHE":history.source==="bundled"?"HISTORY · BUNDLED":"HISTORY · OFFLINE";
      source.textContent=label;
      source.className="pill "+(history.bars?.length ? "good" : "warn");
    }
    const signals=(state.data&&state.data.signals)||[];
    drawChart(history,signals,(state.data&&state.data.journal)||[],signals.find(x=>x.symbol===symbol)?.realtime_price);
    await loadQuote(symbol);
  }catch(e){
    if(request===state.historyRequest){
      $("chartEmpty").style.display="flex";
      $("chartEmpty").textContent="Storico non disponibile: "+e;
    }
  }
}

async function refresh(force=false){
  $("stamp").textContent="scansione…";
  try{
    const res=await fetch("/api/state?force="+(force?"1":"0"),{cache:"no-store"});
    const data=await res.json();
    render(data);
    renderFocus(data);
    scheduleRefresh(data.scan_in_progress ? 3 : (data.refresh_seconds||20));
  }catch(e){
    renderFocus({ok:false,error:String(e),config:{},refresh_seconds:20});
    scheduleRefresh(20);
  }
}
let refreshTimer=null;
function scheduleRefresh(seconds){
  if(refreshTimer)clearInterval(refreshTimer);
  refreshTimer=setInterval(()=>refresh(false),Math.max(10,Number(seconds||20))*1000);
}
$("refresh").addEventListener("click",()=>refresh(true));
function openSelectedAsset(){const value=String($("asset").value||"").trim();if(!value)return;const allowed=new Set(state.marketSymbols||[]);if(!allowed.has(value)){ $("asset").setCustomValidity("Simbolo non presente nei mercati attivi.");$("asset").reportValidity();return;}$("asset").setCustomValidity("");state.selected=value;loadHistory(value);renderTimeline((state.data&&state.data.signals)||[],(state.data&&state.data.journal)||[]);}$("asset").addEventListener("change",openSelectedAsset);$("asset").addEventListener("keydown",e=>{if(e.key==="Enter"){e.preventDefault();openSelectedAsset();}});$("loadAsset").addEventListener("click",openSelectedAsset);$("range").addEventListener("change",()=>loadHistory(state.selected));
window.addEventListener("resize",()=>{if(state.history)drawChart(state.history,(state.data&&state.data.signals)||[],(state.data&&state.data.journal)||[],(state.data&&state.data.signals||[]).find(x=>x.symbol===state.selected)?.realtime_price);});
initAmbientFX();
initAlphaMotion();
refresh(true);
</script>
</body>
</html>
"""


def make_handler(terminal: SignalTerminal):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AdaptiveSignalTerminal/1.0"

        def _send(
            self,
            body: bytes,
            status: int = 200,
            content_type: str = "text/html; charset=utf-8",
        ):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send(HTML.encode("utf-8"))
                return
            if parsed.path == "/api/health":
                body = json.dumps(
                    terminal.health(), separators=(",", ":")
                ).encode("utf-8")
                self._send(
                    body, content_type="application/json; charset=utf-8"
                )
                return
            if parsed.path == "/api/state":
                query = parse_qs(parsed.query)
                force = query.get("force", ["0"])[0] == "1"
                body = json.dumps(
                    terminal._terminal_state(force=force, background=True),
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                self._send(
                    body, content_type="application/json; charset=utf-8"
                )
                return
            if parsed.path == "/api/quote":
                query = parse_qs(parsed.query)
                symbol = query.get("symbol", [terminal.settings.symbol])[0]
                cached = terminal._cached_state or {}
                configured = (
                    getattr(terminal.settings, "live_symbols", ())
                    or (terminal.settings.symbol,)
                )
                active_symbols = [
                    x.get("symbol") for x in cached.get("signals", [])
                    if isinstance(x, dict) and x.get("symbol")
                ]
                discovered = set((cached.get("config") or {}).get("market_symbols") or [])
                allowed = set(configured) | discovered | {str(x) for x in active_symbols}
                if symbol not in allowed:
                    self._send(
                        b'{"error":"symbol_not_configured"}',
                        status=400,
                        content_type="application/json; charset=utf-8",
                    )
                    return
                body = json.dumps(
                    terminal._quote(symbol),
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                self._send(body, content_type="application/json; charset=utf-8")
                return
            if parsed.path == "/api/history":
                query = parse_qs(parsed.query)
                symbol = query.get("symbol", [terminal.settings.symbol])[0]
                cached = terminal._cached_state or {}
                configured = (
                    getattr(terminal.settings, "live_symbols", ())
                    or (terminal.settings.symbol,)
                )
                active_symbols = [
                    x.get("symbol") for x in cached.get("signals", [])
                    if isinstance(x, dict) and x.get("symbol")
                ]
                discovered = set((cached.get("config") or {}).get("market_symbols") or [])
                allowed = set(configured) | discovered | {str(x) for x in active_symbols}
                if symbol not in allowed:
                    self._send(
                        b'{"error":"symbol_not_configured","bars":[]}',
                        status=400,
                        content_type="application/json; charset=utf-8",
                    )
                    return
                try:
                    limit = int(query.get("limit", [terminal.history_bars])[0])
                except ValueError:
                    limit = terminal.history_bars
                body = json.dumps(
                    terminal._history(symbol, limit),
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                self._send(
                    body, content_type="application/json; charset=utf-8"
                )
                return
            if parsed.path == "/favicon.ico":
                self._send(b"", status=204, content_type="image/x-icon")
                return
            self._send(
                b"Not found", status=404, content_type="text/plain; charset=utf-8"
            )

        def log_message(self, fmt, *args):
            print(
                f"[signal-terminal] {self.address_string()} - {fmt % args}"
            )

    return Handler


def validate_dashboard_markup() -> None:
    """Fail fast on malformed critical markup before a packaged terminal is released."""
    required = (
        '<section class="legacy-hidden decision-deck" id="decisionDeck" aria-live="polite">',
        '<div class="legacy-hidden panel chart-panel" id="market">',
        '<div class="legacy-hidden panel" id="timeline">',
        '<div class="legacy-hidden panel" id="detail">',
        '<div class="legacy-hidden panel" id="journal">',
        '<div class="legacy-hidden panel" id="evidencePanel">',
        '<aside id="inspectorDrawer" class="legacy-hidden inspector-drawer" aria-label="Decision inspector" aria-hidden="true">',
    )
    missing = [token for token in required if token not in HTML]
    if missing:
        raise RuntimeError("dashboard_markup_missing:" + ",".join(missing))

    # These are the critical opening tags most likely to break the single-file UI.
    malformed_prefixes = (
        '<section class="legacy-hidden decision-deck" id="decisionDeck" aria-live="polite"\n',
        '<div class="legacy-hidden panel chart-panel" id="market"\n',
        '<div class="legacy-hidden panel" id="timeline"\n',
        '<div class="legacy-hidden panel" id="detail"\n',
        '<div class="legacy-hidden panel" id="journal"\n',
        '<div class="legacy-hidden panel" id="evidencePanel"\n',
        '<aside id="inspectorDrawer" class="legacy-hidden inspector-drawer" aria-label="Decision inspector" aria-hidden="true"\n',
    )
    found = [token for token in malformed_prefixes if token in HTML]
    if found:
        raise RuntimeError("dashboard_markup_unclosed_tag")

def parse_args():
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument(
        "--config",
        default=None,
        help="Settings YAML path (defaults to repository/config.yaml)",
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help="Bind address (default: localhost)"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--refresh", type=int, default=DEFAULT_REFRESH, help="Refresh TTL in seconds"
    )
    parser.add_argument(
        "--history-bars", type=int, default=DEFAULT_HISTORY_BARS
    )
    parser.add_argument(
        "--symbols", default=None, help="Comma-separated symbol override"
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not open browser automatically"
    )
    parser.add_argument(
        "--once", action="store_true", help="Print one JSON state and exit"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Validate frozen/source startup without contacting the market.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke_test:
        validate_dashboard_markup()
        print("dashboard-smoke-ok")
        return
    settings = load_settings(args.config or ROOT / "config.yaml")
    if args.symbols:
        settings.live_symbols = tuple(
            x.strip() for x in args.symbols.split(",") if x.strip()
        )

    terminal = SignalTerminal(
        settings,
        ROOT,
        refresh_seconds=args.refresh,
        history_bars=args.history_bars,
    )
    if args.once:
        print(
            json.dumps(
                terminal._terminal_state(force=True),
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    try:
        server = ThreadingHTTPServer((args.host, args.port), make_handler(terminal))
    except OSError as exc:
        if args.port != DEFAULT_PORT:
            raise
        # A stale terminal or another local service may already occupy 8765.
        server = ThreadingHTTPServer((args.host, 0), make_handler(terminal))
        args.port = int(server.server_address[1])
        if sys.stdout is not None:
            print(f"Port {DEFAULT_PORT} was busy ({exc}); using {args.port}.")
    url = (
        f"http://{args.host if args.host not in {'0.0.0.0','::'} else '127.0.0.1'}:"
        f"{args.port}/"
    )
    if sys.stdout is not None:
        print(f"{APP_TITLE} v{__version__}")
        print(
            f"Read-only · exchange={settings.exchange} · timeframe={settings.timeframe}"
        )
        print(f"Dashboard: {url}")
        print("No order endpoints are exposed by this process.")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if sys.stdout is not None:
            print("\nStopping signal terminal.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
