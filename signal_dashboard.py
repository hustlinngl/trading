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
import json
import math
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

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
        self.refresh_seconds = max(10, int(refresh_seconds))
        self.history_bars = max(80, int(history_bars))
        self._lock = threading.Lock()
        self._cached_state: dict | None = None
        self._cached_at = 0.0
        self.root.joinpath("logs").mkdir(parents=True, exist_ok=True)

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
            exchange = exchange_client(
                getattr(self.settings, "exchange", "binance"), sandbox=False
            )
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
            exchange = exchange_client(
                getattr(self.settings, "exchange", "binance"), sandbox=False
            )
            payload = self._ticker_row(symbol, exchange.fetch_ticker(symbol))
            payload["generated_at"] = datetime.now(timezone.utc).isoformat()
            return payload
        except Exception as exc:
            return {"symbol": symbol, "error": f"{type(exc).__name__}:{exc}"}

    def _history(self, symbol: str, limit: int | None = None) -> dict:
        bars = max(80, min(1000, int(limit or self.history_bars)))
        try:
            exchange = exchange_client(
                getattr(self.settings, "exchange", "binance"), sandbox=False
            )
            frame = fetch_ohlcv(
                exchange, symbol, self.settings.timeframe, bars
            )
            if frame is None or frame.empty:
                return {"symbol": symbol, "bars": [], "error": "empty_data"}
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
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            return {
                "symbol": symbol,
                "bars": [],
                "error": f"{type(exc).__name__}:{exc}",
            }

    def _journal(self) -> list[dict]:
        path = self.root / "logs" / "live_signal_history.jsonl"
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

    def _terminal_state(self, force: bool = False) -> dict:
        with self._lock:
            now = time.time()
            if (
                not force
                and self._cached_state is not None
                and now - self._cached_at < self.refresh_seconds
            ):
                return self._cached_state

            started = time.monotonic()
            try:
                symbols = list(
                    dict.fromkeys(
                        getattr(self.settings, "live_symbols", ())
                        or (self.settings.symbol,)
                    )
                )[: int(getattr(self.settings, "live_max_symbols", 15))]

                assessments = scan_top5(
                    self.settings, str(self.root), symbols=symbols
                )
                write_live_snapshot(assessments, str(self.root))
                append_live_signal_history(assessments, str(self.root))

                try:
                    outcome_update = update_live_signal_outcomes(
                        self.settings, str(self.root)
                    )
                except Exception as exc:
                    outcome_update = {
                        "updated": 0,
                        "open": None,
                        "closed": None,
                        "error": f"{type(exc).__name__}:{exc}",
                    }

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
                        float(x.get("confidence", 0.0) or 0.0),
                        float(
                            (x.get("decision") or {}).get("score", -9.0)
                            or -9.0
                        ),
                    ),
                    reverse=True,
                )

                active = sum(
                    str(x.get("signal")) in {"LONG", "SHORT"} for x in signals
                )
                compatible = sum(
                    bool((x.get("bundle") or {}).get("compatible"))
                    and bool((x.get("bundle") or {}).get("manifest_ready"))
                    for x in signals
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
                    "config": {
                        "exchange": self.settings.exchange,
                        "timeframe": self.settings.timeframe,
                        "primary_symbol": self.settings.symbol,
                        "live_symbols": symbols,
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
                        "assets_scanned": len(signals),
                        "active_signals": active,
                        "waits": len(signals) - active,
                        "compatible_bundles": compatible,
                        "fresh_data_assets": fresh,
                        "terminal_ready": bool(signals) and compatible > 0,
                    },
                    "signals": signals,
                    "journal": self._journal(),
                    "outcome_update": _json_safe(outcome_update),
                    "notes": [
                        "Sola lettura: il terminale non espone API per ordini.",
                        "Ogni scan usa il gate live/paper del motore addestrato.",
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

            self._cached_state = _json_safe(state)
            self._cached_at = time.time()
            return self._cached_state

    def health(self) -> dict:
        return {
            "ok": True,
            "version": __version__,
            "cached": self._cached_state is not None,
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
<title>Adaptive AI Signal Terminal</title>
<style>
:root{
  --bg:#070a0f;--panel:#0d121a;--panel2:#111823;--line:#202b38;
  --text:#eef3f8;--muted:#8492a4;--green:#45e39a;--red:#ff6f88;
  --amber:#ffd166;--blue:#78b8ff;--shadow:0 20px 70px rgba(0,0,0,.28);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:
radial-gradient(1000px 500px at 15% -10%,#172638 0,transparent 62%),
radial-gradient(900px 500px at 100% 0,#141b28 0,transparent 58%),
var(--bg);color:var(--text);
font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:1480px;margin:auto;padding:26px 24px 56px}
.top{display:flex;justify-content:space-between;gap:24px;align-items:flex-end;margin-bottom:18px}
.eyebrow{font-size:10px;text-transform:uppercase;letter-spacing:.2em;color:var(--muted)}
h1{font-size:36px;line-height:1;margin:6px 0 9px;letter-spacing:-.03em}
.sub{max-width:930px;color:var(--muted)}
.actions{display:flex;gap:9px;align-items:center}
button,select{border:1px solid var(--line);background:var(--panel2);color:var(--text);border-radius:11px;padding:9px 12px}
button{cursor:pointer}
button:hover,select:hover{border-color:#405065}
#stamp{font-size:12px;color:var(--muted);white-space:nowrap}
.metrics{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px}
.card,.panel{background:linear-gradient(180deg,rgba(16,23,33,.95),rgba(10,14,21,.98));
border:1px solid var(--line);box-shadow:var(--shadow)}
.card{border-radius:15px;padding:15px}
.metric-label,.label{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.13em}
.metric-value{margin-top:6px;font-size:24px;font-weight:800;letter-spacing:-.02em}
.good{color:var(--green)}.bad{color:var(--red)}.warn{color:var(--amber)}
.panel{border-radius:18px;margin-top:14px;overflow:hidden}
.panel-head{display:flex;justify-content:space-between;gap:14px;align-items:center;padding:15px 17px;border-bottom:1px solid var(--line)}
.title{font-weight:750;font-size:15px}
.small{font-size:11px;color:var(--muted)}
.layout{display:grid;grid-template-columns:1.5fr .5fr;gap:14px}
.chart-panel{min-height:500px}
.chart-tools{display:flex;gap:8px;align-items:center}
#chart{display:block;width:100%;height:430px}
.chart-wrap{position:relative;padding:10px 12px 12px}
.chart-empty{display:flex;align-items:center;justify-content:center;height:430px;color:var(--muted)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--muted);padding:0 12px 12px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}
.controls{display:flex;gap:8px;align-items:center}
table{border-collapse:collapse;width:100%}
th,td{padding:11px 12px;text-align:left;border-bottom:1px solid rgba(32,43,56,.72);vertical-align:top}
th{font-size:9px;text-transform:uppercase;letter-spacing:.11em;color:var(--muted);position:sticky;top:0;background:#0b1017}
.table-wrap{overflow:auto}
.signal{font-weight:850;letter-spacing:.08em}
.signal-long{color:var(--green)}.signal-short{color:var(--red)}.signal-wait{color:var(--amber)}.signal-flat{color:var(--muted)}
.pill{display:inline-flex;align-items:center;gap:6px;padding:4px 8px;border-radius:999px;border:1px solid var(--line);font-size:10px}
.dot{width:7px;height:7px;border-radius:50%;background:currentColor}
.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;padding:14px}
.info{border:1px solid var(--line);border-radius:12px;background:#0d141d;padding:13px}
.info .big{font-size:20px;font-weight:800;margin-top:4px}
.reasons{display:flex;gap:5px;flex-wrap:wrap;margin-top:6px}
.reason{padding:3px 6px;border:1px solid #283443;border-radius:7px;background:#171f29;color:#a9b4c2;font-size:10px}
.num{font-variant-numeric:tabular-nums}
.footer{margin-top:14px;color:var(--muted);font-size:11px}
#cursor{position:absolute;pointer-events:none;display:none;background:#101822;border:1px solid #2b3949;border-radius:9px;padding:7px 9px;font-size:10px;box-shadow:var(--shadow)}
@media(max-width:1180px){.metrics{grid-template-columns:repeat(3,minmax(0,1fr))}.layout{grid-template-columns:1fr}}
@media(max-width:760px){.wrap{padding:18px 13px 40px}.top{align-items:flex-start;flex-direction:column}h1{font-size:29px}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.grid3{grid-template-columns:1fr}.chart-panel{min-height:430px}#chart{height:350px}}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div>
      <div class="eyebrow">Pre-alpha · signal intelligence</div>
      <h1>Adaptive AI Signal Terminal</h1>
      <div class="sub">Radar read-only basato sul training del repository: dati pubblici, gate conservativi, provenance verificata e feedback prequentiale. Nessuna funzione di esecuzione ordini.</div>
    </div>
    <div class="actions">
      <span id="stamp">Connessione…</span>
      <button id="refresh">Aggiorna</button>
    </div>
  </div>

  <div class="metrics" id="metrics"></div>

  <div class="layout">
    <div class="panel chart-panel">
      <div class="panel-head">
        <div>
          <div class="title">Market cockpit</div>
          <div class="small">Candele storiche + marker segnali, con ticker realtime separato dal close usato dal modello.</div>
        </div>
        <div class="chart-tools">
          <span id="livePrice" class="pill good">REALTIME —</span>
          <select id="asset"></select>
          <select id="range"><option value="120">120</option><option value="240" selected>240</option><option value="480">480</option></select>
        </div>
      </div>
      <div class="chart-wrap" id="chartWrap">
        <canvas id="chart"></canvas>
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

  <div class="panel">
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

  <div class="panel">
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

  <div class="panel">
    <div class="panel-head"><div><div class="title">Training / deployment evidence</div><div class="small">Un segnale è ammissibile solo con bundle compatibile e provenance coerente.</div></div></div>
    <div class="grid3" id="evidence"></div>
  </div>

  <div class="footer" id="footer"></div>
</div>

<script>
const $ = (id) => document.getElementById(id);
const state = { data:null, history:null, selected:null, historyRequest:0 };

function esc(v){return String(v??"").replace(/[&<>"]/g,c=>c==="&"?"&amp;":c==="<"?"&lt;":c===">"?"&gt;":"&quot;");}
function pct(v,d=1){return v==null||Number.isNaN(Number(v))?"—":(Number(v)*100).toFixed(d)+"%";}
function num(v,d=3){return v==null||Number.isNaN(Number(v))?"—":Number(v).toFixed(d);}
function age(v){return v==null||Number.isNaN(Number(v))?"—":Number(v).toFixed(1)+"m";}
function cls(sig){return sig==="LONG"?"signal-long":sig==="SHORT"?"signal-short":sig==="WAIT"?"signal-wait":"signal-flat";}
function pill(ok,label){return '<span class="pill '+(ok?'good':'warn')+'"><span class="dot"></span>'+esc(label)+'</span>';}

function populateAssets(signals){
  const sel=$("asset");
  const existing=Array.from(sel.options).map(x=>x.value);
  const values=signals.map(x=>x.symbol);
  if(values.join("|")!==existing.join("|")){
    sel.innerHTML=values.map(s=>'<option value="'+esc(s)+'">'+esc(s)+'</option>').join("");
  }
  if(!state.selected || !values.includes(state.selected)) state.selected=values[0]||null;
  sel.value=state.selected||"";
}

function renderMetrics(data){
  const s=data.summary||{};
  const q=data.config||{};
  $("metrics").innerHTML=[
    ["Terminal",data.ok?"READY":"WAIT",data.ok?"good":"bad"],
    ["Segnali",s.active_signals??0,s.active_signals>0?"good":""],
    ["WAIT",s.waits??0,s.waits>0?"warn":""],
    ["Bundle compatibili",s.compatible_bundles??0,s.compatible_bundles>0?"good":"warn"],
    ["Dati freschi",s.fresh_data_assets??0,s.fresh_data_assets>0?"good":"warn"],
    ["Timeframe",q.timeframe||"—",""]
  ].map(x=>'<div class="card"><div class="metric-label">'+x[0]+'</div><div class="metric-value '+x[2]+'">'+esc(x[1])+'</div></div>').join("");
  $("stamp").textContent=(data.generated_at?new Date(data.generated_at).toLocaleTimeString():"—")+" · "+(q.exchange||"—");
}

function renderRadar(signals){
  $("radarRows").innerHTML=signals.map(r=>{
    const price=r.realtime_price??r.price;
    return '<tr><td><strong>'+esc(r.symbol)+'</strong><div class="small">'+num(price,2)+'</div></td>'+
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
    return '<tr>'+
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

function render(data){
  state.data=data;
  renderMetrics(data);
  if(!data.ok){
    $("radarRows").innerHTML='<tr><td colspan="3"><span class="signal signal-wait">WAIT</span></td></tr>';
    $("detailRows").innerHTML='<tr><td colspan="11" class="small">'+esc(data.error||"Terminale non disponibile")+'</td></tr>';
    renderJournal(data); renderEvidence([]);
    return;
  }
  const signals=data.signals||[];
  populateAssets(signals);
  renderRadar(signals); renderDetail(signals); renderJournal(data); renderEvidence(signals);
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
  const pad={l:58,r:18,t:18,b:32}, cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  const lows=bars.map(x=>x.l), highs=bars.map(x=>x.h);
  let lo=Math.min(...lows), hi=Math.max(...highs);
  const rp=Number(realtimePrice); if(Number.isFinite(rp)){lo=Math.min(lo,rp);hi=Math.max(hi,rp);}
  const span=Math.max(hi-lo,1e-9), extra=span*.08; lo-=extra; hi+=extra;
  const xAt=i=>pad.l+(i/(bars.length-1))*cw;
  const yAt=v=>pad.t+(1-(v-lo)/(hi-lo))*ch;

  ctx.strokeStyle="rgba(132,146,164,.14)";ctx.lineWidth=1;
  for(let i=0;i<=5;i++){const y=pad.t+(i/5)*ch;ctx.beginPath();ctx.moveTo(pad.l,y);ctx.lineTo(W-pad.r,y);ctx.stroke();}
  ctx.font="10px system-ui";ctx.fillStyle="#718096";ctx.textAlign="right";
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

  ctx.strokeStyle="#78b8ff";ctx.lineWidth=1.7;ctx.beginPath();
  bars.forEach((b,i)=>{const x=xAt(i),y=yAt(b.c);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();

  const first=new Date(bars[0].t), last=new Date(bars[bars.length-1].t);
  ctx.fillStyle="#718096";ctx.textAlign="left";ctx.fillText(first.toLocaleDateString(),pad.l,H-8);ctx.textAlign="right";ctx.fillText(last.toLocaleDateString(),W-pad.r,H-8);

  canvas.onmousemove=(ev)=>{
    const r=canvas.getBoundingClientRect(), mx=ev.clientX-r.left;
    const idx=Math.max(0,Math.min(bars.length-1,Math.round((mx-pad.l)/cw*(bars.length-1))));
    const b=bars[idx], x=xAt(idx);
    const cur=$("cursor");cur.style.display="block";cur.style.left=Math.min(W-150,Math.max(8,x+10))+"px";cur.style.top="12px";
    cur.innerHTML=new Date(b.t).toLocaleString()+"<br>O "+num(b.o,2)+" · H "+num(b.h,2)+" · L "+num(b.l,2)+" · C "+num(b.c,2);
  };
  canvas.onmouseleave=()=>{$("cursor").style.display="none";};
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
    const data=await res.json(); render(data);
    await loadHistory(state.selected);
    scheduleRefresh(data.refresh_seconds||20);
  }catch(e){
    render({ok:false,error:String(e),summary:{},notes:["Impossibile raggiungere il terminale locale."]});
    scheduleRefresh(20);
  }
}
let refreshTimer=null;
function scheduleRefresh(seconds){
  if(refreshTimer)clearInterval(refreshTimer);
  refreshTimer=setInterval(()=>refresh(false),Math.max(10,Number(seconds||20))*1000);
}
$("refresh").addEventListener("click",()=>refresh(true));
$("asset").addEventListener("change",()=>{state.selected=$("asset").value;loadHistory(state.selected);});
$("range").addEventListener("change",()=>loadHistory(state.selected));
window.addEventListener("resize",()=>{if(state.history)drawChart(state.history,(state.data&&state.data.signals)||[],(state.data&&state.data.journal)||[],(state.data&&state.data.signals||[]).find(x=>x.symbol===state.selected)?.realtime_price);});
refresh(true);
setInterval(()=>loadQuote(state.selected),5000);
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
                    terminal._terminal_state(force=force),
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
                allowed = set(
                    getattr(terminal.settings, "live_symbols", ())
                    or (terminal.settings.symbol,)
                )
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
                allowed = set(
                    getattr(terminal.settings, "live_symbols", ())
                    or (terminal.settings.symbol,)
                )
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
    return parser.parse_args()


def main():
    args = parse_args()
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

    server = ThreadingHTTPServer((args.host, args.port), make_handler(terminal))
    url = (
        f"http://{args.host if args.host not in {'0.0.0.0','::'} else '127.0.0.1'}:"
        f"{args.port}/"
    )
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
        print("\nStopping signal terminal.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
