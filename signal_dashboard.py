#!/usr/bin/env python3
"""
Adaptive AI Signal Terminal.

Single-entry local dashboard for trained signal bundles.
It has no order endpoints, no exchange write API and no trading execution path.
It only reads public market data, validates model provenance and serves signals.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import sys
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ai_trading_lab import __version__
from ai_trading_lab.config import Settings, load_settings
from ai_trading_lab.deployment import (
    asset_bundle_dir,
    bundle_compatibility,
    bundle_artifact_fingerprint,
    resolve_signal_bundle,
)
from ai_trading_lab.live import LiveAssessment, append_live_signal_history, scan_top5, write_live_snapshot
from ai_trading_lab.live_tracker import update_live_signal_outcomes


APP_TITLE = "Adaptive AI Segnale Terminale"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_REFRESH = 45


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        if value != value:
            return None
    except Exception:
        pass
    return value.item() if hasattr(value, "item") else value


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _short(value, width=14):
    text = str(value or "")
    return text if len(text) <= width else text[: width - 1] + "…"


def _pct(value):
    try:
        return f"{float(value) * 100:.2f}%"
    except Exception:
        return "—"


def _num(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "—"


def _signal_rank(signal: str) -> int:
    return {"LONG": 3, "SHORT": 2, "WAIT": 1, "FLAT": 0}.get(signal, 0)


def _signal_class(signal: str) -> str:
    return {
        "LONG": "signal-long",
        "SHORT": "signal-short",
        "WAIT": "signal-wait",
        "FLAT": "signal-flat",
    }.get(signal, "signal-flat")


def _status_class(status: str) -> str:
    return "status-ready" if status == "PRONTO" else "status-wait"


class SegnaleTerminale:
    def __init__(self, settings: Settings, root: str | Path = ".", refresh_seconds: int = DEFAULT_REFRESH):
        self.settings = settings
        self.root = Path(root).resolve()
        self.refresh_seconds = max(10, int(refresh_seconds))
        self._lock = threading.Lock()
        self._cached_state = None
        self._cached_at = 0.0
        self._scan_error = None
        self.root.joinpath("logs").mkdir(parents=True, exist_ok=True)

    def _bundle_snapshot(self, symbol: str) -> dict:
        bundle = resolve_signal_bundle(self.settings, self.root, symbol)
        info = {
            "symbol": symbol,
            "bundle": str(bundle),
            "exists": (bundle / "signal_model.joblib").exists(),
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
        info["training_rows"] = meta.get("rows")
        info["training_end"] = meta.get("end")
        info["data_fingerprint"] = meta.get("data_fingerprint")
        info["model_semantics_fingerprint"] = meta.get("model_semantics_fingerprint")
        info["deployment_semantics_fingerprint"] = meta.get("deployment_semantics_fingerprint")
        info["manifest_ready"] = bool(manifest.get("ready", False))
        info["holdout"] = holdout.get("holdout", holdout if isinstance(holdout, dict) else {})
        try:
            info["artifact_fingerprint"] = bundle_artifact_fingerprint(bundle)
        except Exception:
            info["artifact_fingerprint"] = None
        try:
            compatible, reason = bundle_compatibility(self.settings, bundle, symbol)
            info["compatible"] = bool(compatible)
            info["compatibility"] = reason
        except Exception as exc:
            info["compatibility"] = f"{type(exc).__name__}:{exc}"
        return info

    def _assessment_payload(self, assessment: LiveAssessment) -> dict:
        row = assessment.to_dict()
        row["signal_class"] = _signal_class(str(row.get("signal", "WAIT")))
        row["details"] = _json_safe(row.get("details") or {})
        return _json_safe(row)

    def _terminal_state(self, force=False) -> dict:
        now = time.time()
        with self._lock:
            if not force and self._cached_state and now - self._cached_at < self.refresh_seconds:
                return self._cached_state

            started = time.monotonic()
            try:
                configured = list(dict.fromkeys(getattr(self.settings, "live_symbols", ()) or (self.settings.symbol,)))
                configured = configured[: int(getattr(self.settings, "live_max_symbols", 15))]
                assessments = scan_top5(self.settings, str(self.root), symbols=configured)
                write_live_snapshot(assessments, str(self.root))
                append_live_signal_history(assessments, str(self.root))
                try:
                    outcome_update = update_live_signal_outcomes(self.settings, str(self.root))
                except Exception as exc:
                    outcome_update = {"updated": 0, "open": None, "closed": None, "error": f"{type(exc).__name__}:{exc}"}
                signals = []
                for assessment in assessments:
                    payload = self._assessment_payload(assessment)
                    payload["bundle"] = self._bundle_snapshot(assessment.symbol)
                    details = payload.get("details") or {}
                    payload["decision"] = {
                        "score": details.get("score"),
                        "p_up": details.get("p_up"),
                        "expected_return_lcb": details.get("expected_return_lcb"),
                        "expected_return_ucb": details.get("expected_return_ucb"),
                        "meta_success": details.get("meta_success"),
                        "model_disagreement": details.get("model_disagreement"),
                        "regime": details.get("regime"),
                        "analog_n": details.get("analog_n"),
                        "analog_agreement": details.get("analog_agreement"),
                        "trade_window_confidence": details.get("trade_window_confidence"),
                        "trade_window_direction": details.get("trade_window_direction"),
                        "trade_window_ready": details.get("trade_window_ready"),
                        "data_age_minutes": details.get("data_age_minutes"),
                    }
                    signals.append(payload)
                signals.sort(
                    key=lambda x: (
                        _signal_rank(str(x.get("signal", "WAIT"))),
                        float(x.get("confidence", 0.0) or 0.0),
                        float((x.get("decision") or {}).get("score", -9.0) or -9.0),
                    ),
                    reverse=True,
                )
                ready_bundles = sum(
                    1 for row in signals
                    if bool((row.get("bundle") or {}).get("compatible"))
                    and bool((row.get("bundle") or {}).get("manifest_ready"))
                )
                active_signals = sum(
                    1 for row in signals if str(row.get("signal")) in {"LONG", "SHORT"}
                )
                waits = len(signals) - active_signals
                latest_data_age = [
                    float((row.get("decision") or {}).get("data_age_minutes"))
                    for row in signals
                    if (row.get("decision") or {}).get("data_age_minutes") is not None
                ]
                self._scan_error = None
                history_path = self.root / "logs" / "live_signal_history.jsonl"
                journal = []
                if history_path.exists():
                    for line in history_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]:
                        try:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                journal.append(obj)
                        except json.JSONDecodeError:
                            continue
                state = {
                    "ok": True,
                    "app": APP_TITLE,
                    "version": __version__,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "scan_seconds": round(time.monotonic() - started, 3),
                    "refresh_seconds": self.refresh_seconds,
                    "config": {
                        "exchange": self.settings.exchange,
                        "timeframe": self.settings.timeframe,
                        "primary_symbol": self.settings.symbol,
                        "live_symbols": configured,
                        "signal_only_mode": bool(self.settings.signal_only_mode),
                        "paper_only": bool(self.settings.paper_only),
                        "sandbox": bool(self.settings.sandbox),
                        "trade_window_required": bool(self.settings.trade_window_required_for_signal),
                    },
                    "summary": {
                        "assets_scanned": len(signals),
                        "active_signals": active_signals,
                        "waits": waits,
                        "compatible_bundles": ready_bundles,
                        "fresh_data_assets": sum(
                            1 for age in latest_data_age
                            if age <= float(getattr(self.settings, "live_max_data_age_minutes", 30.0))
                        ),
                        "terminal_ready": bool(signals) and ready_bundles > 0,
                    },
                    "signals": signals,
                    "journal": journal,
                    "outcome_update": _json_safe(outcome_update),
                    "notes": [
                        "Segnales are read-only. This terminal has no order-placement endpoint.",
                        "WAIT è l'esito predefinito quando falliscono dati, provenienza, consenso modello o gate di qualità.",
                        "Un bundle compatibile è necessario, ma non dimostra un edge di trading persistente.",
                    ],
                }
            except Exception as exc:
                self._scan_error = f"{type(exc).__name__}:{exc}"
                state = {
                    "ok": False,
                    "app": APP_TITLE,
                    "version": __version__,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "scan_seconds": round(time.monotonic() - started, 3),
                    "refresh_seconds": self.refresh_seconds,
                    "config": {
                        "exchange": self.settings.exchange,
                        "timeframe": self.settings.timeframe,
                        "primary_symbol": self.settings.symbol,
                        "live_symbols": list(getattr(self.settings, "live_symbols", ()) or ()),
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
                    "journal": [],
                    "outcome_update": {},
                    "error": self._scan_error,
                    "notes": [
                        "No signal was produced because the terminal failed chiusi.",
                        "Controlla la connettività ai dati pubblici e i bundle addestrati.",
                    ],
                }

            self._cached_state = _json_safe(state)
            self._cached_at = time.time()
            return self._cached_state

    def health(self) -> dict:
        state = self._cached_state
        return {
            "ok": True,
            "version": __version__,
            "cached": bool(state),
            "cache_age_seconds": round(time.time() - self._cached_at, 1) if state else None,
            "last_scan_ok": bool(state.get("ok")) if state else None,
        }


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="X-Content-Type-Options" content="nosniff">
<title>Adaptive AI Segnale Terminale</title>
<style>
:root {
  --bg:#070a0f; --panel:#0d121a; --panel2:#111823; --line:#1d2733;
  --text:#e8edf4; --muted:#8390a2; --green:#49df9b; --red:#ff6d87;
  --amber:#ffd166; --blue:#74b8ff; --shadow:0 16px 60px rgba(0,0,0,.28);
}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(circle at 20% 0%,#172232 0,#070a0f 36%);color:var(--text);font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:1440px;margin:0 auto;padding:28px 24px 50px}
.top{display:flex;justify-content:space-between;gap:20px;align-items:flex-end;margin-bottom:20px}
.kicker{font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:var(--muted)}
h1{font-size:34px;line-height:1.05;margin:5px 0 8px}
.sub{color:var(--muted);max-width:850px}
.actions{display:flex;gap:10px;align-items:center}
button{border:1px solid var(--line);background:var(--panel2);color:var(--text);padding:10px 14px;border-radius:11px;cursor:pointer}
button:hover{border-color:#3a4a5d}
#stamp{color:var(--muted);font-size:12px}
.grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px;margin:18px 0}
.card{background:linear-gradient(180deg,rgba(17,24,35,.95),rgba(10,14,21,.98));border:1px solid var(--line);border-radius:16px;padding:16px;box-shadow:var(--shadow)}
.label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.12em}
.value{font-size:24px;font-weight:750;margin-top:6px}
.value.good{color:var(--green)} .value.bad{color:var(--red)} .value.warn{color:var(--amber)}
.panel{background:rgba(10,14,21,.82);border:1px solid var(--line);border-radius:18px;margin-top:16px;overflow:hidden}
.panel-head{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:16px 18px;border-bottom:1px solid var(--line)}
.panel-title{font-size:16px;font-weight:700}
.small{color:var(--muted);font-size:12px}
.table-wrap{overflow:auto}
table{width:100%;border-collapse:collapse;min-width:1180px}
th,td{padding:13px 14px;border-bottom:1px solid rgba(29,39,51,.7);text-align:left;vertical-align:top}
th{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.11em;font-weight:650;position:sticky;top:0;background:#0b1017}
tr:last-child td{border-bottom:0}
.sig{font-weight:800;letter-spacing:.08em}
.signal-long{color:var(--green)} .signal-short{color:var(--red)} .signal-wait{color:var(--amber)} .signal-flat{color:var(--muted)}
.tag{display:inline-flex;align-items:center;gap:6px;padding:4px 8px;border-radius:999px;border:1px solid var(--line);font-size:11px}
.dot{width:7px;height:7px;border-radius:50%;background:currentColor}
.status-ready{color:var(--green)} .status-wait{color:var(--amber)}
.reasons{display:flex;flex-wrap:wrap;gap:5px}
.reason{padding:3px 7px;border-radius:7px;background:#171e28;color:#a8b3c2;border:1px solid #26303c;font-size:11px}
.num{font-variant-numeric:tabular-nums}
.tw{font-size:12px}
.tw strong{display:block;margin-bottom:2px}
.two{display:grid;grid-template-columns:1.25fr .75fr;gap:16px}
.cols{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;padding:16px}
.info{padding:14px;border:1px solid var(--line);border-radius:12px;background:#0d141d}
pre{white-space:pre-wrap;word-break:break-word;color:#aeb9c8;font-size:12px;margin:0}
.footer{margin-top:18px;color:var(--muted);font-size:12px}
@media(max-width:1000px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.two{grid-template-columns:1fr}.cols{grid-template-columns:1fr}.top{align-items:flex-start;flex-direction:column}}
</style>
</head>
<body>
<div class="wrap">
  <div class="top">
    <div>
      <div class="kicker">Terminalee segnali pre-alpha</div>
      <h1>Adaptive AI Segnale Terminale</h1>
      <div class="sub">Radar di mercato in sola lettura, alimentato dai bundle addestrati del repository, con controlli di provenienza e gate conservativi.</div>
    </div>
    <div class="actions">
      <span id="stamp">caricamento…</span>
      <button id="refresh">Aggiorna ora</button>
    </div>
  </div>

  <div class="grid" id="summary"></div>

  <div class="panel">
    <div class="panel-head">
      <div>
        <div class="panel-title">Radar segnali</div>
        <div class="small">La dashboard non invia ordini. Un segnale appare solo dopo il superamento degli stessi gate live/paper del motore.</div>
      </div>
      <div class="small" id="config"></div>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Asset</th><th>Segnale</th><th>Confidenza</th><th>Atteso</th><th>Intervallo robusto</th>
            <th>Score</th><th>Meta</th><th>Memoria</th><th>Regime</th><th>Durata</th><th>Dati</th><th>Perché / provenienza</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </div>
  </div>

  <div class="panel">
    <div class="panel-head"><div><div class="panel-title">Segnale Journal</div><div class="small">Solo feedback: i segnali maturati vengono risolti usando le candele pubbliche successive. Nessun ordine viene inviato.</div></div><div class="small" id="journalStatus"></div></div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Ora</th><th>Asset</th><th>Segnale</th><th>Confidenza</th><th>Atteso</th><th>Esito</th><th>Realizzato</th><th>Holding</th></tr></thead>
        <tbody id="journalRows"></tbody>
      </table>
    </div>
  </div>

  <div class="two">
    <div class="panel">
      <div class="panel-head"><div><div class="panel-title">Evidenze training e deployment</div><div class="small">Segnales are tied to asset-local trained artifacts; stale evidence fails chiusi.</div></div></div>
      <div class="cols" id="evidence"></div>
    </div>
    <div class="panel">
      <div class="panel-head"><div><div class="panel-title">Contratto operativo</div><div class="small">Progettato per osservare, non per eseguire.</div></div></div>
      <div class="cols">
        <div class="info"><div class="label">Esecuzione</div><div class="value good">OFF</div><div class="small">Questo processo non espone API per gli ordini.</div></div>
        <div class="info"><div class="label">Postura predefinita</div><div class="value warn">WAIT</div><div class="small">Dati mancanti, evidenze modello o consenso insufficiente producono WAIT.</div></div>
        <div class="info"><div class="label">Stato edge</div><div class="value warn">NON DIMOSTRATO</div><div class="small">La readiness della validazione non dimostra un edge di mercato durevole.</div></div>
      </div>
    </div>
  </div>

  <div class="footer" id="footer"></div>
</div>
<script>
const $ = (id) => document.getElementById(id);
function pct(v,d=1){ return v==null || Number.isNaN(Number(v)) ? "—" : (Number(v)*100).toFixed(d)+"%"; }
function num(v,d=3){ return v==null || Number.isNaN(Number(v)) ? "—" : Number(v).toFixed(d); }
function age(v){ return v==null || Number.isNaN(Number(v)) ? "—" : Number(v).toFixed(1)+"m"; }
function esc(v){ return String(v??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c])); }
function badge(ok,label){ return '<span class="tag '+(ok?'status-ready':'status-wait')+'"><span class="dot"></span>'+esc(label)+'</span>'; }
function _signalClass(signal){ return signal==="LONG"?"signal-long":signal==="SHORT"?"signal-short":signal==="WAIT"?"signal-wait":"signal-flat"; }

function render(data){
  const s=data.summary||{};
  $("stamp").textContent = data.generated_at ? new Date(data.generated_at).toLocaleOraString() : "—";
  $("config").textContent = data.config ? data.config.exchange+" · "+data.config.timeframe+" · "+data.config.primary_symbol : "";
  $("summary").innerHTML = [
    ["Terminale", data.ok ? "PRONTO" : "WAIT", data.ok ? "good":"bad"],
    ["Asset", s.assets_scanned??0, ""],
    ["Segnali live", s.active_signals??0, s.active_signals>0?"good":""],
    ["WAIT", s.waits??0, s.waits>0?"warn":""],
    ["Bundle compatibili", s.compatible_bundles??0, s.compatible_bundles>0?"good":"warn"],
  ].map(x=>'<div class="card"><div class="label">'+x[0]+'</div><div class="value '+x[2]+'">'+x[1]+'</div></div>').join("");

  if(!data.ok){
    $("rows").innerHTML='<tr><td colspan="12"><div class="info"><strong>Terminale failed chiusi</strong><div class="small">'+esc(data.error||"unknown error")+'</div></div></td></tr>';
    $("evidence").innerHTML="";
    $("footer").textContent=(data.notes||[]).join(" · ");
    return;
  }

  $("rows").innerHTML=(data.signals||[]).map(r=>{
    const d=r.decision||{}, b=r.bundle||{};
    const signal=r.signal||"WAIT";
    const reasons=(r.reason_codes||[]).map(x=>'<span class="reason">'+esc(x)+'</span>').join("");
    const lineage = badge(b.compatible && b.manifest_ready, b.compatible && b.manifest_ready ? "evidenza deployabile":"WAIT / provenienza");
    const duration = d.trade_window_direction ? esc(d.trade_window_direction)+" · "+pct(d.trade_window_confidence,0) : "—";
    const robust = d.expected_return_lcb==null ? "—" : num(d.expected_return_lcb,4)+" / "+num(d.expected_return_ucb,4);
    return '<tr>'+
      '<td><strong>'+esc(r.symbol)+'</strong><div class="small">'+num(r.price,2)+'</div></td>'+
      '<td><span class="sig '+esc(r.signal_class)+'">'+esc(signal)+'</span><div class="small">'+esc(r.status||"WAIT")+'</div></td>'+
      '<td class="num">'+pct(r.confidence,1)+'</td>'+
      '<td class="num">'+pct(r.expected_return,2)+'</td>'+
      '<td class="num">'+robust+'</td>'+
      '<td class="num">'+num(d.score,3)+'</td>'+
      '<td class="num">'+pct(d.meta_success,1)+'</td>'+
      '<td class="num">'+(d.analog_n==null?"—":esc(d.analog_n))+' · '+pct(d.analog_agreement,0)+'</td>'+
      '<td>'+esc(d.regime||"—")+'</td>'+
      '<td class="tw"><strong>'+duration+'</strong>'+badge(!!d.trade_window_ready, d.trade_window_ready?"PRONTO":"WAIT")+'</td>'+
      '<td class="num">'+age(d.data_age_minutes)+'</td>'+
      '<td>'+lineage+'<div class="reasons" style="margin-top:7px">'+reasons+'</div></td>'+
    '</tr>';
  }).join("");

  const journal=data.journal||[];
  $("journalRows").innerHTML=journal.length ? journal.slice().reverse().map(r=>{
    const outcome=r.outcome||"OPEN";
    const outcomeClass=outcome==="WIN"?"signal-long":outcome==="LOSS"?"signal-short":outcome==="AMBIGUOUS"?"signal-wait":"signal-flat";
    return '<tr>'+
      '<td class="small">'+esc(r.data_timestamp?new Date(r.data_timestamp).toLocaleString():"—")+'</td>'+
      '<td><strong>'+esc(r.symbol||"—")+'</strong></td>'+
      '<td><span class="sig '+esc(_signalClass(r.signal||"WAIT"))+'">'+esc(r.signal||"WAIT")+'</span></td>'+
      '<td class="num">'+pct(r.confidence,1)+'</td>'+
      '<td class="num">'+pct(r.expected_return,2)+'</td>'+
      '<td><span class="sig '+outcomeClass+'">'+esc(outcome)+'</span></td>'+
      '<td class="num">'+pct(r.realized_return,2)+'</td>'+
      '<td class="num">'+(r.holding_hours==null?"—":num(r.holding_hours,1)+"h")+'</td>'+
    '</tr>';
  }).join("") : '<tr><td colspan="8"><span class="small">Nessun segnale registrato.</span></td></tr>';
  const ou=data.outcome_update||{};
  $("journalStatus").textContent=ou.error ? "feedback non disponibile" : "aggiornati "+(ou.updated??0)+" · aperti "+(ou.open??0)+" · chiusi "+(ou.closed??0);

  $("evidence").innerHTML=(data.signals||[]).map(r=>{
    const b=r.bundle||{}, h=b.holdout||{};
    return '<div class="info">'+
      '<div class="label">'+esc(r.symbol)+'</div>'+
      '<div style="margin:7px 0">'+badge(!!b.compatible && !!b.manifest_ready, b.compatible&&b.manifest_ready?"PRONTO":"WAIT")+'</div>'+
      '<div class="small">righe training: '+esc(b.training_rows??"—")+'</div>'+
      '<div class="small">holdout: '+pct(h.total_return ?? h.net_compounded_return,2)+' · trade: '+esc(h.trades??h.trades_taken??"—")+'</div>'+
      '<div class="small">dataset: '+esc((b.data_fingerprint||"—").slice(0,12))+'</div>'+
      '<div class="small">artefatti: '+esc((b.artifact_fingerprint||"—").slice(0,12))+'</div>'+
      '<div class="small">compatibilità: '+esc(b.compatibility||"—")+'</div>'+
    '</div>';
  }).join("");

  $("footer").textContent=(data.notes||[]).join(" · ")+" · refresh "+data.refresh_seconds+"s";
}

let refreshHandle=null;
function scheduleRefresh(seconds){
  if(refreshHandle) clearInterval(refreshHandle);
  refreshHandle=setInterval(()=>refresh(false),Math.max(10,Number(seconds||45))*1000);
}

async function refresh(force=false){
  $("stamp").textContent="scansione…";
  try{
    const res=await fetch("/api/state?force="+(force?"1":"0"),{cache:"no-store"});
    const data=await res.json();
    render(data);
    scheduleRefresh(data.refresh_seconds||45);
  }catch(e){
    render({ok:false,error:String(e),summary:{},notes:["Il browser non riesce a raggiungere il terminale locale dei segnali."]});
    scheduleRefresh(45);
  }
}
$("refresh").addEventListener("click",()=>refresh(true));
refresh(false);
</script>
</body>
</html>
"""


def make_handler(terminal: SegnaleTerminale):
    class Handler(BaseHTTPRequestHandler):
        server_version = "AdaptiveSegnaleTerminale/1.0"

        def _send(self, body: bytes, status=200, content_type="text/html; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in {"/", "/index.html"}:
                self._send(HTML.encode("utf-8"))
                return
            if parsed.path == "/api/health":
                body = json.dumps(terminal.health(), separators=(",", ":")).encode("utf-8")
                self._send(body, content_type="application/json; charset=utf-8")
                return
            if parsed.path == "/api/state":
                query = parse_qs(parsed.query)
                force = query.get("force", ["0"])[0] == "1"
                body = json.dumps(terminal._terminal_state(force=force), separators=(",", ":"), allow_nan=False).encode("utf-8")
                self._send(body, content_type="application/json; charset=utf-8")
                return
            if parsed.path == "/favicon.ico":
                self._send(b"", status=204, content_type="image/x-icon")
                return
            self._send(b"Not found", status=404, content_type="text/plain; charset=utf-8")

        def log_message(self, fmt, *args):
            print(f"[signal-terminal] {self.address_string()} - {fmt % args}")

    return Handler


def parse_args():
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--config", default=None, help="Settings YAML path (defaults to repository/config.yaml)")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HTTP port")
    parser.add_argument("--refresh", type=int, default=DEFAULT_REFRESH, help="Segnale refresh TTL in seconds")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbol override")
    parser.add_argument("--no-browser", action="store_true", help="Do not aperti the dashboard automatically")
    parser.add_argument("--once", action="store_true", help="Print one JSON snapshot and exit")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = args.config or str(ROOT / "config.yaml")
    settings = load_settings(config_path)
    if args.symbols:
        settings.live_symbols = tuple(x.strip() for x in args.symbols.split(",") if x.strip())

    terminal = SegnaleTerminale(settings, ".", refresh_seconds=args.refresh)
    if args.once:
        print(json.dumps(terminal._terminal_state(force=True), indent=2, ensure_ascii=False))
        return
    server = ThreadingHTTPServer((args.host, args.port), make_handler(terminal))
    url = f"http://{args.host if args.host not in {'0.0.0.0','::'} else '127.0.0.1'}:{args.port}/"
    print(f"{APP_TITLE} v{__version__}")
    print(f"Modalità sola lettura · exchange={settings.exchange} · timeframe={settings.timeframe}")
    print(f"Dashboard: {url}")
    print("Questo processo non espone endpoint per ordini.")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArresto del terminale segnali.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
