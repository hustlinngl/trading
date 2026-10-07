from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json, numpy as np, pandas as pd

from .data import exchange_client, fetch_ohlcv, timeframe_offset
from .data_quality import audit_market_data
from .engine import AdaptiveEngine
from .fingerprint import strong_dataset_fingerprint
from .policy import live_signal_gate
from .trade_window import assess_trade_window
from .deployment import resolve_signal_bundle, resolve_trade_window_model, bundle_compatibility

@dataclass
class LiveAssessment:
    symbol:str; timestamp:str; status:str; signal:str; confidence:float; expected_return:float; price:float; reason_codes:list[str]; data_fingerprint:str
    details:dict[str, object] = field(default_factory=dict)
    def to_dict(self): return asdict(self)

def _load_bundled_history(root, symbol, timeframe, limit):
    slug = symbol.replace("/", "_").replace(":", "_")
    path = Path(root) / "data" / "historical" / f"{slug}_{timeframe}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        frame=pd.read_csv(path)
        if "timestamp" not in frame:
            return pd.DataFrame()
        frame["timestamp"]=pd.to_datetime(frame["timestamp"],utc=True)
        frame=frame.set_index("timestamp").sort_index()
        return frame.tail(max(80,int(limit)))
    except (OSError,ValueError,TypeError):
        return pd.DataFrame()


def assess_symbol(settings,root=".",symbol=None,exchange=None,*,skip_network=False):
    symbol=symbol or settings.symbol
    df=None
    data_error=None
    if exchange is not None:
        try:
            df=fetch_ohlcv(exchange,symbol,settings.timeframe,int(getattr(settings,"live_lookback_bars",600)))
        except Exception as exc:
            data_error=exc
    elif not skip_network:
        try:
            ex=exchange_client(getattr(settings,"exchange","binance"),sandbox=False)
            df=fetch_ohlcv(ex,symbol,settings.timeframe,int(getattr(settings,"live_lookback_bars",600)))
        except Exception as exc:
            data_error=exc

    if df is None or df.empty:
        df=_load_bundled_history(root,symbol,settings.timeframe,int(getattr(settings,"live_lookback_bars",600)))
    if df is None or df.empty:
        reason=f"data_fetch:{type(data_error).__name__}:{data_error}" if data_error else "empty_data"
        return LiveAssessment(symbol,"","WAIT","FLAT",0.0,0.0,0.0,[reason],"")

    stamp=df.index[-1].isoformat()
    price=float(df.close.iloc[-1])
    fp=strong_dataset_fingerprint(df)
    
    try:
        quality=audit_market_data(df, settings.timeframe)
        age_minutes=max(0.0,(pd.Timestamp.now(tz="UTC")-pd.Timestamp(df.index[-1])).total_seconds()/60.0)
        quality_reasons=[f"data_quality:{reason}" for reason in quality.reasons]
        if not quality.passed:
            return LiveAssessment(symbol,stamp,"WAIT","FLAT",0.0,0.0,price,quality_reasons or ["data_quality"],fp,{"data_quality":quality.to_dict()})
        if age_minutes > float(getattr(settings,"live_max_data_age_minutes",30.0)):
            return LiveAssessment(symbol,stamp,"WAIT","FLAT",0.0,0.0,price,[f"stale_data:{age_minutes:.1f}m"],fp,{"data_age_minutes":age_minutes})

        model_dir=resolve_signal_bundle(settings,root,symbol)
        if not (model_dir/"signal_model.joblib").exists():
            return LiveAssessment(symbol,stamp,"WAIT","FLAT",0.0,0.0,price,["model_missing"],fp,{"bundle":str(model_dir)})
        compatible, compatibility_reason=bundle_compatibility(settings,model_dir,symbol)
        if not compatible:
            return LiveAssessment(symbol,stamp,"WAIT","FLAT",0.0,0.0,price,[compatibility_reason],fp,{"bundle":str(model_dir),"compatibility":compatibility_reason})

        eng=AdaptiveEngine(settings).load(model_dir)
        feat=eng.features(df)
        pred=eng.predict_frame(feat)
        if pred.empty:
            return LiveAssessment(symbol,stamp,"WAIT","FLAT",0.0,0.0,price,["empty_prediction"],fp,{"bundle":str(model_dir)})

        last=pred.iloc[-1].copy()
        tw_path = resolve_trade_window_model(settings,root,symbol)
        tw = assess_trade_window(df, settings, tw_path, symbol=symbol)
        for key, value in tw.items():
            last[key] = value
        signal,reasons=live_signal_gate(last,settings)
        p=float(last.get("p_up",0.5))
        confidence=p if signal=="LONG" else (1.0-p if signal=="SHORT" else 0.0)
        # Rank only after the hard gate, using directional conservative edge first.
        lcb=float(last.get("expected_return_lcb", last.get("expected_return", 0.0)))
        ucb=float(last.get("expected_return_ucb", last.get("expected_return", 0.0)))
        robust_edge=lcb if signal=="LONG" else (-ucb if signal=="SHORT" else 0.0)
        score=float(last.get("score", 0.0))
        tw_conf=float(last.get("trade_window_confidence", 0.0))
        probability_floor=max(
            float(getattr(settings, "signal_confidence_threshold", 0.82)),
            float(getattr(settings, "signal_probability_threshold", 0.72)),
        )
        edge_floor=float(getattr(settings, "signal_min_expected_return", 0.003))
        score_floor=float(getattr(settings, "signal_min_score", 0.22))
        tw_floor=float(getattr(settings, "trade_window_min_confidence", 0.80))
        # Rank the margin above the active gates, not an unrelated absolute scale.
        edge_strength=max(0.0, min(1.0, (robust_edge-edge_floor)/max(edge_floor*4.0, 0.006)))
        confidence_strength=max(0.0, min(1.0, (confidence-probability_floor)/max(1.0-probability_floor, 1e-6)))
        score_strength=max(0.0, min(1.0, (score-score_floor)/max(1.0-score_floor, 1e-6)))
        tw_strength=max(0.0, min(1.0, (tw_conf-tw_floor)/max(1.0-tw_floor, 1e-6)))
        selection_score=(
            0.50 * edge_strength
            + 0.25 * confidence_strength
            + 0.15 * score_strength
            + 0.10 * tw_strength
        )
        detail_keys=(
            "p_up","expected_return","expected_return_lcb","expected_return_ucb","score",
            "meta_success","model_disagreement","return_disagreement","regime",
            "regime_persistence","analog_edge","analog_agreement","analog_dispersion","analog_n",
            "trade_window_available","trade_window_ready","trade_window_direction",
            "trade_window_confidence","trade_window_reason",
        )
        details={k:last.get(k) for k in detail_keys if k in last.index}
        details.update({
            "data_age_minutes": age_minutes,
            "bundle": str(model_dir),
            "compatibility": "ok",
            "robust_directional_edge": robust_edge,
            "selection_score": selection_score,
        })
        return LiveAssessment(
            symbol,stamp,"SIGNAL" if signal!="FLAT" else "WAIT",signal,
            confidence,float(last.get("expected_return",0.0)),price,
            (reasons + ([str(last.get("trade_window_reason"))] if tw.get("trade_window_reason") not in {None, "ok", "disabled"} and str(last.get("trade_window_reason")) else []) or quality_reasons),fp,
            details,
        )
    except Exception as exc:
        return LiveAssessment(symbol,stamp,"WAIT","FLAT",0.0,0.0,price,[f"runtime:{type(exc).__name__}:{exc}"],fp)

def discover_live_universe(settings, root=".", exchange=None, symbols=None):
    """Build the complete live universe before ranking.

    When exchange metadata is available it is authoritative: only currently active,
    supported market types are candidates. Local bundles are an eligibility layer, not
    a substitute for a missing live market. When offline, configured/local assets are
    retained as an explicit fallback universe.
    """
    configured = list(dict.fromkeys(symbols or getattr(settings, "live_symbols", ()) or [settings.symbol]))
    discovered = []
    market_counts = {}
    exchange_ready = False
    ex = exchange
    if ex is not None:
        try:
            markets = getattr(ex, "markets", {}) or {}
            allowed_types = set(getattr(settings, "live_market_types", ("spot", "swap", "future", "margin", "option")))
            if isinstance(markets, dict):
                exchange_ready = bool(markets)
            for market in markets.values():
                if not isinstance(market, dict) or market.get("active") is False:
                    continue
                symbol = str(market.get("symbol") or "").strip()
                if not symbol:
                    continue
                mtype = str(market.get("type") or ("swap" if market.get("swap") else "future" if market.get("future") else "spot"))
                if allowed_types and mtype not in allowed_types:
                    continue
                if not bool(market.get("contract")) and mtype in {"swap", "future"}:
                    continue
                discovered.append(symbol)
                market_counts[mtype] = market_counts.get(mtype, 0) + 1
        except Exception:
            exchange_ready = False

    # Offline fallback: include every locally known model/historical asset.
    roots = (Path(root) / "models" / "assets", Path(root) / "data" / "historical")
    local = []
    asset_root = roots[0]
    if asset_root.exists():
        for path in asset_root.iterdir():
            if path.is_dir():
                local.append(path.name.replace("_", "/"))
    history_root = roots[1]
    if history_root.exists():
        for path in history_root.glob("*_*.csv"):
            stem = path.stem.rsplit("_", 1)[0]
            if stem.endswith(("_USDT", "_USDC", "_FDUSD")):
                local.append(stem.replace("_", "/"))

    # An explicit symbol list is a true scope override. Online scans intersect it
    # with the authoritative active exchange universe; offline scans use it directly.
    if symbols is not None:
        requested = set(configured)
        universe_sources = (
            [symbol for symbol in discovered if symbol in requested]
            if exchange_ready
            else configured
        )
    else:
        universe_sources = discovered if exchange_ready else (configured + local)
    universe = list(dict.fromkeys(universe_sources))
    model_backed = []
    eligible = []
    for symbol in universe:
        try:
            bundle = resolve_signal_bundle(settings, root, symbol)
            if not (bundle / "signal_model.joblib").exists():
                continue
            model_backed.append(symbol)
            compatible, _ = bundle_compatibility(settings, bundle, symbol)
            if compatible:
                eligible.append(symbol)
        except Exception:
            continue

    limit = int(getattr(settings, "live_max_symbols", 0))
    if limit > 0:
        eligible = eligible[:limit]
    return {
        "symbols": eligible,
        "universe_symbols": list(universe),
        "discovered_markets": len(universe),
        "exchange_market_metadata": bool(exchange_ready),
        "model_backed_markets": len(model_backed),
        "model_eligible_markets": len(eligible),
        "ineligible_model_markets": max(0, len(model_backed) - len(eligible)),
        "uncovered_markets": max(0, len(universe) - len(model_backed)),
        "market_counts": market_counts,
    }


def scan_top5(settings, root=".", symbols=None, *, exchange=None, cache=None, return_meta=False, progress_callback=None):
    ex = exchange
    network_unavailable = ex is None
    if ex is None:
        try:
            ex = exchange_client(getattr(settings, "exchange", "binance"), sandbox=False)
            network_unavailable = False
        except Exception:
            network_unavailable = True

    universe = discover_live_universe(settings, root, ex, symbols)
    candidates = universe["symbols"]
    out = []
    now = pd.Timestamp.now(tz="UTC")
    cache = cache if cache is not None else {}
    try:
        bar_delta = timeframe_offset(settings.timeframe)
    except Exception:
        bar_delta = pd.Timedelta(0)

    reused = 0
    refreshed = 0
    waits = 0
    failures = 0
    evaluated = 0
    total_candidates = len(candidates)
    if callable(progress_callback):
        try:
            progress_callback({
                "stage": "scanning",
                "evaluated": 0,
                "total": total_candidates,
                "reused": 0,
                "refreshed": 0,
                "signals": 0,
                "waits": 0,
                "failures": 0,
            })
        except Exception:
            pass

    def emit_progress(symbol=None):
        if not callable(progress_callback):
            return
        try:
            progress_callback({
                "stage": "scanning",
                "evaluated": evaluated,
                "total": total_candidates,
                "reused": reused,
                "refreshed": refreshed,
                "signals": len(out),
                "waits": waits,
                "failures": failures,
                "symbol": symbol,
            })
        except Exception:
            pass

    def cached_assessment(symbol):
        cached = cache.get(symbol)
        if cached is None or bar_delta <= pd.Timedelta(0):
            return None
        try:
            stamp = pd.Timestamp(cached.timestamp)
            if stamp.tzinfo is None:
                stamp = stamp.tz_localize("UTC")
            return cached if stamp + bar_delta > now else None
        except Exception:
            return None

    # Fast path first: reuse every still-current assessment without touching the network
    # or model stack. Only misses enter the bounded worker pool.
    pending = []
    for symbol in candidates:
        assessment = cached_assessment(symbol)
        if assessment is not None:
            reused += 1
            evaluated += 1
            if assessment.signal in {"LONG", "SHORT"} and assessment.status == "SIGNAL":
                out.append(assessment)
            else:
                waits += 1
                if any(str(reason).startswith(("runtime:", "data_fetch:")) for reason in assessment.reason_codes):
                    failures += 1
            emit_progress(symbol)
        else:
            pending.append(symbol)

    worker_cap = max(1, min(8, int(getattr(settings, "live_scan_workers", 6))))
    workers = min(worker_cap, len(pending)) if pending else 0

    def evaluate(symbol):
        return assess_symbol(
            settings, root, symbol, exchange=ex, skip_network=network_unavailable
        )

    if workers:
        # CCXT REST calls are I/O-bound; a small bounded pool dramatically reduces
        # full-universe scan wall time while keeping request pressure finite.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="live-scan") as pool:
            futures = {pool.submit(evaluate, symbol): symbol for symbol in pending}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    assessment = future.result()
                    cache[symbol] = assessment
                    refreshed += 1
                    if assessment.signal in {"LONG", "SHORT"} and assessment.status == "SIGNAL":
                        out.append(assessment)
                    else:
                        waits += 1
                        if any(str(reason).startswith(("runtime:", "data_fetch:")) for reason in assessment.reason_codes):
                            failures += 1
                except Exception:
                    waits += 1
                    failures += 1
                finally:
                    evaluated += 1
                    emit_progress(symbol)

    out.sort(
        key=lambda x: (
            float((x.details or {}).get("selection_score", 0.0)),
            float((x.details or {}).get("robust_directional_edge", 0.0)),
            float(x.confidence),
            str(x.timestamp),
        ),
        reverse=True,
    )
    picks = out[:5]
    if return_meta:
        return picks, {
            "universe_total": int(universe["discovered_markets"]),
            "universe_model_backed": int(universe["model_backed_markets"]),
            "universe_model_eligible": int(universe.get("model_eligible_markets", len(candidates))),
            "universe_ineligible_model": int(universe.get("ineligible_model_markets", 0)),
            "universe_uncovered": int(universe.get("uncovered_markets", 0)),
            "universe_evaluated": int(len(candidates)),
            "market_symbols": list(universe.get("universe_symbols", candidates)),
            "universe_signals": int(len(out)),
            "universe_waits": int(waits),
            "assessment_failures": int(failures),
            "market_counts": universe["market_counts"],
            "exchange_market_metadata": bool(universe.get("exchange_market_metadata", ex is not None)),
            "universe_mode": "all_active_markets" if universe.get("exchange_market_metadata", ex is not None) else "local_fallback_universe",
            "assessments_reused": int(reused),
            "assessments_refreshed": int(refreshed),
            "scan_completed": int(evaluated),
            "scan_total": int(total_candidates),
            "scan_workers": int(workers or 0),
        }
    return picks

def write_live_snapshot(results,root="."):
    p=Path(root)/"logs"/"live_snapshot.json"; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps([r.to_dict() for r in results],indent=2,default=str),encoding="utf-8"); return p

HISTORY_NAME="live_signal_history.jsonl"
def append_live_signal_history(results,root="."):
    p=Path(root)/"logs"/HISTORY_NAME; p.parent.mkdir(parents=True,exist_ok=True)
    seen=set()
    if p.exists():
        for line in p.read_text(encoding="utf-8",errors="replace").splitlines()[-5000:]:
            try:
                obj=json.loads(line)
                seen.add((str(obj.get("symbol")),str(obj.get("timestamp"))))
            except json.JSONDecodeError:
                continue
    added=0
    with p.open("a",encoding="utf-8") as fh:
        for r in results:
            key=(str(r.symbol),str(r.timestamp))
            if key in seen:
                continue
            fh.write(json.dumps(r.to_dict(),default=str)+"\n")
            seen.add(key); added+=1
    return {"path":str(p),"added":int(added),"duplicates_skipped":int(len(results)-added)}
