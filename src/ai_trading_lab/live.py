from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json, numpy as np, pandas as pd

from .data import exchange_client, fetch_ohlcv
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
    path=Path(root)/"data"/"historical"/f"{symbol.replace("/", "_").replace(":", "_")}_{timeframe}.csv"
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
        detail_keys=(
            "p_up","expected_return","expected_return_lcb","expected_return_ucb","score",
            "meta_success","model_disagreement","return_disagreement","regime",
            "regime_persistence","analog_edge","analog_agreement","analog_dispersion","analog_n",
            "trade_window_available","trade_window_ready","trade_window_direction",
            "trade_window_confidence","trade_window_reason",
        )
        details={k:last.get(k) for k in detail_keys if k in last.index}
        details.update({"data_age_minutes":age_minutes,"bundle":str(model_dir),"compatibility":"ok"})
        return LiveAssessment(
            symbol,stamp,"SIGNAL" if signal!="FLAT" else "WAIT",signal,
            confidence,float(last.get("expected_return",0.0)),price,
            (reasons + ([str(last.get("trade_window_reason"))] if tw.get("trade_window_reason") not in {None, "ok", "disabled"} and str(last.get("trade_window_reason")) else []) or quality_reasons),fp,
            details,
        )
    except Exception as exc:
        return LiveAssessment(symbol,stamp,"WAIT","FLAT",0.0,0.0,price,[f"runtime:{type(exc).__name__}:{exc}"],fp)

def scan_top5(settings,root=".",symbols=None):
    ex=None
    network_unavailable=False
    try:
        ex=exchange_client(getattr(settings,"exchange","binance"),sandbox=False)
    except Exception:
        network_unavailable=True

    configured=list(symbols or getattr(settings,"live_symbols",()) or [settings.symbol])
    asset_root=Path(root)/"models"/"assets"
    bundled_root=Path(root)/"data"/"historical"
    discovered=[]
    if asset_root.exists():
        for path in asset_root.iterdir():
            if path.is_dir():
                name=path.name.replace("_","/")
                if name.endswith(("/USDT","/USDC","/FDUSD")):
                    discovered.append(name)
    if bundled_root.exists():
        for path in bundled_root.glob("*_*.csv"):
            stem=path.stem.rsplit("_",1)[0]
            if stem.endswith(("_USDT","_USDC","_FDUSD")):
                discovered.append(stem.replace("_","/"))
    universe=list(dict.fromkeys(configured+discovered))
    max_symbols=max(5,int(getattr(settings,"live_max_symbols",15)))
    out=[]
    for symbol in universe[:max_symbols]:
        try:
            assessment=assess_symbol(
                settings,root,symbol,exchange=ex,skip_network=network_unavailable
            )
            if assessment.signal in {"LONG","SHORT"} and assessment.status=="SIGNAL":
                out.append(assessment)
        except Exception:
            continue
    out.sort(
        key=lambda x: (
            float(x.confidence),
            float(x.expected_return),
            str(x.timestamp),
        ),
        reverse=True,
    )
    return out[:5]


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
