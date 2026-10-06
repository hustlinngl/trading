from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json, numpy as np, pandas as pd

from .data import exchange_client, fetch_ohlcv
from .engine import AdaptiveEngine
from .fingerprint import strong_dataset_fingerprint

@dataclass
class LiveAssessment:
    symbol:str; timestamp:str; status:str; signal:str; confidence:float; expected_return:float; price:float; reason_codes:list[str]; data_fingerprint:str
    def to_dict(self): return asdict(self)

def _model_dir(settings, root):
    return Path(root)/getattr(settings,"model_dir","models/champion")

def assess_symbol(settings,root=".",symbol=None,exchange=None):
    symbol=symbol or settings.symbol
    ex=exchange or exchange_client(getattr(settings,"exchange","binance"),sandbox=False)
    df=fetch_ohlcv(ex,symbol,settings.timeframe,int(getattr(settings,"live_lookback_bars",600)))
    stamp=df.index[-1].isoformat(); price=float(df.close.iloc[-1]); fp=strong_dataset_fingerprint(df)
    model_dir=_model_dir(settings,root)
    if not (model_dir/"signal_model.joblib").exists():
        return LiveAssessment(symbol,stamp,"WAIT","FLAT",0.0,0.0,price,["champion_missing"],fp)
    try:
        eng=AdaptiveEngine(settings); eng.load(model_dir)
        feat=eng.features(df) if hasattr(eng,"features") else df
        pred=eng.model.predict(feat).iloc[-1]
        p=float(pred.get("p_up",0.5)); er=float(pred.get("expected_return",0.0)); thr=float(getattr(settings,"probability_threshold",.57)); min_er=float(getattr(settings,"min_expected_return",.0015))
        sig="LONG" if p>=thr and er>=min_er else ("SHORT" if p<=1-thr and er<=-min_er else "FLAT")
        reasons=[] if sig!="FLAT" else ["policy_gate"]
        return LiveAssessment(symbol,stamp,"SIGNAL" if sig!="FLAT" else "WAIT",sig,max(p,1-p) if sig!="FLAT" else 0.0,er,price,reasons,fp)
    except Exception as exc:
        return LiveAssessment(symbol,stamp,"WAIT","FLAT",0.0,0.0,price,[f"runtime:{exc}"],fp)

def scan_top5(settings,root=".",symbols=None):
    ex=exchange_client(getattr(settings,"exchange","binance"),sandbox=False)
    symbols=list(symbols or getattr(settings,"live_symbols",()) or [settings.symbol])[:int(getattr(settings,"live_max_symbols",15))]
    out=[]
    for symbol in dict.fromkeys(symbols):
        try: out.append(assess_symbol(settings,root,symbol,exchange=ex))
        except Exception as exc: out.append(LiveAssessment(symbol,pd.Timestamp.now(tz="UTC").isoformat(),"WAIT","FLAT",0.0,0.0,float("nan"),[str(exc)],""))
    out.sort(key=lambda x:(x.status=="SIGNAL",x.confidence),reverse=True)
    return out[:5]

def write_live_snapshot(results,root="."):
    p=Path(root)/"logs"/"live_snapshot.json"; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps([r.to_dict() for r in results],indent=2,default=str),encoding="utf-8"); return p

HISTORY_NAME="live_signal_history.jsonl"
def append_live_signal_history(results,root="."):
    p=Path(root)/"logs"/HISTORY_NAME; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a",encoding="utf-8") as fh:
        for r in results: fh.write(json.dumps(r.to_dict(),default=str)+"\n")
    return p
