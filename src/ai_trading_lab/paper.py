from __future__ import annotations

from pathlib import Path
import json, time
import pandas as pd

from .data import exchange_client, fetch_ohlcv
from .engine import AdaptiveEngine
from .fingerprint import strong_dataset_fingerprint

def one_iteration(settings, root: str | Path = "."):
    root=Path(root); (root/"logs").mkdir(parents=True,exist_ok=True)
    result={"timestamp":pd.Timestamp.now(tz="UTC").isoformat(),"symbol":settings.symbol,"timeframe":settings.timeframe,"status":"WAIT","reason":[],"paper_only":True,"sandbox":True}
    model=Path(getattr(settings,"model_dir","models/champion"))/"signal_model.joblib"
    if not model.exists():
        result["reason"].append("champion_missing")
        return result
    try:
        ex=exchange_client(getattr(settings,"exchange","binance"),sandbox=False)
        df=fetch_ohlcv(ex,settings.symbol,settings.timeframe,int(getattr(settings,"paper_lookback_bars",600)))
        eng=AdaptiveEngine(settings); eng.load(Path(getattr(settings,"model_dir","models/champion")))
        feat=eng.features(df) if hasattr(eng,"features") else df
        pred=eng.model.predict(feat)
        last=pred.iloc[-1]
        p=float(last.get("p_up",0.5)); er=float(last.get("expected_return",0.0))
        thr=float(getattr(settings,"probability_threshold",0.57)); min_er=float(getattr(settings,"min_expected_return",0.0015))
        if p>=thr and er>=min_er: result.update({"status":"LONG","confidence":p,"expected_return":er})
        elif p<=1-thr and er<=-min_er: result.update({"status":"SHORT","confidence":1-p,"expected_return":er})
        else: result["reason"].append("policy_gate")
        result["data_fingerprint"]=strong_dataset_fingerprint(df)
    except Exception as exc:
        result["reason"].append(str(exc))
    (root/"logs"/"paper_last.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8"); return result

def paper_daemon(settings,cycles=None,sleep_seconds=60,root="."):
    n=0
    while cycles is None or n<int(cycles):
        one_iteration(settings,root); n+=1
        if cycles is not None and n>=int(cycles): break
        time.sleep(max(5,int(sleep_seconds)))
