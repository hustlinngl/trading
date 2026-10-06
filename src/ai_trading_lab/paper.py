from __future__ import annotations

from pathlib import Path
import json, time
import pandas as pd

from .data import exchange_client, fetch_ohlcv
from .data_quality import audit_market_data
from .engine import AdaptiveEngine
from .fingerprint import strong_dataset_fingerprint
from .policy import live_signal_gate
from .trade_window import assess_trade_window
from .deployment import resolve_signal_bundle, resolve_trade_window_model, bundle_compatibility

def one_iteration(settings, root: str | Path = "."):
    root=Path(root); (root/"logs").mkdir(parents=True,exist_ok=True)
    result={"timestamp":pd.Timestamp.now(tz="UTC").isoformat(),"symbol":settings.symbol,"timeframe":settings.timeframe,"status":"WAIT","signal":"FLAT","reason":[],"paper_only":True,"sandbox":True}
    model=resolve_signal_bundle(settings,root,settings.symbol)/"signal_model.joblib"
    if not model.exists():
        result["reason"].append("model_missing")
        (root/"logs"/"paper_last.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
        return result
    try:
        ex=exchange_client(getattr(settings,"exchange","binance"),sandbox=False)
        df=fetch_ohlcv(ex,settings.symbol,settings.timeframe,int(getattr(settings,"live_lookback_bars",600)))
        quality=audit_market_data(df,settings.timeframe)
        result["data_quality"]=quality.to_dict()
        result["data_fingerprint"]=strong_dataset_fingerprint(df)
        result["price"]=float(df["close"].iloc[-1])
        if not quality.passed:
            result["reason"]=[f"data_quality:{x}" for x in quality.reasons] or ["data_quality"]
        else:
            age_minutes=max(0.0,(pd.Timestamp.now(tz="UTC")-pd.Timestamp(df.index[-1])).total_seconds()/60.0)
            result["data_age_minutes"]=age_minutes
            if age_minutes > float(getattr(settings,"live_max_data_age_minutes",30.0)):
                result["reason"]=[f"stale_data:{age_minutes:.1f}m"]
            else:
                model_dir=model.parent
                compatible, compatibility_reason=bundle_compatibility(settings,model_dir,settings.symbol)
                if not compatible:
                    result["reason"]=[compatibility_reason]
                    (root/"logs"/"paper_last.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
                    return result
                eng=AdaptiveEngine(settings).load(model_dir)
                feat=eng.features(df)
                pred=eng.predict_frame(feat)
                if pred.empty:
                    result["reason"]=["empty_prediction"]
                else:
                    last=pred.iloc[-1].copy()
                    tw_path = resolve_trade_window_model(settings,root,settings.symbol)
                    tw = assess_trade_window(df, settings, tw_path)
                    for key, value in tw.items():
                        last[key] = value
                    signal,reasons=live_signal_gate(last,settings)
                    p=float(last.get("p_up",0.5))
                    confidence=p if signal=="LONG" else (1.0-p if signal=="SHORT" else 0.0)
                    result.update({"status":"SIGNAL" if signal!="FLAT" else "WAIT","signal":signal,"confidence":confidence,"expected_return":float(last.get("expected_return",0.0)),"score":float(last.get("score",0.0)),"meta_success":float(last.get("meta_success",0.5)),"model_disagreement":float(last.get("model_disagreement",0.0)),"analog_neighbors":int(last.get("analog_n",0)),"analog_agreement":float(last.get("analog_agreement",0.0)),"reason":reasons})
    except Exception as exc:
        result["reason"].append(f"runtime:{type(exc).__name__}:{exc}")
    (root/"logs"/"paper_last.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    return result

def paper_daemon(settings,cycles=None,sleep_seconds=60,root="."):
    n=0
    while cycles is None or n<int(cycles):
        one_iteration(settings,root); n+=1
        if cycles is not None and n>=int(cycles): break
        time.sleep(max(5,int(sleep_seconds)))
