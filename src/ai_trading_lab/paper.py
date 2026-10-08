from __future__ import annotations

from pathlib import Path
import json, time
import pandas as pd

from .data import exchange_client, fetch_ohlcv
from .data_quality import audit_market_data
from .inference import InferenceBundle, InferenceBundleError
from .fingerprint import strong_dataset_fingerprint
from .policy import live_signal_gate
from .trade_window import assess_trade_window
from .deployment import resolve_signal_bundle, resolve_trade_window_model, bundle_compatibility

def one_iteration(settings, root: str | Path = "."):
    root=Path(root); (root/"logs").mkdir(parents=True,exist_ok=True)
    result={"timestamp":pd.Timestamp.now(tz="UTC").isoformat(),"symbol":settings.symbol,"timeframe":settings.timeframe,"status":"WAIT","signal":"FLAT","reason":[],"paper_only":True,"sandbox":True}
    previous_path=root/"logs"/"paper_last.json"
    previous={}
    if previous_path.exists():
        try:
            previous=json.loads(previous_path.read_text(encoding="utf-8"))
        except Exception:
            previous={}
    model=resolve_signal_bundle(settings,root,settings.symbol)/"signal_model.joblib"
    if not model.exists():
        result["reason"].append("model_missing")
        (root/"logs"/"paper_last.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
        return result
    try:
        ex=exchange_client(getattr(settings,"exchange","binance"),sandbox=False)
        df=fetch_ohlcv(ex,settings.symbol,settings.timeframe,int(getattr(settings,"live_lookback_bars",600)))
        quality=audit_market_data(df,settings.timeframe)
        result["data_timestamp"]=df.index[-1].isoformat()
        result["data_quality"]=quality.to_dict()
        if str(previous.get("data_timestamp","")) == str(result["data_timestamp"]):
            result["reason"]=["same_completed_candle"]
            previous_path.write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
            return result
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
                from .live import _funding_snapshot
                funding = _funding_snapshot(ex, settings.symbol, settings)
                if (
                    funding["funding_data_missing"]
                    and bool(getattr(settings, "require_funding_data_for_derivatives", True))
                    and funding["market_type"] in {"swap", "future", "perpetual", "unknown"}
                ):
                    result["reason"]=["funding_data_missing"]
                    result["market_type"]=funding["market_type"]
                    result["funding_error"]=funding.get("funding_error")
                    previous_path.write_text(
                        json.dumps(result,indent=2,default=str),encoding="utf-8"
                    )
                    return result
                model_dir=model.parent
                try:
                    bundle=InferenceBundle.load(settings,root,settings.symbol)
                except InferenceBundleError as exc:
                    result["reason"]=[exc.reason]
                    (root/"logs"/"paper_last.json").write_text(
                        json.dumps(result,indent=2,default=str),encoding="utf-8"
                    )
                    return result
                feat,pred=bundle.predict(df,strict=True)
                if pred.empty:
                    result["reason"]=["empty_prediction"]
                else:
                    last=pred.iloc[-1].copy()
                    tw_path = resolve_trade_window_model(settings,root,settings.symbol)
                    tw = assess_trade_window(df, settings, tw_path)
                    for key, value in tw.items():
                        last[key] = value
                    last["funding_cost_return"] = float(
                        funding.get("funding_cost_return", 0.0) or 0.0
                    )
                    last["funding_data_missing"] = bool(
                        funding.get("funding_data_missing", False)
                    )
                    last["market_type"] = funding.get("market_type", "spot")
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
