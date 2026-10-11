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
from .deployment import resolve_signal_bundle, resolve_trade_window_model

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
                try:
                    bundle=InferenceBundle.load(settings,root,settings.symbol)
                except InferenceBundleError as exc:
                    result["reason"]=[exc.reason]
                    (root/"logs"/"paper_last.json").write_text(
                        json.dumps(result,indent=2,default=str),encoding="utf-8"
                    )
                    return result
                feat = bundle.features(df)
                strict_pred = bundle.predict_frame(feat, strict=True, profile="strict")
                research_pred = bundle.predict_frame(feat, strict=True, profile="research")
                if strict_pred.empty or research_pred.empty:
                    result["reason"] = ["empty_prediction"]
                else:
                    tw_path = resolve_trade_window_model(settings, root, settings.symbol)
                    tw = assess_trade_window(df, settings, tw_path)
                    market_type = str(funding.get("market_type", "spot") or "spot")
                    funding_cost = float(funding.get("funding_cost_return", 0.0) or 0.0)
                    funding_missing = bool(funding.get("funding_data_missing", False))

                    def enrich(predictions):
                        row = predictions.iloc[-1].copy()
                        for key, value in tw.items():
                            row[key] = value
                        row["funding_cost_return"] = funding_cost
                        row["funding_data_missing"] = funding_missing
                        row["market_type"] = market_type
                        return row

                    strict_row = enrich(strict_pred)
                    research_row = enrich(research_pred)
                    strict_signal, strict_reasons = live_signal_gate(
                        strict_row, settings, profile="strict"
                    )
                    research_signal, research_reasons = live_signal_gate(
                        research_row, settings, profile="research"
                    )
                    configured_profile = str(
                        getattr(settings, "paper_signal_profile", "strict")
                    ).strip().lower()
                    if configured_profile not in {"strict", "research"}:
                        configured_profile = "strict"
                        result["reason"] = ["invalid_paper_signal_profile"]
                        selected_signal, selected_reasons, last = "FLAT", ["invalid_paper_signal_profile"], strict_row
                    elif configured_profile == "research":
                        selected_signal, selected_reasons, last = research_signal, research_reasons, research_row
                    else:
                        selected_signal, selected_reasons, last = strict_signal, strict_reasons, strict_row

                    selected_short = selected_signal == "SHORT"
                    side_prefix = "short_" if selected_short else ""
                    confidence = (
                        float(last.get("p_short", 0.0)) if selected_short
                        else float(last.get("p_up", 0.5)) if selected_signal == "LONG"
                        else 0.0
                    )
                    result.update({
                        "status": "SIGNAL" if selected_signal != "FLAT" else "WAIT",
                        "signal": selected_signal,
                        "confidence": confidence,
                        "selected_profile": configured_profile,
                        "probability_baseline": float(last.get(
                            "short_probability_baseline" if selected_short else "probability_baseline", 0.5
                        )),
                        "p_up": float(last.get("p_up", 0.5)),
                        "p_short": float(last.get("p_short", 0.0)),
                        "expected_return": float(last.get(f"{side_prefix}expected_return", 0.0)),
                        "score": float(last.get(f"{side_prefix}score", last.get("score", 0.0))),
                        "meta_success": float(last.get(f"{side_prefix}meta_success", 0.5)),
                        "model_disagreement": float(last.get(f"{side_prefix}model_disagreement", 0.0)),
                        "analog_neighbors": int(last.get(f"{side_prefix}analog_n", 0)),
                        "analog_agreement": float(last.get(f"{side_prefix}analog_agreement", 0.0)),
                        "reason": selected_reasons,
                        "policy_comparison": {
                            "strict": {
                                "signal": strict_signal,
                                "reasons": strict_reasons,
                                "policy_diagnostics": strict_pred.attrs.get("policy_diagnostics", {}),
                            },
                            "research": {
                                "signal": research_signal,
                                "reasons": research_reasons,
                                "policy_diagnostics": research_pred.attrs.get("policy_diagnostics", {}),
                            },
                            "publication_note": "Research profile is paper-only; the live terminal always uses strict gates.",
                        },
                        "trade_window": {
                            key: value for key, value in tw.items()
                        },
                        "funding": {
                            "market_type": market_type,
                            "funding_data_missing": funding_missing,
                            "funding_cost_return": funding_cost,
                        },
                    })
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
