from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from .features import make_features
from .fingerprint import strong_dataset_fingerprint

def timeframe_minutes(timeframe):
    tf=str(timeframe).lower()
    return {"1m":1,"3m":3,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440}.get(tf,15)

def _atr(df,period=14):
    high=df["high"].astype(float); low=df["low"].astype(float); close=df["close"].astype(float)
    prev=close.shift(1); tr=pd.concat([(high-low),(high-prev).abs(),(low-prev).abs()],axis=1).max(axis=1)
    return tr.rolling(period,min_periods=period).mean()

def _barrier_labels(df,horizon_bars=96,min_bars=12,pt_atr=1.25,sl_atr=.90):
    x=df.copy(); x["atr_14"]=_atr(x); rows=[]
    for i in range(len(x)-horizon_bars-1):
        atr=float(x["atr_14"].iloc[i]) if np.isfinite(x["atr_14"].iloc[i]) else np.nan
        if not np.isfinite(atr) or atr<=0: rows.append((0,0)); continue
        entry=float(x["open"].iloc[i+1]); up=entry+pt_atr*atr; dn=entry-sl_atr*atr; out=0
        for j in range(i+1,min(len(x),i+horizon_bars+1)):
            hu=float(x["high"].iloc[j])>=up; hd=float(x["low"].iloc[j])<=dn; held=j-i
            if hu and hd: out=0; break
            if hu: out=1 if held>=min_bars else 0; break
            if hd: out=-1 if held>=min_bars else 0; break
        rows.append((out,abs(out)))
    return pd.DataFrame(rows,index=x.index[:len(rows)],columns=["direction","margin"])

def _wilson_lower_bound(successes, total, z=1.6448536269514722):
    n=int(total)
    if n<=0: return 0.0
    phat=float(successes)/n
    denom=1.0+(z*z)/n
    centre=phat+(z*z)/(2*n)
    radius=z*np.sqrt(max(0.0,phat*(1.0-phat)/n+(z*z)/(4*n*n)))
    return float((centre-radius)/denom)


def train_trade_window_backbone(df,settings,holdout_frac=.15,save_path=None):
    base_minutes=float(getattr(settings,"base_bar_minutes",timeframe_minutes(getattr(settings,"timeframe","15m"))))
    min_hours=float(getattr(settings,"trade_window_min_hours",3.0))
    max_hours=float(getattr(settings,"trade_window_max_hours",24.0))
    min_bars=max(1,int(round(min_hours*60/base_minutes)))
    max_bars=max(min_bars,int(round(max_hours*60/base_minutes)))
    x,_,_=make_features(df,max_bars,external_feature_lag_bars=getattr(settings,"external_feature_lag_bars",1))
    lab=_barrier_labels(df,max_bars,min_bars)
    common=x.index.intersection(lab.index)
    x=x.loc[common]; lab=lab.loc[common]
    mask=lab["direction"]!=0
    x=x.loc[mask]; y=lab.loc[mask,"direction"].map({-1:0,1:1}).astype(int)
    if len(x)<80:
        return {"production_ready":False,"reason":"insufficient_labeled_rows","rows":int(len(x))}
    original_positions=df.index.get_indexer(x.index)
    split_pos=int(len(df)*(1-float(holdout_frac)))
    train_mask=(original_positions>=0) & ((original_positions+max_bars)<split_pos)
    holdout_mask=(original_positions>=split_pos)
    train_x=x.loc[train_mask]
    train_y=y.loc[train_mask]
    holdout_x=x.loc[holdout_mask]
    holdout_y=y.loc[holdout_mask]
    if len(train_x)<40 or len(holdout_x)<20:
        return {"production_ready":False,"reason":"insufficient_purged_split","rows":int(len(x)),"train_rows":int(len(train_x)),"holdout_rows":int(len(holdout_x))}
    model=ExtraTreesClassifier(
        n_estimators=300,min_samples_leaf=12,max_features="sqrt",
        random_state=int(settings.seed),n_jobs=-1,class_weight="balanced"
    )
    model.fit(train_x,train_y)
    p=model.predict_proba(holdout_x)[:,1]
    active=(p>=.62)|(p<=.38)
    pred=(p>=.62).astype(int)
    truth=holdout_y.to_numpy()
    successes=int(np.sum(pred[active]==truth[active])) if active.any() else 0
    precision=float(successes/max(1,int(active.sum()))) if active.any() else 0.0
    support=int(active.sum())
    wilson=_wilson_lower_bound(successes,support)

    # Economic holdout: direction is evaluated on the realized close return over the
    # full specialist horizon, net of conservative round-trip costs.
    raw_cost_bps=2.0*(float(getattr(settings,"fee_bps",0.0))+float(getattr(settings,"slippage_bps",0.0)))
    raw_cost_bps += 2.0*float(getattr(settings,"impact_bps_per_sqrt",0.0))*np.sqrt(float(np.clip(getattr(settings,"max_participation_pct",0.10),0.0,1.0)))
    net_returns=[]
    positions=df.index.get_indexer(holdout_x.index)
    for local_i, is_active in enumerate(active):
        if not bool(is_active): continue
        pos=int(positions[local_i])
        if pos<0 or pos+1>=len(df): continue
        exit_pos=min(len(df)-1,pos+max_bars)
        entry=float(df["open"].iloc[pos+1]); exit_price=float(df["close"].iloc[exit_pos])
        direction=1.0 if int(pred[local_i])==1 else -1.0
        gross=direction*(exit_price/entry-1.0)
        net_returns.append(gross-raw_cost_bps/10000.0)
    net_arr=np.asarray(net_returns,dtype=float)
    mean_net=float(net_arr.mean()) if len(net_arr) else 0.0
    compounded_net=float(np.prod(1.0+net_arr)-1.0) if len(net_arr) and np.all(net_arr>-1.0) else -1.0
    target_precision=float(getattr(settings,"trade_window_target_precision",0.80))
    min_wilson=float(getattr(settings,"trade_window_min_holdout_wilson",0.60))
    min_net=float(getattr(settings,"trade_window_min_net_return",0.0005))
    ready=(
        precision>=target_precision
        and wilson>=min_wilson
        and support>=int(getattr(settings,"trade_window_min_holdout_trades",12))
        and len(net_returns)>=int(getattr(settings,"trade_window_min_holdout_trades",12))
        and mean_net>=min_net
        and (compounded_net>0.0 if bool(getattr(settings,"trade_window_require_positive_holdout_backtest",True)) else True)
    )
    report={
        "production_ready":bool(ready),
        "holdout_precision":precision,
        "holdout_wilson_lower":wilson,
        "holdout_signals":support,
        "holdout_net_return_mean":mean_net,
        "holdout_net_return_compounded":compounded_net,
        "holdout_economic_observations":int(len(net_returns)),
        "target_precision":target_precision,
        "min_wilson":min_wilson,
        "min_net_return":min_net,
        "rows":int(len(x)),
        "train_rows":int(len(train_x)),
        "holdout_rows":int(len(holdout_x)),
        "min_hours":min_hours,
        "max_hours":max_hours,
        "symbol":str(getattr(settings,"symbol","")),
        "timeframe":str(getattr(settings,"timeframe","15m")),
        "data_fingerprint":strong_dataset_fingerprint(df),
        "model_semantics_fingerprint":__import__("ai_trading_lab.deployment", fromlist=["model_semantics_fingerprint"]).model_semantics_fingerprint(settings),
        "deployment_semantics_fingerprint":__import__("ai_trading_lab.deployment", fromlist=["deployment_semantics_fingerprint"]).deployment_semantics_fingerprint(settings),
        "trained_at":pd.Timestamp.now(tz="UTC").isoformat(),
    }
    if save_path is not None:
        pth=Path(save_path); pth.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump({"model":model,"feature_columns":list(x.columns),"report":report},pth)
    return report


def assess_trade_window(df, settings, model_path=None, symbol=None):
    """Evaluate the persisted 3–24h specialist as a final direction/quality verifier."""
    path = Path(model_path or getattr(settings, "trade_window_model_path", "models/champion/trade_window_specialist.joblib"))
    out = {
        "trade_window_available": False,
        "trade_window_ready": False,
        "trade_window_direction": "FLAT",
        "trade_window_confidence": 0.0,
        "trade_window_reason": "disabled",
    }
    if not bool(getattr(settings, "trade_window_enabled", True)):
        return out
    if not path.exists():
        out["trade_window_reason"] = "artifact_missing"
        return out
    try:
        artifact = joblib.load(path)
        model = artifact["model"]
        feature_columns = list(artifact.get("feature_columns", []))
        report = artifact.get("report", {}) or {}
        expected_symbol = str(symbol or getattr(settings, "symbol", ""))
        artifact_symbol = report.get("symbol")
        artifact_timeframe = report.get("timeframe")
        if artifact_symbol and str(artifact_symbol) != expected_symbol:
            out["trade_window_reason"] = "symbol_mismatch"
            return out
        if artifact_timeframe and str(artifact_timeframe) != str(getattr(settings, "timeframe", "15m")):
            out["trade_window_reason"] = "timeframe_mismatch"
            return out
        base_minutes = float(getattr(settings, "base_bar_minutes", timeframe_minutes(getattr(settings, "timeframe", "15m"))))
        max_bars = max(1, int(round(float(getattr(settings, "trade_window_max_hours", 24.0)) * 60.0 / base_minutes)))
        features, _, _ = make_features(
            df,
            max_bars,
            external_feature_lag_bars=getattr(settings, "external_feature_lag_bars", 1),
        )
        x = features.reindex(columns=feature_columns).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        if x.empty:
            out["trade_window_reason"] = "empty_features"
            return out
        proba = model.predict_proba(x.iloc[[-1]])[0]
        p_up = float(proba[1]) if len(proba) > 1 else float(proba[0])
        direction = "LONG" if p_up >= 0.5 else "SHORT"
        confidence = max(p_up, 1.0 - p_up)
        min_conf = float(getattr(settings, "trade_window_min_confidence", 0.80))
        production_ready = bool(report.get("production_ready", False))
        active = confidence >= min_conf
        out.update({
            "trade_window_available": True,
            "trade_window_ready": bool(production_ready and active),
            "trade_window_direction": direction if active else "FLAT",
            "trade_window_confidence": confidence,
            "trade_window_report_precision": float(report.get("holdout_precision", 0.0) or 0.0),
            "trade_window_reason": "ok" if production_ready and active else (
                "not_production_ready" if not production_ready else "below_confidence"
            ),
        })
        return out
    except Exception as exc:
        out["trade_window_reason"] = f"{type(exc).__name__}:{exc}"
        return out
