from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from .features import make_features

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

def train_trade_window_backbone(df,settings,holdout_frac=.15,save_path=None):
    base_minutes=float(getattr(settings,"base_bar_minutes",15)); min_hours=float(getattr(settings,"trade_window_min_hours",3.0)); max_hours=float(getattr(settings,"trade_window_max_hours",24.0))
    min_bars=max(1,int(round(min_hours*60/base_minutes))); max_bars=max(min_bars,int(round(max_hours*60/base_minutes)))
    x,_,_=make_features(df,max_bars,external_feature_lag_bars=getattr(settings,"external_feature_lag_bars",1)); lab=_barrier_labels(df,max_bars,min_bars)
    common=x.index.intersection(lab.index); x=x.loc[common]; lab=lab.loc[common]; mask=lab["direction"]!=0
    x=x.loc[mask]; y=lab.loc[mask,"direction"].map({-1:0,1:1}).astype(int)
    if len(x)<80: return {"production_ready":False,"reason":"insufficient_labeled_rows","rows":int(len(x))}
    cut=max(40,min(len(x)-20,int(len(x)*(1-holdout_frac))))
    model=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=12,max_features="sqrt",random_state=int(settings.seed),n_jobs=-1,class_weight="balanced"); model.fit(x.iloc[:cut],y.iloc[:cut])
    p=model.predict_proba(x.iloc[cut:])[:,1]; active=(p>=.62)|(p<=.38); pred=(p>=.62).astype(int); truth=y.iloc[cut:].to_numpy()
    precision=float(np.mean(pred[active]==truth[active])) if active.any() else 0.0; support=int(active.sum())
    ready=precision>=float(getattr(settings,"trade_window_min_precision",.60)) and support>=int(getattr(settings,"trade_window_min_holdout_trades",12))
    report={"production_ready":bool(ready),"holdout_precision":precision,"holdout_signals":support,"rows":int(len(x)),"min_hours":min_hours,"max_hours":max_hours}
    if save_path is not None:
        pth=Path(save_path); pth.parent.mkdir(parents=True,exist_ok=True); joblib.dump({"model":model,"feature_columns":list(x.columns),"report":report},pth)
    return report
