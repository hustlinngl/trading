from __future__ import annotations
from dataclasses import dataclass,asdict
import numpy as np
import pandas as pd
@dataclass
class QualityReport:
    rows:int; missing_pct:float; duplicate_timestamps:int; non_monotonic:bool; negative_volume_rows:int; invalid_ohlc_rows:int; gap_rows:int; max_gap_bars:int; gap_ratio:float; coverage_ratio:float; stale_tail_bars:int; inferred_timeframe:str; median_bar_minutes:float; coverage_days:float; timeframe_match:bool; reasons:tuple[str,...]=(); passed:bool=False
    def to_dict(self): return asdict(self)
def timeframe_minutes(timeframe):
    text=str(timeframe or "15m").strip().lower(); units={"m":1.0,"h":60.0,"d":1440.0,"w":10080.0}
    if not text or text[-1] not in units:return 15.0
    try:return float(text[:-1] or 1)*units[text[-1]]
    except ValueError:return 15.0
def format_timeframe(minutes):
    m=float(minutes)
    if m>=10080 and abs(m/10080-round(m/10080))<1e-6:return f"{int(round(m/10080))}w"
    if m>=1440 and abs(m/1440-round(m/1440))<1e-6:return f"{int(round(m/1440))}d"
    if m>=60 and abs(m/60-round(m/60))<1e-6:return f"{int(round(m/60))}h"
    return f"{int(round(m))}m"
def infer_timeframe(df):
    if not isinstance(df.index,pd.DatetimeIndex) or len(df)<3:return "unknown",0.0
    d=pd.Series(df.index[1:]-df.index[:-1]).dt.total_seconds().div(60.0); d=d[(d>0)&np.isfinite(d)]
    if d.empty:return "unknown",0.0
    med=float(d.median()); return format_timeframe(med),med
def audit_market_data(df,timeframe="15m"):
    req=["open","high","low","close","volume"]; missing=100.0*float(df[req].isna().any(axis=1).mean()) if len(df) else 100.0
    dups=int(df.index.duplicated().sum()) if hasattr(df.index,"duplicated") else 0; nonmono=not bool(df.index.is_monotonic_increasing); neg=int((pd.to_numeric(df["volume"],errors="coerce")<0).sum())
    op,hi=pd.to_numeric(df["open"],errors="coerce"),pd.to_numeric(df["high"],errors="coerce"); lo,cl=pd.to_numeric(df["low"],errors="coerce"),pd.to_numeric(df["close"],errors="coerce")
    ex=pd.concat([op,cl],axis=1); invalid=int(((hi<ex.max(axis=1))|(lo>ex.min(axis=1))|(lo<=0)|(op<=0)|(cl<=0)).sum()); inferred,med=infer_timeframe(df); exp=timeframe_minutes(timeframe); match=bool(med>0 and abs(med-exp)<=max(.5,exp*.05))
    gap_rows=max_gap=0; gap_ratio=coverage_ratio=0.0
    if len(df)>=2 and isinstance(df.index,pd.DatetimeIndex):
        diffs=pd.Series(df.index[1:]-df.index[:-1]).dt.total_seconds().to_numpy(float)/(exp*60.0); extra=np.maximum(0.0,np.ceil(diffs)-1.0); gap_rows=int(np.sum(extra>0)); max_gap=int(np.max(extra)) if len(extra) else 0; gap_ratio=float(np.sum(extra)/max(1.0,len(df)-1))
        span=max(1.0,(df.index[-1]-df.index[0]).total_seconds()/(exp*60.0)); coverage_ratio=float(np.clip(len(df)/(span+1.0),0.0,1.0))
    stale=0
    if len(df)>1:
        tail=pd.to_numeric(df["close"].tail(min(40,len(df))),errors="coerce")
        for v in tail.diff().abs().fillna(np.nan).to_numpy()[::-1]:
            if np.isfinite(v) and float(v)==0.0:stale+=1
            else:break
    days=max(0.0,float((df.index[-1]-df.index[0]).total_seconds())/86400.0) if len(df)>=2 and isinstance(df.index,pd.DatetimeIndex) else 0.0
    reasons=[]
    if len(df)<=100:reasons.append("insufficient_rows")
    if missing>=.5:reasons.append("missing_values")
    if dups:reasons.append("duplicate_timestamps")
    if nonmono:reasons.append("non_monotonic")
    if neg:reasons.append("negative_volume")
    if invalid:reasons.append("invalid_ohlc")
    if gap_ratio>.0002:reasons.append("excessive_gap_ratio")
    if max_gap>4:reasons.append("long_data_gap")
    if not match:reasons.append("timeframe_mismatch")
    if coverage_ratio and coverage_ratio<.995:reasons.append("low_time_coverage")
    return QualityReport(len(df),missing,dups,nonmono,neg,invalid,gap_rows,max_gap,gap_ratio,coverage_ratio,stale,inferred,med,days,match,tuple(reasons),not reasons)