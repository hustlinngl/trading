from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from .data import exchange_client, fetch_ohlcv
from .trade_window import _atr, timeframe_minutes

HISTORY_NAME="live_signal_history.jsonl"

def _load_history(path:Path)->list[dict[str,Any]]:
    if not path.exists(): return []
    rows=[]
    for line in path.read_text(encoding="utf-8",errors="replace").splitlines():
        try:
            obj=json.loads(line); rows.append(obj) if isinstance(obj,dict) else None
        except json.JSONDecodeError: pass
    return rows

def _write_history(path:Path,rows): 
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text("".join(json.dumps(r,default=str)+"\n" for r in rows),encoding="utf-8"); tmp.replace(path)

def _resolve_result(df,signal,timeframe,pt_atr,sl_atr,fee_bps,slippage_bps,max_bars):
    stamp=pd.Timestamp(signal.get("data_timestamp","")); stamp=stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    idx=df.index.searchsorted(stamp)
    if idx>=len(df) or pd.Timestamp(df.index[idx])!=stamp or idx+1>=len(df): return None
    entry=float(df["open"].iloc[idx+1]); atr=float(_atr(df).iloc[idx])
    if not np.isfinite(atr) or atr<=0: return None
    side=str(signal.get("signal","")); 
    if side not in {"LONG","SHORT"}: return None
    upper=entry+pt_atr*atr; lower=entry-sl_atr*atr; end_i=min(len(df)-1,idx+1+max_bars); cost=2*(fee_bps+slippage_bps)/10000.0; min_bars=int(signal.get("_min_bars",1) or 1)
    for j in range(idx+1,end_i+1):
        up=float(df["high"].iloc[j])>=upper; down=float(df["low"].iloc[j])<=lower; held=j-(idx+1)+1
        if up and down: return {"outcome":"AMBIGUOUS","realized_return":0.0,"holding_hours":held*timeframe_minutes(timeframe)/60}
        if side=="LONG" and up: return {"outcome":"WIN" if held>=min_bars else "EARLY","realized_return":upper/entry-1-cost,"holding_hours":held*timeframe_minutes(timeframe)/60}
        if side=="LONG" and down: return {"outcome":"LOSS" if held>=min_bars else "EARLY","realized_return":lower/entry-1-cost,"holding_hours":held*timeframe_minutes(timeframe)/60}
        if side=="SHORT" and down: return {"outcome":"WIN" if held>=min_bars else "EARLY","realized_return":entry/lower-1-cost,"holding_hours":held*timeframe_minutes(timeframe)/60}
        if side=="SHORT" and up: return {"outcome":"LOSS" if held>=min_bars else "EARLY","realized_return":entry/upper-1-cost,"holding_hours":held*timeframe_minutes(timeframe)/60}
    if end_i>=len(df)-1: return None
    exit_price=float(df["close"].iloc[end_i]); ret=(exit_price/entry-1 if side=="LONG" else entry/exit_price-1)-cost
    return {"outcome":"TIMEOUT","realized_return":ret,"holding_hours":(end_i-(idx+1)+1)*timeframe_minutes(timeframe)/60}

def update_live_signal_outcomes(settings,root=".",exchange=None,max_records=500):
    root=Path(root); path=root/"logs"/HISTORY_NAME; rows=_load_history(path)
    open_rows=[r for r in rows if r.get("status")=="SIGNAL" and not r.get("outcome")]
    if not open_rows: return {"updated":0,"open":0,"closed":sum(bool(r.get("outcome")) for r in rows)}
    ex=exchange or exchange_client(getattr(settings,"exchange","binance"),sandbox=False); updated=0
    for rec in open_rows:
        try:
            df=fetch_ohlcv(ex,str(rec["symbol"]),settings.timeframe,max(300,int(getattr(settings,"live_lookback_bars",600))))
            df=df.sort_index(); rec["_min_bars"]=max(1,int(round(float(getattr(settings,"trade_window_min_hours",3))*60/timeframe_minutes(settings.timeframe))))
            resolved=_resolve_result(df,rec,settings.timeframe,float(getattr(settings,"trade_window_pt_atr",1.25)),float(getattr(settings,"trade_window_sl_atr",.90)),float(getattr(settings,"fee_bps",7)),float(getattr(settings,"slippage_bps",5)),int(round(float(getattr(settings,"trade_window_max_hours",24))*60/timeframe_minutes(settings.timeframe))))
            if resolved:
                rec.update(resolved); rec.pop("_min_bars",None); rec["resolved_at"]=pd.Timestamp.now(tz="UTC").isoformat(); stamp=pd.Timestamp(rec["data_timestamp"]); stamp=stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC"); ei=df.index.searchsorted(stamp)+1
                if ei<len(df): rec["entry_timestamp"]=str(df.index[ei]); rec["entry_price"]=float(df["open"].iloc[ei])
                updated+=1
        except Exception as exc: rec.pop("_min_bars",None); rec["tracking_error"]=str(exc)
    for rec in rows: rec.pop("_min_bars",None)
    rows=rows[-max_records:]; _write_history(path,rows)
    return {"updated":updated,"open":sum(1 for r in rows if r.get("status")=="SIGNAL" and not r.get("outcome")),"closed":sum(bool(r.get("outcome")) for r in rows)}
