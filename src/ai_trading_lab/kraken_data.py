from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json
import requests, pandas as pd

BASE_URL="https://api.kraken.com/0/public"; MAX_ROWS=720
@dataclass(frozen=True)
class DataProvenance:
    provider:str; endpoint:str; symbol:str; pair:str; timeframe:str; requested_since:str|None; retrieved_at:str; rows:int; first_timestamp:str|None; last_timestamp:str|None; complete_history:bool; notes:str=""
    def to_dict(self): return asdict(self)

def kraken_pair(symbol):
    raw=str(symbol).upper().replace("-","/")
    aliases={"BTC/USDT":"XBTUSDT","BTC/USD":"XBTUSD","ETH/USDT":"ETHUSDT","ETH/USD":"ETHUSD","SOL/USDT":"SOLUSDT","SOL/USD":"SOLUSD"}
    return aliases.get(raw,raw.replace("/",""))

def _minutes(tf): return {"1m":1,"5m":5,"15m":15,"30m":30,"1h":60,"4h":240,"1d":1440,"1w":10080}[str(tf).lower()]

def fetch_ohlcv(symbol,timeframe="4h",limit=700,since=None):
    tf=str(timeframe).lower(); interval=_minutes(tf); limit=min(MAX_ROWS,max(1,int(limit))); pair=kraken_pair(symbol)
    params={"pair":pair,"interval":interval}
    if since is not None: params["since"]=int(pd.Timestamp(since).timestamp())
    r=requests.get(BASE_URL+"/OHLC",params=params,timeout=30); r.raise_for_status(); data=r.json()
    if data.get("error"): raise RuntimeError("; ".join(data["error"]))
    result=data.get("result",{}); key=next((k for k in result if k!="last"),None)
    if not key: raise RuntimeError("Kraken returned no OHLC series")
    rows=result[key][-limit:]; df=pd.DataFrame(rows,columns=["timestamp","open","high","low","close","vwap","volume","count"])
    df["timestamp"]=pd.to_datetime(df["timestamp"],unit="s",utc=True)
    for c in ["open","high","low","close","volume"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.set_index("timestamp")[["open","high","low","close","volume"]].dropna().sort_index()
    now=pd.Timestamp.now(tz="UTC"); delta=pd.Timedelta(minutes=interval)
    if len(df) and df.index[-1]+delta>now: df=df.iloc[:-1]
    prov=DataProvenance("kraken",BASE_URL+"/OHLC",symbol,pair,tf,str(since) if since is not None else None,datetime.now(timezone.utc).isoformat(),len(df),df.index[0].isoformat() if len(df) else None,df.index[-1].isoformat() if len(df) else None,False,"Public OHLC is bounded to 720 recent candles.")
    return df,prov

def fetch_ticker(symbol):
    pair=kraken_pair(symbol); r=requests.get(BASE_URL+"/Ticker",params={"pair":pair},timeout=20); r.raise_for_status(); data=r.json()
    if data.get("error"): raise RuntimeError("; ".join(data["error"]))
    result=data.get("result",{}); key=next(iter(result),None)
    raw=result.get(key,{}) if key else {}
    return {"provider":"kraken","symbol":symbol,"pair":pair,"last":float(raw.get("c",[0])[0]),"bid":float(raw.get("b",[0])[0]),"ask":float(raw.get("a",[0])[0]),"retrieved_at":datetime.now(timezone.utc).isoformat()}

def fingerprint_frame(df):
    payload=df.to_csv(date_format="%Y-%m-%dT%H:%M:%S%z",float_format="%.12g").encode()
    return hashlib.sha256(payload).hexdigest()

def save_provenance(provenance,path):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(provenance.to_dict(),indent=2),encoding="utf-8")
