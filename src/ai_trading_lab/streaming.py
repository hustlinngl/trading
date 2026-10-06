from __future__ import annotations
import asyncio,json
from dataclasses import dataclass,asdict
from datetime import datetime,timezone
from pathlib import Path
from typing import Any,AsyncIterator
@dataclass
class StreamEvent:
    received_at:str; stream:str; event_time_ms:int; symbol:str; event_type:str; payload:dict[str,Any]
    def row(self): return asdict(self)
class BinancePublicStream:
    def __init__(self,symbol="BTC/USDT",streams=("bookTicker","aggTrade"),base_url="wss://stream.binance.com:9443/stream"):
        self.symbol=symbol.replace("/","").lower(); self.streams=tuple(x.lower() for x in streams); self.base_url=base_url
    def url(self): return self.base_url+"?streams="+"/".join(f"{self.symbol}@{s}" for s in self.streams)
    async def events(self,reconnect_seconds=3.0)->AsyncIterator[StreamEvent]:
        try: import websockets
        except ImportError as exc: raise RuntimeError("websockets is not installed; install the stream extra") from exc
        while True:
            try:
                async with websockets.connect(self.url(),ping_interval=20,ping_timeout=20,max_queue=10000) as ws:
                    async for raw in ws:
                        msg=json.loads(raw); data=msg.get("data",msg); yield StreamEvent(datetime.now(timezone.utc).isoformat(),str(msg.get("stream","")),int(data.get("E") or data.get("T") or 0),str(data.get("s",self.symbol)).upper(),str(data.get("e",data.get("stream","unknown"))),data)
            except Exception: await asyncio.sleep(max(1.0,reconnect_seconds))
async def collect_to_jsonl(collector,path,max_events=None):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); count=0
    with p.open("a",encoding="utf-8") as fh:
        async for event in collector.events():
            fh.write(json.dumps(event.row(),default=str)+"\n"); count+=1
            if max_events is not None and count>=max_events: break
    return count
