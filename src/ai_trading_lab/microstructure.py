from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
import pandas as pd

@dataclass
class OrderBookFeatures:
    timestamp: str
    mid: float
    spread_bps: float
    microprice_gap_bps: float
    imbalance_5: float
    imbalance_20: float
    imbalance_50: float
    bid_depth_50: float
    ask_depth_50: float
    depth_ratio_50: float
    impact_buy_10k_bps: float
    impact_sell_10k_bps: float
    top_bid_qty: float
    top_ask_qty: float
    def to_dict(self) -> dict[str, Any]: return asdict(self)

def _weighted_fill(levels: list[list[float]], notional: float, side: str, mid: float) -> float:
    remaining=max(float(notional),0.0)
    if remaining<=0 or mid<=0: return 0.0
    paid=0.0; filled_qty=0.0
    for price,qty in levels:
        price,qty=float(price),float(qty)
        if price<=0 or qty<=0: continue
        take_qty=min(qty,remaining/price); paid+=take_qty*price; filled_qty+=take_qty; remaining-=take_qty*price
        if remaining<=1e-9: break
    if filled_qty<=0 or remaining>1e-9: return float("nan")
    vwap=paid/filled_qty
    return max(0.0,(vwap/mid-1.0)*1e4) if side=="buy" else max(0.0,(1.0-vwap/mid)*1e4)

def snapshot_orderbook(exchange,symbol:str,limit:int=50,impact_notional:float=10_000.0)->OrderBookFeatures:
    book=exchange.fetch_order_book(symbol,limit=limit)
    bids=[(float(p),float(q)) for p,q in (book.get("bids") or []) if float(q)>0]
    asks=[(float(p),float(q)) for p,q in (book.get("asks") or []) if float(q)>0]
    if not bids or not asks: raise RuntimeError(f"Empty order book for {symbol}")
    best_bid,best_ask=bids[0][0],asks[0][0]; mid=(best_bid+best_ask)/2.0
    spread_bps=(best_ask-best_bid)/mid*1e4; bid_qty,ask_qty=bids[0][1],asks[0][1]; denom=bid_qty+ask_qty
    microprice=(best_ask*bid_qty+best_bid*ask_qty)/denom if denom else mid
    def imbalance(n:int)->float:
        b=sum(q for _,q in bids[:n]); a=sum(q for _,q in asks[:n]); d=b+a
        return float((b-a)/d) if d else 0.0
    bid_depth=sum(q for _,q in bids[:50]); ask_depth=sum(q for _,q in asks[:50])
    depth_ratio=bid_depth/max(ask_depth,1e-12); ts=book.get("timestamp") or pd.Timestamp.utcnow().value//10**6
    return OrderBookFeatures(pd.to_datetime(ts,unit="ms",utc=True).isoformat(),float(mid),float(spread_bps),
        float((microprice/mid-1.0)*1e4),imbalance(5),imbalance(20),imbalance(50),float(bid_depth),float(ask_depth),
        float(depth_ratio),float(_weighted_fill(asks,impact_notional,"buy",mid)),float(_weighted_fill(bids,impact_notional,"sell",mid)),
        float(bid_qty),float(ask_qty))

def trade_tape_features(exchange,symbol:str,limit:int=500)->dict[str,float]:
    try: trades=exchange.fetch_trades(symbol,limit=limit)
    except Exception: return {"tape_buy_share":np.nan,"tape_imbalance":np.nan,"tape_vwap_gap_bps":np.nan,"tape_count":0.0,"tape_notional":0.0}
    if not trades: return {"tape_buy_share":np.nan,"tape_imbalance":np.nan,"tape_vwap_gap_bps":np.nan,"tape_count":0.0,"tape_notional":0.0}
    rows=[]
    for t in trades:
        price=pd.to_numeric(pd.Series([t.get("price")]),errors="coerce").iloc[0]; amount=pd.to_numeric(pd.Series([t.get("amount")]),errors="coerce").iloc[0]
        if not np.isfinite(price) or not np.isfinite(amount) or amount<=0: continue
        side=str(t.get("side") or "").lower(); sign=1.0 if side=="buy" else (-1.0 if side=="sell" else 0.0); rows.append((float(price),float(amount),sign))
    if not rows: return {"tape_buy_share":np.nan,"tape_imbalance":np.nan,"tape_vwap_gap_bps":np.nan,"tape_count":0.0,"tape_notional":0.0}
    a=np.asarray(rows,float); notionals=a[:,0]*a[:,1]; signed=notionals*a[:,2]; total=notionals.sum()
    buy_share=notionals[a[:,2]>0].sum()/total if total else 0.5; vwap=(a[:,0]*a[:,1]).sum()/max(a[:,1].sum(),1e-12)
    return {"tape_buy_share":float(buy_share),"tape_imbalance":float(signed.sum()/max(total,1e-12)),"tape_vwap_gap_bps":float((a[-1,0]/vwap-1.0)*1e4),"tape_count":float(len(rows)),"tape_notional":float(total)}

def collect_microstructure(exchange,symbol:str,limit:int=50,impact_notional:float=10_000.0)->dict[str,float]:
    ob=snapshot_orderbook(exchange,symbol,limit=limit,impact_notional=impact_notional)
    return {**ob.to_dict(),**trade_tape_features(exchange,symbol)}
