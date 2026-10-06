from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from .cross_asset import cross_asset_snapshot, fetch_cross_asset_bars
from .microstructure import collect_microstructure

@dataclass
class MarketState:
    timestamp:str; symbol:str; price:float; regime_hint:str
    microstructure:dict[str,float]; derivatives:dict[str,float]; cross_asset:dict[str,float]
    macro:dict[str,float]; external:dict[str,float]; quality:dict[str,Any]
    def flatten(self)->dict[str,Any]:
        out={"timestamp":self.timestamp,"symbol":self.symbol,"price":self.price,"regime_hint":self.regime_hint}
        for prefix,values in (("micro",self.microstructure),("deriv",self.derivatives),("cross",self.cross_asset),("macro",self.macro),("ext",self.external)):
            for k,v in values.items():
                if isinstance(v,(int,float,np.floating)) and np.isfinite(v): out[f"{prefix}_{k}"]=float(v)
        out["quality_json"]=str(self.quality); return out

def derivative_snapshot(exchange,symbol:str)->dict[str,float]:
    out={"funding_rate":np.nan,"open_interest":np.nan}
    try:
        market=exchange.market(symbol); contract_symbol=symbol
        if not market.get("contract"):
            candidates=[m for m in exchange.markets.values() if m.get("base")==market.get("base") and m.get("quote")==market.get("quote") and m.get("contract") and m.get("swap")]
            if candidates: contract_symbol=candidates[0]["symbol"]
        fr=exchange.fetch_funding_rate(contract_symbol) if exchange.has.get("fetchFundingRate") else {}
        oi=exchange.fetch_open_interest(contract_symbol) if exchange.has.get("fetchOpenInterest") else {}
        if fr.get("fundingRate") is not None: out["funding_rate"]=float(fr["fundingRate"])
        val=oi.get("openInterestValue") or oi.get("openInterestAmount")
        if val is not None: out["open_interest"]=float(val)
    except Exception: pass
    return out

def build_market_state(exchange,symbol:str,timeframe:str,price:float,macro:dict[str,float]|None=None,external:dict[str,float]|None=None,cross_symbols:list[str]|None=None,orderbook_levels:int=50,impact_notional:float=10_000.0)->MarketState:
    symbols=cross_symbols or [symbol,"ETH/USDT","SOL/USDT"]
    def get_micro():
        try: return collect_microstructure(exchange,symbol,limit=orderbook_levels,impact_notional=impact_notional)
        except Exception: return {}
    def get_cross():
        try:
            frames=fetch_cross_asset_bars(exchange,[s for s in symbols if s!=symbol]+[symbol],timeframe,limit=300)
            return cross_asset_snapshot(frames,symbol)
        except Exception: return {}
    with ThreadPoolExecutor(max_workers=2,thread_name_prefix="state") as pool:
        f_micro=pool.submit(get_micro); f_cross=pool.submit(get_cross); micro=f_micro.result(); cross=f_cross.result()
    deriv=derivative_snapshot(exchange,symbol)
    macro={k:float(v) for k,v in (macro or {}).items() if isinstance(v,(int,float)) and np.isfinite(v)}
    external={k:float(v) for k,v in (external or {}).items() if isinstance(v,(int,float)) and np.isfinite(v)}
    quality={"microstructure_available":bool(micro),"derivatives_available":bool(np.isfinite(deriv.get("funding_rate",np.nan)) or np.isfinite(deriv.get("open_interest",np.nan))),"cross_asset_available":bool(cross),"macro_available":bool(macro),"external_available":bool(external)}
    return MarketState(pd.Timestamp.utcnow().isoformat(),symbol,float(price),"unknown",micro,deriv,cross,macro,external,quality)
