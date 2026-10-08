from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json, re
from typing import Iterable
import pandas as pd
import requests

@dataclass(frozen=True)
class MarketSpec:
    symbol:str; source:str; asset_class:str; start:str; end:str; timeframe:str="1d"; market:str="spot"; enabled:bool=True; priority:int=50; notes:str=""
    def to_dict(self): return asdict(self)

DEFAULT_CRYPTO_CORE=(
    "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT",
    "ADA/USDT","DOGE/USDT","AVAX/USDT","LINK/USDT","DOT/USDT",
    "LTC/USDT","BCH/USDT","ATOM/USDT","UNI/USDT","ETC/USDT",
    "NEAR/USDT","APT/USDT","ARB/USDT","OP/USDT","FIL/USDT",
    "INJ/USDT","SUI/USDT","SEI/USDT","TIA/USDT","AAVE/USDT",
    "MKR/USDT","CRV/USDT","ALGO/USDT","ICP/USDT","XLM/USDT",
    "HBAR/USDT","TRX/USDT","EOS/USDT","XTZ/USDT","VET/USDT",
    "SAND/USDT","MANA/USDT","GRT/USDT","RUNE/USDT","THETA/USDT",
)
DEFAULT_MACRO_SERIES=("DFF","DGS2","DGS10","DTWEXBGS","VIXCLS","BAMLH0A0HYM2","T10Y2Y")

def build_seed_universe(start:str,end:str,crypto_symbols:Iterable[str]=DEFAULT_CRYPTO_CORE)->list[MarketSpec]:
    specs=[MarketSpec(s,"binance","crypto",start,end,"1d","spot",True,100-i) for i,s in enumerate(crypto_symbols)]
    specs += [MarketSpec(s,"fred","macro",start,end,"1d","macro",True,60-i) for i,s in enumerate(DEFAULT_MACRO_SERIES)]
    return specs

def validate_spec(spec:MarketSpec)->None:
    if pd.Timestamp(spec.start)>pd.Timestamp(spec.end): raise ValueError(f"start > end for {spec}")
    if spec.source not in {"binance","fred","sec","cftc","treasury","alpha_vantage"}: raise ValueError(f"unsupported source: {spec.source}")

def write_manifest(specs:Iterable[MarketSpec],path:str|Path)->Path:
    specs=list(specs); [validate_spec(s) for s in specs]; p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps([s.to_dict() for s in specs],indent=2),encoding="utf-8"); return p

def read_manifest(path:str|Path)->list[MarketSpec]:
    rows=json.loads(Path(path).read_text(encoding="utf-8")); return [MarketSpec(**r) for r in rows if r.get("enabled",True)]

def build_liquid_intraday_candidates(daily_stats:pd.DataFrame,top_n:int=25)->list[str]:
    required={"symbol","median_quote_volume","coverage_ratio"}; missing=required-set(daily_stats.columns)
    if missing: raise ValueError(f"missing daily liquidity columns: {sorted(missing)}")
    df=daily_stats.replace([float("inf"),float("-inf")],pd.NA).dropna(subset=list(required)); df=df[(df["median_quote_volume"]>0)&(df["coverage_ratio"]>=0.95)]
    return list(df.sort_values(["median_quote_volume","coverage_ratio"],ascending=False).head(int(top_n))["symbol"])

def discover_binance_symbols(prefix:str="data/spot/monthly/klines/")->list[str]:
    urls=[f"https://data.binance.vision/?prefix={prefix}",f"https://data.binance.vision/index.html?prefix={prefix}"]; last=None
    for url in urls:
        try:
            r=requests.get(url,timeout=30,headers={"User-Agent":"adaptive-ai-trading-lab/0.6"}); r.raise_for_status(); html=r.text
            found=set(re.findall(rf"{re.escape(prefix)}([A-Z0-9]+?)/",html)) or set(re.findall(rf"<Key>{re.escape(prefix)}([A-Z0-9]+?)/",html))
            if found: return sorted(found)
        except Exception as exc: last=exc
    raise RuntimeError(f"Binance Vision symbol discovery failed: {last}")

def expand_binance_universe(symbols:Iterable[str],start:str,end:str,timeframe:str="1d",market:str="spot")->list[MarketSpec]:
    out=[]
    for rank,sym in enumerate(sorted(set(symbols))):
        if not sym.endswith(("USDT","USDC","FDUSD")): continue
        base=sym.replace("/",""); quote="USDT" if base.endswith("USDT") else ("USDC" if base.endswith("USDC") else "FDUSD"); asset=base[:-len(quote)]
        out.append(MarketSpec(f"{asset}/{quote}","binance","crypto",start,end,timeframe,market,True,max(1,10000-rank)))
    return out

def source_batches(specs:Iterable[MarketSpec])->dict[str,list[MarketSpec]]:
    batches={}
    for s in specs: batches.setdefault(s.source,[]).append(s)
    return batches

def summarize_manifest(specs:Iterable[MarketSpec])->dict:
    rows=list(specs); by_source={}; by_asset={}
    for s in rows: by_source[s.source]=by_source.get(s.source,0)+1; by_asset[s.asset_class]=by_asset.get(s.asset_class,0)+1
    return {"total":len(rows),"by_source":by_source,"by_asset_class":by_asset}
