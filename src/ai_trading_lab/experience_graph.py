from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
from typing import Any, Iterable
import numpy as np
import pandas as pd
from .memory_store import ResearchMemoryStore

@dataclass
class TransitionEvidence:
    state_from: str
    state_to: str
    action: str
    n: int
    mean_return: float
    median_return: float
    win_rate: float
    downside_deviation: float
    def to_dict(self) -> dict[str, Any]: return asdict(self)

class ExperienceGraph:
    DEFAULT_SIGNATURE_KEYS=["ret_1","ret_3","ret_24","ret_96","vol_24","vol_96","atr_pct","atr_ratio","adx_14","efficiency_24","bb_z","bb_width","volume_trend","vwap_gap","taker_buy_imbalance","micro_imbalance_5","micro_imbalance_20","micro_spread_bps","deriv_funding_rate","deriv_open_interest","cross_asset_mean_return","cross_asset_dispersion","cross_asset_breadth"]
    def __init__(self,path:str|Path="data/memory.sqlite",bins:int=5):
        self.path=Path(path); self.store=ResearchMemoryStore(self.path); self.bins=max(3,int(bins))
    @staticmethod
    def _finite(value:Any)->float|None:
        try:
            v=float(value); return v if np.isfinite(v) else None
        except Exception: return None
    @classmethod
    def signature(cls,row:pd.Series|dict[str,Any],keys:Iterable[str])->str:
        vals=[]
        for key in keys:
            v=row.get(key) if hasattr(row,"get") else None; fv=cls._finite(v)
            if fv is None: vals.append(f"{key}=NA"); continue
            q=np.sign(fv)*np.log1p(abs(fv)); bucket=int(np.clip(np.floor((q+3.0)*2.0),0,11)); vals.append(f"{key}={bucket}")
        return hashlib.sha256("|".join(vals).encode("utf-8")).hexdigest()[:16]
    def add_state(self,symbol:str,observed_at:str,payload:dict[str,Any],keys:Iterable[str]|None=None)->str:
        if keys is None:
            preferred=[k for k in self.DEFAULT_SIGNATURE_KEYS if k in payload]; fallback=[k for k,v in payload.items() if isinstance(v,(int,float,np.floating)) and k not in preferred and not str(k).startswith('_')]; keys=(preferred+fallback)[:24]
        keys=list(keys); sig=self.signature(payload,keys); sid=hashlib.sha256(f"{symbol}|{observed_at}|{sig}".encode()).hexdigest()[:20]
        with self.store.connect() as cx:
            cx.execute("INSERT OR IGNORE INTO states(state_id,observed_at,symbol,signature,payload_json) VALUES(?,?,?,?,?)",(sid,observed_at,symbol,sig,json.dumps(payload,sort_keys=True,default=str))); cx.commit()
        return sid
    def learn_episode(self,df:pd.DataFrame,state_features:pd.DataFrame,actions:pd.Series,future_returns:pd.Series,symbol:str="UNKNOWN",horizon_bars:int=8,regime:pd.Series|None=None,max_rows:int=5000)->int:
        frame=state_features.copy().replace([np.inf,-np.inf],np.nan).ffill().fillna(0.0)
        idx=frame.index.intersection(df.index).intersection(actions.index).intersection(future_returns.index)
        if regime is not None: idx=idx.intersection(regime.index)
        idx=idx[future_returns.reindex(idx).notna().to_numpy()][-max_rows:]
        if len(idx)<3: return 0
        preferred=[c for c in self.DEFAULT_SIGNATURE_KEYS if c in frame.columns]; fallback=[c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and c not in preferred and not str(c).startswith('_')]; numeric_keys=(preferred+fallback)[:24]
        state_rows=[]; transition_rows=[]; prev_sig=None; prev_time=None; prev_payload=None; position_map={pd.Timestamp(t):int(i) for i,t in enumerate(df.index)}
        for ts in idx:
            payload={k:float(frame.at[ts,k]) for k in numeric_keys if np.isfinite(frame.at[ts,k])}; payload["price"]=float(df.at[ts,"close"]); payload["_signature_schema"]=1
            if regime is not None: payload["regime"]=str(regime.at[ts])
            sig=self.signature(payload,numeric_keys); sid=hashlib.sha256(f"{symbol}|{pd.Timestamp(ts).isoformat()}|{sig}".encode()).hexdigest()[:20]
            state_rows.append((sid,pd.Timestamp(ts).isoformat(),symbol,sig,json.dumps(payload,sort_keys=True,default=str)))
            if prev_sig is not None and prev_time is not None:
                ret_from=self._finite(future_returns.at[prev_time])
                if ret_from is not None:
                    prev_pos=position_map.get(pd.Timestamp(prev_time),0); outcome_pos=min(len(df.index)-1,prev_pos+1+int(horizon_bars)); outcome_time=pd.Timestamp(df.index[outcome_pos]).isoformat()
                    transition_rows.append((symbol,pd.Timestamp(ts).isoformat(),prev_sig,sig,str(actions.at[prev_time]) if pd.notna(actions.at[prev_time]) else "FLAT",float(ret_from),int(horizon_bars),json.dumps({"regime_from":(prev_payload or {}).get("regime"),"action_at":pd.Timestamp(prev_time).isoformat(),"outcome_at":outcome_time},default=str),2))
            prev_sig,prev_time,prev_payload=sig,ts,payload
        with self.store.connect() as cx:
            cx.execute("PRAGMA journal_mode=WAL"); cx.execute("PRAGMA synchronous=NORMAL")
            cx.executemany("INSERT OR IGNORE INTO states(state_id,observed_at,symbol,signature,payload_json) VALUES(?,?,?,?,?)",state_rows)
            before_transitions=cx.total_changes
            cx.executemany("INSERT OR IGNORE INTO transitions(symbol,observed_at,state_from,state_to,action,realized_return,hold_bars,metadata_json,transition_version) VALUES(?,?,?,?,?,?,?,?,?)",transition_rows)
            tcount=max(0,int(cx.total_changes-before_transitions)); cx.commit()
        return tcount
    def transition_evidence(self,state_from:str,action:str|None=None,limit:int=10000,include_legacy:bool=False)->pd.DataFrame:
        q="SELECT state_from,state_to,action,realized_return,hold_bars,metadata_json FROM transitions WHERE state_from=?"; params=[state_from]
        if not include_legacy: q+=" AND transition_version>=2"
        if action is not None: q+=" AND action=?"; params.append(action)
        q+=" ORDER BY observed_at DESC LIMIT ?"; params.append(int(limit))
        with self.store.connect() as cx: return pd.read_sql_query(q,cx,params=params)
    def summarize(self,state_from:str,action:str,min_n:int=5,include_legacy:bool=False)->TransitionEvidence|None:
        df=self.transition_evidence(state_from,action,include_legacy=include_legacy)
        if len(df)<min_n: return None
        r=df["realized_return"].to_numpy(float); downside=r[r<0]
        return TransitionEvidence(state_from,str(df["state_to"].mode().iloc[0]),action,int(len(r)),float(r.mean()),float(np.median(r)),float(np.mean(r>0)),float(np.sqrt(np.mean(downside**2))) if len(downside) else 0.0)
    def best_actions(self,state_from:str,min_n:int=5,include_legacy:bool=False)->list[dict[str,Any]]:
        clause="" if include_legacy else " AND transition_version>=2"
        with self.store.connect() as cx:
            df=pd.read_sql_query("SELECT action, COUNT(*) n, AVG(realized_return) mean_return, AVG(CASE WHEN realized_return>0 THEN 1.0 ELSE 0.0 END) win_rate FROM transitions WHERE state_from=?"+clause+" GROUP BY action ORDER BY mean_return DESC",cx,params=(state_from,))
        df=df[df["n"]>=min_n]; return df.to_dict(orient="records") if not df.empty else []
    def recent_path(self,symbol:str,limit:int=100)->list[dict[str,Any]]:
        with self.store.connect() as cx:
            df=pd.read_sql_query("SELECT observed_at,state_from,state_to,action,realized_return,hold_bars FROM transitions WHERE symbol=? ORDER BY observed_at DESC LIMIT ?",cx,params=(symbol,int(limit)))
        return df.to_dict(orient="records")
