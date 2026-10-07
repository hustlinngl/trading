from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, json, sqlite3
from typing import Any
import numpy as np, pandas as pd
from .time_utils import utcnow
from .memory_store import ResearchMemoryStore

@dataclass
class Experiment:
    experiment_id:str; created_at:str; kind:str; hypothesis:str; config:dict[str,Any]; parent_id:str|None=None; status:str="proposed"

class GrowthRegistry:
    """Append-oriented persistent memory for autonomous research."""
    def __init__(self,path:str|Path="data/memory.sqlite"): self.path=Path(path); self.store=ResearchMemoryStore(self.path)
    @staticmethod
    def stable_id(*parts:Any)->str: return hashlib.sha256("|".join(map(str,parts)).encode()).hexdigest()[:20]
    def propose(self,kind,hypothesis,config,parent_id=None):
        eid=self.stable_id(kind,hypothesis,json.dumps(config,sort_keys=True),parent_id or "")
        e=Experiment(eid,utcnow(),kind,hypothesis,config,parent_id)
        with self.store.connect() as cx:
            cx.execute("INSERT OR IGNORE INTO experiments VALUES (?,?,?,?,?,?,?)",(e.experiment_id,e.created_at,e.kind,e.hypothesis,json.dumps(e.config,sort_keys=True),e.parent_id,e.status)); cx.commit()
        return eid
    def set_status(self,eid,status):
        with self.store.connect() as cx: cx.execute("UPDATE experiments SET status=? WHERE experiment_id=?",(status,eid)); cx.commit()
    def add_evaluation(self,eid,metric,value,split="",regime="",details=None):
        with self.store.connect() as cx: cx.execute("INSERT INTO evaluations(experiment_id,evaluated_at,split,metric,value,regime,details_json) VALUES(?,?,?,?,?,?,?)",(eid,utcnow(),split,metric,float(value),regime,json.dumps(details or {},default=str))); cx.commit()
    def add_model_version(self,role,status,score,metrics,artifact_path="",parent_version=None):
        vid=self.stable_id(role,utcnow(),artifact_path)
        with self.store.connect() as cx: cx.execute("INSERT INTO model_versions VALUES (?,?,?,?,?,?,?)",(vid,utcnow(),role,parent_version,status,None if score is None else float(score),json.dumps(metrics,default=str),artifact_path)); cx.commit()
        return vid
    def add_event(self,event_id,provider,event_time,feature,horizon):
        with self.store.connect() as cx: cx.execute("INSERT OR IGNORE INTO event_outcomes(event_id,provider,event_time,feature_json,horizon) VALUES(?,?,?,?,?)",(event_id,provider,event_time,json.dumps(feature,sort_keys=True),int(horizon))); cx.commit()
    def open_events(self,limit=5000):
        with self.store.connect() as cx: return pd.read_sql_query("SELECT * FROM event_outcomes WHERE status='open' ORDER BY event_time LIMIT ?",cx,params=(int(limit),))
    def settle_events(self,prices,default_horizon=8):
        if prices.empty: return 0
        p=prices.sort_index(); settled=0
        with self.store.connect() as cx:
            rows=cx.execute("SELECT id,event_time,horizon FROM event_outcomes WHERE status='open'").fetchall()
            for rid,event_time,horizon in rows:
                try: t=pd.Timestamp(event_time); t=t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
                except Exception: continue
                pos=p.index.searchsorted(t,side="right"); h=int(horizon or default_horizon)
                if pos>=len(p) or pos+h>=len(p): continue
                base=float(p.iloc[pos]); fut=float(p.iloc[pos+h])
                if not np.isfinite(base) or not np.isfinite(fut) or base==0: continue
                ret=fut/base-1.0; cx.execute("UPDATE event_outcomes SET realized_return=?,direction=?,status='settled' WHERE id=?",(float(ret),float(np.sign(ret)),int(rid))); settled+=1
            cx.commit()
        return settled
    def add_market_state(self,symbol,observed_at,price,payload):
        sid=self.stable_id(symbol,observed_at,round(float(price),8),json.dumps(payload,sort_keys=True,default=str))
        with self.store.connect() as cx: cx.execute("INSERT OR IGNORE INTO market_states(state_id,observed_at,symbol,price,payload_json) VALUES(?,?,?,?,?)",(sid,observed_at,symbol,float(price),json.dumps(payload,sort_keys=True,default=str))); cx.commit()
        return sid
    def recent_market_states(self,symbol,limit=500):
        with self.store.connect() as cx: return pd.read_sql_query("SELECT * FROM market_states WHERE symbol=? ORDER BY observed_at DESC LIMIT ?",cx,params=(symbol,int(limit)))

class StrategyBandit:
    def __init__(self,registry,prior_alpha=2.0,prior_beta=2.0): self.registry=registry; self.a=float(prior_alpha); self.b=float(prior_beta)
    def sample_weights(self,strategy_ids,seed=42):
        rng=np.random.default_rng(seed)
        if not strategy_ids: return {}
        vals={}
        with self.registry.store.connect() as cx:
            for sid in strategy_ids:
                row=cx.execute("SELECT SUM(CASE WHEN metric='win' THEN value ELSE 0 END),SUM(CASE WHEN metric='loss' THEN value ELSE 0 END) FROM evaluations WHERE experiment_id=?",(sid,)).fetchone()
                vals[sid]=float(rng.beta(self.a+float(row[0] or 0),self.b+float(row[1] or 0)))
        total=sum(vals.values()) or 1.0; return {k:v/total for k,v in vals.items()}
