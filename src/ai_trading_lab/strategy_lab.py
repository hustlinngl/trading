from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

FEATURES = [
    'rsi_14', 'macd_hist', 'macd_hist_z', 'vwap_gap', 'ema_gap_12', 'ema_gap_24', 'ema_gap_48', 'ema_gap_96',
    'ret_3', 'ret_6', 'ret_12', 'ret_24', 'ret_48', 'vol_12', 'vol_24', 'atr_ratio', 'volume_z_24', 'volume_trend',
    'range_pct', 'range_z_48', 'close_location', 'dist_high_48', 'dist_low_48', 'adx_14', 'bb_z', 'bb_width',
    'efficiency_24', 'sign_entropy_48', 'obv_z', 'htf1h_ema_gap', 'htf4h_ema_gap'
]

@dataclass(frozen=True)
class Rule:
    feature: str
    op: str
    threshold: float
    weight: float = 1.0
    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        s = df[self.feature]
        return {'<': s < self.threshold, '>': s > self.threshold}.get(self.op, pd.Series(False,index=df.index))

@dataclass
class Candidate:
    rules: tuple[Rule,...]
    direction: str
    def signal(self, df: pd.DataFrame) -> pd.Series:
        masks=[r.evaluate(df) for r in self.rules]
        m=np.logical_and.reduce(masks) if masks else np.zeros(len(df),dtype=bool)
        return pd.Series(m,index=df.index,dtype=bool)

def random_candidate(rng, stats):
    available=[f for f in FEATURES if f in stats]
    n=int(rng.integers(2,min(6,len(available)+1))); rules=[]
    for f in rng.choice(available,size=n,replace=False):
        s=stats[f].dropna()
        if len(s)<50: continue
        threshold=float(s.quantile(float(rng.uniform(0.10,0.90))))
        rules.append(Rule(f,'>' if rng.random()<0.5 else '<',threshold,float(rng.uniform(0.5,1.5))))
    return Candidate(tuple(rules),'long' if rng.random()<0.5 else 'short')

def mutate(c,rng,stats):
    rules=list(c.rules)
    if not rules: return random_candidate(rng,stats)
    j=int(rng.integers(0,len(rules))); r=rules[j]; s=stats[r.feature].dropna()
    rules[j]=Rule(r.feature,'>' if rng.random()<0.5 else '<',float(s.quantile(float(rng.uniform(0.03,0.97)))),float(rng.uniform(0.5,1.5)))
    if rng.random()<0.25 and len(rules)<6:
        extra=random_candidate(rng,stats)
        if extra.rules: rules.append(extra.rules[0])
    if rng.random()<0.12 and len(rules)>2: rules.pop(int(rng.integers(0,len(rules))))
    direction=c.direction if rng.random()<0.90 else ('short' if c.direction=='long' else 'long')
    return Candidate(tuple(rules),direction)

def score_candidate(c,df,future_ret,cost_bps:float=0.0):
    sig=c.signal(df)
    if sig.sum()<30: return -999.0
    r=future_ret[sig].astype(float)
    if c.direction=='short': r=-r
    r=r.dropna()
    if cost_bps>0 and len(r): r=r-(2.0*float(cost_bps)/10_000.0)
    if len(r)<30: return -999.0
    mean,std=float(r.mean()),float(r.std()); t=mean/(std/np.sqrt(len(r))+1e-9); hit=float((r>0).mean())
    q25,q75=float(r.quantile(0.25)),float(r.quantile(0.75))
    complexity_penalty=0.04*max(0,len(c.rules)-3); concentration_penalty=0.20 if len(r)<60 else 0.0
    return float(t+2.0*mean*100+0.7*(hit-0.5)+0.2*np.tanh((q75+q25)*100)-complexity_penalty-concentration_penalty)

def evolve(df,future_ret,population_size=500,generations=30,seed=42,cost_bps:float=0.0):
    rng=np.random.default_rng(seed); pop=[random_candidate(rng,df) for _ in range(population_size)]; best=[]
    for _ in range(generations):
        scored=[(score_candidate(c,df,future_ret,cost_bps=cost_bps),c) for c in pop]; scored.sort(key=lambda x:x[0],reverse=True)
        elites=[c for _,c in scored[:max(20,population_size//12)]]; best.extend(scored[:20]); pop=elites.copy()
        while len(pop)<population_size: pop.append(mutate(elites[int(rng.integers(0,len(elites)))],rng,df))
    best.sort(key=lambda x:x[0],reverse=True); unique=[]; seen=set()
    for score,c in best:
        sig=(c.direction,tuple((r.feature,r.op,round(r.threshold,5)) for r in c.rules))
        if sig in seen: continue
        seen.add(sig); unique.append((score,c))
        if len(unique)>=30: break
    return unique
