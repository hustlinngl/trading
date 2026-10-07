from __future__ import annotations

from itertools import combinations
import math
import numpy as np
from scipy.stats import norm

def _clean_returns(returns) -> np.ndarray:
    r=np.asarray(returns,dtype=float); return r[np.isfinite(r)]

def _sample_sharpe(returns,periods_per_year:float)->float:
    r=_clean_returns(returns)
    if len(r)<2: return 0.0
    sd=float(np.std(r,ddof=1))
    return 0.0 if sd<=1e-12 else float(np.mean(r)/sd*math.sqrt(max(float(periods_per_year),1.0)))

def _sr_standard_error(sr:float,skew:float,kurtosis:float,n:int)->float:
    if n<=1: return float("inf")
    variance=(1.0-skew*sr+((kurtosis-1.0)/4.0)*sr*sr)/max(n-1,1)
    return math.sqrt(max(1e-12,variance))

def probabilistic_sharpe_ratio(returns,periods_per_year:float=365.0,benchmark_sr:float=0.0)->float:
    r=_clean_returns(returns); n=len(r)
    if n<30: return 0.5
    sr=_sample_sharpe(r,periods_per_year); centered=r-np.mean(r); m2=float(np.mean(centered**2))
    if m2<=1e-18: return 0.5
    skew=float(np.mean(centered**3)/(m2**1.5)); kurt=float(np.mean(centered**4)/(m2**2)); se=_sr_standard_error(sr,skew,kurt,n)
    return float(np.clip(norm.cdf((sr-float(benchmark_sr))/se),0.0,1.0))

def expected_max_sharpe_under_null(trials:int,*,euler_gamma:float=0.5772156649015329)->float:
    m=max(1,int(trials))
    if m==1: return 0.0
    q1=norm.ppf(max(1e-12,1.0-1.0/m)); q2=norm.ppf(max(1e-12,1.0-1.0/(m*math.e)))
    return float((1.0-euler_gamma)*q1+euler_gamma*q2)

def deflated_sharpe_ratio(returns,trials:int,periods_per_year:float=365.0,benchmark_sr:float=0.0)->float:
    r=_clean_returns(returns); n=len(r)
    if n<30: return 0.5
    sr=_sample_sharpe(r,periods_per_year); centered=r-np.mean(r); m2=float(np.mean(centered**2))
    if m2<=1e-18: return 0.5
    skew=float(np.mean(centered**3)/(m2**1.5)); kurt=float(np.mean(centered**4)/(m2**2)); se=_sr_standard_error(sr,skew,kurt,n)
    adjusted=float(benchmark_sr)+expected_max_sharpe_under_null(trials)
    return float(np.clip(norm.cdf((sr-adjusted)/se),0.0,1.0))

def selection_adjusted_psr(returns,trials:int,periods_per_year:float=365.0,benchmark_sr:float=0.0)->float:
    return deflated_sharpe_ratio(returns,trials,periods_per_year,benchmark_sr)

def combinatorial_pbo(score_matrix,partitions:int=8,seed:int=42,max_combinations:int=2000)->dict:
    x=np.asarray(score_matrix,dtype=float)
    if x.ndim!=2: raise ValueError("score_matrix must be 2-D")
    x=np.where(np.isfinite(x),x,np.nan); rows,candidates=x.shape; parts=max(4,int(partitions))
    if rows<parts or candidates<2 or rows<6:
        return {"available":False,"reason":"insufficient_blocks_or_candidates","pbo":None,"combinations":0,"mean_logit_rank":None}
    block_ids=np.array_split(np.arange(rows),parts); rng=np.random.default_rng(int(seed)); combos=list(combinations(range(parts),parts//2))
    if len(combos)>int(max_combinations):
        picks=rng.choice(len(combos),size=int(max_combinations),replace=False); combos=[combos[int(i)] for i in np.sort(picks)]
    logits=[]; ranks=[]; all_blocks=set(range(parts))
    for left in combos:
        left=set(left); right=all_blocks-left
        train_idx=np.concatenate([block_ids[i] for i in sorted(left)]); test_idx=np.concatenate([block_ids[i] for i in sorted(right)])
        train_scores=np.nanmean(x[train_idx],axis=0)
        if not np.isfinite(train_scores).any(): continue
        winner=int(np.nanargmax(train_scores)); test_scores=x[test_idx,:]; winner_score=float(np.nanmean(test_scores[:,winner]))
        if not np.isfinite(winner_score): continue
        percentile=float(np.mean(test_scores<=winner_score)); percentile=min(1.0-1e-9,max(1e-9,percentile))
        logits.append(math.log(percentile/(1.0-percentile))); ranks.append(percentile)
    if not logits:
        return {"available":False,"reason":"no_valid_partitions","pbo":None,"combinations":0,"mean_logit_rank":None}
    arr=np.asarray(logits,float)
    return {"available":True,"pbo":float(np.mean(arr<0.0)),"combinations":int(len(arr)),"mean_logit_rank":float(np.mean(arr)),
            "median_percentile_rank":float(np.median(ranks)),"selected_below_test_median_rate":float(np.mean(arr<0.0))}

def bootstrap_max_drawdown(returns,reps:int=1000,block:int=8,seed:int=42)->dict:
    r=_clean_returns(returns)
    if len(r)<10: return {"median_drawdown":0.0,"p95_drawdown":0.0,"prob_drawdown_below_20pct":0.0}
    rng=np.random.default_rng(seed); block=max(1,int(block)); n_blocks=int(math.ceil(len(r)/block)); dds=np.empty(int(reps),dtype=float)
    for j in range(int(reps)):
        starts=rng.integers(0,len(r),size=n_blocks); idx=np.concatenate([(s+np.arange(block))%len(r) for s in starts])[:len(r)]
        path=np.cumprod(1.0+r[idx]); dd=path/np.maximum.accumulate(path)-1.0; dds[j]=float(dd.min())
    return {"median_drawdown":float(np.quantile(dds,.50)),"p95_drawdown":float(np.quantile(dds,.05)),
            "prob_drawdown_below_20pct":float(np.mean(dds<-.20))}
