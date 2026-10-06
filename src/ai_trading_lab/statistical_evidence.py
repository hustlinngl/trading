from __future__ import annotations
import math
import numpy as np
from scipy.stats import norm

def probabilistic_sharpe_ratio(returns, periods_per_year=365.0, benchmark_sr=0.0):
    r=np.asarray(returns,dtype=float); r=r[np.isfinite(r)]; n=len(r)
    if n<30: return 0.5
    sd=float(np.std(r,ddof=1))
    if sd<=1e-12: return 1.0 if float(np.mean(r))>0 else 0.0
    sr=float(np.mean(r)/sd*math.sqrt(max(periods_per_year,1.0))); centered=r-np.mean(r); m2=float(np.mean(centered**2))
    if m2<=1e-18: return 0.5
    skew=float(np.mean(centered**3)/(m2**1.5)); kurt=float(np.mean(centered**4)/(m2**2)); denom=math.sqrt(max(1e-12,(1-skew*sr+((kurt-1)/4)*sr*sr)/max(n-1,1))); z=(sr-benchmark_sr)/denom
    return float(np.clip(norm.cdf(z),0,1))

def selection_adjusted_psr(returns,trials,periods_per_year=365.0,benchmark_sr=0.0):
    base=probabilistic_sharpe_ratio(returns,periods_per_year,benchmark_sr); m=max(1,int(trials)); r=np.asarray(returns,dtype=float); r=r[np.isfinite(r)]
    if len(r)<30: return base
    sd=float(np.std(r,ddof=1)); se_sr=1.0/math.sqrt(max(len(r)-1,1)); penalty=math.sqrt(2*math.log(max(m,2)))*se_sr; sr=float(np.mean(r)/max(sd,1e-12)*math.sqrt(max(periods_per_year,1.0))); adjusted=sr-penalty
    return float(np.clip(norm.cdf(adjusted/max(se_sr,1e-12)),0,1))

def bootstrap_max_drawdown(returns,reps=1000,block=8,seed=42):
    r=np.asarray(returns,dtype=float); r=r[np.isfinite(r)]
    if len(r)<10: return {"median_drawdown":0.0,"p95_drawdown":0.0,"prob_drawdown_below_20pct":0.0}
    rng=np.random.default_rng(seed); block=max(1,int(block)); n_blocks=int(math.ceil(len(r)/block)); dds=np.empty(int(reps))
    for j in range(int(reps)):
        starts=rng.integers(0,len(r),size=n_blocks); idx=np.concatenate([(s+np.arange(block))%len(r) for s in starts])[:len(r)]; path=np.cumprod(1+r[idx]); dds[j]=float((path/np.maximum.accumulate(path)-1).min())
    return {"median_drawdown":float(np.quantile(dds,0.5)),"p95_drawdown":float(np.quantile(dds,0.05)),"prob_drawdown_below_20pct":float(np.mean(dds<-0.20))}
