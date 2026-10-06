from __future__ import annotations

import numpy as np
import pandas as pd


def _block_bootstrap_mean(values, rng, reps=1500, block=8):
    values=np.asarray(values,dtype=float)
    if len(values)==0: return 0.0,(0.0,0.0)
    starts=rng.integers(0,len(values),size=(reps,int(np.ceil(len(values)/block)))); samples=[]
    for row in starts:
        idx=np.concatenate([((s+np.arange(block))%len(values)) for s in row])[:len(values)]; samples.append(values[idx].mean())
    arr=np.asarray(samples)
    return float(values.mean()),(float(np.quantile(arr,0.025)),float(np.quantile(arr,0.975)))


def compare_equity(a: pd.Series, b: pd.Series, seed=42, reps=2000, block=8):
    r_a=a.pct_change().fillna(0).to_numpy(float); r_b=b.reindex(a.index).pct_change().fillna(0).to_numpy(float); n=min(len(r_a),len(r_b)); diff=(r_a[:n]-r_b[:n]).astype(float)
    rng=np.random.default_rng(seed); mean,ci=_block_bootstrap_mean(diff,rng,reps=reps,block=block)
    if len(diff):
        starts=rng.integers(0,len(diff),size=(int(reps),int(np.ceil(len(diff)/max(1,block))))); boot=np.empty(int(reps))
        for j,row in enumerate(starts):
            idx=np.concatenate([((s+np.arange(max(1,block)))%len(diff)) for s in row])[:len(diff)]; boot[j]=diff[idx].mean()
        prob=float(np.mean(boot>0))
    else: prob=0.5
    return {'mean_return_diff':mean,'ci_low':ci[0],'ci_high':ci[1],'bootstrap_probability_mean_diff_gt_0':prob}


def robust_score(stats: dict) -> float:
    total=float(np.clip(stats.get('total_return',0.0),-1,1)); sharpe=float(np.clip(stats.get('sharpe_like',0.0),-6,6)); sortino=float(np.clip(stats.get('sortino_like',0.0),-8,8)); pf=min(float(stats.get('profit_factor',0.0)),4.0); win=float(np.clip(stats.get('win_rate',0.5),0,1)); dd=float(np.clip(stats.get('max_drawdown',0.0),-1,0)); excess=float(np.clip(stats.get('excess_return',total),-1,1)); concentration=float(np.clip(stats.get('trade_pnl_concentration',0.0),0,1)); trades=max(0,int(stats.get('trades',0)))
    reliability=min(1.0,np.sqrt(trades/50.0)); core=0.60*total+0.40*excess+0.10*sharpe+0.06*sortino+0.04*(pf-1)+0.10*(win-0.5)+0.75*dd-0.18*concentration
    sparse_floor=min(-0.06,0.20*min(0.0,total))
    return float(reliability*core+(1.0-reliability)*sparse_floor)


def aggregate_fold_stats(folds: pd.DataFrame) -> dict:
    if folds.empty: return {'folds':0,'median_score':-np.inf}
    scores=folds['robust_score'].to_numpy(float)
    return {'folds':int(len(folds)),'median_score':float(np.median(scores)),'mean_score':float(np.mean(scores)),'positive_folds':int(np.sum(scores>0)),'median_return':float(folds['total_return'].median()),'worst_drawdown':float(folds['max_drawdown'].min()),'median_sharpe':float(folds['sharpe_like'].median()),'total_trades':int(folds['trades'].sum())}
