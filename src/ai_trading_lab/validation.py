from __future__ import annotations

import numpy as np
import pandas as pd


def _block_bootstrap_samples(values, rng, reps=1500, block=8):
    """Return circular block-bootstrap means using memory-bounded vectorized chunks."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    reps = max(1, int(reps))
    if n == 0:
        return np.zeros(reps, dtype=float)
    block = max(1, int(block))
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block, dtype=np.int64)
    chunk_size = max(1, min(reps, 2_000_000 // max(1, n_blocks * block)))
    samples = np.empty(reps, dtype=float)
    for first in range(0, reps, chunk_size):
        count = min(chunk_size, reps - first)
        starts = rng.integers(0, n, size=(count, n_blocks), dtype=np.int64)
        indices = (starts[:, :, None] + offsets[None, None, :]) % n
        indices = indices.reshape(count, -1)[:, :n]
        samples[first:first + count] = values[indices].mean(axis=1)
    return samples


def _block_bootstrap_mean(values, rng, reps=1500, block=8):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0, (0.0, 0.0)
    samples = _block_bootstrap_samples(values, rng, reps=reps, block=block)
    return float(values.mean()), (
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def compare_equity(a: pd.Series, b: pd.Series, seed=42, reps=2000, block=8):
    r_a = a.pct_change().fillna(0).to_numpy(dtype=float)
    r_b = b.reindex(a.index).pct_change().fillna(0).to_numpy(dtype=float)
    n = min(len(r_a), len(r_b))
    diff = (r_a[:n] - r_b[:n]).astype(float)
    diff = diff[np.isfinite(diff)]
    if not len(diff):
        return {
            "mean_return_diff": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "bootstrap_probability_mean_diff_gt_0": 0.5,
        }
    samples = _block_bootstrap_samples(
        diff, np.random.default_rng(seed), reps=reps, block=block
    )
    return {
        "mean_return_diff": float(diff.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "bootstrap_probability_mean_diff_gt_0": float(np.mean(samples > 0)),
    }

def robust_score(stats: dict) -> float:
    total=float(np.clip(stats.get('total_return',0.0),-1,1)); sharpe=float(np.clip(stats.get('sharpe_like',0.0),-6,6)); sortino=float(np.clip(stats.get('sortino_like',0.0),-8,8)); pf=min(float(stats.get('profit_factor',0.0)),4.0); win=float(np.clip(stats.get('win_rate',0.5),0,1)); dd=float(np.clip(stats.get('max_drawdown',0.0),-1,0)); excess=float(np.clip(stats.get('excess_return',total),-1,1)); concentration=float(np.clip(stats.get('trade_pnl_concentration',0.0),0,1)); trades=max(0,int(stats.get('trades',0)))
    reliability=min(1.0,np.sqrt(trades/50.0)); core=0.60*total+0.40*excess+0.10*sharpe+0.06*sortino+0.04*(pf-1)+0.10*(win-0.5)+0.75*dd-0.18*concentration
    sparse_floor=min(-0.06,0.20*min(0.0,total))
    return float(reliability*core+(1.0-reliability)*sparse_floor)


def aggregate_fold_stats(folds: pd.DataFrame) -> dict:
    if folds.empty: return {'folds':0,'median_score':-np.inf}
    scores=folds['robust_score'].to_numpy(float)
    return {'folds':int(len(folds)),'median_score':float(np.median(scores)),'mean_score':float(np.mean(scores)),'positive_folds':int(np.sum(scores>0)),'median_return':float(folds['total_return'].median()),'worst_drawdown':float(folds['max_drawdown'].min()),'median_sharpe':float(folds['sharpe_like'].median()),'total_trades':int(folds['trades'].sum())}
