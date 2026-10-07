from __future__ import annotations

import math
import numpy as np


def duration_preference(hours: float, *, min_hours=3.0, max_hours=24.0, preferred_min_hours=3.0, preferred_max_hours=4.0) -> float:
    h=float(hours); lo=float(min_hours); hi=max(lo,float(max_hours)); p_lo=float(preferred_min_hours); p_hi=max(p_lo,float(preferred_max_hours))
    if not math.isfinite(h) or h<=0 or h<lo or h>hi: return 0.0
    if p_lo<=h<=p_hi: return 1.0
    mid=0.5*(p_lo+p_hi); span=max(0.5,p_hi-p_lo)
    return float(np.clip(math.exp(-abs(h-mid)/(3.0*span)),0.0,1.0))


def live_data_quality_score(age_minutes: float, gap_rows: int, rows: int, *, max_age_minutes=30.0) -> float:
    age=max(0.0,float(age_minutes)); gaps=max(0,int(gap_rows)); n=max(1,int(rows))
    freshness=float(np.clip(1.0-age/max(float(max_age_minutes),1.0),0.0,1.0))
    continuity=float(np.clip(math.exp(-30.0*gaps/n),0.0,1.0))
    return float(np.clip(0.70*freshness+0.30*continuity,0.0,1.0))


def opportunity_quality_score(*, confidence, p_direction, robust_expected_return, meta_success=0.5, memory_consensus=0.0, trade_window_confidence=0.0, duration_preference=0.0, model_disagreement=0.0, data_quality=1.0, min_expected_return=0.003) -> float:
    conf=float(np.clip(confidence,0,1)); p=float(np.clip((float(p_direction)-0.5)/0.5,0,1))
    er=float(np.clip(float(robust_expected_return)/max(float(min_expected_return),1e-6),0,1.5)/1.5)
    meta=float(np.clip(meta_success,0,1)); memory=float(np.clip(memory_consensus,0,1)); tw=float(np.clip(trade_window_confidence,0,1)); dur=float(np.clip(duration_preference,0,1)); data=float(np.clip(data_quality,0,1))
    disagreement=float(np.clip(float(model_disagreement)/0.05,0,1))
    base=0.24*conf+0.17*p+0.19*er+0.12*meta+0.13*memory+0.07*tw+0.04*dur+0.04*data
    return float(np.clip(base-0.08*disagreement,0,1))


def quality_grade(score: float) -> str:
    value=float(score)
    return "A" if value>=0.82 else "B" if value>=0.72 else "C" if value>=0.60 else "D" if value>=0.48 else "E"


def robust_performance_utility(stats: dict, *, benchmark_return=None, min_trades=20, max_drawdown=-0.25) -> float:
    total=float(stats.get("total_return",stats.get("net_compounded_return",0.0))); bench=float(benchmark_return if benchmark_return is not None else stats.get("benchmark_return",0.0)); excess=total-bench
    sharpe=float(np.clip(stats.get("sharpe_like",0.0),-3,3)); sortino=float(np.clip(stats.get("sortino_like",0.0),-3,3)); dd=float(stats.get("max_drawdown",max_drawdown-1.0))
    trades=max(0,int(stats.get("trades",stats.get("trades_taken",0)))); sample=float(np.clip(math.sqrt(trades/max(1,int(min_trades))),0,1))
    dd_penalty=max(0.0,float(max_drawdown)-dd)/max(abs(float(max_drawdown)),1e-6); concentration=float(np.clip(stats.get("top_trade_share",0.0),0,1))
    raw=0.42*np.tanh(excess*12)+0.22*(sharpe/3)+0.16*(sortino/3)+0.10*min(1.0,max(0.0,float(stats.get("profit_factor",0.0))-1.0))-0.10*concentration-0.22*dd_penalty
    return float(np.clip(raw*(0.35+0.65*sample),-1,1))
