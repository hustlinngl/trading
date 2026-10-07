from __future__ import annotations
import numpy as np

def promotion_gate(stats:dict,champion_score:float,*,min_folds:int=4,min_trades:int=50,max_dd:float=-0.20,min_positive_fold_ratio:float=0.65,min_bootstrap_prob:float=0.58,min_score:float=0.0,require_score_improvement:bool=True)->dict:
    folds=int(stats.get("folds",0)); trades=int(stats.get("total_trades",0)); score=float(stats.get("score",stats.get("median_score",-np.inf)))
    positive_ratio=float(stats.get("positive_fold_ratio",stats.get("positive_folds",0)/max(folds,1))); bootstrap_prob=float(stats.get("bootstrap_superiority_prob",0.0))
    checks={
        "enough_folds":folds>=min_folds,
        "enough_trades":trades>=min_trades,
        "drawdown_ok":float(stats.get("worst_drawdown",-1))>=max_dd,
        "cross_fold_consistency":positive_ratio>=min_positive_fold_ratio,
        "bootstrap_evidence":bootstrap_prob>=min_bootstrap_prob,
        "score_above_floor":score>=float(min_score),
    }
    checks["score_improves"]=True if not require_score_improvement else score>champion_score+0.01
    return {"approved":all(checks.values()),"checks":checks,"score":score,"champion_score":champion_score,"require_score_improvement":bool(require_score_improvement)}
