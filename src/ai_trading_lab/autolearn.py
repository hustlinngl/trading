from __future__ import annotations
from pathlib import Path
import json, numpy as np, pandas as pd
from .engine import AdaptiveEngine
from .research import walk_forward
from .features import make_oos_features
from .evaluation import run_configured_backtest
from .objectives import robust_performance_utility
from .promotion import promotion_gate
from .growth import GrowthRegistry
from .fingerprint import strong_dataset_fingerprint

def evaluate_engine(df,settings):
    folds=walk_forward(df,settings,independent_test=True)
    if folds.empty: return -np.inf,{"folds":0,"_fold_frame":folds}
    score=float(folds["robust_score"].median()-0.20*abs(folds["robust_score"].std(ddof=0)))
    fold_scores=folds["robust_score"].to_numpy(float)
    champion_hint=float(getattr(settings,"_champion_score_hint",-np.inf))
    if not np.isfinite(champion_hint):
        bootstrap_prob=1.0
    else:
        rng=np.random.default_rng(int(getattr(settings,"seed",42)))
        medians=np.empty(2000,dtype=float)
        for i in range(len(medians)):
            medians[i]=float(np.median(fold_scores[rng.integers(0,len(fold_scores),len(fold_scores))]))
        bootstrap_prob=float(np.mean(medians>champion_hint))
    return score,{"folds":int(len(folds)),"median_score":float(folds["robust_score"].median()),"score_std":float(folds["robust_score"].std(ddof=0)),"positive_folds":int((folds["robust_score"]>0).sum()),"positive_fold_ratio":float((folds["robust_score"]>0).mean()),"worst_drawdown":float(folds["max_drawdown"].min()),"total_trades":int(folds["trades"].sum()),"median_return":float(folds["total_return"].median()),"median_sharpe":float(folds["sharpe_like"].median()),"bootstrap_superiority_prob":bootstrap_prob,"_fold_frame":folds}

def _final_holdout_eval(df,settings):
    frac=float(np.clip(getattr(settings,"final_holdout_frac",0.15),0.05,0.30)); cut=int(len(df)*(1-frac))
    if len(df)-cut<max(20,int(getattr(settings,"base_min_holdout_trades",20))): return {"passed":False,"reason":"insufficient_holdout_rows"}
    tuning,holdout=df.iloc[:cut].copy(),df.iloc[cut:].copy(); engine=AdaptiveEngine(settings); engine.fit(tuning)
    features=make_oos_features(tuning,holdout,settings.horizon_bars,external_feature_lag_bars=getattr(settings,"external_feature_lag_bars",1))
    pred=engine.model.predict(features); regimes=engine.regimes.transform(features); persistence=engine.regimes.persistence(features); probs=engine.regimes.semantic_probabilities(features); analog=engine.memory.query_many(features)
    from .meta import MetaPolicy
    meta_p=engine.meta.predict_proba(MetaPolicy.frame(pred,features,regimes,analog,regime_persistence=persistence,regime_probs=probs))
    from .policy import decide_actions
    actions,_=decide_actions(pred,regimes,analog,meta_p,settings,regime_persistence=persistence,regime_probs=probs)
    bt=holdout.copy(); bt["atr_14"]=features["atr_14"]; result=run_configured_backtest(bt,actions,settings); stats=dict(result.stats)
    utility=robust_performance_utility(stats,min_trades=int(getattr(settings,"base_min_holdout_trades",20)),max_drawdown=float(getattr(settings,"base_max_holdout_drawdown",-0.25)))
    checks={"minimum_trades":int(stats.get("trades",0))>=int(getattr(settings,"base_min_holdout_trades",20)),"positive_return":float(stats.get("total_return",0))>0 if bool(getattr(settings,"base_require_positive_holdout_return",True)) else True,"profit_factor":float(stats.get("profit_factor",0))>=float(getattr(settings,"base_min_holdout_profit_factor",1.0)),"drawdown":float(stats.get("max_drawdown",-1))>=float(getattr(settings,"base_max_holdout_drawdown",-0.25)),"utility":float(utility)>=float(getattr(settings,"base_min_holdout_utility",0))}
    return {"passed":bool(all(checks.values())),"checks":checks,"stats":stats,"utility":float(utility),"holdout_rows":len(holdout)}

def auto_update(df,settings,model_dir="models"):
    mdir=Path(model_dir); mdir.mkdir(parents=True,exist_ok=True); state_path=mdir/"promotion_state.json"; state=json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}; champion=float(state.get("score",-np.inf)); fp=strong_dataset_fingerprint(df)
    if (mdir/"signal_model.joblib").exists() and state.get("data_fingerprint")==fp: return {"promoted":False,"skipped":True,"skip_reason":"dataset_unchanged","data_fingerprint":fp,"champion_score":champion,"challenger_score":champion}
    setattr(settings,"_champion_score_hint",champion)
    score,stats=evaluate_engine(df,settings); stats["score"]=score; stats["positive_fold_ratio"]=stats.get("positive_folds",0)/max(stats.get("folds",1),1)
    gate=promotion_gate(stats,champion,min_folds=getattr(settings,"growth_min_folds",4),min_trades=max(settings.min_trades_promotion,getattr(settings,"growth_min_trades",50)),max_dd=settings.max_promotion_drawdown,min_positive_fold_ratio=getattr(settings,"growth_min_positive_fold_ratio",.65),min_bootstrap_prob=getattr(settings,"growth_min_bootstrap_probability",.58))
    holdout=_final_holdout_eval(df,settings); gate["final_holdout"]=bool(holdout.get("passed")); gate["approved"]=bool(gate.get("approved") and holdout.get("passed"))
    result={"challenger_score":score,"champion_score":champion,"stats":stats,"promotion_gate":gate,"final_holdout":holdout,"promoted":bool(gate["approved"]),"data_fingerprint":fp}
    if gate["approved"]:
        engine=AdaptiveEngine(settings); engine.fit(df); engine.save(mdir); state={"score":score,"stats":stats,"version":int(state.get("version",0))+1,"data_fingerprint":fp,"final_holdout":holdout}; state_path.write_text(json.dumps(state,indent=2,default=str),encoding="utf-8")
        try: GrowthRegistry(getattr(settings,"memory_db","data/memory.sqlite")).add_model_version("signal","promoted",score,stats,str(mdir))
        except Exception: pass
    else: state.update({"data_fingerprint":fp,"last_challenger_score":score,"last_stats":stats,"last_final_holdout":holdout}); state_path.write_text(json.dumps(state,indent=2,default=str),encoding="utf-8")
    return result
