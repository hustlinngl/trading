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
from .deployment import model_semantics_fingerprint, deployment_semantics_fingerprint, refresh_deployment_manifest

def evaluate_engine(df,settings):
    folds=walk_forward(df,settings,independent_test=True)
    if folds.empty: return -np.inf,{"folds":0,"_fold_frame":folds}
    score=float(folds["robust_score"].median()-0.20*abs(folds["robust_score"].std(ddof=0)))
    return score,{"folds":int(len(folds)),"median_score":float(folds["robust_score"].median()),"score_std":float(folds["robust_score"].std(ddof=0)),"positive_folds":int((folds["robust_score"]>0).sum()),"positive_fold_ratio":float((folds["robust_score"]>0).mean()),"worst_drawdown":float(folds["max_drawdown"].min()),"total_trades":int(folds["trades"].sum()),"median_return":float(folds["total_return"].median()),"median_sharpe":float(folds["sharpe_like"].median()),"_fold_frame":folds}

def _final_holdout_eval(df,settings,engine=None):
    frac=float(np.clip(getattr(settings,"final_holdout_frac",0.15),0.05,0.30)); cut=int(len(df)*(1-frac))
    if len(df)-cut<max(20,int(getattr(settings,"base_min_holdout_trades",20))): return {"passed":False,"reason":"insufficient_holdout_rows"}
    tuning,holdout=df.iloc[:cut].copy(),df.iloc[cut:].copy()
    engine=engine or AdaptiveEngine(settings)
    if not getattr(engine.model,"ready",False): engine.fit(tuning)
    features=make_oos_features(tuning,holdout,settings.horizon_bars,external_feature_lag_bars=getattr(settings,"external_feature_lag_bars",1))
    pred=engine.predict_frame(features); actions=pred["action"] if "action" in pred else pd.Series("FLAT",index=holdout.index)
    bt=holdout.copy(); bt["atr_14"]=features["atr_14"]; result=run_configured_backtest(bt,actions,settings); stats=dict(result.stats)
    utility=robust_performance_utility(stats,min_trades=int(getattr(settings,"base_min_holdout_trades",20)),max_drawdown=float(getattr(settings,"base_max_holdout_drawdown",-0.25)))
    checks={"minimum_trades":int(stats.get("trades",0))>=int(getattr(settings,"base_min_holdout_trades",20)),"positive_return":float(stats.get("total_return",0))>0 if bool(getattr(settings,"base_require_positive_holdout_return",True)) else True,"profit_factor":float(stats.get("profit_factor",0))>=float(getattr(settings,"base_min_holdout_profit_factor",1.0)),"drawdown":float(stats.get("max_drawdown",-1))>=float(getattr(settings,"base_max_holdout_drawdown",-0.25)),"utility":float(utility)>=float(getattr(settings,"base_min_holdout_utility",0))}
    return {"passed":bool(all(checks.values())),"checks":checks,"stats":stats,"utility":float(utility),"holdout_rows":len(holdout)}

def auto_update(df,settings,model_dir="models"):
    mdir=Path(model_dir)
    mdir.mkdir(parents=True,exist_ok=True)
    state_path=mdir/"promotion_state.json"
    state=json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    previous_score=float(state.get("score",-np.inf))
    fp=strong_dataset_fingerprint(df)
    model_semantics=model_semantics_fingerprint(settings)
    deployment_semantics=deployment_semantics_fingerprint(settings)

    if (
        (mdir/"signal_model.joblib").exists()
        and state.get("data_fingerprint")==fp
        and state.get("model_semantics_fingerprint")==model_semantics
        and state.get("deployment_semantics_fingerprint")==deployment_semantics
    ):
        return {
            "promoted":False,
            "skipped":True,
            "skip_reason":"dataset_and_configuration_unchanged",
            "data_fingerprint":fp,
            "champion_score":previous_score,
            "challenger_score":previous_score,
            "model_semantics_fingerprint":model_semantics,
            "deployment_semantics_fingerprint":deployment_semantics,
        }

    score,stats=evaluate_engine(df,settings)
    stats["score"]=score
    stats["positive_fold_ratio"]=stats.get("positive_folds",0)/max(stats.get("folds",1),1)

    fold_scores=stats.get("_fold_frame",pd.DataFrame())
    if isinstance(fold_scores,pd.DataFrame) and not fold_scores.empty:
        rng=np.random.default_rng(int(getattr(settings,"seed",42)))
        arr=fold_scores["robust_score"].to_numpy(float)
        boot=np.empty(2000,dtype=float)
        for i in range(len(boot)):
            boot[i]=float(np.median(arr[rng.integers(0,len(arr),len(arr))]))
        stats["bootstrap_superiority_prob"]=float(np.mean(boot>0.0))
    else:
        stats["bootstrap_superiority_prob"]=0.0

    gate=promotion_gate(
        stats,
        previous_score,
        min_folds=getattr(settings,"growth_min_folds",4),
        min_trades=max(settings.min_trades_promotion,getattr(settings,"growth_min_trades",50)),
        max_dd=settings.max_promotion_drawdown,
        min_positive_fold_ratio=getattr(settings,"growth_min_positive_fold_ratio",.65),
        min_bootstrap_prob=getattr(settings,"growth_min_bootstrap_probability",.58),
        min_score=0.0,
        require_score_improvement=False,
    )

    holdout=_final_holdout_eval(df,settings)
    champion_holdout={}
    holdout_superiority=True
    if mdir/"signal_model.joblib".exists():
        try:
            from .deployment import bundle_compatibility
            compatible,reason=bundle_compatibility(settings,mdir,settings.symbol)
            if not compatible:
                champion_holdout={"passed":False,"reason":reason,"utility":-np.inf}
            else:
                champion_engine=AdaptiveEngine(settings).load(mdir)
                champion_holdout=_final_holdout_eval(df,settings,champion_engine)
        except Exception as exc:
            champion_holdout={"passed":False,"reason":f"{type(exc).__name__}:{exc}","utility":-np.inf}
        holdout_superiority=float(holdout.get("utility",-np.inf))>float(champion_holdout.get("utility",-np.inf))+0.01

    gate["final_holdout"]=bool(holdout.get("passed"))
    gate["checks"]["holdout_superiority"]=bool(holdout_superiority)
    gate["approved"]=bool(gate.get("approved") and holdout.get("passed") and holdout_superiority)

    result={
        "challenger_score":score,
        "champion_score":previous_score,
        "stats":stats,
        "promotion_gate":gate,
        "final_holdout":holdout,
        "champion_holdout":champion_holdout,
        "promoted":bool(gate["approved"]),
        "data_fingerprint":fp,
    }
    if gate["approved"]:
        engine=AdaptiveEngine(settings)
        engine.fit(df)
        engine.save(mdir)
        # Promotion changes the executable artifact, so refresh all provenance/evidence
        # artifacts before the bundle can become signal-eligible again.
        (mdir/"base_training_meta.json").write_text(json.dumps({
            "symbol":str(settings.symbol),
            "rows":int(len(df)),
            "start":str(df.index.min()),
            "end":str(df.index.max()),
            "timeframe":str(settings.timeframe),
            "data_fingerprint":fp,
            "model_semantics_fingerprint":model_semantics_fingerprint(settings),
            "deployment_semantics_fingerprint":deployment_semantics_fingerprint(settings),
            "trained_at":pd.Timestamp.now(tz="UTC").isoformat(),
            "promotion_source":"auto_update",
        },indent=2,default=str),encoding="utf-8")
        (mdir/"base_holdout_report.json").write_text(json.dumps({
            "symbol":str(settings.symbol),
            "rows":int(len(df)),
            "train_rows":int(len(df)-int(holdout.get("holdout_rows",0))),
            "data_fingerprint":fp,
            "holdout_rows":int(holdout.get("holdout_rows",0)),
            "holdout":holdout.get("stats",{}),
            "utility":float(holdout.get("utility",-np.inf)),
        },indent=2,default=str),encoding="utf-8")
        try:
            from .trade_window import train_trade_window_backbone
            duration_path=mdir/"trade_window_specialist.joblib"
            duration_report=train_trade_window_backbone(df,settings,holdout_frac=float(getattr(settings,"final_holdout_frac",0.15)),save_path=duration_path)
            (mdir/"trade_window_training_report.json").write_text(json.dumps(duration_report,indent=2,default=str),encoding="utf-8")
        except Exception as exc:
            duration_report={"production_ready":False,"reason":f"{type(exc).__name__}:{exc}"}
            (mdir/"trade_window_training_report.json").write_text(json.dumps(duration_report,indent=2,default=str),encoding="utf-8")
        manifest_root=Path(mdir).parents[2] if len(Path(mdir).parts)>=3 else "."
        deployment_manifest=refresh_deployment_manifest(settings,manifest_root)
        state={
            "score":score,
            "stats":stats,
            "version":int(state.get("version",0))+1,
            "data_fingerprint":fp,
            "model_semantics_fingerprint":model_semantics,
            "deployment_semantics_fingerprint":deployment_semantics,
            "final_holdout":holdout,
            "champion_holdout":champion_holdout,
            "deployment_manifest":deployment_manifest,
            "duration_report":duration_report,
        }
        state_path.write_text(json.dumps(state,indent=2,default=str),encoding="utf-8")
        result["deployment_manifest"]=deployment_manifest
        result["deployment_ready"]=bool(deployment_manifest.get("ready",False))
        result["duration_report"]=duration_report
        try:
            GrowthRegistry(getattr(settings,"memory_db","data/memory.sqlite")).add_model_version("signal","promoted",score,stats,str(mdir))
        except Exception:
            pass
    else:
        state.update({
            "data_fingerprint":fp,
            "model_semantics_fingerprint":model_semantics,
            "deployment_semantics_fingerprint":deployment_semantics,
            "last_challenger_score":score,
            "last_champion_score":previous_score,
            "last_stats":stats,
            "last_final_holdout":holdout,
            "last_champion_holdout":champion_holdout,
        })
        state_path.write_text(json.dumps(state,indent=2,default=str),encoding="utf-8")
    return result
