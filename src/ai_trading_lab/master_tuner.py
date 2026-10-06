from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import json
import math
import hashlib
import numpy as np
import pandas as pd
import joblib

try:
    import optuna
    HAVE_OPTUNA=True
except Exception:
    HAVE_OPTUNA=False

from .engine import AdaptiveEngine
from .features import make_oos_features
from .meta import MetaPolicy
from .policy import decide_actions
from .evaluation import run_configured_backtest, make_risk
from .backtest import run_backtest
from .validation import robust_score
from .purged_cv import walk_forward_splits
from .statistical_evidence import selection_adjusted_psr, deflated_sharpe_ratio, combinatorial_pbo, bootstrap_max_drawdown
from .fingerprint import strong_dataset_fingerprint

@dataclass
class FoldSnapshot:
    fold:int
    test:pd.DataFrame
    features:pd.DataFrame
    pred:pd.DataFrame
    regimes:pd.Series
    analog:pd.DataFrame
    meta_p:np.ndarray
    regime_persistence:pd.Series|None=None
    regime_probs:pd.DataFrame|None=None
    conformal_abs_residuals:np.ndarray|None=None
    conformal_scaled_residuals:np.ndarray|None=None

def representative_fold_ids(all_ids, target):
    ids=list(dict.fromkeys(int(x) for x in all_ids)); n=max(0,int(target))
    if n<=0 or not ids: return set()
    if len(ids)<=n: return set(ids)
    pos=np.linspace(0,len(ids)-1,n).round().astype(int)
    return {ids[int(i)] for i in pos}

def _fold_cache_key(df,settings):
    fields={'schema':6,'data':strong_dataset_fingerprint(df),'train':int(settings.walk_forward_train_bars),'test':int(settings.walk_forward_test_bars),'step':int(settings.walk_forward_step),'purge':int(getattr(settings,'validation_purge_bars',12)),'min_train':int(settings.min_train_rows),'horizon':int(settings.horizon_bars),'seed':int(settings.seed),'xgb':int(getattr(settings,'xgb_estimators',240)),'lgbm':int(getattr(settings,'lgbm_estimators',240)),'hist':int(getattr(settings,'hist_max_iter',260)),'regime_n_init':int(getattr(settings,'regime_n_init',5)),'memory_k':int(getattr(settings,'memory_k',32)),'feature_lag':int(getattr(settings,'external_feature_lag_bars',1)),'fee_bps':float(getattr(settings,'fee_bps',0.0)),'slippage_bps':float(getattr(settings,'slippage_bps',0.0)),'fold_subset':sorted(int(x) for x in only_folds) if only_folds is not None else None}
    return hashlib.sha256(json.dumps(fields,sort_keys=True).encode()).hexdigest()[:24]

def _collect_folds(df,settings,max_folds=None,only_folds=None):
    purge=int(getattr(settings,'validation_purge_bars',settings.horizon_bars+max(1,settings.horizon_bars//2))); cache_root=Path(getattr(settings,'master_tuning_cache_dir','data/master_cache')); cache_root.mkdir(parents=True,exist_ok=True); cache_path=cache_root/f"folds_{_fold_cache_key(df,settings,only_folds)}.joblib"; cached_all=None
    if cache_path.exists():
        try:
            cached_all=joblib.load(cache_path)
            if not isinstance(cached_all,list) or not all(isinstance(x,FoldSnapshot) for x in cached_all): cached_all=None
        except Exception: cached_all=None
    if cached_all is None:
        built=[]
        splits=list(walk_forward_splits(len(df),settings.walk_forward_train_bars,settings.walk_forward_test_bars,settings.walk_forward_step,purge,settings.min_train_rows,independent_test=True))
        if only_folds is not None: splits=[sp for sp in splits if sp.fold in only_folds]
        if max_folds is not None: splits=splits[:max(0,int(max_folds))]
        for sp in splits:
            train=df.iloc[sp.train_start:sp.train_end].copy(); test=df.iloc[sp.test_start:sp.test_end].copy(); engine=AdaptiveEngine(settings); engine.fit(train); feat=make_oos_features(train,test,settings.horizon_bars,external_feature_lag_bars=getattr(settings,'external_feature_lag_bars',1)); pred=engine.model.predict(feat); regimes=engine.regimes.transform(feat); rp=engine.regimes.persistence(feat); rprob=engine.regimes.semantic_probabilities(feat); analog=engine.memory.query_many(feat); meta_x=MetaPolicy.frame(pred,feat,regimes,analog,regime_persistence=rp,regime_probs=rprob); meta_p=engine.meta.predict_proba(meta_x); built.append(FoldSnapshot(sp.fold,test,feat,pred,regimes,analog,meta_p,rp,rprob,getattr(engine.model,'conformal_abs_residuals_',None),getattr(engine.model,'conformal_scaled_residuals_',None)))
        try: joblib.dump(built,cache_path,compress=3)
        except Exception: pass
        cached_all=built
    out=[]
    for snap in cached_all:
        if only_folds is not None and snap.fold not in only_folds: continue
        out.append(snap)
        if max_folds is not None and len(out)>=max_folds: break
    return out

def _actions(snapshot,settings,params):
    pred=snapshot.pred.copy(); er=pred.expected_return.to_numpy(float)
    if snapshot.conformal_scaled_residuals is not None and len(snapshot.conformal_scaled_residuals) and 'atr_pct' in snapshot.features:
        level=float(np.clip(params.get('conformal_level',getattr(settings,'conformal_level',0.90)),0.50,0.999)); q=float(np.quantile(snapshot.conformal_scaled_residuals,level,method='higher')); margin=np.maximum(np.abs(snapshot.features.atr_pct.to_numpy(float)),1e-6)*q; direction=np.where(pred.p_up.to_numpy(float)>=0.5,1.0,-1.0); pred['expected_return_lcb']=er-margin*direction
    actions,_=decide_actions(pred,snapshot.regimes,snapshot.analog,snapshot.meta_p,settings,regime_persistence=snapshot.regime_persistence,regime_probs=snapshot.regime_probs,**params); return actions

def _evaluate_params(folds,settings,params,cost_multipliers=(1.0,1.25,1.5)):
    rows=[]
    for snap in folds:
        for cm in cost_multipliers:
            bt=snap.test.copy(); bt['atr_14']=snap.features.atr_14; res=run_configured_backtest(bt,_actions(snap,settings,params),settings,stop_atr_mult=params['stop_atr_mult'],take_profit_rr=params['take_profit_rr'],max_holding_bars=params['max_holding_bars'],cost_multiplier=cm); row={**res.stats,'fold':snap.fold,'cost_multiplier':cm,'robust_score':robust_score(res.stats)}
            if not res.trades.empty:
                entry_times=pd.to_datetime(res.trades.entry_timestamp,utc=True); regime_index=pd.DatetimeIndex(snap.regimes.index); positions=regime_index.searchsorted(entry_times,side='left')-1; valid=positions>=0; trade_reg=pd.Series(index=entry_times,dtype=object)
                if valid.any(): trade_reg.loc[valid]=snap.regimes.iloc[positions[valid]].to_numpy()
                counts=trade_reg.dropna().value_counts(); means=res.trades.assign(_reg=trade_reg.to_numpy()).dropna(subset=['_reg']).groupby('_reg').pnl_net.mean(); row['regime_trade_coverage']=min(1.0,len(counts)/5.0); row['positive_regime_ratio']=float(np.mean(means.to_numpy(float)>0)) if len(means) else 0.0; row['worst_regime_trade_pnl']=float(means.min()) if len(means) else 0.0
            else: row.update({'regime_trade_coverage':0.0,'positive_regime_ratio':0.0,'worst_regime_trade_pnl':0.0})
            rows.append(row)
    frame=pd.DataFrame(rows)
    if frame.empty: return {'objective':-1e9,'rows':frame,'median_score':-1e9,'score_std':1e9,'positive_ratio':0.0,'worst_drawdown':-1.0}
    fold_scores=frame.groupby('fold').robust_score.median(); fold_trade_counts=frame.groupby('fold').trades.sum(); q25=float(fold_scores.quantile(0.25)); median=float(fold_scores.median()); worst=float(fold_scores.min()); coverage=float(np.mean(np.clip(fold_trade_counts.to_numpy(float)/10.0,0,1))); positive=float((fold_scores>0).mean()); score_std=float(fold_scores.std(ddof=0))
    normal=frame.loc[np.isclose(frame.cost_multiplier,1.0)].groupby('fold').robust_score.median(); stress=frame.loc[frame.cost_multiplier>1].groupby('fold').robust_score.median(); common=normal.index.intersection(stress.index); cost_degradation=float(np.median((normal.loc[common]-stress.loc[common]).to_numpy(float))) if len(common) else 0.0
    search_count=max(1,int(getattr(settings,'_tuning_trial_count',40))); selection_penalty=min(0.20,score_std*math.sqrt(2*math.log(max(search_count,2)))/math.sqrt(max(len(fold_scores),1))); regime_cov=float(frame.regime_trade_coverage.median()); regime_pos=float(frame.positive_regime_ratio.median())
    objective=float(0.43*q25+0.27*median+0.14*worst+0.05*positive+0.07*coverage+0.03*regime_cov+0.03*regime_pos-0.10*score_std-0.08*max(0,cost_degradation-0.05)-selection_penalty); trades=int(frame.trades.sum()); worst_dd=float(frame.max_drawdown.min())
    if trades<30: objective-=0.40*(30-trades)/30
    if worst_dd<-0.20: objective-=1.5*(-0.20-worst_dd)
    return {'objective':objective,'rows':frame,'median_score':float(frame.robust_score.median()),'score_std':float(score_std),'positive_ratio':positive,'worst_drawdown':worst_dd,'total_trades':trades,'q25_score':q25,'fold_trade_coverage':coverage,'cost_degradation':cost_degradation,'selection_penalty':float(selection_penalty),'regime_coverage':regime_cov,'positive_regime_ratio':regime_pos}

def _suggest(trial):
    return {'probability_threshold':trial.suggest_float('probability_threshold',0.52,0.66),'min_expected_return':trial.suggest_float('min_expected_return',0.00005,0.0075,log=True),'decision_threshold':trial.suggest_float('decision_threshold',0.02,0.24),'meta_threshold':trial.suggest_float('meta_threshold',0.50,0.62),'stop_atr_mult':trial.suggest_float('stop_atr_mult',1.2,2.8),'take_profit_rr':trial.suggest_float('take_profit_rr',1.3,3.8),'max_holding_bars':trial.suggest_int('max_holding_bars',16,96,step=8),'regime_weight':trial.suggest_float('regime_weight',0,0.12),'uncertainty_penalty_mult':trial.suggest_float('uncertainty_penalty_mult',0.5,4.0),'memory_weight':trial.suggest_float('memory_weight',0.05,0.28),'meta_weight':trial.suggest_float('meta_weight',0.08,0.28),'conviction_weight':trial.suggest_float('conviction_weight',0.20,0.45),'edge_weight':trial.suggest_float('edge_weight',0.18,0.42),'conformal_blend':trial.suggest_float('conformal_blend',0,1),'conformal_level':trial.suggest_float('conformal_level',0.70,0.95)}

def _placebo_test(folds,settings,params,seed=42,n=6):
    rng=np.random.default_rng(seed); real=_evaluate_params(folds,settings,params,cost_multipliers=(1.0,))['objective']; null_values=[]
    for _ in range(max(1,int(n))):
        shuffled=[]
        for snap in folds:
            order=rng.permutation(len(snap.test)); pred=snap.pred.iloc[order].copy(); pred.index=snap.pred.index; analog=snap.analog.iloc[order].copy(); analog.index=snap.analog.index; regimes=snap.regimes.iloc[order].copy(); regimes.index=snap.regimes.index; persistence=None if snap.regime_persistence is None else snap.regime_persistence.iloc[order]; probs=None if snap.regime_probs is None else snap.regime_probs.iloc[order]
            if persistence is not None: persistence.index=snap.regime_persistence.index
            if probs is not None: probs.index=snap.regime_probs.index
            shuffled.append(FoldSnapshot(snap.fold,snap.test,snap.features,pred,regimes,analog,np.asarray(snap.meta_p)[order],persistence,probs,snap.conformal_abs_residuals,snap.conformal_scaled_residuals))
        null_values.append(float(_evaluate_params(shuffled,settings,params,cost_multipliers=(1.0,))['objective']))
    arr=np.asarray(null_values,float); q95=float(np.quantile(arr,0.95)) if len(arr) else float('inf')
    return {'real_objective':float(real),'placebo_mean':float(np.mean(arr)) if len(arr) else 0.0,'placebo_q95':q95,'placebo_max':float(np.max(arr)) if len(arr) else 0.0,'placebo_samples':int(len(arr)),'passes_95pct_negative_control':bool(real>q95)}

def _stability_test(folds,settings,best,seed=42,n=12):
    rng=np.random.default_rng(seed); base=_evaluate_params(folds,settings,best,cost_multipliers=(1,1.25,1.5))['objective']; vals=[]
    for _ in range(int(n)):
        p=dict(best)
        for key in ['probability_threshold','decision_threshold','meta_threshold','regime_weight','uncertainty_penalty_mult','memory_weight','meta_weight','conviction_weight','edge_weight']:
            scale=0.04 if key!='uncertainty_penalty_mult' else 0.08; p[key]=float(p[key]*(1+rng.normal(0,scale)))
        p['probability_threshold']=float(np.clip(p['probability_threshold'],0.50,0.80)); p['meta_threshold']=float(np.clip(p['meta_threshold'],0.50,0.75)); p['decision_threshold']=float(np.clip(p['decision_threshold'],0.01,0.50)); p['min_expected_return']=float(max(1e-5,p['min_expected_return']*(1+rng.normal(0,0.04)))); p['stop_atr_mult']=float(max(0.8,p['stop_atr_mult']*(1+rng.normal(0,0.05)))); p['take_profit_rr']=float(max(1.0,p['take_profit_rr']*(1+rng.normal(0,0.05)))); p['max_holding_bars']=int(max(8,round(p['max_holding_bars']*(1+rng.normal(0,0.08))/8)*8)); vals.append(_evaluate_params(folds,settings,p,cost_multipliers=(1.25,1.5))['objective'])
    vals=np.asarray(vals,float); return {'base_objective':float(base),'median_perturbed':float(np.median(vals)),'worst_perturbed':float(np.min(vals)),'degradation_median':float(base-np.median(vals)),'degradation_worst':float(base-np.min(vals)),'stable':bool(np.median(vals)>=base-0.10 and np.min(vals)>=base-0.35)}

def master_tune(df,settings,trials=40,final_holdout_frac=0.15,stability_samples=8,tuning_folds=6,save_path=None):
    if not 0.05<=final_holdout_frac<=0.30: raise ValueError('final_holdout_frac must be between 0.05 and 0.30')
    cut=int(len(df)*(1-final_holdout_frac)); tuning_df=df.iloc[:cut].copy(); holdout_df=df.iloc[cut:].copy()
    split_meta=list(walk_forward_splits(len(tuning_df),settings.walk_forward_train_bars,settings.walk_forward_test_bars,settings.walk_forward_step,int(getattr(settings,'validation_purge_bars',settings.horizon_bars+max(1,settings.horizon_bars//2))),settings.min_train_rows,independent_test=True))
    if not split_meta: raise ValueError('Not enough data for purged master tuning folds')
    all_ids=[sp.fold for sp in split_meta]
    if len(all_ids)<=int(tuning_folds): selected_ids=set(all_ids)
    else: selected_ids={all_ids[int(i)] for i in np.linspace(0,len(all_ids)-1,int(tuning_folds)).round().astype(int)}
    folds=_collect_folds(tuning_df,settings,only_folds=selected_ids)
    if not folds: raise ValueError('Selected master tuning folds could not be built')
    if not HAVE_OPTUNA: raise RuntimeError('Optuna is required for master tuning')
    sampler_kwargs={'seed':settings.seed}
    try: sampler_kwargs.update({'multivariate':True,'group':True})
    except Exception: pass
    setattr(settings,'_tuning_trial_count',max(1,int(trials))); study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(**sampler_kwargs))
    baseline={'probability_threshold':float(getattr(settings,'probability_threshold',0.57)),'min_expected_return':float(getattr(settings,'min_expected_return',0.0015)),'decision_threshold':float(getattr(settings,'decision_threshold',0.16)),'meta_threshold':float(getattr(settings,'meta_threshold',0.53)),'stop_atr_mult':float(getattr(settings,'stop_atr_mult',1.8)),'take_profit_rr':float(getattr(settings,'take_profit_rr',2.2)),'max_holding_bars':int(getattr(settings,'max_holding_bars',48)),'regime_weight':float(getattr(settings,'regime_weight',0.08)),'uncertainty_penalty_mult':float(getattr(settings,'uncertainty_penalty_mult',2.0)),'memory_weight':float(getattr(settings,'memory_weight',0.16)),'meta_weight':float(getattr(settings,'meta_weight',0.18)),'conviction_weight':float(getattr(settings,'conviction_weight',0.34)),'edge_weight':float(getattr(settings,'edge_weight',0.30)),'conformal_blend':float(getattr(settings,'conformal_blend',0.60)),'conformal_level':float(getattr(settings,'conformal_level',0.90))}
    try: study.enqueue_trial(baseline)
    except Exception: pass
    cache={}
    trial_fold_scores={}
    def objective(trial):
        params=_suggest(trial); rep=_evaluate_params(folds,settings,params,cost_multipliers=(1.0,1.5)); cache[trial.number]=rep
        if isinstance(rep.get('rows'),pd.DataFrame) and not rep['rows'].empty:
            trial_fold_scores[int(trial.number)] = rep['rows'].groupby('fold').robust_score.median().to_dict()
        return float(rep['objective'])
    study.optimize(objective,n_trials=max(1,int(trials)),show_progress_bar=False)
    completed={int(t.number) for t in study.trials if getattr(t.state,'name',str(t.state).split('.')[-1])=='COMPLETE'}; evaluated=[cache.get(n,{}) for n in sorted(completed) if n in cache]
    if not evaluated: raise RuntimeError('Master tuning produced no completed trial evaluations; refuse to select an arbitrary candidate.')
    if max(int(x.get('total_trades',0)) for x in evaluated)<=0: raise RuntimeError('Master tuning found no trading activity on the tuning folds. Refuse to select an arbitrary zero-trade candidate.')
    best=dict(study.best_params); rep=cache.get(study.best_trial.number) or _evaluate_params(folds,settings,best)
    if selected_ids==set(all_ids): folds_all=folds
    else: folds_all=sorted(folds+_collect_folds(tuning_df,settings,only_folds=set(all_ids)-selected_ids),key=lambda x:x.fold)
    full_rep=_evaluate_params(folds_all,settings,best,cost_multipliers=(1.0,1.25,1.5));
    pbo_evidence={'available':False,'reason':'insufficient_trial_fold_matrix','pbo':None,'combinations':0};
    try:
        trial_ids=sorted(trial_fold_scores.keys()); fold_ids=sorted({int(k) for d in trial_fold_scores.values() for k in d.keys()})
        if len(trial_ids)>=8 and len(fold_ids)>=6:
            mat=np.full((len(fold_ids),len(trial_ids)),np.nan,dtype=float)
            for j,tid in enumerate(trial_ids):
                for i,fid in enumerate(fold_ids):
                    if fid in trial_fold_scores[tid]: mat[i,j]=float(trial_fold_scores[tid][fid])
            pbo_evidence=combinatorial_pbo(mat,partitions=min(8,len(fold_ids)),seed=settings.seed)
    except Exception as exc:
        pbo_evidence={'available':False,'reason':f'{type(exc).__name__}: {exc}','pbo':None,'combinations':0}
    stability=_stability_test(folds,settings,best,seed=settings.seed,n=stability_samples); placebo=_placebo_test(folds,settings,best,seed=settings.seed+17,n=min(6,max(3,stability_samples//2)))
    tuned=replace(settings,**{k:v for k,v in best.items() if hasattr(settings,k)}); engine=AdaptiveEngine(tuned); engine.fit(tuning_df); feat=make_oos_features(tuning_df,holdout_df,tuned.horizon_bars,external_feature_lag_bars=getattr(tuned,'external_feature_lag_bars',1)); pred=engine.model.predict(feat); regimes=engine.regimes.transform(feat); analog=engine.memory.query_many(feat); rp=engine.regimes.persistence(feat); rprob=engine.regimes.semantic_probabilities(feat); meta_p=engine.meta.predict_proba(MetaPolicy.frame(pred,feat,regimes,analog,regime_persistence=rp,regime_probs=rprob)); snap=FoldSnapshot(-1,holdout_df,feat,pred,regimes,analog,meta_p,rp,rprob,getattr(engine.model,'conformal_abs_residuals_',None),getattr(engine.model,'conformal_scaled_residuals_',None)); final=_evaluate_params([snap],tuned,best,cost_multipliers=(1,1.25,1.5))
    final_stats=final['rows'].copy() if isinstance(final.get('rows'),pd.DataFrame) else pd.DataFrame(); final_evidence={}
    try:
        final_risk=make_risk(tuned,stop_atr_mult=best['stop_atr_mult'],take_profit_rr=best['take_profit_rr']); final_bt=run_backtest(holdout_df.assign(atr_14=feat['atr_14']),_actions(snap,tuned,best),final_risk,tuned.initial_cash,fee_bps=tuned.fee_bps,slippage_bps=tuned.slippage_bps,max_holding_bars=int(best['max_holding_bars']),intrabar_barriers=getattr(tuned,'intrabar_barriers',True),impact_bps_per_sqrt=getattr(tuned,'impact_bps_per_sqrt',1.5),force_daily_loss_exit=getattr(tuned,'force_daily_loss_exit',True),short_borrow_bps_per_bar=getattr(tuned,'short_borrow_bps_per_bar',0.0)); bar_returns=final_bt.equity.pct_change().replace([np.inf,-np.inf],np.nan).dropna().to_numpy(float)
        if len(bar_returns)>=30:
            periods=365.25*24*3600/max(1.0,pd.Series(holdout_df.index).diff().dropna().dt.total_seconds().median()) if len(holdout_df.index)>1 else 365.0
            final_evidence={'selection_adjusted_psr_proxy':selection_adjusted_psr(bar_returns,len(study.trials),periods_per_year=periods),
            'deflated_sharpe_ratio':deflated_sharpe_ratio(bar_returns,len(study.trials),periods_per_year=periods),'path_stress':bootstrap_max_drawdown(bar_returns,reps=min(1000,max(200,stability_samples*100)),block=max(1,int(best['max_holding_bars'])//4),seed=settings.seed)}
    except Exception as exc: final_evidence={'error':f'{type(exc).__name__}: {exc}'}
    result={'data':{'rows_total':len(df),'rows_tuning':len(tuning_df),'rows_final_holdout':len(holdout_df),'final_holdout_start':str(holdout_df.index[0]),'final_holdout_end':str(holdout_df.index[-1])},'best_params':best,'study_best_value':float(study.best_value),'trials':len(study.trials),'tuning_summary':{k:v for k,v in rep.items() if k!='rows'},'full_tuning_set_verification':{k:v for k,v in full_rep.items() if k!='rows'},'full_tuning_fold_rows': full_rep['rows'].to_dict(orient='records'),'stability':stability,'negative_control_placebo':placebo,'final_holdout':{k:v for k,v in final.items() if k!='rows'},'final_holdout_summary':final_stats.to_dict(orient='records'),'final_holdout_rows':final['rows'].to_dict(orient='records'),'final_statistical_evidence':final_evidence,'probability_of_backtest_overfitting':pbo_evidence,'trial_count_ledger':int(len(study.trials)),'folds_used_for_optimization':[{'fold':x.fold,'start':str(x.test.index[0]),'end':str(x.test.index[-1])} for x in folds],'all_tuning_folds':[{'fold':x.fold,'start':str(x.test.index[0]),'end':str(x.test.index[-1])} for x in folds_all],'candidate_fold_count':len(all_ids),'representative_audit_fold_count':len(audit_ids),'optimization_fold_count':len(selected_ids),'interpretation':'No finite tuning run can establish a perfect or future-proof optimum. The objective rewards robustness, cost tolerance and parameter plateau stability while preserving an untouched final holdout.'}
    if save_path:
        path=Path(save_path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    return result
