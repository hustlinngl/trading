from __future__ import annotations
import numpy as np
import pandas as pd
from .engine import AdaptiveEngine
from .features import make_features, make_oos_features
from .strategy_lab import evolve, score_candidate
from .validation import robust_score, aggregate_fold_stats
from .evaluation import run_configured_backtest

def walk_forward(df: pd.DataFrame, settings, independent_test=False):
    rows=[]; train_bars=int(settings.walk_forward_train_bars); test_bars=int(settings.walk_forward_test_bars); step=max(int(settings.walk_forward_step),test_bars if independent_test else int(settings.walk_forward_step)); purge=int(getattr(settings,'validation_purge_bars',int(settings.horizon_bars)+max(1,int(settings.horizon_bars)//2))); cursor=train_bars+purge
    while cursor+test_bars<=len(df):
        test_start=cursor; train_end=test_start-purge; train_start=max(0,train_end-train_bars); train=df.iloc[train_start:train_end].copy(); test=df.iloc[test_start:test_start+test_bars].copy()
        if len(train)<int(getattr(settings,'min_train_rows',1500)): cursor+=step; continue
        engine=AdaptiveEngine(settings); engine.fit(train); test_features=make_oos_features(train,test,settings.horizon_bars,external_feature_lag_bars=getattr(settings,'external_feature_lag_bars',1)); pred=engine.model.predict(test_features); regimes=engine.regimes.transform(test_features); rp=engine.regimes.persistence(test_features); rprob=engine.regimes.semantic_probabilities(test_features); analog=engine.memory.query_many(test_features)
        from .meta import MetaPolicy, combine
        meta_x=MetaPolicy.frame(pred,test_features,regimes,analog,regime_persistence=rp,regime_probs=rprob); meta_p=engine.meta.predict_proba(meta_x); actions=[]
        for i,r in enumerate(pred.itertuples()):
            direction=1 if r.p_up>=0.5 else -1; directed_p=r.p_up if direction>0 else 1-r.p_up; directed_er_raw=r.expected_return if direction>0 else -r.expected_return; directed_lcb=r.expected_return_lcb if direction>0 else -r.expected_return_lcb; blend=float(np.clip(getattr(settings,'conformal_blend',0.60),0,1)); directed_er=directed_er_raw-blend*(directed_er_raw-directed_lcb); directed_mem=analog.iloc[i].edge if direction>0 else -analog.iloc[i].edge
            score=combine(directed_p,directed_er,directed_mem,str(regimes.iloc[i]),float(meta_p[i]),float(r.model_disagreement),float(analog.iloc[i].agreement),direction=direction,regime_weight=getattr(settings,'regime_weight',0.08),uncertainty_penalty_mult=getattr(settings,'uncertainty_penalty_mult',2.0),memory_weight=getattr(settings,'memory_weight',0.16),meta_weight=getattr(settings,'meta_weight',0.18),conviction_weight=getattr(settings,'conviction_weight',0.34),edge_weight=getattr(settings,'edge_weight',0.30))['score']
            hurdle=max(float(getattr(settings,'min_expected_return',0.0015)),(2*(float(getattr(settings,'fee_bps',0))+float(getattr(settings,'slippage_bps',0)))+2*float(getattr(settings,'impact_bps_per_sqrt',0))*np.sqrt(max(0,float(getattr(settings,'max_participation_pct',0.10))))+float(getattr(settings,'min_edge_after_cost_bps',5.0)))/10000.0)
            ok=score>=settings.decision_threshold and directed_p>=settings.probability_threshold and directed_er>=hurdle and meta_p[i]>=settings.meta_threshold
            actions.append('LONG' if ok and direction>0 else ('SHORT' if ok else 'FLAT'))
        bt=test.copy(); bt['atr_14']=test_features['atr_14']; result=run_configured_backtest(bt,pd.Series(actions,index=test.index),settings); rows.append({'train_start':str(train.index[0]),'train_end':str(train.index[-1]),'test_start':str(test.index[0]),'test_end':str(test.index[-1]),'purge_bars':purge,'median_regime':regimes.mode().iloc[0],**result.stats,'robust_score':robust_score(result.stats)}); cursor+=step
    return pd.DataFrame(rows)

def strategy_discovery(df,settings):
    split=int(len(df)*0.75); train_df,valid_df=df.iloc[:split],df.iloc[split:]; train_x,_,train_ret=make_features(train_df,settings.horizon_bars); valid_x=make_oos_features(train_df,valid_df,settings.horizon_bars,external_feature_lag_bars=getattr(settings,'external_feature_lag_bars',1)); valid_ret=valid_df.close.shift(-(int(settings.horizon_bars)+1))/valid_df.open.shift(-1)-1; best=evolve(train_x,train_ret,settings.population_size,settings.generations,settings.seed); serial=[{'train_score':float(s),'validation_score':float(score_candidate(c,valid_x,valid_ret)),'direction':c.direction,'rules':[r.__dict__ for r in c.rules]} for s,c in best]; serial.sort(key=lambda z:z['validation_score'],reverse=True); return serial

def research_report(df,settings):
    folds=walk_forward(df,settings); return {'folds':folds.to_dict('records'),'aggregate':aggregate_fold_stats(folds)}
