from __future__ import annotations
import numpy as np
import pandas as pd
from .engine import AdaptiveEngine
from .features import make_features, make_oos_features
from .strategy_lab import evolve, score_candidate
from .validation import robust_score, aggregate_fold_stats
from .evaluation import run_configured_backtest
from .labels import triple_barrier_labels
from .research_ledger import register_trials

def walk_forward(df: pd.DataFrame, settings, independent_test=False):
    rows=[]; train_bars=int(settings.walk_forward_train_bars); test_bars=int(settings.walk_forward_test_bars); step=max(int(settings.walk_forward_step),test_bars if independent_test else int(settings.walk_forward_step)); purge=int(getattr(settings,'validation_purge_bars',int(settings.horizon_bars)+max(1,int(settings.horizon_bars)//2))); cursor=train_bars+purge
    while cursor+test_bars<=len(df):
        test_start=cursor; train_end=test_start-purge; train_start=max(0,train_end-train_bars); train=df.iloc[train_start:train_end].copy(); test=df.iloc[test_start:test_start+test_bars].copy()
        if len(train)<int(getattr(settings,'min_train_rows',1500)): cursor+=step; continue
        engine=AdaptiveEngine(settings); engine.fit(train); test_features=make_oos_features(train,test,settings.horizon_bars,external_feature_lag_bars=getattr(settings,'external_feature_lag_bars',1)); pred=engine.model.predict(test_features); regimes=engine.regimes.transform(test_features); rp=engine.regimes.persistence(test_features); rprob=engine.regimes.semantic_probabilities(test_features); analog=engine.memory.query_many(test_features)
        from .meta import MetaPolicy
        from .policy import decide_actions
        meta_x=MetaPolicy.frame(pred,test_features,regimes,analog,regime_persistence=rp,regime_probs=rprob)
        meta_p=engine.meta.predict_proba(meta_x)
        actions,_=decide_actions(
            pred,
            regimes,
            analog,
            meta_p,
            settings,
            regime_persistence=rp,
            regime_probs=rprob,
        )
        bt=test.copy(); bt['atr_14']=test_features['atr_14']; result=run_configured_backtest(bt,pd.Series(actions,index=test.index),settings); rows.append({'train_start':str(train.index[0]),'train_end':str(train.index[-1]),'test_start':str(test.index[0]),'test_end':str(test.index[-1]),'purge_bars':purge,'median_regime':regimes.mode().iloc[0],**result.stats,'robust_score':robust_score(result.stats)}); cursor+=step
    return pd.DataFrame(rows)

def strategy_discovery(df,settings):
    split=int(len(df)*0.75); train_df,valid_df=df.iloc[:split],df.iloc[split:]; train_x,_,_=make_features(train_df,settings.horizon_bars); train_tb=triple_barrier_labels(train_df,settings.horizon_bars,settings.pt_atr,settings.sl_atr); valid_x=make_oos_features(train_df,valid_df,settings.horizon_bars,external_feature_lag_bars=getattr(settings,'external_feature_lag_bars',1)); valid_tb=triple_barrier_labels(valid_df,settings.horizon_bars,settings.pt_atr,settings.sl_atr); train_ret=train_tb['tb_return']; valid_ret=valid_tb['tb_return']
    impact=float(getattr(settings,'impact_bps_per_sqrt',0.0))*np.sqrt(max(0.0,float(getattr(settings,'max_participation_pct',0.10))))
    research_cost_bps=float(getattr(settings,'fee_bps',0.0))+float(getattr(settings,'slippage_bps',0.0))+impact
    best=evolve(train_x,train_ret,settings.population_size,settings.generations,settings.seed,cost_bps=research_cost_bps)
    ledger_path=getattr(settings,'research_ledger_path','data/research_ledger.json')
    register_trials(ledger_path,int(settings.population_size)*int(settings.generations),kind='strategy_evolution',metadata={'split_ratio':0.75,'cost_bps':research_cost_bps})
    serial=[{'train_score':float(s),'validation_score':float(score_candidate(c,valid_x,valid_ret,cost_bps=research_cost_bps)),'direction':c.direction,'rules':[r.__dict__ for r in c.rules]} for s,c in best]; serial.sort(key=lambda z:z['validation_score'],reverse=True); return serial

def research_report(df,settings):
    folds=walk_forward(df,settings); return {'folds':folds.to_dict('records'),'aggregate':aggregate_fold_stats(folds)}
