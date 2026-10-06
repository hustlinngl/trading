from __future__ import annotations

import numpy as np
import pandas as pd

from .meta import MetaPolicy


def decide_actions(pred, regime, analog, meta_p, settings, *, regime_persistence=None, regime_probs=None, **overrides):
    def val(name, default):
        value=overrides.get(name,getattr(settings,name,default)); return float(value) if isinstance(default,float) else value
    probability_threshold=float(val('probability_threshold',0.57)); min_expected_return=float(val('min_expected_return',0.0015))
    fee_bps=float(getattr(settings,'fee_bps',0.0)); slippage_bps=float(getattr(settings,'slippage_bps',0.0)); impact_bps=float(getattr(settings,'impact_bps_per_sqrt',0.0)); max_participation=float(np.clip(getattr(settings,'max_participation_pct',0.10),0,1))
    hurdle_bps=2*(fee_bps+slippage_bps)+2*impact_bps*np.sqrt(max_participation)+max(0.0,float(getattr(settings,'min_edge_after_cost_bps',5.0)))
    effective_min_expected_return=max(min_expected_return,hurdle_bps/10000)
    decision_threshold=float(val('decision_threshold',0.16)); meta_threshold=float(val('meta_threshold',0.53)); conformal_blend=float(np.clip(val('conformal_blend',0.60),0,1))
    regime_weight=float(val('regime_weight',0.08)); uncertainty_penalty_mult=float(val('uncertainty_penalty_mult',2.0)); memory_weight=float(val('memory_weight',0.16)); meta_weight=float(val('meta_weight',0.18)); conviction_weight=float(val('conviction_weight',0.34)); edge_weight=float(val('edge_weight',0.30))
    p_up=pred['p_up'].to_numpy(float); er=pred['expected_return'].to_numpy(float); er_lcb=pred['expected_return_lcb'].to_numpy(float) if 'expected_return_lcb' in pred else er
    model_disagreement=pred.get('model_disagreement',pd.Series(0.0,index=pred.index)).to_numpy(float); agreement=analog['agreement'].to_numpy(float); memory_edge=analog['edge'].to_numpy(float); meta_p=np.asarray(meta_p,float)
    direction=np.where(p_up>=0.5,1.0,-1.0); p_dir=np.where(direction>0,p_up,1-p_up); er_dir=np.where(direction>0,er,-er); er_lcb_dir=np.where(direction>0,er_lcb,-er_lcb); er_robust=er_dir-conformal_blend*(er_dir-er_lcb_dir); mem_dir=np.where(direction>0,memory_edge,-memory_edge)
    regime_vec=regime.map({'trend_up':1.0,'high_vol_up':0.45,'range_or_down':-0.15,'high_vol_down':-1.0,'transition':0.0,'unknown':0.0}).fillna(0.0).to_numpy(float)
    weights=np.clip(np.asarray([conviction_weight,edge_weight,memory_weight,meta_weight,regime_weight],dtype=float),0,None); weights/=max(float(weights.sum()),1e-12)
    score=weights[0]*((p_dir-0.5)*2)+weights[1]*np.tanh(er_robust*50)+weights[2]*np.tanh(mem_dir*50)+weights[3]*((meta_p-0.5)*2)+weights[4]*regime_vec*direction+(agreement-0.5)*0.20-np.minimum(0.35,np.maximum(0.0,model_disagreement*uncertainty_penalty_mult))
    ok=(score>=decision_threshold)&(p_dir>=probability_threshold)&(er_robust>=effective_min_expected_return)&(meta_p>=meta_threshold)
    action=np.full(len(pred),'FLAT',dtype=object); action[ok&(direction>0)]='LONG'; action[ok&(direction<0)]='SHORT'
    score_series=pd.Series(score,index=pred.index,name='score'); score_series.attrs['effective_min_expected_return']=float(effective_min_expected_return); score_series.attrs['economic_hurdle_bps']=float(hurdle_bps)
    return pd.Series(action,index=pred.index,name='action'),score_series


def make_actions(engine,features,settings,probability_threshold=None,min_expected_return=None,decision_threshold=None,meta_threshold=None):
    pred=engine.model.predict(features); regime=engine.regimes.transform(features); regime_persistence=engine.regimes.persistence(features); regime_probs=engine.regimes.semantic_probabilities(features); analog=engine.memory.query_many(features)
    meta_x=MetaPolicy.frame(pred,features,regime,analog,regime_persistence=regime_persistence,regime_probs=regime_probs); meta_p=engine.meta.predict_proba(meta_x)
    overrides={k:v for k,v in {'probability_threshold':probability_threshold,'min_expected_return':min_expected_return,'decision_threshold':decision_threshold,'meta_threshold':meta_threshold}.items() if v is not None}
    actions,scores=decide_actions(pred,regime,analog,meta_p,settings,regime_persistence=regime_persistence,regime_probs=regime_probs,**overrides); actions.attrs['score']=scores
    return actions
