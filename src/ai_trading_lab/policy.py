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

def live_signal_gate(row, settings):
    """Apply the strictest live/paper signal gates without changing research semantics."""
    p_up = float(row.get("p_up", 0.5))
    expected_return = float(row.get("expected_return", 0.0))
    expected_return_lcb = float(row.get("expected_return_lcb", expected_return))
    action = str(row.get("action", "FLAT"))
    direction = 1.0 if p_up >= 0.5 else -1.0
    p_direction = p_up if direction > 0 else 1.0 - p_up
    robust_expected_return = direction * expected_return_lcb
    reasons = []

    if action not in {"LONG", "SHORT"}:
        reasons.append("base_policy")

    if bool(getattr(settings, "signal_only_mode", True)):
        probability_floor = max(
            float(getattr(settings, "signal_confidence_threshold", 0.82)),
            float(getattr(settings, "signal_probability_threshold", 0.72)),
        )
        if p_direction < probability_floor:
            reasons.append("signal_probability")
        if robust_expected_return < float(getattr(settings, "signal_min_expected_return", 0.003)):
            reasons.append("signal_expected_return")
        if float(row.get("meta_success", 0.5)) < float(getattr(settings, "signal_meta_threshold", 0.62)):
            reasons.append("signal_meta")
        if float(row.get("score", 0.0)) < float(getattr(settings, "signal_min_score", 0.22)):
            reasons.append("signal_score")
        if float(row.get("model_disagreement", 0.0)) > float(getattr(settings, "signal_max_disagreement", 0.05)):
            reasons.append("model_disagreement")
        if int(row.get("analog_n", 0)) < int(getattr(settings, "signal_memory_min_neighbors", 16)):
            reasons.append("memory_neighbors")
        if float(row.get("analog_agreement", 0.0)) < float(getattr(settings, "signal_memory_min_agreement", 0.70)):
            reasons.append("memory_agreement")

    if reasons:
        return "FLAT", reasons
    return ("LONG" if direction > 0 else "SHORT"), []


def make_actions(engine,features,settings,probability_threshold=None,min_expected_return=None,decision_threshold=None,meta_threshold=None):
    pred=engine.model.predict(features); regime=engine.regimes.transform(features); regime_persistence=engine.regimes.persistence(features); regime_probs=engine.regimes.semantic_probabilities(features); analog=engine.memory.query_many(features)
    meta_x=MetaPolicy.frame(pred,features,regime,analog,regime_persistence=regime_persistence,regime_probs=regime_probs); meta_p=engine.meta.predict_proba(meta_x)
    overrides={k:v for k,v in {'probability_threshold':probability_threshold,'min_expected_return':min_expected_return,'decision_threshold':decision_threshold,'meta_threshold':meta_threshold}.items() if v is not None}
    actions,scores=decide_actions(pred,regime,analog,meta_p,settings,regime_persistence=regime_persistence,regime_probs=regime_probs,**overrides); actions.attrs['score']=scores
    return actions
