from __future__ import annotations

import numpy as np
import pandas as pd

from .meta import MetaPolicy


def decide_actions(pred, regime, analog, meta_p, settings, *, regime_persistence=None, regime_probs=None, **overrides):
    def val(name, default):
        value=overrides.get(name,getattr(settings,name,default)); return float(value) if isinstance(default,float) else value
    probability_threshold=float(val('probability_threshold',0.57)); min_expected_return=float(val('min_expected_return',0.0015))
    fee_bps=float(getattr(settings,'fee_bps',0.0)); slippage_bps=float(getattr(settings,'slippage_bps',0.0)); impact_bps=float(getattr(settings,'impact_bps_per_sqrt',0.0)); max_participation=float(np.clip(getattr(settings,'max_participation_pct',0.10),0,1))
    borrow_bps_per_bar=float(max(0.0,getattr(settings,'short_borrow_bps_per_bar',0.0)))
    max_holding=int(max(1,getattr(settings,'max_holding_bars',96)))
    base_hurdle_bps=2*(fee_bps+slippage_bps)+2*impact_bps*np.sqrt(max_participation)+max(0.0,float(getattr(settings,'min_edge_after_cost_bps',5.0)))
    short_borrow_hurdle_bps=borrow_bps_per_bar*max_holding
    direction_hint=np.where(pred['p_up'].to_numpy(float)>=0.5,1.0,-1.0)
    hurdle_bps=np.where(direction_hint>0,base_hurdle_bps,base_hurdle_bps+short_borrow_hurdle_bps)
    effective_min_expected_return=np.maximum(min_expected_return,hurdle_bps/10000)
    decision_threshold=float(val('decision_threshold',0.16)); meta_threshold=float(val('meta_threshold',0.53)); conformal_blend=float(np.clip(val('conformal_blend',0.60),0,1))
    regime_weight=float(val('regime_weight',0.08)); uncertainty_penalty_mult=float(val('uncertainty_penalty_mult',2.0)); memory_weight=float(val('memory_weight',0.16)); meta_weight=float(val('meta_weight',0.18)); conviction_weight=float(val('conviction_weight',0.34)); edge_weight=float(val('edge_weight',0.30))
    p_up=pred['p_up'].to_numpy(float); er=pred['expected_return'].to_numpy(float); er_lcb=pred['expected_return_lcb'].to_numpy(float) if 'expected_return_lcb' in pred else er
    model_disagreement=pred.get('model_disagreement',pd.Series(0.0,index=pred.index)).to_numpy(float); agreement=analog['agreement'].to_numpy(float); memory_edge=analog['edge'].to_numpy(float); meta_p=np.asarray(meta_p,float)
    direction=np.where(p_up>=0.5,1.0,-1.0); p_dir=np.where(direction>0,p_up,1-p_up); er_dir=np.where(direction>0,er,-er); er_ucb=pred['expected_return_ucb'].to_numpy(float) if 'expected_return_ucb' in pred else er; er_lcb_dir=np.where(direction>0,er_lcb,-er_ucb); er_robust=er_dir-conformal_blend*(er_dir-er_lcb_dir); mem_dir=np.where(direction>0,memory_edge,-memory_edge)
    regime_vec=regime.map({'trend_up':1.0,'high_vol_up':0.45,'range_or_down':-0.15,'high_vol_down':-1.0,'transition':0.0,'unknown':0.0}).fillna(0.0).to_numpy(float)
    weights=np.clip(np.asarray([conviction_weight,edge_weight,memory_weight,meta_weight,regime_weight],dtype=float),0,None); weights/=max(float(weights.sum()),1e-12)
    score=weights[0]*((p_dir-0.5)*2)+weights[1]*np.tanh(er_robust*50)+weights[2]*np.tanh(mem_dir*50)+weights[3]*((meta_p-0.5)*2)+weights[4]*regime_vec*direction+(agreement-0.5)*0.20-np.minimum(0.35,np.maximum(0.0,model_disagreement*uncertainty_penalty_mult))
    ok=(score>=decision_threshold)&(p_dir>=probability_threshold)&(er_robust>=effective_min_expected_return)&(meta_p>=meta_threshold)
    action=np.full(len(pred),'FLAT',dtype=object); action[ok&(direction>0)]='LONG'; action[ok&(direction<0)]='SHORT'
    score_series=pd.Series(score,index=pred.index,name='score')
    score_series.attrs['effective_min_expected_return']=float(np.max(effective_min_expected_return)) if len(effective_min_expected_return) else 0.0
    score_series.attrs['economic_hurdle_bps']=float(np.max(hurdle_bps)) if len(hurdle_bps) else 0.0

    def summarize(values):
        arr=np.asarray(values,dtype=float)
        arr=arr[np.isfinite(arr)]
        if not len(arr):
            return {"n":0}
        return {
            "n":int(len(arr)),
            "min":float(np.min(arr)),
            "p50":float(np.quantile(arr,0.50)),
            "p90":float(np.quantile(arr,0.90)),
            "p99":float(np.quantile(arr,0.99)),
            "max":float(np.max(arr)),
        }

    # Keep diagnostics attached to the internal Series so holdout evaluation can
    # distinguish an overly selective policy from a backtest/execution defect.
    score_series.attrs["diagnostics"]={
        "rows":int(len(pred)),
        "pass_probability":int(np.sum(p_dir>=probability_threshold)),
        "pass_robust_return":int(np.sum(er_robust>=effective_min_expected_return)),
        "pass_meta":int(np.sum(meta_p>=meta_threshold)),
        "pass_score":int(np.sum(score>=decision_threshold)),
        "pass_all_policy_gates":int(np.sum(ok)),
        "thresholds":{
            "directional_probability":float(probability_threshold),
            "minimum_robust_return_min":float(np.min(effective_min_expected_return)) if len(effective_min_expected_return) else 0.0,
            "minimum_robust_return_max":float(np.max(effective_min_expected_return)) if len(effective_min_expected_return) else 0.0,
            "meta_success":float(meta_threshold),
            "score":float(decision_threshold),
        },
        "direction_return_sign_agreement":{
            "n":int(len(p_up)),
            "matches":int(np.sum((p_up>=0.5)==(er>=0.0))),
            "rate":float(np.mean((p_up>=0.5)==(er>=0.0))) if len(p_up) else 0.0,
        },
        "distributions":{
            "directional_probability":summarize(p_dir),
            "raw_expected_return":summarize(er),
            "directional_expected_return":summarize(er_dir),
            "expected_return_lcb":summarize(er_lcb),
            "expected_return_ucb":summarize(er_ucb),
            "robust_expected_return":summarize(er_robust),
            "meta_success":summarize(meta_p),
            "score":summarize(score),
            "model_disagreement":summarize(model_disagreement),
            "analog_agreement":summarize(agreement),
        },
    }
    return pd.Series(action,index=pred.index,name='action'),score_series

def live_signal_gate(row, settings):
    """Apply live/paper gates with explicit missing-data rejection."""
    required = (
        "p_up","expected_return","expected_return_lcb",
        "meta_success","score","model_disagreement","analog_n","analog_agreement",
    )
    missing = []
    for key in required:
        value = row.get(key)
        if value is None:
            missing.append(key)
            continue
        try:
            if not np.isfinite(float(value)):
                missing.append(key)
        except (TypeError, ValueError):
            missing.append(key)
    if missing:
        return "FLAT", ["missing_prediction:" + ",".join(missing)]

    p_up = float(row["p_up"])
    expected_return = float(row["expected_return"])
    expected_return_lcb = float(row["expected_return_lcb"])
    expected_return_ucb = float(row.get("expected_return_ucb", expected_return))
    action = str(row.get("action", "FLAT"))
    direction = 1.0 if p_up >= 0.5 else -1.0
    p_direction = p_up if direction > 0 else 1.0 - p_up
    funding_drag = max(0.0, float(row.get("funding_cost_return", 0.0) or 0.0))
    robust_expected_return = expected_return_lcb if direction > 0 else -expected_return_ucb
    robust_expected_return -= funding_drag
    reasons = []

    if action not in {"LONG", "SHORT"}:
        reasons.append("base_policy")
    if bool(row.get("funding_data_missing", False)):
        reasons.append("funding_data_missing")
    if direction < 0 and bool(getattr(settings, "require_short_borrow_cost", True)) and float(getattr(settings, "short_borrow_bps_per_bar", 0.0)) <= 0.0:
        reasons.append("short_borrow_cost_missing")

    if bool(getattr(settings, "signal_only_mode", True)):
        probability_floor = max(
            float(getattr(settings, "signal_confidence_threshold", 0.82)),
            float(getattr(settings, "signal_probability_threshold", 0.72)),
        )
        if p_direction < probability_floor:
            reasons.append("signal_probability")
        if robust_expected_return < float(getattr(settings, "signal_min_expected_return", 0.003)):
            reasons.append("signal_expected_return")
        if float(row["meta_success"]) < float(getattr(settings, "signal_meta_threshold", 0.62)):
            reasons.append("signal_meta")
        if float(row["score"]) < float(getattr(settings, "signal_min_score", 0.22)):
            reasons.append("signal_score")
        if float(row["model_disagreement"]) > float(getattr(settings, "signal_max_disagreement", 0.05)):
            reasons.append("model_disagreement")
        if int(float(row["analog_n"])) < int(getattr(settings, "signal_memory_min_neighbors", 16)):
            reasons.append("memory_neighbors")
        if float(row["analog_agreement"]) < float(getattr(settings, "signal_memory_min_agreement", 0.70)):
            reasons.append("memory_agreement")

    if bool(getattr(settings, "trade_window_required_for_signal", False)):
        tw_available = bool(row.get("trade_window_available", False))
        tw_ready = bool(row.get("trade_window_ready", False))
        tw_direction = str(row.get("trade_window_direction", "FLAT"))
        tw_confidence = float(row.get("trade_window_confidence", 0.0))
        if not tw_available:
            reasons.append("trade_window_missing")
        elif not tw_ready:
            reasons.append("trade_window_not_ready")
        elif tw_direction != ("LONG" if direction > 0 else "SHORT"):
            reasons.append("trade_window_disagreement")
        elif tw_confidence < float(getattr(settings, "trade_window_min_confidence", 0.80)):
            reasons.append("trade_window_confidence")

    if reasons:
        return "FLAT", reasons
    return ("LONG" if direction > 0 else "SHORT"), []
def make_actions(engine,features,settings,probability_threshold=None,min_expected_return=None,decision_threshold=None,meta_threshold=None):
    pred=engine.model.predict(features); regime=engine.regimes.transform(features); regime_persistence=engine.regimes.persistence(features); regime_probs=engine.regimes.semantic_probabilities(features); analog=engine.memory.query_many(features)
    meta_x=MetaPolicy.frame(pred,features,regime,analog,regime_persistence=regime_persistence,regime_probs=regime_probs); meta_p=engine.meta.predict_proba(meta_x)
    overrides={k:v for k,v in {'probability_threshold':probability_threshold,'min_expected_return':min_expected_return,'decision_threshold':decision_threshold,'meta_threshold':meta_threshold}.items() if v is not None}
    actions,scores=decide_actions(pred,regime,analog,meta_p,settings,regime_persistence=regime_persistence,regime_probs=regime_probs,**overrides); actions.attrs['score']=scores; actions.attrs['diagnostics']=scores.attrs.get('diagnostics',{})
    return actions
