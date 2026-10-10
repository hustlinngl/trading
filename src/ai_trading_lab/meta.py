from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

def cost_aware_meta_target(future_ret,p_up,round_trip_cost):
    """Label meta success only when directional realized return clears the supplied economic hurdle."""
    ret=np.asarray(future_ret,float); prob=np.asarray(p_up,float); direction=np.where(prob>=0.5,1.0,-1.0)
    hurdle=np.maximum(np.asarray(round_trip_cost,float),0.0)
    return ((ret*direction)>hurdle).astype(int)

class MetaPolicy:
    def __init__(self,seed=42):
        self.seed=seed; self.model=HistGradientBoostingClassifier(max_iter=180,learning_rate=0.035,max_leaf_nodes=15,l2_regularization=2.0,random_state=seed); self.columns=[]; self.fill_values=None; self.ready=False
    @staticmethod
    def frame(pred,features,regimes,analog,regime_persistence=None,regime_probs=None):
        out=pd.DataFrame(index=pred.index)
        # The meta policy is trained on out-of-sample ensemble scores, before the
        # probability calibrator is applied. Prefer that same raw score at inference;
        # the calibrated probability remains available to the execution policy.
        probability_col = 'p_up_raw' if 'p_up_raw' in pred else 'p_up'
        out['p_up'] = pred[probability_col]
        for c in ['expected_return','model_disagreement','return_disagreement']: out[c]=pred[c]
        # The meta-training OOS frame intentionally has no calibrated return bounds.
        # Keep these compatibility columns neutral at inference too, rather than
        # feeding deployment-only conformal intervals into features trained as copies
        # of expected_return. Return uncertainty is still used by the execution policy.
        out['expected_return_lcb'] = pred['expected_return']
        out['expected_return_ucb'] = pred['expected_return']
        out['analog_edge']=analog['edge']; out['analog_agreement']=analog['agreement']; out['analog_dispersion']=analog['dispersion']
        out['regime_trend_up']=(regimes=='trend_up').astype(float); out['regime_high_vol_up']=(regimes=='high_vol_up').astype(float); out['regime_high_vol_down']=(regimes=='high_vol_down').astype(float); out['regime_range']=(regimes=='range_or_down').astype(float); out['regime_transition']=(regimes=='transition').astype(float)
        out['regime_persistence']=pd.Series(regime_persistence,index=out.index).astype(float) if regime_persistence is not None else 0.0
        if regime_probs is not None:
            for c in regime_probs.columns: out[f'regime_prob_{c}']=regime_probs[c].reindex(out.index).astype(float)
        for c in ['atr_pct','atr_ratio','adx_14','bb_z','bb_width','efficiency_24','sign_entropy_48','volume_trend','htf1h_ema_gap','htf4h_ema_gap']:
            if c in features: out[c]=features[c]
        return out.replace([np.inf,-np.inf],np.nan)
    def fit(self,X,y):
        self.columns=list(X.columns); clean=X[self.columns].replace([np.inf,-np.inf],np.nan); self.fill_values=clean.median(numeric_only=True); clean=clean.fillna(self.fill_values).fillna(0.0); self.model.fit(clean[self.columns],y.astype(int)); self.ready=True; return self
    def predict_proba(self,X):
        if not self.ready: return np.full(len(X),0.5)
        clean=X[self.columns].replace([np.inf,-np.inf],np.nan)
        if self.fill_values is not None: clean=clean.fillna(self.fill_values)
        return self.model.predict_proba(clean.fillna(0.0)[self.columns])[:,1]

def combine(p_up,expected_return,analog_edge,regime,meta_success=0.5,model_disagreement=0.0,analog_agreement=0.5,direction=1,regime_weight=0.08,uncertainty_penalty_mult=2.0,memory_weight=0.16,meta_weight=0.18,conviction_weight=0.34,edge_weight=0.30):
    regime_vector={'trend_up':1.0,'high_vol_up':0.45,'range_or_down':-0.15,'high_vol_down':-1.0,'transition':0.0,'unknown':0.0}.get(regime,0.0); direction=1 if int(direction)>=0 else -1
    regime_bias=float(regime_weight)*regime_vector*direction; conviction=(p_up-0.5)*2.0; edge_component=np.tanh(expected_return*50); memory_component=np.tanh(analog_edge*50); meta_component=(meta_success-0.5)*2.0; uncertainty_penalty=min(0.35,max(0.0,model_disagreement*float(uncertainty_penalty_mult))); memory_confidence=(analog_agreement-0.5)*0.20
    weights=np.asarray([conviction_weight,edge_weight,memory_weight,meta_weight,regime_weight],float); w=np.clip(weights,0.0,None)/max(float(np.sum(np.clip(weights,0.0,None))),1e-9)
    normalized_conviction,normalized_edge,normalized_memory,normalized_meta,normalized_regime=w
    score=normalized_conviction*conviction+normalized_edge*edge_component+normalized_memory*memory_component+normalized_meta*meta_component+normalized_regime*regime_vector*direction+memory_confidence-uncertainty_penalty
    return {'score':float(score),'regime_bias':float(normalized_regime*regime_vector*direction),'weight_sum':1.0}
