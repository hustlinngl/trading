from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

def cost_aware_meta_target(future_ret,p_up,round_trip_cost):
    """Legacy direction-selection target retained for external callers."""
    ret=np.asarray(future_ret,float); prob=np.asarray(p_up,float); direction=np.where(prob>=0.5,1.0,-1.0)
    hurdle=np.maximum(np.asarray(round_trip_cost,float),0.0)
    return ((ret*direction)>hurdle).astype(int)


def side_aware_meta_target(position_return, round_trip_cost):
    """Label whether one side-specific executable return clears its cost hurdle."""
    ret = np.asarray(position_return, dtype=float)
    hurdle = np.maximum(np.asarray(round_trip_cost, dtype=float), 0.0)
    return (ret > hurdle).astype(int)

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
