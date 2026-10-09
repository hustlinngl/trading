from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .features import make_features
from .labels import triple_barrier_labels
from .regimes import RegimeDetector
from .models import SignalModel
from .memory import AnalogMemory
from .meta import MetaPolicy, cost_aware_meta_target
from .policy import decide_actions
from .efficiency import audit_features
from .execution_semantics import validate_execution_alignment


@dataclass
class EngineArtifacts:
    features: pd.DataFrame
    labels: pd.DataFrame
    regimes: pd.Series
    model: SignalModel
    memory: AnalogMemory
    meta: MetaPolicy
    predictions: pd.DataFrame


def directional_target_from_returns(target_ret: pd.Series) -> pd.Series:
    """Create the binary direction target aligned with simulated realized returns."""
    returns = pd.to_numeric(target_ret, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return returns.gt(0.0).astype(float).where(returns.notna())


def _pre_calibration_core_mask(index, calibration_start, purge_bars):
    """Match meta-learning's training boundary to SignalModel's purged calibration split."""
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    if len(index) == 0:
        return pd.Series(dtype=bool, index=index)
    positions = np.arange(len(index))
    try:
        cal_pos = int(index.searchsorted(pd.Timestamp(calibration_start), side="left"))
    except Exception:
        cal_pos = len(index)
    core_end = max(0, cal_pos - max(0, int(purge_bars)))
    return pd.Series(positions < core_end, index=index, dtype=bool)


class AdaptiveEngine:
    def __init__(self, settings):
        self.settings=settings
        self.regimes=RegimeDetector(settings.seed,n_init=getattr(settings,'regime_n_init',5))
        self.model=SignalModel(settings.seed,xgb_estimators=getattr(settings,'xgb_estimators',240),lgbm_estimators=getattr(settings,'lgbm_estimators',240),hist_max_iter=getattr(settings,'hist_max_iter',260))
        self.model.conformal_level=float(getattr(settings,'conformal_level',0.90))
        self.memory=AnalogMemory(k=getattr(settings,'memory_k',32),exclusion_bars=int(getattr(settings,'memory_exclusion_bars',max(1,getattr(settings,'validation_purge_bars',settings.horizon_bars))))); self.meta=MetaPolicy(settings.seed); self.meta_regimes=None

    def features(self, df):
        """Build the canonical feature frame used by both research and live inference."""
        features, _, _ = make_features(
            df,
            self.settings.horizon_bars,
            external_feature_lag_bars=getattr(self.settings, "external_feature_lag_bars", 1),
        )
        return features

    def load(self, out_dir):
        """Load a persisted bundle into this engine instance."""
        loaded = type(self).load_bundle(self.settings, out_dir)
        self.__dict__.update(loaded.__dict__)
        return self

    def fit(self,df):
        validate_execution_alignment(self.settings)
        features,_,_raw_future_ret=make_features(df,self.settings.horizon_bars,external_feature_lag_bars=getattr(self.settings,'external_feature_lag_bars',1))
        tb=triple_barrier_labels(df,self.settings.horizon_bars,self.settings.pt_atr,self.settings.sl_atr)
        target_ret=tb['tb_return']
        # p_up is consumed as a directional probability by decide_actions. Train it
        # against the sign of the simulated, executable return rather than
        # "take-profit barrier hit vs stop/timeout", which conflates neutral
        # outcomes with bearish ones.
        y=directional_target_from_returns(target_ret)
        cal_n=max(48,int(max(1,y.notna().sum())*0.15)); valid_idx=y.notna(); valid_positions=np.flatnonzero(valid_idx.to_numpy()); selection_mask=pd.Series(False,index=features.index)
        if len(valid_positions)>cal_n+100: selection_mask.iloc[valid_positions[:-cal_n]]=True
        else: selection_mask.loc[valid_idx.index]=valid_idx
        pruned,efficiency_report=audit_features(features,y,corr_threshold=float(getattr(self.settings,'efficiency_corr_threshold',0.995)),selection_mask=selection_mask); self.feature_efficiency=efficiency_report
        self.model.fit(pruned,y,target_ret,purge_bars=int(getattr(self.settings,'validation_purge_bars',self.settings.horizon_bars)))
        cal_oos=getattr(self.model,'calibration_oos_',None)
        if cal_oos is not None and not cal_oos.empty:
            first_cal=cal_oos.index[0]; core_idx=_pre_calibration_core_mask(features.index,first_cal,int(getattr(self.settings,'validation_purge_bars',self.settings.horizon_bars))); self.meta_regimes=RegimeDetector(self.settings.seed,n_init=getattr(self.settings,'regime_n_init',5)); self.meta_regimes.fit(features.loc[core_idx])
        self.regimes.fit(features); regime=self.regimes.transform(features); regime_persistence=self.regimes.persistence(features); model_features=pruned.reindex(index=features.index)
        meta_analog=None
        if cal_oos is not None and not cal_oos.empty:
            first_cal=cal_oos.index[0]; core_idx=_pre_calibration_core_mask(model_features.index,first_cal,int(getattr(self.settings,'validation_purge_bars',self.settings.horizon_bars))); meta_memory=AnalogMemory(k=getattr(self.settings,'memory_k',32),exclusion_bars=int(getattr(self.settings,'memory_exclusion_bars',max(1,getattr(self.settings,'validation_purge_bars',self.settings.horizon_bars))))); meta_memory.fit(model_features.loc[core_idx],target_ret.loc[core_idx],information_weighted=bool(getattr(self.settings,'memory_information_weighted',False))); meta_analog=meta_memory.query_many(model_features.loc[cal_oos.index])
        self.memory.fit(model_features,target_ret,information_weighted=bool(getattr(self.settings,'memory_information_weighted',False))); base=self.model.predict(model_features); analog=self.memory.query_many(model_features,exclude_self=True,exclusion_bars=int(getattr(self.settings,'memory_exclusion_bars',max(1,getattr(self.settings,'validation_purge_bars',self.settings.horizon_bars)))));
        if cal_oos is not None and not cal_oos.empty and meta_analog is not None:
            idx=cal_oos.index.intersection(features.index); meta_base=cal_oos.loc[idx]; mm=self.meta_regimes if self.meta_regimes is not None else self.regimes; mr=mm.transform(features.loc[idx]); mp=mm.persistence(features.loc[idx]); mprob=mm.semantic_probabilities(features.loc[idx])
            meta_x=MetaPolicy.frame(meta_base,features.loc[idx],mr,meta_analog.loc[idx],regime_persistence=mp,regime_probs=mprob)
            fee_bps=float(getattr(self.settings,'fee_bps',0.0)); slip_bps=float(getattr(self.settings,'slippage_bps',0.0)); impact_bps=float(getattr(self.settings,'impact_bps_per_sqrt',0.0)); participation=float(np.clip(getattr(self.settings,'max_participation_pct',0.10),0.0,1.0)); edge_bps=float(getattr(self.settings,'min_edge_after_cost_bps',5.0)); borrow_bps=float(max(0.0,getattr(self.settings,'short_borrow_bps_per_bar',0.0)))*int(max(1,getattr(self.settings,'max_holding_bars',96)))
            base_cost=(2*(fee_bps+slip_bps)+2*impact_bps*np.sqrt(participation)+max(0.0,edge_bps))/10000.0
            directional_cost=base_cost+np.where(meta_base.p_up.to_numpy(float)<0.5,borrow_bps/10000.0,0.0)
            meta_y=cost_aware_meta_target(target_ret.loc[idx].to_numpy(),meta_base.p_up.to_numpy(),directional_cost); valid=target_ret.loc[idx].notna().to_numpy()
            if valid.sum()>=100 and np.unique(meta_y[valid]).size>1: self.meta.fit(meta_x.loc[valid],pd.Series(meta_y[valid],index=idx[valid]))
        meta_detector=self.meta_regimes if self.meta_regimes is not None else self.regimes
        meta_regime=meta_detector.transform(features)
        meta_persistence=meta_detector.persistence(features)
        meta_probs=meta_detector.semantic_probabilities(features)
        meta_all=self.meta.predict_proba(MetaPolicy.frame(base,features,meta_regime,analog,regime_persistence=meta_persistence,regime_probs=meta_probs))
        action_series,score_series=decide_actions(base,regime,analog,meta_all,self.settings,regime_persistence=regime_persistence,regime_probs=self.regimes.semantic_probabilities(features))
        pred=base.copy(); pred['regime']=regime; pred['analog_edge']=analog.edge; pred['analog_agreement']=analog.agreement; pred['analog_dispersion']=analog.dispersion; pred['analog_n']=analog.n; pred['regime_persistence']=regime_persistence; pred['meta_success']=meta_all; pred['score']=score_series; pred['action']=action_series
        return EngineArtifacts(features,tb,regime,self.model,self.memory,self.meta,pred.replace([np.inf,-np.inf],np.nan))

    @classmethod
    def load_bundle(cls,settings,out_dir):
        p=Path(out_dir)
        if not (p/'signal_model.joblib').exists(): raise FileNotFoundError(f'Missing signal_model.joblib in {p}')
        import joblib
        obj=cls(settings); obj.model=SignalModel.load(p/'signal_model.joblib'); obj.memory=AnalogMemory.load(p/'analog_memory.joblib'); obj.memory.exclusion_bars=int(getattr(settings,'memory_exclusion_bars',max(1,getattr(settings,'validation_purge_bars',settings.horizon_bars)))); obj.regimes=joblib.load(p/'regime_detector.joblib')
        obj.meta_regimes=joblib.load(p/'meta_regime_detector.joblib') if (p/'meta_regime_detector.joblib').exists() else obj.regimes
        obj.meta=joblib.load(p/'meta_policy.joblib')
        try: obj.feature_efficiency=joblib.load(p/'feature_efficiency.joblib')
        except Exception: obj.feature_efficiency=None
        return obj

    def predict_frame(self,features,*,strict=True):
        if features is None or features.empty: return pd.DataFrame(index=getattr(features,'index',None))
        if strict:
            last=features.iloc[-1]
            missing=[c for c in self.model.feature_cols if c in features.columns and pd.isna(last.get(c))]
            if missing: raise ValueError('missing_live_features:' + ','.join(missing))
        base=self.model.predict(features); regime=self.regimes.transform(features); persistence=self.regimes.persistence(features); probs=self.regimes.semantic_probabilities(features); analog=self.memory.query_many(features)
        meta_detector=self.meta_regimes if self.meta_regimes is not None else self.regimes
        meta_regime=meta_detector.transform(features); meta_persistence=meta_detector.persistence(features); meta_probs=meta_detector.semantic_probabilities(features)
        meta_x=MetaPolicy.frame(base,features,meta_regime,analog,regime_persistence=meta_persistence,regime_probs=meta_probs); meta_p=self.meta.predict_proba(meta_x); actions,scores=decide_actions(base,regime,analog,meta_p,self.settings,regime_persistence=persistence,regime_probs=probs)
        pred=base.copy(); pred['regime']=regime; pred['analog_edge']=analog.edge; pred['analog_agreement']=analog.agreement; pred['analog_dispersion']=analog.dispersion; pred['analog_n']=analog.n; pred['regime_persistence']=persistence; pred['meta_success']=meta_p; pred['score']=scores; pred['action']=actions
        pred=pred.replace([np.inf,-np.inf],np.nan)
        if strict:
            required=('p_up','expected_return','expected_return_lcb','expected_return_ucb','model_disagreement','meta_success','score','analog_edge','analog_agreement','analog_n')
            missing_outputs=[c for c in required if c not in pred.columns or not pd.notna(pred.iloc[-1][c])]
            if missing_outputs: raise ValueError('missing_prediction_outputs:' + ','.join(missing_outputs))
            if not pd.notna(pred.iloc[-1].get('regime')) or not pd.notna(pred.iloc[-1].get('regime_persistence')): raise ValueError('missing_prediction_regime')
        return pred

    def save(self,out_dir):
        p=Path(out_dir); p.mkdir(parents=True,exist_ok=True); self.model.save(p/'signal_model.joblib'); self.memory.save(p/'analog_memory.joblib')
        import joblib
        joblib.dump(self.regimes,p/'regime_detector.joblib'); joblib.dump(self.meta_regimes if self.meta_regimes is not None else self.regimes,p/'meta_regime_detector.joblib'); joblib.dump(self.meta,p/'meta_policy.joblib'); joblib.dump(getattr(self,'feature_efficiency',None),p/'feature_efficiency.joblib')
