from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class RegimeDetector:
    def __init__(self, random_state=42, n_components=5, n_init=5):
        self.scaler=StandardScaler()
        self.gmm=GaussianMixture(n_components=n_components,random_state=random_state,n_init=int(n_init),covariance_type='full')
        self.cols=['ret_24','vol_24','ema_gap_48','atr_pct','adx_14','efficiency_24']
        self.fitted=False; self.cluster_map={}; self.raw_to_canonical={}; self.canonical_to_raw={}; self.transition_matrix=None
        self.semantic_columns=['trend_up','high_vol_up','high_vol_down','range_or_down','transition','unknown']
    @staticmethod
    def _sigmoid(x): x=np.clip(np.asarray(x,float),-30,30); return 1/(1+np.exp(-x))
    def _stable_semantic_logits(self,features):
        clean=features[self.cols].replace([np.inf,-np.inf],np.nan).copy()
        if hasattr(self.scaler,'mean_'):
            for i,c in enumerate(self.cols): clean[c]=clean[c].fillna(float(self.scaler.mean_[i]))
        z=self.scaler.transform(clean); ret,vol,ema,atr,adx,eff=z.T
        logits=np.column_stack([0.95*ret+0.70*ema+0.35*adx-0.25*vol,0.90*vol+0.55*ret+0.35*ema,0.90*vol-0.55*ret-0.35*ema,-0.55*np.abs(ret)-0.45*np.abs(ema)-0.35*adx-0.55*eff,-0.35*np.abs(ret)-0.25*np.abs(ema)+0.20*vol,np.full(len(clean),-0.5)]) / 1.35
        logits=logits-np.max(logits,axis=1,keepdims=True); exp=np.exp(np.clip(logits,-30,30)); probs=exp/np.maximum(exp.sum(axis=1,keepdims=True),1e-12)
        return pd.DataFrame(probs,index=features.index,columns=self.semantic_columns)
    def fit(self,features):
        z=features[self.cols].replace([np.inf,-np.inf],np.nan).dropna()
        if len(z)<100: raise ValueError('Not enough observations to fit regime detector')
        self.scaler.fit(z); self.gmm.fit(self.scaler.transform(z)); means=pd.DataFrame(self.gmm.means_,columns=self.cols)
        def key(i): r=means.loc[i]; return (float(r.ret_24),float(r.vol_24),float(r.adx_14),float(r.ema_gap_48),int(i))
        order=sorted(range(self.gmm.n_components),key=key); self.raw_to_canonical={int(raw):int(can) for can,raw in enumerate(order)}; self.canonical_to_raw={int(can):int(raw) for can,raw in enumerate(order)}
        med_vol,med_adx=means.vol_24.median(),means.adx_14.median()
        for i,row in means.iterrows():
            if row.vol_24>=med_vol and row.ret_24>=0 and row.ema_gap_48>=0: label='high_vol_up'
            elif row.vol_24>=med_vol and row.ret_24<0: label='high_vol_down'
            elif row.adx_14>=med_adx and row.ema_gap_48>=0: label='trend_up'
            elif row.adx_14<med_adx and row.efficiency_24<0.35: label='range_or_down'
            else: label='transition'
            self.cluster_map[int(self.raw_to_canonical[int(i)])]=label
        seq_raw=self.gmm.predict(self.scaler.transform(z)); seq=np.array([self.raw_to_canonical[int(s)] for s in seq_raw],dtype=int); tm=np.ones((self.gmm.n_components,self.gmm.n_components))*0.5
        for a,b in zip(seq[:-1],seq[1:]): tm[int(a),int(b)]+=1
        self.transition_matrix=tm/tm.sum(axis=1,keepdims=True); self.fitted=True; return self
    def transform(self,features):
        if not self.fitted: raise RuntimeError('RegimeDetector is not fitted')
        return self._stable_semantic_logits(features).idxmax(axis=1).astype(object)
    def probabilities(self,features):
        valid=features[self.cols].replace([np.inf,-np.inf],np.nan).fillna(0.0); probs=self.gmm.predict_proba(self.scaler.transform(valid)); ordered=np.zeros_like(probs)
        for raw,can in self.raw_to_canonical.items(): ordered[:,int(can)]=probs[:,int(raw)]
        return pd.DataFrame(ordered,index=features.index,columns=[f'cluster_{i}' for i in range(ordered.shape[1])])
    def semantic_probabilities(self,features):
        if not self.fitted: raise RuntimeError('RegimeDetector is not fitted')
        return self._stable_semantic_logits(features)
    def persistence(self,features):
        if not self.fitted or self.transition_matrix is None: return pd.Series(0.0,index=features.index)
        valid=features[self.cols].replace([np.inf,-np.inf],np.nan).fillna(0.0); seq_raw=self.gmm.predict(self.scaler.transform(valid)); seq=np.array([self.raw_to_canonical[int(c)] for c in seq_raw],dtype=int); vals=np.array([self.transition_matrix[int(c),int(c)] for c in seq],float)
        return pd.Series(vals,index=features.index)
