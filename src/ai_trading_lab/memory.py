from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


class AnalogMemory:
    """Nearest-neighbour episodic memory of market states and subsequent outcomes."""
    def __init__(self, k: int = 32, max_points: int = 15000):
        self.k = k; self.max_points = max_points; self.scaler = RobustScaler(); self.nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
        self.feature_cols=[]; self.matrix=None; self.outcomes=None; self.fill_values=None; self.timestamps=None; self.feature_weights=None; self.weight_power=0.5; self.fitted=False

    def fit(self, features, future_ret, sample_weight=None, information_weighted=True, weight_floor=0.25):
        numeric=[c for c in features.columns if features[c].dtype!='object' and c!='target']; self.feature_cols=numeric
        z=features[numeric].replace([np.inf,-np.inf],np.nan).copy(); self.fill_values=z.median(numeric_only=True); z=z.fillna(self.fill_values).fillna(0.0)
        mask=future_ret.notna() & z.notna().all(axis=1); x=z.loc[mask]; y=future_ret.loc[mask].to_numpy(float)
        if len(x)>self.max_points:
            idx=np.linspace(0,len(x)-1,self.max_points).astype(int); x,y=x.iloc[idx],y[idx]
        self.timestamps=x.index.copy()
        if information_weighted and len(x)>=64:
            corr=x.corrwith(pd.Series(y,index=x.index),method='spearman').abs().fillna(0.0); strength=corr/max(float(corr.max()),1e-9)
            weights=float(max(0,min(1,weight_floor)))+(1-float(max(0,min(1,weight_floor))))*strength; weights=weights/max(float(weights.mean()),1e-9); self.feature_weights=weights.astype(float)
        else: self.feature_weights=pd.Series(1.0,index=x.columns,dtype=float)
        scaled=self.scaler.fit_transform(x); metric_weights=np.sqrt(self.feature_weights.reindex(x.columns).fillna(1.0).to_numpy(float)); self.matrix=scaled*metric_weights[None,:]
        self.outcomes=y; self.nn=NearestNeighbors(n_neighbors=min(self.k,max(1,len(self.matrix))),metric='euclidean'); self.nn.fit(self.matrix); self.fitted=True; return self

    def query_many(self, frame, *, exclude_self=False):
        if not self.fitted or self.matrix is None: return pd.DataFrame({'edge':0.0,'dispersion':0.0,'agreement':0.0,'n':0},index=frame.index)
        x=frame.reindex(columns=self.feature_cols).replace([np.inf,-np.inf],np.nan); x=x.fillna(self.fill_values).fillna(0.0); zx=self.scaler.transform(x)
        metric_weights=np.sqrt(self.feature_weights.reindex(x.columns).fillna(1.0).to_numpy(float)) if self.feature_weights is not None else 1.0; zx=zx*metric_weights
        dist,idx=self.nn.kneighbors(zx)
        if exclude_self and self.timestamps is not None and len(frame):
            frame_ts=pd.Index(frame.index); mem_ts=pd.Index(self.timestamps)
            for row_i,ts in enumerate(frame_ts):
                matches=np.flatnonzero(mem_ts==ts)
                if len(matches):
                    keep=idx[row_i]!=int(matches[0]); filtered=idx[row_i][keep]; filtered_dist=dist[row_i][keep]
                    if len(filtered)<min(self.k,len(self.matrix)):
                        wider=min(len(self.matrix),max(self.k+8,2*self.k)); d2,i2=self.nn.kneighbors(zx[row_i:row_i+1],n_neighbors=wider); keep2=i2[0]!=int(matches[0]); filtered,filtered_dist=i2[0][keep2],d2[0][keep2]
                    idx[row_i],dist[row_i]=filtered[:self.k],filtered_dist[:self.k]
        y=self.outcomes[idx]; w=1.0/(1.0+dist); edge=np.average(y,axis=1,weights=w); disp=np.sqrt(np.average((y-edge[:,None])**2,axis=1,weights=w)); agreement=np.mean(np.sign(y)==np.sign(edge[:,None]),axis=1)
        return pd.DataFrame({'edge':edge,'dispersion':disp,'agreement':agreement,'n':idx.shape[1]},index=frame.index)

    def query(self,row):
        result=self.query_many(pd.DataFrame([row]),exclude_self=True)
        r=result.iloc[0]
        return {'edge':float(r['edge']),'dispersion':float(r['dispersion']),'agreement':float(r['agreement']),'n':int(r['n'])}

    def score(self,row): return self.query(row)['edge']
    def save(self,path): p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); joblib.dump(self,p)
    @staticmethod
    def load(path): return joblib.load(path)
