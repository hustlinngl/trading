from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


class AnalogMemory:
    """Nearest-neighbour episodic memory of market states and subsequent outcomes."""
    def __init__(self, k: int = 32, max_points: int = 15000, exclusion_bars: int = 0):
        self.k = int(k); self.max_points = int(max_points); self.exclusion_bars = max(0, int(exclusion_bars))
        self.scaler = RobustScaler(); self.nn = NearestNeighbors(n_neighbors=max(1, int(k)), metric='euclidean')
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

    def query_many(self, frame, *, exclude_self=False, exclusion_bars=None):
        if not self.fitted or self.matrix is None:
            return pd.DataFrame({'edge':np.nan,'dispersion':np.nan,'agreement':np.nan,'n':0},index=frame.index)
        x=frame.reindex(columns=self.feature_cols).replace([np.inf,-np.inf],np.nan)
        x=x.fillna(self.fill_values).fillna(0.0)
        zx=self.scaler.transform(x)
        metric_weights=np.sqrt(self.feature_weights.reindex(x.columns).fillna(1.0).to_numpy(float)) if self.feature_weights is not None else 1.0
        zx=zx*metric_weights

        exclusion=max(0,int(getattr(self,'exclusion_bars',0) if exclusion_bars is None else exclusion_bars))
        requested=min(len(self.matrix),max(self.k,2*self.k+2*exclusion+8))
        dist,idx=self.nn.kneighbors(zx,n_neighbors=requested)
        mem_ts=pd.Index(self.timestamps) if self.timestamps is not None else pd.Index([])

        for row_i,ts in enumerate(pd.Index(frame.index)):
            if not (exclusion>0 and len(mem_ts)):
                if exclude_self and len(mem_ts):
                    matches=np.flatnonzero(mem_ts==ts)
                    if len(matches):
                        anchor=int(matches[0])
                        keep=np.abs(idx[row_i]-anchor)>0
                        idx[row_i],dist[row_i]=idx[row_i][keep][:self.k],dist[row_i][keep][:self.k]
                continue
            matches=np.flatnonzero(mem_ts==ts)
            if not len(matches):
                continue
            anchor=int(matches[0])
            keep=np.abs(idx[row_i]-anchor)>exclusion
            filtered,filtered_dist=idx[row_i][keep],dist[row_i][keep]
            if len(filtered)<min(self.k,len(self.matrix)) and len(self.matrix)>requested:
                all_dist,all_idx=self.nn.kneighbors(zx[row_i:row_i+1],n_neighbors=len(self.matrix))
                keep_all=np.abs(all_idx[0]-anchor)>exclusion
                filtered,filtered_dist=all_idx[0][keep_all],all_dist[0][keep_all]
            idx[row_i],dist[row_i]=filtered[:self.k],filtered_dist[:self.k]

        edges=[]; dispersions=[]; agreements=[]; counts=[]
        for row_i in range(len(idx)):
            y=self.outcomes[idx[row_i]]
            d=dist[row_i]
            finite=np.isfinite(y)
            y=y[finite]; d=d[finite]
            n=int(len(y)); counts.append(n)
            if n==0:
                edges.append(np.nan); dispersions.append(np.nan); agreements.append(np.nan)
                continue
            w=1.0/(1.0+d)
            edge=float(np.average(y,weights=w))
            edges.append(edge)
            dispersions.append(float(np.sqrt(np.average((y-edge)**2,weights=w))))
            agreements.append(float(np.mean(np.sign(y)==np.sign(edge))))
        return pd.DataFrame({'edge':edges,'dispersion':dispersions,'agreement':agreements,'n':counts},index=frame.index)

    def query(self,row):
        result=self.query_many(pd.DataFrame([row]),exclude_self=True,exclusion_bars=getattr(self,'exclusion_bars',0))
        r=result.iloc[0]
        return {'edge':float(r['edge']),'dispersion':float(r['dispersion']),'agreement':float(r['agreement']),'n':int(r['n'])}

    def score(self,row): return self.query(row)['edge']
    def save(self,path): p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); joblib.dump(self,p)
    @staticmethod
    def load(path): return joblib.load(path)
