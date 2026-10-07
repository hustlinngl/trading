from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, mean_squared_error

@dataclass
class PrequentialReport:
    rows_scored: int
    directional_accuracy: float
    logloss: float
    rmse: float
    drift_score: float
    def to_dict(self): return asdict(self)

class PrequentialLearner:
    """Online shadow learner using predict-then-update prequential evaluation."""
    def __init__(self, seed: int = 42, warmup_rows: int = 64):
        self.seed=int(seed); self.warmup_rows=max(16,int(warmup_rows)); self.scaler=StandardScaler()
        self.clf=SGDClassifier(loss="log_loss",penalty="elasticnet",l1_ratio=0.08,alpha=2e-4,average=True,random_state=seed)
        self.reg=SGDRegressor(loss="huber",penalty="elasticnet",l1_ratio=0.02,alpha=2e-4,average=True,random_state=seed)
        self.feature_cols=[]; self.ready=False; self.seen=0; self.last_timestamp=None
    def partial_update(self,X:pd.DataFrame,y_cls:pd.Series,y_ret:pd.Series,weight:pd.Series|None=None)->PrequentialReport:
        if self.feature_cols==[]: self.feature_cols=[c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        mask=y_cls.notna() & y_ret.notna(); X=X.loc[mask].copy().sort_index(); yc=y_cls.loc[mask].astype(int); yr=y_ret.loc[mask].astype(float)
        if self.ready and self.last_timestamp is not None:
            last=pd.Timestamp(self.last_timestamp); idx=X.index
            if last.tzinfo is not None and getattr(idx,"tz",None) is None: idx=idx.tz_localize("UTC")
            X=X.loc[idx>last]; yc=yc.loc[X.index]; yr=yr.loc[X.index]
        if len(X)==0: return PrequentialReport(0,0.0,0.0,0.0,0.0)
        xn_raw=X.reindex(columns=self.feature_cols).replace([np.inf,-np.inf],np.nan).ffill().fillna(0.0)
        p_out=[]; r_out=[]; y_out=[]; sample_weight=None if weight is None else np.asarray(weight.reindex(X.index),dtype=float)
        if not self.ready:
            warm=min(self.warmup_rows,len(xn_raw))
            if warm<self.warmup_rows and self.seen==0: return PrequentialReport(0,0.0,0.0,0.0,0.0)
            warm_x=xn_raw.iloc[:warm]; self.scaler.fit(warm_x); z0=self.scaler.transform(warm_x)
            w0=None if sample_weight is None else sample_weight[:warm]
            self.clf.partial_fit(z0,yc.iloc[:warm].to_numpy(),classes=np.array([0,1]),sample_weight=w0)
            self.reg.partial_fit(z0,yr.iloc[:warm].to_numpy(),sample_weight=w0); self.ready=True; self.seen+=warm
            xn_raw=xn_raw.iloc[warm:]; yc=yc.iloc[warm:]; yr=yr.iloc[warm:]
            if sample_weight is not None: sample_weight=sample_weight[warm:]
        for i in range(len(xn_raw)):
            z=self.scaler.transform(xn_raw.iloc[[i]]); p=float(self.clf.predict_proba(z)[:,1][0]); r=float(self.reg.predict(z)[0])
            p_out.append(p); r_out.append(r); y_out.append((int(yc.iloc[i]),float(yr.iloc[i]))); w=None if sample_weight is None else sample_weight[i:i+1]
            self.clf.partial_fit(z,np.array([yc.iloc[i]]),sample_weight=w); self.reg.partial_fit(z,np.array([yr.iloc[i]]),sample_weight=w); self.seen+=1
        self.last_timestamp=pd.Timestamp(X.index[-1]).isoformat()
        if not p_out: return PrequentialReport(0,0.0,0.0,0.0,0.0)
        yb=np.array([a for a,_ in y_out],dtype=int); yr_true=np.array([b for _,b in y_out],dtype=float); p_arr=np.clip(np.asarray(p_out),1e-6,1-1e-6)
        acc=float(np.mean((p_arr>=0.5)==yb)); ll=float(log_loss(yb,p_arr,labels=[0,1])) if len(np.unique(yb))>1 else 0.0; rmse=float(mean_squared_error(yr_true,r_out)**0.5)
        drift=float(min(1.0,abs(acc-0.5)*2.0))
        return PrequentialReport(len(p_out),acc,ll,rmse,drift)
    def predict(self,X:pd.DataFrame)->pd.DataFrame:
        if not self.ready: raise RuntimeError("Online learner is not initialized")
        xn=self._clean(X); return pd.DataFrame({"online_p_up":self.clf.predict_proba(xn)[:,1],"online_expected_return":self.reg.predict(xn)},index=X.index)
    def _clean(self,X:pd.DataFrame,fit_scaler:bool=False)->np.ndarray:
        x=X.reindex(columns=self.feature_cols).replace([np.inf,-np.inf],np.nan).ffill().fillna(0.0)
        if fit_scaler: return self.scaler.fit_transform(x)
        return self.scaler.transform(x)
    def save(self,path:str|Path): p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); joblib.dump(self,p)
    @staticmethod
    def load(path:str|Path)->"PrequentialLearner": return joblib.load(path)
