from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

@dataclass
class EventStudyResult:
    event_type: str
    n: int
    mean_event_return: float
    baseline_return: float
    abnormal_return: float
    t_stat: float
    bootstrap_prob_positive: float
    def to_dict(self):
        return asdict(self)

class EventEmbeddingMemory:
    def __init__(self, n_features: int = 256):
        self.vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False, norm=None, ngram_range=(1, 2), stop_words="english")
        self.vectors = None; self.docs = []; self._norm = None
    def fit(self, docs: list[dict]):
        self.docs = list(docs)
        texts = [str(d.get("text") or d.get("title") or "") for d in self.docs]
        self.vectors = normalize(self.vectorizer.transform(texts), norm="l2") if texts else None
        return self
    def query(self, text: str, k: int = 8) -> list[dict]:
        if self.vectors is None or not self.docs: return []
        q = normalize(self.vectorizer.transform([text]), norm="l2")
        scores = (self.vectors @ q.T).toarray().ravel()
        order = np.argsort(scores)[::-1][:max(1, int(k))]
        out = []
        for i in order:
            row = dict(self.docs[int(i)]); row["semantic_similarity"] = float(scores[int(i)]); out.append(row)
        return out

def event_study(prices: pd.Series, events: pd.DataFrame, horizons=(1, 4, 8, 16), seed: int = 42) -> pd.DataFrame:
    if prices.empty or events.empty:
        return pd.DataFrame(columns=["event_type", "n", "mean_event_return", "baseline_return", "abnormal_return", "t_stat", "bootstrap_prob_positive"])
    p = prices.sort_index().astype(float); rows=[]; rng=np.random.default_rng(seed)
    for _, e in events.iterrows():
        try:
            t=pd.Timestamp(e["event_time"]); t=t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        except Exception: continue
        pos=p.index.searchsorted(t,side="right"); h=int(e.get("horizon",8))
        if pos>=len(p) or pos+h>=len(p): continue
        base=float(p.iloc[pos]); fut=float(p.iloc[pos+h])
        if base<=0 or not np.isfinite(base) or not np.isfinite(fut): continue
        rows.append((str(e.get("event_type","unknown")),fut/base-1.0,pos,h))
    if not rows: return pd.DataFrame()
    frame=pd.DataFrame(rows,columns=["event_type","event_return","pos","horizon"]); outputs=[]
    for event_type,g in frame.groupby("event_type"):
        vals=g["event_return"].to_numpy(float); h=max(1,int(g["horizon"].median()))
        future_series=p.pct_change(h).shift(-h); excluded=np.zeros(len(p),dtype=bool)
        for pos,hh in zip(g["pos"].astype(int),g["horizon"].astype(int)):
            lo=max(0,pos-h); hi=min(len(p),pos+max(h,hh)+1); excluded[lo:hi]=True
        base_mask=future_series.notna().to_numpy() & ~excluded; future=future_series.to_numpy(float)[base_mask]
        baseline_samples=rng.choice(len(future),size=min(2000,len(future)),replace=False) if len(future)>200 else np.arange(len(future))
        baseline=float(np.nanmean(future[baseline_samples])) if len(future) else 0.0
        abnormal=vals-baseline; sd=float(np.std(abnormal,ddof=1)) if len(abnormal)>1 else 0.0
        t_stat=float(np.mean(abnormal)/(sd/np.sqrt(len(abnormal)))) if sd>0 else 0.0
        boot=rng.choice(abnormal,size=(1500,len(abnormal)),replace=True).mean(axis=1) if len(abnormal) else np.array([])
        prob=float(np.mean(boot>0)) if len(boot) else 0.0
        outputs.append(EventStudyResult(str(event_type),int(len(vals)),float(vals.mean()),baseline,float(abnormal.mean()),t_stat,prob).to_dict())
    return pd.DataFrame(outputs).sort_values("abnormal_return",key=np.abs,ascending=False)
