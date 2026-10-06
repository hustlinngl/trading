from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, dataclass
import hashlib, json
import numpy as np
import pandas as pd

from .features import make_features
from .growth import GrowthRegistry
from .experience_graph import ExperienceGraph
from .causal_memory import MatchedEventMemory
from .online_learning import PrequentialLearner
from .drift import drift_report
from .time_utils import utcnow
from .strategy_lab import evolve
from .regimes import RegimeDetector
from .fingerprint import strong_dataset_fingerprint

@dataclass
class Hypothesis:
    key: str
    thesis: str
    feature_family: str
    complexity: int
    created_at: str
    evidence: list[str]
    status: str = "candidate"

@dataclass
class CandidateReport:
    key: str
    auc: float
    directional_hit: float
    mean_return: float
    robust: float
    folds: int
    complexity: int
    accepted: bool

class CognitionEngine:
    """Unified shadow research loop. It can learn and propose experiments, but cannot place orders."""

    def __init__(self, settings, root: str | Path = "."):
        self.settings=settings; self.root=Path(root)
        for p in (self.root/"data", self.root/"logs", self.root/"models"): p.mkdir(parents=True,exist_ok=True)
        self.memory_path=Path(getattr(settings,"memory_db","data/memory.sqlite")); self.memory_path=self.memory_path if self.memory_path.is_absolute() else self.root/self.memory_path
        self.registry=GrowthRegistry(self.memory_path)
        self.graph=ExperienceGraph(self.memory_path)
        self.causal=MatchedEventMemory(k=int(getattr(settings,"causal_match_k",5)),exclusion_bars=int(getattr(settings,"causal_exclusion_bars",getattr(settings,"horizon_bars",8))),seed=int(settings.seed))
        self.online_path=self.root/"models"/"online_shadow.joblib"; self.state_path=self.root/"data"/"cognition_state.json"

    def _state(self):
        try: return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception: return {}
    def _save(self,section,fp,payload,query=None):
        s=self._state(); s[section]={"data_fingerprint":fp,"query":query,"payload":payload,"updated_at":utcnow()}
        self.state_path.write_text(json.dumps(s,indent=2,default=str),encoding="utf-8")
    def _cache(self,section,fp,query=None):
        x=self._state().get(section,{})
        return x.get("payload") if x.get("data_fingerprint")==fp and (query is None or x.get("query")==query) else None

    def discover_hypotheses(self,query:str):
        bases=[("trend","Trend continuation may strengthen with higher-timeframe alignment.",1),("reversion","Reversion may improve when price is stretched and efficiency is low.",1),("breakout","Breakouts may improve after range compression plus volume expansion.",2),("risk_off","Risk-off state may weaken long signals.",2),("event","Event pressure may favor wider barriers and smaller size.",2)]
        evidence=[str(query)[:160]] if query else []
        return [Hypothesis(hashlib.sha256((family+"|"+thesis).encode()).hexdigest()[:12],thesis,family,complexity,utcnow(),evidence) for family,thesis,complexity in bases]

    def evaluate_feature_families(self,df):
        x,y,ret=make_features(df,self.settings.horizon_bars,external_feature_lag_bars=getattr(self.settings,"external_feature_lag_bars",1))
        if x.empty: return []
        valid=y.notna() & ret.notna(); x=x.loc[valid]; y=y.loc[valid].astype(int); ret=ret.loc[valid].astype(float)
        if len(x)<200: return []
        from sklearn.ensemble import ExtraTreesClassifier
        split=max(100,int(len(x)*0.70)); clf=ExtraTreesClassifier(n_estimators=200,min_samples_leaf=12,max_features="sqrt",random_state=int(self.settings.seed),n_jobs=-1)
        clf.fit(x.iloc[:split],y.iloc[:split]); p=clf.predict_proba(x.iloc[split:])[:,1]; sig=np.where(p>=.55,1,np.where(p<=.45,-1,0)); target=np.sign(ret.iloc[split:].to_numpy())
        active=sig!=0; hit=float(np.mean(sig[active]==target[active])) if active.any() else 0.0; mean_ret=float(np.mean(sig[active]*ret.iloc[split:].to_numpy()[active])) if active.any() else 0.0
        try:
            from sklearn.metrics import roc_auc_score; auc=float(roc_auc_score(y.iloc[split:],p))
        except Exception: auc=.5
        return [asdict(CandidateReport("native",auc,hit,mean_ret,mean_ret,max(1,len(x)-split),1,mean_ret>0 and int(active.sum())>=20))]

    def evolve_strategies(self,df):
        try:
            out=evolve(df,self.settings); return out if isinstance(out,list) else []
        except Exception: return []

    def observe_external(self,query,budget):
        """Collect optional external intelligence without giving it execution authority."""
        from .external_adapters import WebSearchRouter, FredAdapter, SecAdapter

        result={"query":str(query),"search":[],"provider_health":{},"macro":{"series":{}},"sec":{"filings":{}},"event_summary":{}}
        try:
            if bool(getattr(self.settings,"external_deep_search",True)):
                router=WebSearchRouter(timeout=25)
                queries=[str(query)]
                for suffix in ("market structure volatility","rates liquidity macro","regulation ETF flows"):
                    if len(queries)<int(getattr(budget,"search_queries",1)):
                        queries.append(f"{query} {suffix}")
                docs=[]
                for q in queries:
                    docs.extend(router.search(q,n=int(getattr(budget,"search_results_per_query",6)),deep=bool(getattr(budget,"deep_search",True))))
                unique={}
                for doc in docs:
                    key=doc.url or f"{doc.provider}:{doc.title}"
                    if key not in unique or float(doc.score)>float(unique[key].score): unique[key]=doc
                event_totals={}
                for doc in sorted(unique.values(),key=lambda d:float(d.score),reverse=True)[:int(getattr(budget,"search_results_per_query",6))*2]:
                    from .external_intelligence import extract_event_terms
                    features=extract_event_terms(doc.title+" "+doc.text)
                    for k,v in features.items(): event_totals[k]=event_totals.get(k,0.0)+float(v)
                    result["search"].append({"provider":doc.provider,"retrieved_at":doc.retrieved_at,"published_at":doc.published_at,"title":doc.title[:240],"url":doc.url,"score":float(doc.score),"features":features})
                result["provider_health"]=router.health.to_dict()
                result["event_summary"]=event_totals
            if getattr(self.settings,"fred_series",()):
                fred=FredAdapter()
                now=pd.Timestamp.now(tz="UTC")
                start=(now-pd.Timedelta(days=90)).date().isoformat()
                for sid in tuple(getattr(self.settings,"fred_series",())):
                    try:
                        frame=fred.series(sid,start=start,end=now.date().isoformat())
                        if not frame.empty:
                            values=pd.to_numeric(frame,errors="coerce").dropna()
                            result["macro"]["series"][sid]={"last":float(values.iloc[-1]),"previous":float(values.iloc[-2]) if len(values)>1 else None,"observed_at":str(frame.index[-1])}
                    except Exception as exc:
                        result["macro"]["series"][sid]={"error":f"{type(exc).__name__}:{exc}"}
            if getattr(self.settings,"sec_ciks",()):
                sec=SecAdapter()
                for cik in tuple(getattr(self.settings,"sec_ciks",())):
                    try:
                        frame=sec.submissions(cik)
                        if not frame.empty:
                            latest=frame.sort_values("filingDate",ascending=False).head(10)
                            result["sec"]["filings"][str(cik)]=latest[[c for c in ("accessionNumber","filingDate","form","primaryDocument") if c in latest]].to_dict(orient="records")
                    except Exception as exc:
                        result["sec"]["filings"][str(cik)]={"error":f"{type(exc).__name__}:{exc}"}
        except Exception as exc:
            result["error"]=f"{type(exc).__name__}:{exc}"
        if not result["search"] and not result["macro"]["series"] and not result["sec"]["filings"] and "error" not in result:
            result["note"]="No external provider credentials or configured SEC CIKs were available; market-only research remains valid."
        return result

    def run_research(self,df,query):
        fp=strong_dataset_fingerprint(df); cached=self._cache("research",fp,query)
        if cached is not None: return {**cached,"skipped":True,"skip_reason":"dataset_unchanged"}
        hs=self.discover_hypotheses(query); fc=self.evaluate_feature_families(df); sc=self.evolve_strategies(df)
        payload={"timestamp":utcnow(),"data_fingerprint":fp,"hypotheses":[asdict(h) for h in hs],"feature_candidates":[asdict(x) for x in fc],"strategy_candidates":sc[:20]}
        self._save("research",fp,payload,query); (self.root/"logs"/"autonomous_cycle.json").write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8"); return payload

    def run_growth(self,df,query,strategy_candidates=None):
        from .compute_policy import adaptive_budget
        budget=adaptive_budget(df,base_queries=int(getattr(self.settings,"growth_search_queries",4)))
        return {"timestamp":utcnow(),"research_budget":budget.to_dict(),"external":self.observe_external(query,budget),"experiments_proposed":[],"discoveries_recorded":len(strategy_candidates or []),"next_action":"validate candidates with fresh walk-forward windows; never promote on a single backtest"}

    def run_evolution(self,df,hypotheses=None):
        fp=strong_dataset_fingerprint(df); cached=self._cache("evolution",fp)
        if cached is not None: return {**cached,"skipped":True,"skip_reason":"dataset_unchanged"}
        x,y,ret=make_features(df,self.settings.horizon_bars,external_feature_lag_bars=getattr(self.settings,"external_feature_lag_bars",1))
        regime=RegimeDetector(self.settings.seed).fit(x).transform(x) if not x.empty else pd.Series(index=x.index,dtype=str)
        graph_features=x.select_dtypes(include=[np.number]) if not x.empty else x; actions=pd.Series("FLAT",index=x.index)
        added=self.graph.learn_episode(df,graph_features,actions,ret,symbol=getattr(self.settings,"symbol","UNKNOWN"),horizon_bars=self.settings.horizon_bars,regime=regime) if len(x) else 0
        online=PrequentialLearner(self.settings.seed,int(getattr(self.settings,"online_warmup_rows",64)))
        if self.online_path.exists():
            try: online=PrequentialLearner.load(self.online_path)
            except Exception: pass
        try: report=online.partial_update(x,y,ret).to_dict(); online.save(self.online_path)
        except Exception as exc: report={"error":str(exc)}
        drift={"drift_ratio":0.0}
        if len(graph_features)>80:
            mid=len(graph_features)//2; drift=drift_report(graph_features.iloc[:mid],graph_features.iloc[mid:],list(graph_features.columns[:40]))
        payload={"timestamp":utcnow(),"data_fingerprint":fp,"graph_transitions_added":added,"online_shadow":report,"state_drift":drift,"safety":{"online_model_is_shadow_only":True,"live_promotion_authority":"champion_challenger_gate"}}
        self._save("evolution",fp,payload); (self.root/"logs"/"deep_evolution.json").write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8"); return payload
