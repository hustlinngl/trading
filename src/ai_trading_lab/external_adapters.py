from __future__ import annotations
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from .external_intelligence import ExaClient,TavilyClient,FredClient,SecClient,extract_event_terms
from .time_utils import utcnow
@dataclass
class ExternalSignal:
    provider:str; observed_at:str; published_at:str|None; title:str; url:str; text:str; score:float; features:dict[str,float]
class ProviderHealth:
    def __init__(self): self.stats={}
    def record(self,provider,success):
        s=self.stats.setdefault(provider,{"ok":0.0,"fail":0.0}); s["ok" if success else "fail"]+=1.0
    def reliability(self,provider):
        s=self.stats.get(provider,{"ok":0.0,"fail":0.0}); return (s["ok"]+2.0)/(s["ok"]+s["fail"]+4.0)
    def load(self,payload):
        if isinstance(payload,dict): self.stats={str(k):{"ok":float(v.get("ok",0.0)),"fail":float(v.get("fail",0.0))} for k,v in payload.items() if isinstance(v,dict)}
        return self
    def to_dict(self): return {k:dict(v) for k,v in self.stats.items()}
def event_features(text): return extract_event_terms(text)
def _as_signal(doc): return ExternalSignal(doc.provider,doc.retrieved_at,doc.published_at,doc.title,doc.url,doc.text,float(doc.score or 0.0),extract_event_terms(doc.text))
class WebSearchRouter:
    def __init__(self,timeout=25,providers=("exa","tavily")):
        self.timeout=timeout; self.health=ProviderHealth()
        allowed={"exa","tavily"}
        selected=[p for p in providers if str(p).lower() in allowed]
        self.clients={p:(ExaClient(timeout=timeout) if p=="exa" else TavilyClient(timeout=timeout)) for p in dict.fromkeys(selected)}
    def _search_provider(self,provider,query,n,deep):
        try:
            docs=self.clients[provider].search(query,num_results=n,deep=deep); self.health.record(provider,bool(docs)); return [_as_signal(d) for d in docs]
        except Exception: self.health.record(provider,False); return []
    def search(self,query,n=6,deep=True):
        with ThreadPoolExecutor(max_workers=2,thread_name_prefix="intel") as pool:
            futs=[pool.submit(self._search_provider,p,query,n,deep) for p in self.clients]
            out=[] 
            for fut in as_completed(futs): out.extend(fut.result())
        best={}
        for item in out:
            key=item.url or f"{item.provider}:{item.title}"
            if key not in best or item.score>best[key].score: best[key]=item
        return list(best.values())
class FredAdapter:
    def __init__(self,api_key=None,timeout=25): self.client=FredClient(api_key=api_key,timeout=timeout)
    def series(self,series_id,start=None,end=None):
        frame=self.client.observations(series_id,start,end); return pd.Series(dtype=float,name=series_id) if frame.empty else frame.set_index("date")["value"].rename(series_id)
class SecAdapter:
    def __init__(self,user_agent=None,timeout=25): self.client=SecClient(user_agent=user_agent,timeout=timeout)
    def submissions(self,cik): return self.client.submissions(cik)
__all__=["ExternalSignal","ProviderHealth","WebSearchRouter","FredAdapter","SecAdapter","event_features","utcnow"]