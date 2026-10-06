from __future__ import annotations

from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class ResearchBudget:
    volatility_z: float
    drift_score: float
    event_pressure: float
    level: str
    search_queries: int
    search_results_per_query: int
    deep_search: bool
    extra_event_study: bool

    def to_dict(self):
        return asdict(self)


def adaptive_budget(df: pd.DataFrame, base_queries: int = 4, base_results: int = 6,
                    drift_score: float = 0.0, event_pressure: float = 0.0) -> ResearchBudget:
    if len(df) < 64 or "close" not in df:
        return ResearchBudget(0.0, drift_score, event_pressure, "normal", max(1, base_queries//2), base_results, False, False)
    ret = df["close"].pct_change().dropna()
    vol = float(ret.rolling(48).std().iloc[-1]) if len(ret) >= 48 else float(ret.std())
    baseline = float(ret.rolling(min(480, len(ret))).std().iloc[-1]) if len(ret) >= 100 else max(float(ret.std()), 1e-9)
    vz = vol / max(baseline, 1e-9)
    pressure = max(float(event_pressure), 0.0)
    if vz > 1.8 or drift_score > 0.75 or pressure > 0.75:
        level = "deep"; q = max(4, base_queries); n = max(8, base_results); deep = True; extra = True
    elif vz > 1.25 or drift_score > 0.40 or pressure > 0.35:
        level = "elevated"; q = max(3, base_queries); n = max(6, base_results); deep = True; extra = True
    else:
        level = "normal"; q = max(1, base_queries // 2); n = max(4, base_results // 2); deep = False; extra = False
    return ResearchBudget(float(vz), float(drift_score), pressure, level, int(q), int(n), deep, extra)
