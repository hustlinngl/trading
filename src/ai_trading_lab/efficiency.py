from __future__ import annotations
from dataclasses import dataclass, asdict
import time
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

@dataclass
class EfficiencyReport:
    original_features: int
    retained_features: int
    redundant_features: int
    low_information_features: int
    leakage_suspects: list[str]
    fit_seconds: float
    def to_dict(self):
        return asdict(self)

def audit_features(X: pd.DataFrame, y: pd.Series | None = None, corr_threshold: float = 0.995, selection_mask: pd.Series | None = None):
    z = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    if z.empty:
        return z, EfficiencyReport(0, 0, 0, 0, [], 0.0)
    leakage = [c for c in z.columns if any(tok in c.lower() for tok in ("future", "target", "label", "forward", "lead_"))]
    missing = z.isna().mean()
    low_info = [c for c in z.columns if missing[c] > 0.35 or z[c].nunique(dropna=True) <= 2]
    ranked = list(z.columns)
    selection = z if selection_mask is None else z.loc[selection_mask.reindex(z.index).fillna(False)]
    if y is not None:
        try:
            valid = y.notna()
            if selection_mask is not None: valid = valid & selection_mask.reindex(y.index).fillna(False)
            vals = z.loc[valid].ffill().fillna(0.0)
            if len(vals) >= 200 and y.loc[valid].nunique() > 1:
                tail = min(len(vals), 5000)
                mi = mutual_info_classif(vals.iloc[-tail:], y.loc[valid].iloc[-tail:].astype(int), random_state=42)
                ranked = [vals.columns[i] for i in np.argsort(mi)[::-1]] + [c for c in ranked if c not in vals.columns]
        except Exception:
            pass
    keep, redundant = [], []
    corr = selection[ranked].corr().abs()
    for c in ranked:
        if c in low_info or c in leakage: continue
        if any(corr.loc[c, k] >= corr_threshold for k in keep if c in corr.index and k in corr.index):
            redundant.append(c)
        else: keep.append(c)
    return z[keep].copy(), EfficiencyReport(len(z.columns), len(keep), len(redundant), len(low_info), leakage, 0.0)

def timed_transform(fn, *args, **kwargs):
    t0=time.perf_counter(); result=fn(*args, **kwargs); return result, time.perf_counter()-t0
