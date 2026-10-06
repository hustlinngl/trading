from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def population_stability_index(base: pd.Series, current: pd.Series, bins: int = 10) -> float:
    b = pd.Series(base).dropna().astype(float); c = pd.Series(current).dropna().astype(float)
    if len(b) < 50 or len(c) < 20: return 0.0
    cuts = np.unique(np.quantile(b, np.linspace(0, 1, bins + 1)))
    if len(cuts) < 3: return 0.0
    cuts[0], cuts[-1] = -np.inf, np.inf
    bp = np.histogram(b, cuts)[0] / len(b); cp = np.histogram(c, cuts)[0] / len(c)
    bp = np.clip(bp, 1e-6, None); cp = np.clip(cp, 1e-6, None)
    return float(np.sum((cp - bp) * np.log(cp / bp)))

def drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], psi_threshold: float = 0.20, ks_alpha: float = 0.01) -> dict:
    rows = []
    for col in columns:
        if col not in reference or col not in current: continue
        psi = population_stability_index(reference[col], current[col])
        a = reference[col].dropna().astype(float); b = current[col].dropna().astype(float)
        p = float(ks_2samp(a, b).pvalue) if len(a) and len(b) else 1.0
        rows.append({'feature': col, 'psi': psi, 'ks_pvalue': p, 'drift': bool(psi >= psi_threshold or p < ks_alpha)})
    frame = pd.DataFrame(rows)
    return {'features': rows, 'drifted_features': int(frame['drift'].sum()) if not frame.empty else 0, 'drift_ratio': float(frame['drift'].mean()) if not frame.empty else 0.0}
