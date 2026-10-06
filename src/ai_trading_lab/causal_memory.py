from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.stats import ttest_1samp


@dataclass
class MatchedEffect:
    event_type: str
    n_treated: int
    n_controls: int
    horizon: int
    treated_mean: float
    control_mean: float
    effect: float
    t_stat: float
    prob_effect_positive: float
    caveat: str = "Observational matched estimate; not proof of causality."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MatchedEventMemory:
    """Causal-style event memory based on state matching.

    It compares event windows with nearby, non-event market states using the same
    observable state variables. This can reduce confounding from market regime,
    but it is deliberately labelled observational rather than causal inference.
    """

    def __init__(self, k: int = 5, exclusion_bars: int = 8, seed: int = 42, control_lookback_bars: int = 1500):
        self.k = max(2, int(k))
        self.exclusion_bars = max(1, int(exclusion_bars))
        self.seed = int(seed)
        self.control_lookback_bars = max(self.k * 4, int(control_lookback_bars))

    @staticmethod
    def _event_type(text: str) -> str:
        t = str(text or "").lower()
        rules = {
            "rates": ("rate", "fomc", "fed", "yield"),
            "inflation": ("inflation", "cpi", "pce"),
            "earnings": ("earnings", "guidance", "revenue"),
            "regulation": ("sec", "regulation", "ban", "approval", "lawsuit"),
            "crypto_flow": ("etf", "flow", "inflow", "outflow", "fund"),
            "security": ("hack", "exploit", "breach"),
            "geopolitics": ("tariff", "sanction", "war", "election"),
        }
        for key, words in rules.items():
            if any(w in t for w in words):
                return key
        return "other"

    def study(self, prices: pd.Series, events: pd.DataFrame, state_features: pd.DataFrame,
              horizon: int = 8, max_events: int = 250) -> pd.DataFrame:
        if prices.empty or events.empty or state_features.empty:
            return pd.DataFrame()
        p = prices.sort_index().astype(float)
        x = state_features.sort_index().select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        common = p.index.intersection(x.index)
        p, x = p.loc[common], x.loc[common]
        if len(x) < 100:
            return pd.DataFrame()
        prepared = []; used_positions: set[int] = set()
        for _, e in events.head(max_events).iterrows():
            try:
                t = pd.Timestamp(e.get("event_time") or e.get("published_at"))
                t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
            except Exception:
                continue
            pos = int(p.index.searchsorted(t, side="left")) - 1
            if pos <= 0 or pos + horizon >= len(p):
                continue
            lo, hi = pos - self.exclusion_bars, pos + self.exclusion_bars
            for j in range(max(0, lo), min(len(p), hi + 1)):
                used_positions.add(j)
            prepared.append((pos, self._event_type(e.get("title") or e.get("text") or e.get("query"))))
        if not prepared:
            return pd.DataFrame()
        scaler_cols = list(x.columns[: min(40, len(x.columns))]); rows = []; rng = np.random.default_rng(self.seed)
        for pos, event_type in prepared:
            exclusion = max(self.exclusion_bars, int(horizon))
            lo = max(exclusion, pos - self.control_lookback_bars); hi = pos - exclusion
            candidates = np.arange(lo, hi, dtype=int)
            if len(candidates) < self.k:
                continue
            ref = x.iloc[candidates][scaler_cols].to_numpy(float)
            med = np.median(ref, axis=0); mad = np.median(np.abs(ref - med), axis=0); scale = np.where(mad < 1e-9, 1.0, 1.4826 * mad)
            zc = (ref - med) / scale; zt = (x.iloc[pos][scaler_cols].to_numpy(float) - med) / scale
            nn = NearestNeighbors(n_neighbors=min(self.k, len(candidates)), metric="euclidean").fit(zc)
            _, inds = nn.kneighbors(zt.reshape(1, -1)); controls = candidates[inds[0]]
            treated_ret = float(p.iloc[pos + horizon] / p.iloc[pos] - 1.0)
            control_ret = (p.iloc[controls + horizon].to_numpy() / p.iloc[controls].to_numpy() - 1.0)
            rows.append((event_type, treated_ret, float(np.mean(control_ret))))
        frame = pd.DataFrame(rows, columns=["event_type", "treated", "control"]); out = []
        for event_type, g in frame.groupby("event_type"):
            diffs = (g["treated"] - g["control"]).to_numpy(float)
            if len(diffs) == 0: continue
            stat = float(ttest_1samp(diffs, 0.0).statistic) if len(diffs) > 1 and np.std(diffs, ddof=1) > 0 else 0.0
            boots = rng.choice(diffs, size=(1200, len(diffs)), replace=True).mean(axis=1)
            out.append(MatchedEffect(str(event_type), int(len(g)), int(len(g) * self.k), int(horizon),
                                     float(g["treated"].mean()), float(g["control"].mean()), float(diffs.mean()),
                                     stat, float(np.mean(boots > 0))).to_dict())
        return pd.DataFrame(out).sort_values("effect", key=np.abs, ascending=False) if out else pd.DataFrame()
