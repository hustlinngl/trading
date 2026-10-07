from __future__ import annotations
import numpy as np
import pandas as pd

def triple_barrier_labels(df: pd.DataFrame, horizon: int = 8, pt_atr: float = 1.6, sl_atr: float = 1.0) -> pd.DataFrame:
    """Triple-barrier outcomes aligned to executable next-open entry."""
    opn, high, low, close = (df[c].to_numpy(float) for c in ("open","high","low","close"))
    tr = pd.concat([(df['high']-df['low']), (df['high']-df['close'].shift()).abs(), (df['low']-df['close'].shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().to_numpy(float)
    out=np.full(len(df), np.nan); ret=np.full(len(df), np.nan); h=int(horizon)
    for i in range(len(df)-h-1):
        entry_i=i+1
        if not np.isfinite(atr[i]) or atr[i]<=0 or not np.isfinite(opn[entry_i]) or opn[entry_i]<=0: continue
        entry=opn[entry_i]; upper=entry+pt_atr*atr[i]; lower=entry-sl_atr*atr[i]; end_i=min(len(df)-1,entry_i+h)
        label=0; realized=close[end_i]/entry-1
        for j in range(entry_i,end_i+1):
            hit_up=high[j]>=upper; hit_down=low[j]<=lower
            if hit_up and hit_down:
                # Intrabar OHLC cannot tell which barrier was touched first.
                # Mark the outcome ambiguous instead of injecting a directional bias.
                label=np.nan
                realized=np.nan
                break
            if hit_up: label=1; realized=upper/entry-1; break
            if hit_down: label=-1; realized=lower/entry-1; break
        out[i]=label; ret[i]=realized
    return pd.DataFrame({'tb_label':out,'tb_return':ret},index=df.index)
