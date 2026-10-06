from __future__ import annotations

from typing import Iterable
import numpy as np
from .data import timeframe_offset
import pandas as pd


def fetch_cross_asset_bars(exchange, symbols: Iterable[str], timeframe: str, limit: int = 1500) -> dict[str, pd.DataFrame]:
    out = {}
    for symbol in symbols:
        try:
            rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            frame = df.set_index("timestamp").astype(float).sort_index()
            if len(frame) > 1:
                try:
                    delta = timeframe_offset(timeframe)
                    if frame.index[-1] + delta > pd.Timestamp.now(tz='UTC'):
                        frame = frame.iloc[:-1]
                except Exception:
                    pass
            out[symbol] = frame
        except Exception:
            continue
    return out


def cross_asset_snapshot(frames: dict[str, pd.DataFrame], anchor: str, windows=(8, 32, 96)) -> dict[str, float]:
    if anchor not in frames or frames[anchor].empty:
        return {}
    anchor_end = frames[anchor].index.max()
    bounded = {k: v.loc[v.index <= anchor_end, 'close'] for k, v in frames.items()}
    prices = pd.concat(bounded, axis=1).sort_index().ffill(limit=2)
    if prices.empty:
        return {}
    r = prices.pct_change()
    out: dict[str, float] = {}
    anchor_r = r[anchor]
    for symbol in prices.columns:
        if symbol == anchor: continue
        name = symbol.replace("/", "_").replace(":", "_")
        for w in windows:
            corr = anchor_r.rolling(w).corr(r[symbol]).iloc[-1]
            beta = anchor_r.rolling(w).cov(r[symbol]).iloc[-1] / max(anchor_r.rolling(w).var().iloc[-1], 1e-12)
            out[f"corr_{name}_{w}"] = float(corr) if np.isfinite(corr) else 0.0
            out[f"beta_{name}_{w}"] = float(beta) if np.isfinite(beta) else 0.0
    latest = r.tail(min(8, len(r))).mean(); vals = latest.drop(labels=[anchor], errors="ignore").to_numpy(float)
    if len(vals):
        out["cross_asset_mean_return"] = float(np.nanmean(vals)); out["cross_asset_dispersion"] = float(np.nanstd(vals)); out["cross_asset_breadth"] = float(np.nanmean(vals > 0))
    else:
        out.update({"cross_asset_mean_return": 0.0, "cross_asset_dispersion": 0.0, "cross_asset_breadth": 0.5})
    return out
