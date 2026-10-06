from __future__ import annotations

import pandas as pd
import numpy as np

def enrich_derivatives(exchange, symbol: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Best-effort funding/open-interest enrichment when the exchange supports it."""
    out = frame.copy()
    for col in ("funding_rate", "open_interest"):
        out[col] = np.nan
    try:
        if exchange.has.get("fetchFundingRate"):
            fr = exchange.fetch_funding_rate(symbol)
            val = fr.get("fundingRate")
            if val is not None and len(out):
                out.loc[out.index[-1], "funding_rate"] = float(val)
    except Exception:
        pass
    try:
        if exchange.has.get("fetchOpenInterest"):
            oi = exchange.fetch_open_interest(symbol)
            val = oi.get("openInterestAmount") or oi.get("openInterestValue")
            if val is not None and len(out):
                out.loc[out.index[-1], "open_interest"] = float(val)
    except Exception:
        pass
    return out
