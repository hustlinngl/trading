from __future__ import annotations

import hashlib
import json
import numpy as np
from typing import Any
import pandas as pd


def dataset_fingerprint(df: pd.DataFrame, tail: int = 256) -> str:
    if df.empty:
        return "empty"
    frame = df.sort_index().tail(max(1, int(tail)))
    payload: list[Any] = [len(df), str(pd.Timestamp(df.index[-1]))]
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in frame.columns]
    raw = frame[cols].round(12).to_json(date_format="iso", orient="split")
    payload.append(raw)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def strong_dataset_fingerprint(df: pd.DataFrame) -> str:
    """Hash every numeric research input, not only OHLCV."""
    if df.empty:
        return "empty"
    frame=df.sort_index()
    cols=sorted([c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])])
    payload=[len(frame),str(pd.Timestamp(frame.index[0])),str(pd.Timestamp(frame.index[-1])),[(str(c),str(frame[c].dtype)) for c in cols]]
    idx_bytes=frame.index.view("int64").tobytes()
    numeric=frame[cols].replace([np.inf,-np.inf],np.nan) if cols else pd.DataFrame(index=frame.index)
    val_bytes=numeric.to_numpy(dtype="float64",copy=False).tobytes()
    h=hashlib.sha256(); h.update(json.dumps(payload,sort_keys=True).encode("utf-8")); h.update(idx_bytes); h.update(val_bytes)
    return h.hexdigest()[:32]
