from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from .data_quality import audit_market_data, infer_timeframe

BINANCE_HEADERLESS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
]

ALIASES = {
    "timestamp": {"timestamp", "time", "datetime", "date", "open_time", "opentime"},
    "open": {"open", "o"},
    "high": {"high", "h"},
    "low": {"low", "l"},
    "close": {"close", "c", "price"},
    "volume": {"volume", "vol", "v", "base_volume"},
}


def _norm_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def _parse_timestamp(values) -> pd.Series:
    raw = pd.Series(values)
    numeric = pd.to_numeric(raw, errors="coerce")
    result = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]")
    mask = numeric.notna()
    if mask.any():
        magnitude = float(numeric[mask].abs().median())
        if magnitude >= 1e17:
            unit = "ns"
        elif magnitude >= 1e14:
            unit = "us"
        elif magnitude >= 1e11:
            unit = "ms"
        elif magnitude >= 1e8:
            unit = "s"
        else:
            unit = None
        if unit is not None:
            result.loc[mask] = pd.to_datetime(numeric.loc[mask], unit=unit, utc=True, errors="coerce")
    text_mask = ~mask
    if text_mask.any():
        result.loc[text_mask] = pd.to_datetime(raw.loc[text_mask], utc=True, errors="coerce")
    return result


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Dataset is empty")
    cols = {_norm_name(c): c for c in df.columns}
    rename = {}
    for target, candidates in ALIASES.items():
        found = next((cols[c] for c in candidates if c in cols), None)
        if found is not None:
            rename[found] = target
    out = df.rename(columns=rename).copy()
    missing = {"open", "high", "low", "close", "volume"} - set(out.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    if "timestamp" in out.columns:
        ts = _parse_timestamp(out["timestamp"])
        out = out.drop(columns=["timestamp"])
    elif isinstance(out.index, pd.DatetimeIndex):
        ts = pd.to_datetime(out.index, utc=True, errors="coerce")
    else:
        raise ValueError("Dataset needs a timestamp/date column or DatetimeIndex")
    out.index = ts
    out = out[~out.index.isna()].sort_index()
    duplicated = out.index.duplicated(keep=False)
    if duplicated.any():
        dup = out.loc[duplicated]
        check_cols = [c for c in ("open","high","low","close","volume","quote_volume","trades","taker_buy_base_volume","taker_buy_quote_volume") if c in dup.columns]
        if bool((dup.groupby(level=0)[check_cols].nunique(dropna=False).max(axis=1) > 1).any()):
            raise ValueError("Conflicting duplicate timestamps in imported market data")
        out = out[~out.index.duplicated(keep="last")]
    for c in ["open", "high", "low", "close", "volume"]:

        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close", "volume"])
    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (out["volume"] < 0).any():
        raise ValueError("Volume cannot be negative")
    extra=["quote_volume","trades","taker_buy_base_volume","taker_buy_quote_volume"]
    return out[[c for c in ["open","high","low","close","volume"]+extra if c in out.columns]]


def read_market_file(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".parquet", ".pq"}:
        raw = pd.read_parquet(p)
    elif p.suffix.lower() in {".csv", ".txt"}:
        raw = pd.read_csv(p)
        normalized_columns = {_norm_name(c) for c in raw.columns}
        required_aliases = {"open", "high", "low", "close", "volume"}
        if not required_aliases.issubset(normalized_columns):
            try:
                no_header = pd.read_csv(p, header=None)
                if no_header.shape[1] >= 6:
                    cols = BINANCE_HEADERLESS[:no_header.shape[1]]
                    no_header.columns = cols
                    raw = no_header
            except Exception:
                pass
    elif p.suffix.lower() in {".json"}:
        raw = pd.read_json(p)
    else:
        raise ValueError(f"Unsupported data format: {p.suffix}. Use CSV, Parquet or JSON.")
    return normalize_ohlcv(raw)


def symbol_slug(symbol: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(symbol)).strip("_").upper()


def infer_symbol(path: Path) -> str:
    stem = path.stem.upper()
    m = re.search(r"([A-Z0-9]{2,})[_\-]?(USDT|USD|EUR|BTC|ETH)(?:[_\-]|$)", stem)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return stem


def import_market_file(path: str | Path, data_dir: str | Path = "data/imported", symbol: str | None = None) -> dict:
    p = Path(path)
    df = read_market_file(p)
    sym = symbol or infer_symbol(p)
    out_dir = Path(data_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{symbol_slug(sym)}.parquet"
    try:
        df.to_parquet(out)
    except ImportError:
        out = out_dir / f"{symbol_slug(sym)}.csv"
        df.to_csv(out, index_label="timestamp")
    inferred, _ = infer_timeframe(df)
    quality_frame = inferred if inferred != "unknown" else "15m"
    inferred_quality = audit_market_data(df, quality_frame)
    meta = {
        "symbol": sym,
        "path": str(out),
        "rows": int(len(df)),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "columns": list(df.columns),
        "inferred_timeframe": inferred_quality.inferred_timeframe,
        "median_bar_minutes": float(inferred_quality.median_bar_minutes),
        "coverage_days": float(inferred_quality.coverage_days),
        "gap_rows": int(inferred_quality.gap_rows),
        "timeframe_match": bool(inferred_quality.timeframe_match),
        "quality_passed": bool(inferred_quality.passed),
        "quality_issues": list(inferred_quality.reasons),
    }
    registry = out_dir / "registry.json"
    existing = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else {}
    existing[sym] = meta
    tmp_registry = registry.with_suffix(".json.tmp")
    tmp_registry.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    tmp_registry.replace(registry)
    return meta


def import_market_path(path: str | Path, data_dir: str | Path = "data/imported", symbol: str | None = None) -> list[dict]:
    p = Path(path)
    files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.suffix.lower() in {".csv", ".parquet", ".pq", ".json"})
    if not files:
        raise ValueError(f"No supported market files found under {p}")
    return [import_market_file(f, data_dir=data_dir, symbol=symbol if len(files) == 1 else None) for f in files]


def imported_registry(data_dir: str | Path = "data/imported") -> dict:
    p = Path(data_dir) / "registry.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_imported(symbol: str, data_dir: str | Path = "data/imported") -> pd.DataFrame:
    meta = imported_registry(data_dir).get(symbol)
    if not meta:
        raise FileNotFoundError(f"No imported dataset registered for {symbol}")
    p = Path(meta["path"])
    return normalize_ohlcv(pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p, index_col="timestamp", parse_dates=True))
