from __future__ import annotations

import time
from typing import Iterable
from pathlib import Path
import pandas as pd
import numpy as np

def timeframe_offset(timeframe: str):
    tf = str(timeframe).strip().lower()
    import re
    m = re.fullmatch(r'(\d+)(s|min|m|h|d|w)', tf)
    if not m:
        raise ValueError(f'Unsupported timeframe: {timeframe}')
    n, unit = int(m.group(1)), m.group(2)
    unit = {'m':'min'}.get(unit, unit)
    return pd.tseries.frequencies.to_offset(f'{n}{unit}')

def _deduplicate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    dup_mask=df.index.duplicated(keep=False)
    if not dup_mask.any():
        return df.sort_index()
    dupes=df.loc[dup_mask].sort_index()
    compare_cols=[c for c in ('open','high','low','close','volume','quote_volume','trades','taker_buy_base_volume','taker_buy_quote_volume') if c in dupes.columns]
    for ts, group in dupes.groupby(level=0, sort=False):
        first=group.iloc[0]
        for col in compare_cols:
            vals=pd.to_numeric(group[col],errors='coerce').to_numpy(float)
            if not bool(np.all((np.isclose(vals, vals[0], rtol=1e-10, atol=1e-12, equal_nan=True)))):
                raise ValueError(f"Conflicting duplicate market timestamp: {ts} column={col}")
    return df[~df.index.duplicated(keep='last')].sort_index()

try:
    import ccxt
except ImportError:
    ccxt = None

def exchange_client(exchange_id: str = 'binance', sandbox: bool = True, api_key: str | None = None, secret: str | None = None):
    if ccxt is None:
        raise RuntimeError('ccxt is not installed. Install the project before using exchange data.')
    cls = getattr(ccxt, exchange_id)
    params = {'enableRateLimit': True}
    if api_key: params['apiKey'] = api_key
    if secret: params['secret'] = secret
    ex = cls(params)
    if sandbox: ex.set_sandbox_mode(True)
    ex.load_markets()
    return ex

def fetch_ohlcv(exchange, symbol: str, timeframe: str, limit: int = 8000, since: int | None = None, include_unclosed: bool = False) -> pd.DataFrame:
    all_rows = []
    tf_ms = int(exchange.parse_timeframe(timeframe) * 1000)
    cursor = since
    target = int(limit)
    while len(all_rows) < target:
        batch_limit = min(1000, target - len(all_rows))
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=batch_limit)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = int(batch[-1][0])
        next_cursor = last_ts + tf_ms
        if cursor is not None and next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(batch) < batch_limit:
            break
        time.sleep(max(float(getattr(exchange, 'rateLimit', 0)) / 1000.0, 0.0))
    df = pd.DataFrame(all_rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    if df.empty: raise RuntimeError(f'No OHLCV returned for {symbol} {timeframe}')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp').astype(float)
    df = _deduplicate_ohlcv(df)
    return df if include_unclosed else drop_unclosed_tail(df, timeframe)

def drop_unclosed_tail(df: pd.DataFrame, timeframe: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    try:
        delta = timeframe_offset(timeframe)
    except Exception:
        return df
    return df.iloc[:-1] if df.index[-1] + delta > now else df

def cache_ohlcv(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    df.to_parquet(tmp, index=True)
    tmp.replace(p)

def load_cached(path: str | Path, timeframe: str | None = None, *, drop_unclosed: bool = True) -> pd.DataFrame:
    df=pd.read_parquet(path).sort_index()
    if df.index.has_duplicates:
        df=_deduplicate_ohlcv(df)
    return drop_unclosed_tail(df,timeframe) if drop_unclosed and timeframe else df

def fetch_ohlcv_incremental(exchange, symbol: str, timeframe: str, cache_path: str | Path, target_bars: int = 8000, include_unclosed: bool = False) -> pd.DataFrame:
    path = Path(cache_path)
    if not path.exists():
        df = fetch_ohlcv(exchange, symbol, timeframe, target_bars, include_unclosed=include_unclosed)
        cache_ohlcv(df, path)
        return df
    cached = load_cached(path, timeframe=timeframe, drop_unclosed=not include_unclosed)
    if cached.empty:
        df = fetch_ohlcv(exchange, symbol, timeframe, target_bars, include_unclosed=include_unclosed)
        cache_ohlcv(df, path)
        return df
    tf_ms = int(exchange.parse_timeframe(timeframe) * 1000)
    last_ts = int(cached.index[-1].timestamp() * 1000)
    since = max(0, last_ts - 2 * tf_ms)
    try:
        combined = cached.copy()
        cursor = since
        while len(combined) < target_bars:
            need = min(1000, target_bars - len(combined) + 2)
            fresh = fetch_ohlcv(exchange, symbol, timeframe, need, since=cursor)
            if fresh.empty:
                break
            combined = pd.concat([combined, fresh]).sort_index()
            combined = _deduplicate_ohlcv(combined)
            last = int(fresh.index[-1].timestamp() * 1000)
            next_cursor = last + tf_ms
            if next_cursor <= cursor or len(fresh) < need:
                break
            cursor = next_cursor
            time.sleep(max(float(getattr(exchange, 'rateLimit', 0)) / 1000.0, 0.0))
        combined = combined.sort_index()
        if not include_unclosed:
            combined = drop_unclosed_tail(combined, timeframe)
        if len(combined) > target_bars:
            combined = combined.iloc[-target_bars:]
        cache_ohlcv(combined, path)
        return combined
    except Exception:
        # Preserve last known-good cache, but keep it subject to the closed-bar policy.
        fallback=cached.iloc[-target_bars:]
        return drop_unclosed_tail(fallback,timeframe) if not include_unclosed else fallback

def asof_join(base: pd.DataFrame, source: pd.DataFrame, *, source_time: str = "timestamp", columns: Iterable[str] | None = None, lag: pd.Timedelta | None = None, tolerance: pd.Timedelta | None = None, suffix: str = "") -> pd.DataFrame:
    if base.empty or source.empty:
        return base.copy()
    b = base.copy().sort_index()
    src = source.copy()
    if source_time in src.columns:
        src[source_time] = pd.to_datetime(src[source_time], utc=True)
        src = src.set_index(source_time)
    if not isinstance(b.index, pd.DatetimeIndex) or not isinstance(src.index, pd.DatetimeIndex):
        raise TypeError("base and source must use DatetimeIndex or source_time must define one")
    src = src.sort_index()
    if columns is not None:
        src = src[[c for c in columns if c in src.columns]]
    if lag is not None:
        src = src.copy()
        src.index = src.index + lag

    positions = src.index.searchsorted(b.index, side="right") - 1
    valid = positions >= 0
    if tolerance is not None and len(src):
        safe_positions = np.maximum(positions, 0)
        valid &= (b.index - src.index.to_numpy()[safe_positions]) <= tolerance

    aligned = pd.DataFrame(index=b.index)
    for col in src.columns:
        target = col if col not in b.columns else f"{col}{suffix}"
        series = pd.Series(np.nan, index=b.index, dtype="object")
        if valid.any():
            series.iloc[np.flatnonzero(valid)] = src[col].iloc[positions[valid]].to_numpy()
        aligned[target] = series
    return pd.concat([b, aligned], axis=1)

