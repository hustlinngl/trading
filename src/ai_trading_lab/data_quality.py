from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd


@dataclass
class QualityReport:
    rows: int
    missing_pct: float
    duplicate_timestamps: int
    non_monotonic: bool
    negative_volume_rows: int
    invalid_ohlc_rows: int
    gap_rows: int
    stale_tail_bars: int
    inferred_timeframe: str
    median_bar_minutes: float
    coverage_days: float
    timeframe_match: bool
    reasons: tuple[str, ...] = ()
    passed: bool = False

    def to_dict(self):
        return asdict(self)


def timeframe_minutes(timeframe: str) -> float:
    text = str(timeframe or "15m").strip().lower()
    units = {"m": 1.0, "h": 60.0, "d": 1440.0, "w": 10080.0}
    if not text:
        return 15.0
    if text[-1] not in units:
        return 15.0
    try:
        return float(text[:-1] or 1) * units[text[-1]]
    except ValueError:
        return 15.0


def format_timeframe(minutes: float) -> str:
    m = float(minutes)
    if m >= 10080 and abs(m / 10080 - round(m / 10080)) < 1e-6:
        return f"{int(round(m / 10080))}w"
    if m >= 1440 and abs(m / 1440 - round(m / 1440)) < 1e-6:
        return f"{int(round(m / 1440))}d"
    if m >= 60 and abs(m / 60 - round(m / 60)) < 1e-6:
        return f"{int(round(m / 60))}h"
    return f"{int(round(m))}m"


def infer_timeframe(df: pd.DataFrame) -> tuple[str, float]:
    if not isinstance(df.index, pd.DatetimeIndex) or len(df.index) < 3:
        return "unknown", 0.0
    diffs = pd.Series(df.index[1:] - df.index[:-1]).dt.total_seconds().div(60.0)
    diffs = diffs[(diffs > 0) & np.isfinite(diffs)]
    if diffs.empty:
        return "unknown", 0.0
    med = float(diffs.median())
    return format_timeframe(med), med


def audit_market_data(df: pd.DataFrame, timeframe: str = "15m") -> QualityReport:
    req = ["open", "high", "low", "close", "volume"]
    missing = 100.0 * float(df[req].isna().any(axis=1).mean()) if len(df) else 100.0
    dups = int(df.index.duplicated().sum()) if hasattr(df.index, "duplicated") else 0
    nonmono = not bool(df.index.is_monotonic_increasing)
    negvol = int((pd.to_numeric(df["volume"], errors="coerce") < 0).sum())
    op = pd.to_numeric(df["open"], errors="coerce")
    hi = pd.to_numeric(df["high"], errors="coerce")
    lo = pd.to_numeric(df["low"], errors="coerce")
    cl = pd.to_numeric(df["close"], errors="coerce")
    invalid = int(((hi < pd.concat([op, cl], axis=1).max(axis=1)) | (lo > pd.concat([op, cl], axis=1).min(axis=1)) | (lo <= 0) | (op <= 0) | (cl <= 0)).sum())
    inferred, median_minutes = infer_timeframe(df)
    expected_minutes = timeframe_minutes(timeframe)
    timeframe_match = bool(median_minutes > 0 and abs(median_minutes - expected_minutes) <= max(0.5, expected_minutes * 0.05))
    try:
        expected = pd.Timedelta(timeframe_minutes(timeframe), unit="m")
        delta = pd.Series(df.index[1:] - df.index[:-1])
        gaps = int((delta > expected * 1.5).sum())
    except Exception:
        gaps = 0
    stale = 0
    if len(df) > 1:
        tail = pd.to_numeric(df["close"].tail(min(40, len(df))), errors="coerce")
        for v in tail.diff().abs().fillna(np.nan).to_numpy()[::-1]:
            if np.isfinite(v) and float(v) == 0.0:
                stale += 1
            else:
                break
    coverage_days = 0.0
    if len(df) >= 2 and isinstance(df.index, pd.DatetimeIndex):
        coverage_days = max(0.0, float((df.index[-1] - df.index[0]).total_seconds()) / 86400.0)
    expected_rows = max(100, len(df))
    reasons = []
    if len(df) <= 100: reasons.append("insufficient_rows")
    if missing >= 0.5: reasons.append("missing_values")
    if dups: reasons.append("duplicate_timestamps")
    if nonmono: reasons.append("non_monotonic")
    if negvol: reasons.append("negative_volume")
    if invalid: reasons.append("invalid_ohlc")
    if gaps >= max(5, expected_rows * 0.005): reasons.append("too_many_gaps")
    if not timeframe_match: reasons.append("timeframe_mismatch")
    passed = not reasons
    return QualityReport(rows=len(df), missing_pct=missing, duplicate_timestamps=dups, non_monotonic=nonmono, negative_volume_rows=negvol, invalid_ohlc_rows=invalid, gap_rows=gaps, stale_tail_bars=stale, inferred_timeframe=inferred, median_bar_minutes=median_minutes, coverage_days=coverage_days, timeframe_match=timeframe_match, reasons=tuple(reasons), passed=passed)
