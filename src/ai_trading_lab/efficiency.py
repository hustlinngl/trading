from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import time
from typing import Any, Mapping

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


@dataclass
class ExecutionEfficiencySnapshot:
    """Observed execution telemetry; missing observations remain None."""

    slippage_bps: float | None = None
    latency_ms: float | None = None
    sample_count: int = 0
    source: str = "unavailable"
    execution_available: bool = False
    market_data_latency_ms: float | None = None
    configured_slippage_bps: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExecutionEfficiencyTracker:
    """Rolling execution telemetry collector independent from order placement."""

    def __init__(self, max_samples: int = 512):
        self._samples = deque(maxlen=max(16, int(max_samples)))

    def record(
        self,
        *,
        expected_price: float,
        execution_price: float,
        side: str,
        sent_at: float | None = None,
        filled_at: float | None = None,
    ) -> None:
        expected = float(expected_price)
        executed = float(execution_price)
        if not np.isfinite(expected) or expected <= 0 or not np.isfinite(executed):
            return
        direction = 1.0 if str(side).upper() == "BUY" else -1.0
        slippage_bps = direction * (executed / expected - 1.0) * 10_000.0
        latency_ms = None
        if sent_at is not None and filled_at is not None:
            latency_ms = max(
                0.0,
                (float(filled_at) - float(sent_at)) * 1000.0,
            )
        self._samples.append(
            {
                "slippage_bps": float(slippage_bps),
                "latency_ms": latency_ms,
                "recorded_at": time.time(),
            }
        )

    def snapshot(
        self,
        *,
        market_data_latency_ms: float | None = None,
        configured_slippage_bps: float | None = None,
    ) -> dict[str, Any]:
        slippage = [
            x["slippage_bps"]
            for x in self._samples
            if x.get("slippage_bps") is not None
        ]
        latency = [
            x["latency_ms"]
            for x in self._samples
            if x.get("latency_ms") is not None
        ]
        result = ExecutionEfficiencySnapshot(
            slippage_bps=float(np.median(slippage)) if slippage else None,
            latency_ms=float(np.median(latency)) if latency else None,
            sample_count=len(self._samples),
            source=(
                "execution_telemetry"
                if self._samples
                else "unavailable"
            ),
            execution_available=bool(self._samples),
            market_data_latency_ms=(
                float(market_data_latency_ms)
                if market_data_latency_ms is not None
                and np.isfinite(float(market_data_latency_ms))
                else None
            ),
            configured_slippage_bps=(
                float(configured_slippage_bps)
                if configured_slippage_bps is not None
                and np.isfinite(float(configured_slippage_bps))
                else None
            ),
        )
        return result.to_dict()


def execution_efficiency_snapshot(
    telemetry: ExecutionEfficiencyTracker | Mapping[str, Any] | None = None,
    *,
    market_data_latency_ms: float | None = None,
    configured_slippage_bps: float | None = None,
) -> dict[str, Any]:
    """Normalize execution telemetry and live stream latency for dashboards."""
    if telemetry is not None and hasattr(telemetry, "snapshot"):
        try:
            snapshot = telemetry.snapshot(
                market_data_latency_ms=market_data_latency_ms,
                configured_slippage_bps=configured_slippage_bps,
            )
            if isinstance(snapshot, Mapping):
                return dict(snapshot)
        except Exception:
            pass

    source = "context"
    slippage = None
    latency = None
    sample_count = 0
    available = False
    if isinstance(telemetry, Mapping):
        raw_slippage = telemetry.get("slippage_bps")
        raw_latency = telemetry.get("latency_ms")
        try:
            slippage = float(raw_slippage) if raw_slippage is not None else None
        except (TypeError, ValueError):
            slippage = None
        try:
            latency = float(raw_latency) if raw_latency is not None else None
        except (TypeError, ValueError):
            latency = None
        sample_count = int(telemetry.get("sample_count", 0) or 0)
        available = bool(
            telemetry.get(
                "execution_available",
                slippage is not None or latency is not None,
            )
        )
        source = str(telemetry.get("source") or source)
        if "market_data_latency_ms" in telemetry:
            market_data_latency_ms = telemetry.get("market_data_latency_ms")
        if "configured_slippage_bps" in telemetry:
            configured_slippage_bps = telemetry.get("configured_slippage_bps")

    if latency is None and market_data_latency_ms is not None:
        try:
            latency = float(market_data_latency_ms)
            source = "market_stream"
        except (TypeError, ValueError):
            pass

    return ExecutionEfficiencySnapshot(
        slippage_bps=slippage,
        latency_ms=latency,
        sample_count=sample_count,
        source=source,
        execution_available=bool(available),
        market_data_latency_ms=(
            float(latency) if latency is not None else None
        ),
        configured_slippage_bps=(
            float(configured_slippage_bps)
            if configured_slippage_bps is not None
            else None
        ),
    ).to_dict()


def audit_features(
    X: pd.DataFrame,
    y: pd.Series | None = None,
    corr_threshold: float = 0.995,
    selection_mask: pd.Series | None = None,
):
    z = X.select_dtypes(include=[np.number]).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    if z.empty:
        return z, EfficiencyReport(0, 0, 0, 0, [], 0.0)

    leakage = [
        c
        for c in z.columns
        if any(
            tok in c.lower()
            for tok in ("future", "target", "label", "forward", "lead_")
        )
    ]
    missing = z.isna().mean()
    low_info = [
        c
        for c in z.columns
        if missing[c] > 0.35 or z[c].nunique(dropna=True) <= 2
    ]
    ranked = list(z.columns)
    selection = (
        z
        if selection_mask is None
        else z.loc[selection_mask.reindex(z.index).fillna(False)]
    )

    if y is not None:
        try:
            valid = y.notna()
            if selection_mask is not None:
                valid = valid & selection_mask.reindex(y.index).fillna(False)
            vals = z.loc[valid].ffill().fillna(0.0)
            if len(vals) >= 200 and y.loc[valid].nunique() > 1:
                tail = min(len(vals), 5000)
                mi = mutual_info_classif(
                    vals.iloc[-tail:],
                    y.loc[valid].iloc[-tail:].astype(int),
                    random_state=42,
                )
                ranked = [vals.columns[i] for i in np.argsort(mi)[::-1]] + [
                    c for c in ranked if c not in vals.columns
                ]
        except Exception:
            pass

    keep, redundant = [], []
    corr = selection[ranked].corr().abs()
    for c in ranked:
        if c in low_info or c in leakage:
            continue
        if any(
            corr.loc[c, k] >= corr_threshold
            for k in keep
            if c in corr.index and k in corr.index
        ):
            redundant.append(c)
        else:
            keep.append(c)

    return z[keep].copy(), EfficiencyReport(
        len(z.columns),
        len(keep),
        len(redundant),
        len(low_info),
        leakage,
        0.0,
    )


def timed_transform(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0
