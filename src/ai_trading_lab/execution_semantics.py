from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionSemantics:
    horizon_bars: int
    stop_atr_mult: float
    take_profit_rr: float
    purge_bars: int


def aligned_execution_parameters(settings) -> dict[str, float | int]:
    """Return the only execution geometry allowed for the trained label definition."""
    horizon = max(1, int(getattr(settings, "horizon_bars", 8)))
    sl_atr = max(1e-9, float(getattr(settings, "sl_atr", 1.0)))
    pt_atr = max(1e-9, float(getattr(settings, "pt_atr", 1.6)))
    purge = max(horizon, int(getattr(settings, "validation_purge_bars", horizon)))
    return {
        "max_holding_bars": horizon,
        "stop_atr_mult": sl_atr,
        "take_profit_rr": pt_atr / sl_atr,
        "validation_purge_bars": purge,
    }


def execution_semantics(settings) -> ExecutionSemantics:
    p = aligned_execution_parameters(settings)
    return ExecutionSemantics(
        horizon_bars=int(p["max_holding_bars"]),
        stop_atr_mult=float(p["stop_atr_mult"]),
        take_profit_rr=float(p["take_profit_rr"]),
        purge_bars=int(p["validation_purge_bars"]),
    )


def validate_execution_alignment(settings) -> ExecutionSemantics:
    """Fail closed when a runtime/backtest geometry no longer matches training labels."""
    expected = execution_semantics(settings)
    actual = {
        "max_holding_bars": int(getattr(settings, "max_holding_bars", expected.horizon_bars)),
        "stop_atr_mult": float(getattr(settings, "stop_atr_mult", expected.stop_atr_mult)),
        "take_profit_rr": float(getattr(settings, "take_profit_rr", expected.take_profit_rr)),
        "validation_purge_bars": int(getattr(settings, "validation_purge_bars", expected.purge_bars)),
    }
    mismatches = []
    if actual["max_holding_bars"] != expected.horizon_bars:
        mismatches.append(f"max_holding_bars={actual['max_holding_bars']} expected {expected.horizon_bars}")
    if abs(actual["stop_atr_mult"] - expected.stop_atr_mult) > 1e-9:
        mismatches.append(f"stop_atr_mult={actual['stop_atr_mult']} expected {expected.stop_atr_mult}")
    if abs(actual["take_profit_rr"] - expected.take_profit_rr) > 1e-9:
        mismatches.append(f"take_profit_rr={actual['take_profit_rr']} expected {expected.take_profit_rr}")
    if actual["validation_purge_bars"] < expected.horizon_bars:
        mismatches.append(
            f"validation_purge_bars={actual['validation_purge_bars']} expected >= {expected.horizon_bars}"
        )
    if mismatches:
        raise ValueError("execution_semantics_mismatch:" + "; ".join(mismatches))
    return expected
