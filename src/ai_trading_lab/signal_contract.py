from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class DirectSignal:
    """Stable inference contract exposed to paper/live consumers."""
    symbol: str
    timestamp: str
    signal: str
    confidence: float
    expected_return: float
    price: float
    horizon_bars: int
    actionable: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def direct_signal(
    *,
    symbol: str,
    timestamp: str,
    signal: str,
    confidence: float,
    expected_return: float,
    price: float,
    horizon_bars: int,
    reason: str = "",
) -> DirectSignal:
    side = str(signal).upper()
    if side not in {"LONG", "SHORT", "FLAT"}:
        side = "FLAT"
    conf = max(0.0, min(1.0, float(confidence)))
    return DirectSignal(
        symbol=str(symbol),
        timestamp=str(timestamp),
        signal=side,
        confidence=conf,
        expected_return=float(expected_return),
        price=float(price),
        horizon_bars=max(1, int(horizon_bars)),
        actionable=side in {"LONG", "SHORT"},
        reason=str(reason or ""),
    )
