from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any, Mapping


PUBLIC_SIGNALS = frozenset({"LONG", "SHORT", "FLAT"})
PUBLIC_SIGNAL_FIELDS = frozenset(
    {
        "symbol",
        "signal",
        "confidence",
        "expected_return",
        "price",
        "horizon_bars",
    }
)


class SignalContractError(ValueError):
    """Raised when inference output cannot be published as a direct signal."""


@dataclass(frozen=True, slots=True)
class DirectSignal:
    """The only model result allowed to cross the live/user boundary."""

    symbol: str
    signal: str
    confidence: float
    expected_return: float
    price: float
    horizon_bars: int

    def __post_init__(self) -> None:
        if not str(self.symbol).strip():
            raise SignalContractError("invalid_symbol")
        if self.signal not in PUBLIC_SIGNALS:
            raise SignalContractError("invalid_signal")
        for name, value in (
            ("confidence", self.confidence),
            ("expected_return", self.expected_return),
            ("price", self.price),
        ):
            if not isfinite(float(value)):
                raise SignalContractError(f"non_finite_{name}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise SignalContractError("invalid_confidence")
        if float(self.price) <= 0.0:
            raise SignalContractError("invalid_price")
        if int(self.horizon_bars) < 1:
            raise SignalContractError("invalid_horizon")

    def to_dict(self) -> dict[str, Any]:
        """Return exactly the six-field public contract."""
        return asdict(self)


def compile_direct_signal(
    row: Mapping[str, Any],
    *,
    symbol: str,
    timestamp: str,
    price: float,
    horizon_bars: int,
) -> DirectSignal:
    """Compile a validated internal decision into the public direct-signal contract.

    timestamp is intentionally accepted as an internal provenance input but is never
    emitted to the caller-facing payload.
    """
    if not str(timestamp).strip():
        raise SignalContractError("invalid_timestamp")

    required = ("action", "p_up", "expected_return")
    missing = [key for key in required if row.get(key) is None]
    if missing:
        raise SignalContractError("missing:" + ",".join(missing))

    signal = str(row["action"]).upper()
    if signal not in PUBLIC_SIGNALS:
        raise SignalContractError("invalid_action")

    p_up = float(row["p_up"])
    if not isfinite(p_up) or not 0.0 <= p_up <= 1.0:
        raise SignalContractError("invalid_probability")

    expected_return = float(row["expected_return"])
    if not isfinite(expected_return):
        raise SignalContractError("invalid_expected_return")

    # The internal regressor predicts a LONG-side return. A public SHORT
    # result must report the return in the predicted direction instead.
    if signal == "LONG" and p_up < 0.5:
        raise SignalContractError("action_probability_mismatch")
    if signal == "SHORT" and p_up >= 0.5:
        raise SignalContractError("action_probability_mismatch")
    directional_expected_return = expected_return if p_up >= 0.5 else -expected_return

    confidence = 0.0
    if signal == "LONG":
        confidence = p_up
    elif signal == "SHORT":
        confidence = 1.0 - p_up

    return DirectSignal(
        symbol=str(symbol),
        signal=signal,
        confidence=confidence,
        expected_return=directional_expected_return,
        price=float(price),
        horizon_bars=int(horizon_bars),
    )
