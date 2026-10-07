from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .cross_asset import cross_asset_snapshot, fetch_cross_asset_bars
from .efficiency import execution_efficiency_snapshot
from .microstructure import collect_microstructure
from .regimes import RegimeDetector


_REGIME_LABELS = {
    "trend_up": ("Trend", "UP"),
    "high_vol_up": ("High Volatility", "UP"),
    "high_vol_down": ("High Volatility", "DOWN"),
    "range_or_down": ("Range / Mean Reversion", "DOWN"),
    "transition": ("Transition", "NEUTRAL"),
    "unknown": ("Unknown", "NEUTRAL"),
}


@dataclass
class MarketState:
    timestamp: str
    symbol: str
    price: float
    regime_hint: str
    microstructure: dict[str, float]
    derivatives: dict[str, float]
    cross_asset: dict[str, float]
    macro: dict[str, float]
    external: dict[str, float]
    quality: dict[str, Any]
    intelligence: dict[str, Any] = field(default_factory=dict)

    def flatten(self) -> dict[str, Any]:
        out = {
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "price": self.price,
            "regime_hint": self.regime_hint,
        }
        for prefix, values in (
            ("micro", self.microstructure),
            ("deriv", self.derivatives),
            ("cross", self.cross_asset),
            ("macro", self.macro),
            ("ext", self.external),
        ):
            for key, value in values.items():
                if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
                    out[f"{prefix}_{key}"] = float(value)
        out["quality_json"] = str(self.quality)
        if self.intelligence:
            out["intelligence"] = dict(self.intelligence)
        return out


def derivative_snapshot(exchange, symbol: str) -> dict[str, float]:
    out = {"funding_rate": np.nan, "open_interest": np.nan}
    try:
        market = exchange.market(symbol)
        contract_symbol = symbol
        if not market.get("contract"):
            candidates = [
                market_item
                for market_item in exchange.markets.values()
                if market_item.get("base") == market.get("base")
                and market_item.get("quote") == market.get("quote")
                and market_item.get("contract")
                and market_item.get("swap")
            ]
            if candidates:
                contract_symbol = candidates[0]["symbol"]
        funding = (
            exchange.fetch_funding_rate(contract_symbol)
            if exchange.has.get("fetchFundingRate")
            else {}
        )
        open_interest = (
            exchange.fetch_open_interest(contract_symbol)
            if exchange.has.get("fetchOpenInterest")
            else {}
        )
        if funding.get("fundingRate") is not None:
            out["funding_rate"] = float(funding["fundingRate"])
        value = open_interest.get("openInterestValue") or open_interest.get(
            "openInterestAmount"
        )
        if value is not None:
            out["open_interest"] = float(value)
    except Exception:
        pass
    return out


def build_market_state(
    exchange,
    symbol: str,
    timeframe: str,
    price: float,
    macro: dict[str, float] | None = None,
    external: dict[str, float] | None = None,
    cross_symbols: list[str] | None = None,
    orderbook_levels: int = 50,
    impact_notional: float = 10_000.0,
) -> MarketState:
    symbols = cross_symbols or [symbol, "ETH/USDT", "SOL/USDT"]

    def get_micro():
        try:
            return collect_microstructure(
                exchange,
                symbol,
                limit=orderbook_levels,
                impact_notional=impact_notional,
            )
        except Exception:
            return {}

    def get_cross():
        try:
            frames = fetch_cross_asset_bars(
                exchange,
                [s for s in symbols if s != symbol] + [symbol],
                timeframe,
                limit=300,
            )
            return cross_asset_snapshot(frames, symbol)
        except Exception:
            return {}

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="state") as pool:
        micro_future = pool.submit(get_micro)
        cross_future = pool.submit(get_cross)
        micro = micro_future.result()
        cross = cross_future.result()

    derivatives = derivative_snapshot(exchange, symbol)
    macro = {
        key: float(value)
        for key, value in (macro or {}).items()
        if isinstance(value, (int, float)) and np.isfinite(value)
    }
    external = {
        key: float(value)
        for key, value in (external or {}).items()
        if isinstance(value, (int, float)) and np.isfinite(value)
    }
    quality = {
        "microstructure_available": bool(micro),
        "derivatives_available": bool(
            np.isfinite(derivatives.get("funding_rate", np.nan))
            or np.isfinite(derivatives.get("open_interest", np.nan))
        ),
        "cross_asset_available": bool(cross),
        "macro_available": bool(macro),
        "external_available": bool(external),
    }
    return MarketState(
        pd.Timestamp.now(tz="UTC").isoformat(),
        symbol,
        float(price),
        "unknown",
        micro,
        derivatives,
        cross,
        macro,
        external,
        quality,
    )


def _finite_float(value: Any) -> float | None:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    return candidate if np.isfinite(candidate) else None


def _normalize_regime(raw_label: Any) -> tuple[str, str]:
    raw = str(raw_label or "unknown").strip().lower()
    return _REGIME_LABELS.get(
        raw,
        (raw.replace("_", " ").title(), "NEUTRAL"),
    )


def _regime_snapshot(
    *,
    regime_context: Mapping[str, Any] | None = None,
    regime_detector: RegimeDetector | Any | None = None,
    features: pd.DataFrame | None = None,
) -> dict[str, Any]:
    context = dict(regime_context or {})
    raw = (
        context.get("raw_label")
        or context.get("regime_hint")
        or context.get("regime")
    )
    confidence = _finite_float(context.get("confidence"))
    persistence = _finite_float(context.get("persistence"))
    source = str(context.get("source") or "engine.regimes")

    if regime_detector is not None and features is not None and not features.empty:
        try:
            latest = features.tail(1)
            raw = str(regime_detector.transform(latest).iloc[-1])
            source = "regimes.RegimeDetector"
            try:
                probabilities = regime_detector.semantic_probabilities(latest).iloc[-1]
                confidence = float(probabilities.max())
            except Exception:
                pass
            try:
                persistence = float(regime_detector.persistence(latest).iloc[-1])
            except Exception:
                pass
        except Exception:
            pass

    friendly, direction = _normalize_regime(raw)
    raw_value = str(raw or "unknown")
    return {
        "label": friendly,
        "raw_label": raw_value,
        "direction": direction,
        "confidence": confidence,
        "persistence": persistence,
        "source": source,
        "available": raw_value.lower() != "unknown",
    }


def _risk_snapshot(
    risk_engine: Any | None,
    risk_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    context = dict(risk_context or {})
    equity = _finite_float(context.get("equity"))
    peak_equity = _finite_float(context.get("peak_equity"))
    active_notional = _finite_float(
        context.get("active_notional", context.get("gross_notional"))
    )
    leverage = _finite_float(context.get("leverage"))
    drawdown = _finite_float(context.get("drawdown"))
    day_start = _finite_float(context.get("day_start_equity"))

    if day_start is None and risk_engine is not None:
        day_start = _finite_float(
            getattr(risk_engine, "day_start_equity", None)
        )
    if equity is not None and peak_equity is not None and peak_equity > 0:
        drawdown = max(-1.0, min(0.0, equity / peak_equity - 1.0))
    if (
        leverage is None
        and active_notional is not None
        and equity is not None
        and equity > 0
    ):
        leverage = abs(active_notional) / equity

    daily_loss = (
        1.0 - equity / day_start
        if equity is not None and day_start is not None and day_start > 0
        else None
    )

    risk_budget_pct = _finite_float(
        getattr(risk_engine, "risk_per_trade", None)
        if risk_engine is not None
        else context.get("risk_budget_pct")
    )
    max_position_pct = _finite_float(
        getattr(risk_engine, "max_position_pct", None)
        if risk_engine is not None
        else context.get("max_position_pct")
    )
    max_daily_loss_pct = _finite_float(
        getattr(risk_engine, "max_daily_loss_pct", None)
        if risk_engine is not None
        else context.get("max_daily_loss_pct")
    )

    max_notional = (
        equity * max_position_pct
        if equity is not None and max_position_pct is not None
        else None
    )
    utilization = (
        abs(active_notional) / max_notional
        if active_notional is not None and max_notional and max_notional > 0
        else None
    )
    available = any(
        value is not None
        for value in (
            equity,
            peak_equity,
            active_notional,
            leverage,
            drawdown,
        )
    )
    return {
        "drawdown": drawdown,
        "leverage": leverage,
        "equity": equity,
        "peak_equity": peak_equity,
        "active_notional": active_notional,
        "daily_loss": daily_loss,
        "risk_budget_pct": risk_budget_pct,
        "max_position_pct": max_position_pct,
        "max_daily_loss_pct": max_daily_loss_pct,
        "position_limit_utilization": utilization,
        "current_exposure_available": bool(available),
        "source": (
            "risk.RiskEngine"
            if risk_engine is not None
            else "risk_context"
        ),
    }


def fuse_live_dashboard_state(
    *,
    symbol: str,
    price: float | None = None,
    market_snapshot: Mapping[str, Any] | None = None,
    regime_context: Mapping[str, Any] | None = None,
    regime_detector: RegimeDetector | Any | None = None,
    features: pd.DataFrame | None = None,
    risk_engine: Any | None = None,
    risk_context: Mapping[str, Any] | None = None,
    execution_telemetry: Any | Mapping[str, Any] | None = None,
    cognition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one read-only intelligence payload for the live dashboard."""
    market = dict(market_snapshot or {})
    market_price = _finite_float(market.get("price"))
    if market_price is None:
        market_price = _finite_float(price)
    market["price"] = market_price
    market["bid"] = _finite_float(market.get("bid"))
    market["ask"] = _finite_float(market.get("ask"))
    market["latency_ms"] = _finite_float(market.get("latency_ms"))
    market["event_time_ms"] = int(market.get("event_time_ms", 0) or 0)
    market["sequence"] = int(market.get("sequence", 0) or 0)

    regime = _regime_snapshot(
        regime_context=regime_context,
        regime_detector=regime_detector,
        features=features,
    )
    risk = _risk_snapshot(
        risk_engine,
        risk_context,
    )
    configured_slippage = None
    if isinstance(risk_context, Mapping):
        configured_slippage = risk_context.get("configured_slippage_bps")
    efficiency = execution_efficiency_snapshot(
        execution_telemetry,
        market_data_latency_ms=market.get("latency_ms"),
        configured_slippage_bps=configured_slippage,
    )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": str(symbol),
        "price": market_price,
        "market": market,
        "regime": regime,
        "risk": risk,
        "efficiency": efficiency,
        "cognition": dict(cognition or {}),
    }
