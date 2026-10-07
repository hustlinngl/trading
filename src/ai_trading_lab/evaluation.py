from __future__ import annotations

import pandas as pd

from .backtest import run_backtest
from .risk import RiskEngine
from .execution_semantics import validate_execution_alignment
from .data import timeframe_offset


def make_risk(settings, *, stop_atr_mult=None, take_profit_rr=None, cost_multiplier: float = 1.0, max_holding_bars=None):
    semantics = validate_execution_alignment(settings)
    if stop_atr_mult is not None and abs(float(stop_atr_mult) - semantics.stop_atr_mult) > 1e-9:
        raise ValueError("execution_stop_override_mismatch")
    if take_profit_rr is not None and abs(float(take_profit_rr) - semantics.take_profit_rr) > 1e-9:
        raise ValueError("execution_take_profit_override_mismatch")
    if max_holding_bars is not None and int(max_holding_bars) != semantics.horizon_bars:
        raise ValueError("execution_holding_override_mismatch")
    cm = float(cost_multiplier)
    holding = semantics.horizon_bars
    return RiskEngine(
        settings.initial_cash,
        settings.risk_per_trade,
        settings.max_position_pct,
        settings.max_daily_loss_pct,
        float(settings.stop_atr_mult if stop_atr_mult is None else stop_atr_mult),
        float(settings.take_profit_rr if take_profit_rr is None else take_profit_rr),
        float(settings.fee_bps) * cm,
        float(settings.slippage_bps) * cm,
        float(getattr(settings, "max_participation_pct", 0.10)),
        float(getattr(settings, "impact_bps_per_sqrt", 1.5)) * cm,
        float(getattr(settings, "short_borrow_bps_per_bar", 0.0)) * cm,
        holding,
    )


def run_configured_backtest(market: pd.DataFrame, actions: pd.Series, settings, *, stop_atr_mult=None, take_profit_rr=None, max_holding_bars=None, cost_multiplier: float = 1.0, risk=None, market_type: str | None = None):
    if 'atr_14' not in market.columns:
        raise ValueError("market must contain 'atr_14' for configured backtests")
    semantics = validate_execution_alignment(settings)
    cm = float(cost_multiplier)
    effective_market_type = str(market_type or getattr(settings, "backtest_market_type", "spot")).lower()
    derivative = effective_market_type in {"swap", "future", "perpetual"}
    funding_column = str(getattr(settings, "funding_rate_column", "funding_rate"))
    if derivative and bool(getattr(settings, "require_funding_data_for_derivatives", True)):
        if funding_column not in market.columns or not pd.to_numeric(market[funding_column], errors="coerce").notna().any():
            raise ValueError("funding_data_required_for_derivative_backtest")
    try:
        bar_delta = timeframe_offset(getattr(settings, "timeframe", "15m"))
        funding_interval_bars = max(1, int(round(pd.Timedelta(hours=float(getattr(settings, "funding_interval_hours", 8.0))) / bar_delta)))
    except Exception:
        funding_interval_bars = max(1, int(getattr(settings, "funding_interval_bars", 32)))
    holding = semantics.horizon_bars
    bt_risk = risk or make_risk(
        settings,
        stop_atr_mult=stop_atr_mult,
        take_profit_rr=take_profit_rr,
        cost_multiplier=cm,
        max_holding_bars=holding,
    )
    return run_backtest(
        market,
        actions,
        bt_risk,
        settings.initial_cash,
        fee_bps=float(settings.fee_bps) * cm,
        slippage_bps=float(settings.slippage_bps) * cm,
        max_holding_bars=holding,
        intrabar_barriers=getattr(settings, 'intrabar_barriers', True),
        impact_bps_per_sqrt=float(getattr(settings, 'impact_bps_per_sqrt', 1.5)) * cm,
        force_daily_loss_exit=getattr(settings, 'force_daily_loss_exit', True),
        short_borrow_bps_per_bar=float(getattr(settings, 'short_borrow_bps_per_bar', 0.0)) * cm,
        require_short_borrow_cost=bool(getattr(settings,'require_short_borrow_cost',True)),
        funding_rate_column=funding_column,
        funding_interval_bars=funding_interval_bars,
        funding_bps_per_bar=float(getattr(settings, "funding_bps_per_bar", 0.0)) * cm,
        apply_funding=bool(derivative),
    )
