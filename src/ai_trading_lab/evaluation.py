from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .risk import RiskEngine
from .execution_semantics import validate_execution_alignment
from .data import timeframe_offset


def directional_validation_diagnostics(probabilities, realized_returns) -> dict:
    """Measure OOS directional predictions against realized executable returns.

    Returns are the same next-open/triple-barrier targets used to train p_up.
    Non-finite observations are excluded; this report is diagnostic only and
    never relaxes deployment-readiness gates.
    """
    p_up = pd.to_numeric(pd.Series(probabilities), errors="coerce").to_numpy(dtype=float)
    returns = pd.to_numeric(pd.Series(realized_returns), errors="coerce").to_numpy(dtype=float)
    if p_up.size != returns.size:
        raise ValueError("directional_validation_length_mismatch")

    valid = np.isfinite(p_up) & np.isfinite(returns) & (p_up >= 0.0) & (p_up <= 1.0)
    p_up = p_up[valid]
    returns = returns[valid]
    if not len(p_up):
        return {"observations": 0, "reason": "no_finite_oos_observations"}

    actual_up = returns > 0.0
    predicted_up = p_up >= 0.5
    tp = int(np.sum(predicted_up & actual_up))
    fp = int(np.sum(predicted_up & ~actual_up))
    tn = int(np.sum(~predicted_up & ~actual_up))
    fn = int(np.sum(~predicted_up & actual_up))
    positive_support = tp + fn
    negative_support = tn + fp
    accuracy = (tp + tn) / len(p_up)
    balanced_accuracy = (
        0.5 * (tp / positive_support + tn / negative_support)
        if positive_support and negative_support
        else None
    )
    long_support = tp + fp
    short_support = tn + fn
    actual_positive_rate = float(np.mean(actual_up))
    return {
        "observations": int(len(p_up)),
        "directional_accuracy": float(accuracy),
        "balanced_directional_accuracy": (
            float(balanced_accuracy) if balanced_accuracy is not None else None
        ),
        "majority_class_baseline_accuracy": float(max(actual_positive_rate, 1.0 - actual_positive_rate)),
        "probability_brier_score": float(np.mean((p_up - actual_up.astype(float)) ** 2)),
        "long_precision": float(tp / long_support) if long_support else None,
        "short_precision": float(tn / short_support) if short_support else None,
        "actual_positive_return_rate": actual_positive_rate,
        "predicted_up_rate": float(np.mean(predicted_up)),
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
    }


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
