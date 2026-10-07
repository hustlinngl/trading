from __future__ import annotations

import pandas as pd

from .backtest import run_backtest
from .risk import RiskEngine


def make_risk(settings, *, stop_atr_mult=None, take_profit_rr=None, cost_multiplier: float = 1.0, max_holding_bars=None):
    cm = float(cost_multiplier)
    holding = int(settings.max_holding_bars if max_holding_bars is None else max_holding_bars)
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


def run_configured_backtest(market: pd.DataFrame, actions: pd.Series, settings, *, stop_atr_mult=None, take_profit_rr=None, max_holding_bars=None, cost_multiplier: float = 1.0, risk=None):
    if 'atr_14' not in market.columns:
        raise ValueError("market must contain 'atr_14' for configured backtests")
    cm = float(cost_multiplier)
    holding = int(settings.max_holding_bars if max_holding_bars is None else max_holding_bars)
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
    )
