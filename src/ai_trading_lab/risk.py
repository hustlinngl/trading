from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RiskDecision:
    allowed: bool
    qty: float
    stop: float | None
    take_profit: float | None
    reason: str


class RiskEngine:
    def __init__(self, initial_cash, risk_per_trade, max_position_pct, max_daily_loss_pct, stop_atr_mult, rr,
                 fee_bps: float = 0.0, slippage_bps: float = 0.0, max_participation_pct: float = 0.10,
                 impact_bps_per_sqrt: float = 0.0, short_borrow_bps_per_bar: float = 0.0,
                 max_holding_bars: int = 96):
        self.initial_cash = initial_cash
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.stop_atr_mult = stop_atr_mult
        self.rr = rr
        self.fee_bps = float(max(0.0, fee_bps))
        self.slippage_bps = float(max(0.0, slippage_bps))
        self.max_participation_pct = float(min(1.0, max(0.0, max_participation_pct)))
        self.impact_bps_per_sqrt = float(max(0.0, impact_bps_per_sqrt))
        self.short_borrow_bps_per_bar = float(max(0.0, short_borrow_bps_per_bar))
        self.max_holding_bars = max(1, int(max_holding_bars))
        self.day_start_equity = initial_cash

    def reset_day(self, equity: float):
        self.day_start_equity = equity

    def size(self, equity: float, price: float, atr: float, direction: int) -> RiskDecision:
        if equity <= self.day_start_equity * (1 - self.max_daily_loss_pct):
            return RiskDecision(False, 0, None, None, 'daily_loss_limit')
        if atr <= 0 or price <= 0:
            return RiskDecision(False, 0, None, None, 'invalid_volatility')
        stop_distance = self.stop_atr_mult * atr
        risk_cash = equity * self.risk_per_trade
        conservative_impact_bps = 2.0 * self.impact_bps_per_sqrt * (self.max_participation_pct ** 0.5)
        round_trip_cost_per_unit = price * (2.0 * (self.fee_bps + self.slippage_bps) + conservative_impact_bps) / 10_000.0
        conservative_borrow_per_unit = 0.0
        if direction < 0 and self.short_borrow_bps_per_bar > 0:
            conservative_borrow_per_unit = price * self.short_borrow_bps_per_bar * self.max_holding_bars / 10_000.0
        effective_risk_per_unit = stop_distance + round_trip_cost_per_unit + conservative_borrow_per_unit
        qty_by_risk = risk_cash / max(effective_risk_per_unit, 1e-12)
        qty_by_cap = equity * self.max_position_pct / price
        qty = min(qty_by_risk, qty_by_cap)
        if qty <= 0:
            return RiskDecision(False, 0, None, None, 'zero_size')
        stop = price - direction * stop_distance
        take = price + direction * stop_distance * self.rr
        return RiskDecision(True, qty, stop, take, 'ok')
