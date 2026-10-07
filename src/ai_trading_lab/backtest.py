from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    stats: dict


def _fill_prices(price: float, side: int, notional: float, bar_volume: float, fee_bps: float, slippage_bps: float, impact_bps_per_sqrt: float = 1.5):
    participation = min(1.0, abs(notional) / max(bar_volume, 1.0))
    impact = impact_bps_per_sqrt * math.sqrt(participation)
    total_slip = (slippage_bps + impact) / 10_000
    return price * (1 + side * total_slip), fee_bps / 10_000


def _annualization_factor(index: pd.Index) -> float:
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return 365.0
    dt = index.to_series().diff().dropna().median().total_seconds()
    if not np.isfinite(dt) or dt <= 0:
        return 365.0
    return max(1.0, 365.25 * 24 * 3600 / dt)


def _stats(equity: pd.Series, trades: list[dict], bars_per_year: float | None = None, benchmark_return: float = 0.0) -> dict:
    if bars_per_year is None:
        bars_per_year = _annualization_factor(equity.index)
    rets = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    downside = rets.where(rets < 0, 0.0)
    sharpe = float(np.sqrt(bars_per_year) * rets.mean() / (rets.std() + 1e-12))
    sortino = float(np.sqrt(bars_per_year) * rets.mean() / (downside.std() + 1e-12))
    dd = equity / equity.cummax() - 1
    pnls = np.array([t['pnl_net'] for t in trades], dtype=float)
    abs_pnls = np.abs(pnls)
    top_trade_share = float(np.max(abs_pnls) / max(abs_pnls.sum(), 1e-12)) if len(abs_pnls) else 0.0
    gross_win = float(pnls[pnls > 0].sum()) if len(pnls) else 0.0
    gross_loss = float(-pnls[pnls < 0].sum()) if len(pnls) else 0.0
    fees_total = float(sum(t.get('entry_fee', 0.0) + t.get('exit_fee', 0.0) for t in trades)) if trades else 0.0
    slippage_total = float(sum(t.get('slippage_cost', 0.0) for t in trades)) if trades else 0.0
    impact_total = float(sum(t.get('impact_cost', 0.0) for t in trades)) if trades else 0.0
    borrow_total = float(sum(t.get('borrow_cost', 0.0) for t in trades)) if trades else 0.0
    participation = np.array([t.get('entry_participation', 0.0) for t in trades], dtype=float) if trades else np.array([])
    return {
        'total_return': float(equity.iloc[-1] / equity.iloc[0] - 1),
        'benchmark_return': float(benchmark_return),
        'excess_return': float(equity.iloc[-1] / equity.iloc[0] - 1 - benchmark_return),
        'max_drawdown': float(dd.min()),
        'sharpe_like': sharpe,
        'sortino_like': sortino,
        'trades': int(len(trades)),
        'win_rate': float(np.mean(pnls > 0)) if len(pnls) else 0.0,
        'profit_factor': float(gross_win / max(gross_loss, 1e-12)) if gross_loss > 0 else (float('inf') if gross_win > 0 else 0.0),
        'top_trade_share': top_trade_share,
        'trade_pnl_concentration': top_trade_share,
        'trades_per_bar': float(len(trades) / max(len(equity), 1)),
        'avg_trade': float(pnls.mean()) if len(pnls) else 0.0,
        'best_trade': float(pnls.max()) if len(pnls) else 0.0,
        'worst_trade': float(pnls.min()) if len(pnls) else 0.0,
        'gross_pnl': float(gross_win - gross_loss),
        'fees_total': fees_total,
        'slippage_impact_total': float(slippage_total + impact_total),
        'borrow_cost_total': borrow_total,
        'funding_cost_total': float(sum(t.get('funding_cost', 0.0) for t in trades)) if trades else 0.0,
        'avg_entry_participation': float(participation.mean()) if len(participation) else 0.0,
        'p95_entry_participation': float(np.quantile(participation, 0.95)) if len(participation) else 0.0,
        'exposure': float(np.mean([abs(t.get('notional', 0.0)) / max(float(t.get('entry_equity', equity.iloc[0])), 1e-9) for t in trades])) if trades else 0.0,
        'turnover_notional': float(sum(abs(t.get('notional', 0.0)) * 2.0 for t in trades)),
    }


def run_backtest(df: pd.DataFrame, signal: pd.Series, risk, initial_cash: float,
                 fee_bps=7, slippage_bps=5, max_holding_bars: int | None = None,
                 intrabar_barriers: bool = True, impact_bps_per_sqrt: float = 1.5,
                 force_daily_loss_exit: bool = True, short_borrow_bps_per_bar: float = 0.0,
                 funding_rate_column: str = 'funding_rate', funding_interval_bars: int = 32,
                 funding_bps_per_bar: float = 0.0, apply_funding: bool = False,
                 require_short_borrow_cost: bool = False) -> BacktestResult:
    """Event-driven single-asset simulator with strict decision/execution semantics."""
    if not signal.index.equals(df.index):
        signal = signal.reindex(df.index).fillna('FLAT')
    df = df.sort_index().copy()
    signal = signal.reindex(df.index).fillna('FLAT').astype(str)

    cash = float(initial_cash)
    side = 0
    qty = 0.0
    entry = stop = take = None
    entry_idx = None
    entry_ts = None
    entry_fee = 0.0
    entry_equity = float(initial_cash)
    entry_slippage_cost = 0.0
    entry_impact_cost = 0.0
    entry_participation = 0.0
    borrow_cost = 0.0
    funding_cost = 0.0
    equity_curve: list[float] = []
    trades: list[dict] = []

    def mark(close: float) -> float:
        return cash if side == 0 else cash + side * (close - float(entry)) * qty

    def close_position(ts, fill_raw: float, volume_notional: float, reason: str):
        nonlocal cash, side, qty, entry, stop, take, entry_idx, entry_ts, entry_fee, entry_slippage_cost, entry_impact_cost, entry_participation, borrow_cost, funding_cost, entry_equity
        exit_notional_raw = qty * fill_raw
        exit_fill, fee = _fill_prices(fill_raw, -side, exit_notional_raw, volume_notional, fee_bps, slippage_bps, impact_bps_per_sqrt)
        participation_exit = min(1.0, abs(exit_notional_raw) / max(volume_notional, 1.0)) if volume_notional > 0 else 1.0
        exit_impact = abs(qty * fill_raw) * (impact_bps_per_sqrt * math.sqrt(participation_exit) / 10_000.0)
        exit_slippage = abs(qty * fill_raw) * (slippage_bps / 10_000.0)
        gross = side * (exit_fill - float(entry)) * qty
        fee_cash = abs(qty * exit_fill) * fee
        cash += gross - fee_cash
        pnl_net = gross - fee_cash - entry_fee - borrow_cost - funding_cost
        trades.append({
            'timestamp': ts, 'entry_timestamp': entry_ts,
            'side': 'LONG' if side > 0 else 'SHORT',
            'entry': float(entry), 'exit': float(exit_fill), 'qty': float(qty),
            'notional': float(qty * entry), 'entry_equity': float(entry_equity), 'pnl_gross': float(gross),
            'pnl_net': float(pnl_net), 'exit_reason': reason,
            'entry_fee': float(entry_fee), 'exit_fee': float(fee_cash),
            'slippage_cost': float(entry_slippage_cost + exit_slippage),
            'borrow_cost': float(borrow_cost),
            'funding_cost': float(funding_cost),
            'impact_cost': float(entry_impact_cost + exit_impact),
            'entry_participation': float(entry_participation),
            'holding_bars': int(i - int(entry_idx)) if entry_idx is not None else None,
        })
        side = 0; qty = 0.0; entry = stop = take = None
        entry_idx = entry_ts = None; entry_fee = 0.0; entry_equity = float(cash)
        entry_slippage_cost = entry_impact_cost = entry_participation = borrow_cost = funding_cost = 0.0

    for i, (ts, row) in enumerate(df.iterrows()):
        close = float(row['close']); open_ = float(row['open'])
        high = float(row.get('high', close)); low = float(row.get('low', close))
        prev_row = df.iloc[i - 1] if i > 0 else None
        atr = float(prev_row.get('atr_14', np.nan)) if prev_row is not None else np.nan
        if not np.isfinite(atr) or atr <= 0:
            atr = max((float(prev_row['close']) if prev_row is not None else close) * 0.01, 1e-8)
        if prev_row is not None:
            if 'quote_volume' in prev_row.index and pd.notna(prev_row.get('quote_volume')):
                prev_quote_volume = float(prev_row.get('quote_volume', 0.0))
            else:
                prev_quote_volume = float(prev_row.get('volume', 0.0)) * max(float(prev_row.get('close', close)), 1e-12)
        else:
            prev_quote_volume = float(row.get('quote_volume', 0.0)) if 'quote_volume' in row.index and pd.notna(row.get('quote_volume')) else float(row.get('volume', 0.0)) * max(open_, 1e-12)

        if i == 0:
            risk.reset_day(cash)
        elif ts.date() != df.index[i - 1].date():
            risk.reset_day(equity_curve[-1])

        pending = str(signal.iloc[i - 1]) if i > 0 else 'FLAT'
        if side != 0 and pending in {'FLAT', 'LONG', 'SHORT'}:
            reverse = (pending == 'SHORT' and side > 0) or (pending == 'LONG' and side < 0)
            flat_exit = pending == 'FLAT'
            if reverse or flat_exit:
                close_position(ts, open_, prev_quote_volume, 'reverse' if reverse else 'signal_exit')

        if side == 0 and i > 0 and pending in {'LONG', 'SHORT'}:
            direction = 1 if pending == 'LONG' else -1
            if direction < 0 and require_short_borrow_cost and float(short_borrow_bps_per_bar) <= 0.0:
                equity_curve.append(mark(close))
                continue
            approved = risk.size(cash, open_, atr, direction)
            if approved.allowed:
                max_participation = float(getattr(risk, 'max_participation_pct', 0.0) or 0.0)
                if max_participation <= 0.0:
                    max_participation = 1.0
                liquidity_qty = max_participation * prev_quote_volume / max(open_, 1e-12)
                final_qty = min(float(approved.qty), max(0.0, liquidity_qty)) if prev_quote_volume > 0 else 0.0
                if final_qty > 0:
                    notional = final_qty * open_
                    fill, fee = _fill_prices(open_, direction, notional, prev_quote_volume, fee_bps, slippage_bps, impact_bps_per_sqrt)
                    entry_fee = abs(final_qty * fill) * fee
                    participation = min(1.0, abs(notional) / max(prev_quote_volume, 1.0))
                    impact_rate = impact_bps_per_sqrt * math.sqrt(participation) / 10_000.0
                    entry_impact_cost = abs(final_qty * open_) * impact_rate
                    entry_slippage_cost = abs(final_qty * open_) * (slippage_bps / 10_000.0)
                    entry_participation = participation
                    cash -= entry_fee
                    side = direction; qty = final_qty; entry = fill
                    stop_distance = float(getattr(risk, 'stop_atr_mult', 1.8)) * atr
                    stop = fill - direction * stop_distance
                    take = fill + direction * stop_distance * float(getattr(risk, 'rr', 2.0))
                    entry_idx = i; entry_ts = ts
                    entry_equity = float(mark(open_))

        if side != 0:
            holding = i - int(entry_idx)
            if force_daily_loss_exit and mark(open_) <= risk.day_start_equity * (1 - risk.max_daily_loss_pct):
                close_position(ts, open_, prev_quote_volume, 'daily_loss_limit')
                equity_curve.append(mark(close))
                continue
            if side < 0 and holding > 0 and short_borrow_bps_per_bar > 0:
                borrow = abs(qty * open_) * float(short_borrow_bps_per_bar) / 10_000
                cash -= borrow
                borrow_cost += borrow
            if apply_funding and holding > 0:
                interval = max(1, int(funding_interval_bars))
                if holding % interval == 0:
                    rate = row.get(funding_rate_column, np.nan)
                    if pd.notna(rate):
                        funding_rate = float(rate)
                        funding_cashflow = -float(side) * abs(qty * open_) * funding_rate
                        cash += funding_cashflow
                        funding_cost += -funding_cashflow
                    elif funding_bps_per_bar > 0:
                        fallback = abs(qty * open_) * float(funding_bps_per_bar) / 10_000.0
                        cash -= fallback
                        funding_cost += fallback
            time_stop = max_holding_bars is not None and holding >= int(max_holding_bars)
            if time_stop:
                close_position(ts, open_, prev_quote_volume, 'time_stop')
            else:
                hit_stop = hit_take = False
                barrier_fill = None
                if intrabar_barriers:
                    hit_stop = low <= stop if side > 0 else high >= stop
                    hit_take = high >= take if side > 0 else low <= take
                    if hit_stop:
                        barrier_fill = open_ if (open_ <= stop if side > 0 else open_ >= stop) else stop
                    elif hit_take:
                        barrier_fill = open_ if (open_ >= take if side > 0 else open_ <= take) else take
                else:
                    hit_stop = close <= stop if side > 0 else close >= stop
                    hit_take = close >= take if side > 0 else close <= take
                    barrier_fill = close if (hit_stop or hit_take) else None

                if hit_stop or hit_take:
                    reason = 'stop' if hit_stop else 'take_profit'
                    close_position(ts, float(barrier_fill if barrier_fill is not None else close), prev_quote_volume, reason)

        equity_curve.append(mark(close))

    if side != 0:
        final_ts = df.index[-1]
        final = float(df['close'].iloc[-1])
        prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
        prev_quote_volume = float(prev.get('quote_volume', 0.0)) if 'quote_volume' in prev.index and pd.notna(prev.get('quote_volume')) else float(prev.get('volume', 0.0)) * max(float(prev.get('close', final)), 1e-12)
        close_position(final_ts, final, prev_quote_volume, 'end_of_test')
        equity_curve[-1] = cash

    eq = pd.Series(equity_curve, index=df.index, dtype=float)
    trades_df = pd.DataFrame(trades)
    benchmark_return = float(df['close'].iloc[-1] / max(float(df['close'].iloc[0]), 1e-12) - 1.0)
    return BacktestResult(eq, trades_df, _stats(eq, trades, benchmark_return=benchmark_return))
