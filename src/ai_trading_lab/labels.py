from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    df: pd.DataFrame,
    horizon: int = 8,
    pt_atr: float = 1.6,
    sl_atr: float = 1.0,
    *,
    side: int = 1,
) -> pd.DataFrame:
    """Simulate gross executable returns for a long or short position.

    The signal is known after candle i closes; entry occurs at open[i + 1].
    A position with horizon h is held across candles i + 1 through i + h and,
    if no barrier exits first, is liquidated at open[i + h + 1]. That candle's
    high/low are not scanned, matching the backtest's time-stop ordering.

    For side=1 the profit/stop barriers are above/below entry; for side=-1 they
    are below/above entry. Returns are position PnL, so a profitable short has
    a positive return. Opening gaps execute first. If both barriers are touched
    intrabar without a resolving gap, OHLC cannot reveal the path and the
    observation remains NaN rather than guessing.
    """
    h = int(horizon)
    direction = int(side)
    if h < 1:
        raise ValueError("horizon must be >= 1")
    if direction not in {-1, 1}:
        raise ValueError("side must be +1 (long) or -1 (short)")
    if not np.isfinite(pt_atr) or float(pt_atr) <= 0:
        raise ValueError("pt_atr must be finite and > 0")
    if not np.isfinite(sl_atr) or float(sl_atr) <= 0:
        raise ValueError("sl_atr must be finite and > 0")

    opn, high, low = (
        pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        for c in ("open", "high", "low")
    )
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14).mean().to_numpy(float)

    out = np.full(len(df), np.nan)
    realized_returns = np.full(len(df), np.nan)

    for i in range(len(df) - h - 1):
        entry_i = i + 1
        end_open_i = entry_i + h
        if (
            not np.isfinite(atr[i])
            or atr[i] <= 0
            or not np.isfinite(opn[entry_i])
            or opn[entry_i] <= 0
            or not np.isfinite(opn[end_open_i])
            or opn[end_open_i] <= 0
        ):
            continue

        entry = opn[entry_i]
        if direction > 0:
            profit_barrier = entry + float(pt_atr) * atr[i]
            stop_barrier = entry - float(sl_atr) * atr[i]
        else:
            profit_barrier = entry - float(pt_atr) * atr[i]
            stop_barrier = entry + float(sl_atr) * atr[i]

        def position_return(exit_price: float) -> float:
            if direction > 0:
                return exit_price / entry - 1.0
            return (entry - exit_price) / entry

        label = 0
        realized = position_return(opn[end_open_i])

        for j in range(entry_i, end_open_i):
            bar_open = opn[j]
            if not np.isfinite(bar_open) or bar_open <= 0:
                label = np.nan
                realized = np.nan
                break

            if direction > 0:
                gap_hits_profit = bar_open >= profit_barrier
                gap_hits_stop = bar_open <= stop_barrier
            else:
                gap_hits_profit = bar_open <= profit_barrier
                gap_hits_stop = bar_open >= stop_barrier

            if gap_hits_profit:
                label = 1
                realized = position_return(bar_open)
                break
            if gap_hits_stop:
                label = -1
                realized = position_return(bar_open)
                break

            if not np.isfinite(high[j]) or not np.isfinite(low[j]):
                label = np.nan
                realized = np.nan
                break

            hit_profit = (
                high[j] >= profit_barrier
                if direction > 0
                else low[j] <= profit_barrier
            )
            hit_stop = (
                low[j] <= stop_barrier
                if direction > 0
                else high[j] >= stop_barrier
            )
            if hit_profit and hit_stop:
                label = np.nan
                realized = np.nan
                break
            if hit_profit:
                label = 1
                realized = position_return(profit_barrier)
                break
            if hit_stop:
                label = -1
                realized = position_return(stop_barrier)
                break

        out[i] = label
        realized_returns[i] = realized

    return pd.DataFrame(
        {"tb_label": out, "tb_return": realized_returns},
        index=df.index,
    )
