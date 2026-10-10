from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    df: pd.DataFrame,
    horizon: int = 8,
    pt_atr: float = 1.6,
    sl_atr: float = 1.0,
) -> pd.DataFrame:
    """Simulate gross executable returns for next-open entry and triple barriers.

    The signal is known after candle i closes; entry occurs at open[i + 1].
    A position with horizon h is held across candles i + 1 through i + h and,
    if no barrier exits first, is liquidated at open[i + h + 1]. This matches the
    backtest's time-stop ordering: the time-stop at the exit candle's open takes
    precedence over that candle's high/low.

    Opening gaps are evaluated before intrabar extrema because an opening fill
    occurs before the rest of that candle's path. If both barriers are touched
    intrabar without an opening gap, OHLC data cannot reveal which came first, so
    that observation remains ambiguous and is excluded from supervised targets.
    """
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be >= 1")
    if not np.isfinite(pt_atr) or float(pt_atr) <= 0:
        raise ValueError("pt_atr must be finite and > 0")
    if not np.isfinite(sl_atr) or float(sl_atr) <= 0:
        raise ValueError("sl_atr must be finite and > 0")

    opn, high, low, close = (
        pd.to_numeric(df[c], errors="coerce").to_numpy(float)
        for c in ("open", "high", "low", "close")
    )
    tr = pd.concat(
        [
            (df["high"] - df["low"]),
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(14).mean().to_numpy(float)

    out = np.full(len(df), np.nan)
    realized_returns = np.full(len(df), np.nan)

    # end_open_i is where a time-stop exits. That candle's intrabar
    # highs/lows are not scanned, matching run_backtest.
    for i in range(len(df) - h - 1):
        entry_i = i + 1
        end_open_i = entry_i + h
        if (
            not np.isfinite(atr[i])
            or atr[i] <= 0
            or not np.isfinite(opn[entry_i])
            or opn[entry_i] <= 0
            or not np.isfinite(opn[end_open_i])
        ):
            continue

        entry = opn[entry_i]
        upper = entry + float(pt_atr) * atr[i]
        lower = entry - float(sl_atr) * atr[i]
        label = 0
        realized = opn[end_open_i] / entry - 1.0

        # Examine exactly h holding candles, excluding the time-stop candle.
        for j in range(entry_i, end_open_i):
            bar_open = opn[j]
            if not np.isfinite(bar_open):
                label = np.nan
                realized = np.nan
                break

            # Opening gaps execute before the high/low path. These checks also
            # resolve bars whose later range happens to cross the opposite barrier.
            if bar_open >= upper:
                label = 1
                realized = bar_open / entry - 1.0
                break
            if bar_open <= lower:
                label = -1
                realized = bar_open / entry - 1.0
                break

            if not np.isfinite(high[j]) or not np.isfinite(low[j]):
                label = np.nan
                realized = np.nan
                break

            hit_up = high[j] >= upper
            hit_down = low[j] <= lower
            if hit_up and hit_down:
                # OHLC cannot determine the first intrabar barrier; do not guess.
                label = np.nan
                realized = np.nan
                break
            if hit_up:
                label = 1
                realized = upper / entry - 1.0
                break
            if hit_down:
                label = -1
                realized = lower / entry - 1.0
                break

        out[i] = label
        realized_returns[i] = realized

    return pd.DataFrame(
        {"tb_label": out, "tb_return": realized_returns},
        index=df.index,
    )
