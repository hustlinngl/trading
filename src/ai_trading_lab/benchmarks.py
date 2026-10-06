from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .data import load_cached
from .engine import AdaptiveEngine
from .features import make_oos_features
from .evaluation import run_configured_backtest
from .risk import RiskEngine
from .validation import robust_score


def _synthetic_data(settings: Settings, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(int(settings.seed))
    freq = {
        "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min",
        "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D",
    }.get(settings.timeframe, settings.timeframe)
    index = pd.date_range("2024-01-01", periods=int(n), freq=freq, tz="UTC")
    regimes = np.repeat([0, 1, 2, 3], int(n) // 4 + 1)[: int(n)]
    drift = np.choose(regimes, [0.00002, 0.00010, -0.00008, 0.0])
    vol = np.choose(regimes, [0.0025, 0.0035, 0.0045, 0.0060])
    returns = drift + rng.normal(0.0, vol, int(n))
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = close * (1.0 + rng.normal(0.0, 0.0008, int(n)))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.0025, int(n)))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.0025, int(n)))
    volume = rng.lognormal(10.0, 0.40, int(n))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]),
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _baseline_actions(df: pd.DataFrame, kind: str) -> pd.Series:
    close = df["close"].astype(float)
    if kind == "flat":
        return pd.Series("FLAT", index=df.index)
    if kind == "buy_hold":
        return pd.Series("LONG", index=df.index)
    ema = close.ewm(span=48, adjust=False, min_periods=48).mean()
    if kind == "trend":
        return pd.Series(np.where(close > ema, "LONG", np.where(close < ema, "SHORT", "FLAT")), index=df.index)
    if kind == "mean_reversion":
        mean = close.rolling(48, min_periods=48).mean()
        std = close.rolling(48, min_periods=48).std().replace(0.0, np.nan)
        z = ((close - mean) / std).fillna(0.0)
        return pd.Series(np.where(z < -1.0, "LONG", np.where(z > 1.0, "SHORT", "FLAT")), index=df.index)
    raise ValueError(f"Unknown benchmark strategy: {kind}")


def _evaluate_actions(df: pd.DataFrame, actions: pd.Series, settings: Settings) -> dict:
    bt = df.copy()
    bt["atr_14"] = _atr(bt).bfill()
    result = run_configured_backtest(bt, actions, settings)
    return {**result.stats, "robust_score": robust_score(result.stats)}


def _load_or_make_data(settings: Settings, n: int) -> pd.DataFrame:
    cache_path = Path(getattr(settings, "cache_dir", "data")) / (
        f"{settings.symbol.replace('/', '_')}_{settings.timeframe}.parquet"
    )
    if cache_path.exists():
        try:
            cached = load_cached(cache_path)
            if len(cached) >= max(300, int(n)):
                return cached.iloc[-int(n):].copy()
        except Exception:
            pass
    return _synthetic_data(settings, max(int(n), 1200))


def evaluate_suite(settings: Settings, n: int = 3000) -> pd.DataFrame:
    """Run diagnostic controls on either cached real data or deterministic synthetic data.

    This suite is deliberately comparative: it is a sanity check against trivial controls,
    not a hyperparameter objective and not evidence of a deployable edge.
    """
    data = _load_or_make_data(settings, int(n))
    work = replace(settings, min_train_rows=min(int(settings.min_train_rows), max(200, len(data) // 3)))
    split = max(int(work.min_train_rows), int(len(data) * 0.70))
    split = min(split, len(data) - max(100, len(data) // 5))
    train, test = data.iloc[:split].copy(), data.iloc[split:].copy()

    rows: list[dict] = []
    for name in ("flat", "buy_hold", "trend", "mean_reversion"):
        stats = _evaluate_actions(test, _baseline_actions(test, name), work)
        rows.append({"strategy": name, **stats})

    try:
        engine = AdaptiveEngine(work)
        engine.fit(train)
        features = make_oos_features(
            train,
            test,
            work.horizon_bars,
            external_feature_lag_bars=getattr(work, "external_feature_lag_bars", 1),
        )
        actions = engine.predict_frame(features)["action"]
        stats = _evaluate_actions(test, actions, work)
        rows.append({"strategy": "adaptive_engine", **stats})
    except Exception as exc:
        rows.append(
            {
                "strategy": "adaptive_engine",
                "total_return": np.nan,
                "benchmark_return": float(test["close"].iloc[-1] / test["close"].iloc[0] - 1.0),
                "excess_return": np.nan,
                "max_drawdown": np.nan,
                "sharpe_like": np.nan,
                "sortino_like": np.nan,
                "trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "robust_score": -np.inf,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )

    frame = pd.DataFrame(rows)
    preferred = [
        "strategy", "total_return", "benchmark_return", "excess_return",
        "max_drawdown", "sharpe_like", "sortino_like", "trades",
        "win_rate", "profit_factor", "robust_score", "error",
    ]
    return frame[[c for c in preferred if c in frame.columns]]


def save_suite_report(result: pd.DataFrame, path: str | Path = "logs/benchmark_suite.json") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "diagnostic_benchmark_suite",
        "warning": "Benchmark controls are diagnostics, not proof of a production trading edge.",
        "rows": result.replace([np.inf, -np.inf], np.nan).to_dict(orient="records"),
    }
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out
