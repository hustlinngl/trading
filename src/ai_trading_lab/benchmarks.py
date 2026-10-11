from __future__ import annotations

from collections import Counter
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
from .validation import robust_score
from .policy import live_signal_gate


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
    # Benchmark controls must remain causal too; never backfill from future bars.
    return tr.rolling(period, min_periods=1).mean()


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
    bt["atr_14"] = _atr(bt)
    result = run_configured_backtest(bt, actions, settings)
    return {**result.stats, "robust_score": robust_score(result.stats)}


def _load_or_make_data(settings: Settings, n: int) -> tuple[pd.DataFrame, str]:
    """Load a cached market dataset or label the deterministic fallback explicitly."""
    cache_path = Path(getattr(settings, "cache_dir", "data")) / (
        f"{settings.symbol.replace('/', '_')}_{settings.timeframe}.parquet"
    )
    if cache_path.exists():
        try:
            cached = load_cached(cache_path)
            if len(cached) >= max(300, int(n)):
                return cached.iloc[-int(n):].copy(), "cached_dataset"
        except Exception:
            pass
    return _synthetic_data(settings, max(int(n), 1200)), "synthetic_fallback"


def _action_counts(actions: pd.Series) -> dict[str, int]:
    counts = actions.astype(str).value_counts()
    return {
        "action_long_count": int(counts.get("LONG", 0)),
        "action_short_count": int(counts.get("SHORT", 0)),
        "action_flat_count": int(counts.get("FLAT", 0)),
    }


def _core_live_gate_diagnostics(predictions: pd.DataFrame, settings: Settings) -> dict:
    """Diagnose the live signal gate's core filters on base-policy candidates.

    The benchmark does not load the required specialist trade-window artifact, so
    that separate gate is deliberately excluded and called out in the report.
    This is diagnostic only: it does not change the actions used in the backtest.
    """
    policy_diagnostics = predictions.attrs.get("policy_diagnostics", {}) or {}
    candidates = predictions.loc[
        predictions["action"].astype(str).isin({"LONG", "SHORT"})
    ]
    # Disable only the specialist gate for diagnosis, since this benchmark does not
    # load that artifact. The actual live path continues requiring it by default.
    gate_settings = replace(settings, trade_window_required_for_signal=False)
    rejected_reasons: Counter[str] = Counter()
    approved: Counter[str] = Counter()
    for record in candidates.to_dict(orient="records"):
        record.setdefault("market_type", "spot")
        decision, reasons = live_signal_gate(record, gate_settings)
        if decision in {"LONG", "SHORT"}:
            approved[decision] += 1
        else:
            rejected_reasons.update(reasons or ["rejected_without_reason"])
    approved_count = int(sum(approved.values()))
    candidate_count = int(len(candidates))
    return {
        "policy_rows": int(policy_diagnostics.get("rows", len(predictions))),
        "policy_pass_probability": int(policy_diagnostics.get("pass_probability", 0)),
        "policy_pass_robust_return": int(policy_diagnostics.get("pass_robust_return", 0)),
        "policy_pass_meta": int(policy_diagnostics.get("pass_meta", 0)),
        "policy_pass_score": int(policy_diagnostics.get("pass_score", 0)),
        "policy_pass_all_gates": int(policy_diagnostics.get("pass_all_policy_gates", 0)),
        "core_live_gate_candidates": candidate_count,
        "core_live_gate_approved": approved_count,
        "core_live_gate_rejected": candidate_count - approved_count,
        "core_live_gate_rejections": json.dumps(dict(sorted(rejected_reasons.items())), sort_keys=True),
    }


def evaluate_suite(settings: Settings, n: int = 3000) -> pd.DataFrame:
    """Run diagnostic controls on either cached real data or deterministic synthetic data.

    This suite is deliberately comparative: it is a sanity check against trivial controls,
    not a hyperparameter objective and not evidence of a deployable edge.
    """
    data, data_source = _load_or_make_data(settings, int(n))
    work = replace(settings, min_train_rows=min(int(settings.min_train_rows), max(200, len(data) // 3)))
    split = max(int(work.min_train_rows), int(len(data) * 0.70))
    split = min(split, len(data) - max(100, len(data) // 5))
    train, test = data.iloc[:split].copy(), data.iloc[split:].copy()

    rows: list[dict] = []
    for name in ("flat", "buy_hold", "trend", "mean_reversion"):
        actions = _baseline_actions(test, name)
        stats = _evaluate_actions(test, actions, work)
        rows.append({"strategy": name, **_action_counts(actions), **stats})

    try:
        engine = AdaptiveEngine(work)
        engine.fit(train)
        features = make_oos_features(
            train,
            test,
            work.horizon_bars,
            external_feature_lag_bars=getattr(work, "external_feature_lag_bars", 1),
        )
        predictions = engine.predict_frame(features)
        actions = predictions["action"]
        stats = _evaluate_actions(test, actions, work)
        rows.append({
            "strategy": "adaptive_engine",
            **_action_counts(actions),
            **_core_live_gate_diagnostics(predictions, work),
            **stats,
        })
    except Exception as exc:
        rows.append(
            {
                "strategy": "adaptive_engine",
                "action_long_count": 0,
                "action_short_count": 0,
                "action_flat_count": int(len(test)),
                "policy_rows": int(len(test)),
                "policy_pass_probability": 0,
                "policy_pass_robust_return": 0,
                "policy_pass_meta": 0,
                "policy_pass_score": 0,
                "policy_pass_all_gates": 0,
                "core_live_gate_candidates": 0,
                "core_live_gate_approved": 0,
                "core_live_gate_rejected": 0,
                "core_live_gate_rejections": "{}",
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
    metadata = {
        "data_source": data_source,
        "data_rows": int(len(data)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_start": test.index[0].isoformat(),
        "test_end": test.index[-1].isoformat(),
    }
    for key, value in metadata.items():
        frame[key] = value
    frame.attrs["benchmark_metadata"] = metadata
    frame.attrs["diagnostic_scope"] = (
        "OOS test split; core live gates are evaluated on policy candidates without "
        "the separate required trade-window specialist, which this benchmark does not load"
    )
    preferred = [
        "strategy", "data_source", "data_rows", "train_rows", "test_rows",
        "test_start", "test_end", "action_long_count", "action_short_count",
        "action_flat_count", "policy_rows", "policy_pass_probability",
        "policy_pass_robust_return", "policy_pass_meta", "policy_pass_score",
        "policy_pass_all_gates", "core_live_gate_candidates",
        "core_live_gate_approved", "core_live_gate_rejected",
        "core_live_gate_rejections", "total_return", "benchmark_return",
        "excess_return", "max_drawdown", "sharpe_like", "sortino_like",
        "trades", "win_rate", "profit_factor", "robust_score", "error",
    ]
    return frame[[c for c in preferred if c in frame.columns]]


def save_suite_report(result: pd.DataFrame, path: str | Path = "logs/benchmark_suite.json") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(result.attrs.get("benchmark_metadata", {}))
    diagnostic_scope = str(result.attrs.get("diagnostic_scope", "diagnostic-only benchmark"))
    payload = {
        "type": "diagnostic_benchmark_suite",
        "schema_version": 2,
        "warning": (
            "Benchmark controls are diagnostics, not proof of a production trading edge. "
            "A synthetic_fallback dataset is not market evidence. Core live-gate diagnostics "
            "exclude the required trade-window specialist and must not be treated as final "
            "live approval counts."
        ),
        "metadata": metadata,
        "diagnostic_scope": diagnostic_scope,
        "rows": result.replace([np.inf, -np.inf], np.nan).to_dict(orient="records"),
    }
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out
