from __future__ import annotations

from pathlib import Path
import json
import os
import time
import numpy as np
import pandas as pd

from .autolearn import auto_update
from .cognition import CognitionEngine
from .config import Settings
from .data import exchange_client, fetch_ohlcv_incremental
from .data_quality import audit_market_data
from .state_fusion import build_market_state
from .deployment import asset_bundle_dir




def _offline_synthetic_data(settings: Settings, n: int | None = None) -> pd.DataFrame:
    """Create deterministic closed-candle data for network-isolated CI/research smoke runs."""
    rows = int(n or max(getattr(settings, "min_train_rows", 1500) + 512, 2500))
    freq = {
        "1m": "1min",
        "3m": "3min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1D",
    }.get(str(settings.timeframe), str(settings.timeframe))
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except Exception:
        offset = pd.Timedelta(minutes=15)
    end = pd.Timestamp.now(tz="UTC").floor(freq) - offset
    index = pd.date_range(end=end, periods=rows, freq=freq)
    rng = np.random.default_rng(int(settings.seed))
    regimes = np.repeat([0, 1, 2, 3], rows // 4 + 1)[:rows]
    drift = np.choose(regimes, [0.00002, 0.00010, -0.00008, 0.0])
    vol = np.choose(regimes, [0.0025, 0.0035, 0.0045, 0.0060])
    returns = drift + rng.normal(0.0, vol, rows)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = close * (1.0 + rng.normal(0.0, 0.0008, rows))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.0025, rows))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.0025, rows))
    volume = rng.lognormal(10.0, 0.40, rows)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )

def refresh_market_dataset(settings: Settings, root: str | Path = ".", exchange=None) -> pd.DataFrame:
    root = Path(root)
    cache = root / "data" / f"{settings.symbol.replace('/','_')}_{settings.timeframe}.parquet"
    ex = exchange or exchange_client(getattr(settings, "exchange", "binance"), sandbox=False)
    return fetch_ohlcv_incremental(ex, settings.symbol, settings.timeframe, cache, settings.lookback_bars)


def _macro_values_from_growth(growth: dict) -> dict[str, float]:
    """Read macro observations from the external-intelligence result envelope."""
    external = growth.get("external", {}) or {}
    macro = growth.get("macro", {}) or external.get("macro", {}) or {}
    series = macro.get("series", {}) or {}
    values: dict[str, float] = {}
    for name, observation in series.items():
        if not isinstance(observation, dict) or observation.get("last") is None:
            continue
        try:
            value = float(observation["last"])
        except (TypeError, ValueError, OverflowError):
            continue
        if np.isfinite(value):
            values[str(name)] = value
    return values


def autonomous_cycle(settings: Settings, query: str, root: str | Path = ".") -> dict:
    root = Path(root)
    offline = str(os.getenv("AUTONOMOUS_OFFLINE", "")).strip().lower() in {"1", "true", "yes", "on"}

    if offline:
        # CI/offline mode must not contact an exchange or external intelligence provider.
        ex = None
        df = _offline_synthetic_data(settings)
        settings.external_deep_search = False
        settings.fred_series = ()
        settings.sec_ciks = ()
    else:
        ex = exchange_client(getattr(settings, "exchange", "binance"), sandbox=False)
        df = refresh_market_dataset(settings, root, ex)

    quality = audit_market_data(df, settings.timeframe)
    if not quality.passed:
        raise RuntimeError(f"Market data quality gate failed: {quality.to_dict()}")
    if len(df):
        last_bar_age=(pd.Timestamp.now(tz="UTC")-pd.Timestamp(df.index[-1])).total_seconds()/60.0
        max_age=max(float(getattr(settings,"live_max_data_age_minutes",30.0)),2.0*float(getattr(settings,"live_default_refresh_seconds",60))/60.0)
        if last_bar_age > max_age:
            raise RuntimeError(f"Market data is stale: age_minutes={last_bar_age:.1f}, limit={max_age:.1f}")

    cognition = CognitionEngine(settings, root)
    research = cognition.run_research(df, query)
    growth = cognition.run_growth(df, query, research.get("strategy_candidates", []))

    try:
        macro_vals = _macro_values_from_growth(growth)
        ext_health = {}
        provider_health = growth.get("external", {}).get("provider_health", {}) or {}
        for provider, stats in provider_health.items():
            if isinstance(stats, dict):
                ok=float(stats.get("ok",0.0)); fail=float(stats.get("fail",0.0))
                ext_health[f"{provider}_reliability"]=(ok+2.0)/(ok+fail+4.0)
        event_summary = growth.get("external", {}).get("event_summary", {}) or {}
        for key, value in event_summary.items():
            if isinstance(value, (int,float)):
                ext_health[key]=float(value)
        if not offline:
            state = build_market_state(
                ex, settings.symbol, settings.timeframe, float(df["close"].iloc[-1]),
                macro=macro_vals, external=ext_health,
                cross_symbols=list(getattr(settings, "cross_asset_symbols", ()) or ()),
                orderbook_levels=getattr(settings, "orderbook_levels", 50),
                impact_notional=getattr(settings, "orderbook_impact_notional", 10000.0),
            )
            cognition.registry.add_market_state(settings.symbol, state.timestamp, state.price, state.flatten())
            growth["market_state"] = state.flatten()
        else:
            growth["market_state"] = {"mode": "offline", "exchange_access": False}
    except Exception as exc:
        growth["market_state_error"] = str(exc)

    promotion = auto_update(df, settings, model_dir=asset_bundle_dir(root, settings.symbol))
    deep = cognition.run_evolution(df, research.get("hypotheses", [])) if getattr(settings, "deep_evolution_enabled", True) else {}
    result = {"data_quality": quality.to_dict(), "research": research, "growth": growth, "deep_evolution": deep, "promotion": promotion}
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "logs" / "autonomous_status.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def daemon(settings: Settings, query: str, root: str | Path = ".", cycles: int | None = None, sleep_seconds: int = 900):
    n = 0
    while cycles is None or n < cycles:
        try:
            autonomous_cycle(settings, query, root)
        except Exception as exc:
            Path(root, "logs").mkdir(parents=True, exist_ok=True)
            Path(root, "logs", "autonomous_error.json").write_text(json.dumps({"error": str(exc)}), encoding="utf-8")
        n += 1
        if cycles is not None and n >= cycles:
            break
        time.sleep(max(60, int(sleep_seconds)))
