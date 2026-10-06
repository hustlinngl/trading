from __future__ import annotations

from pathlib import Path
import json
import time
import pandas as pd

from .autolearn import auto_update
from .cognition import CognitionEngine
from .config import Settings
from .data import exchange_client, fetch_ohlcv_incremental
from .data_quality import audit_market_data
from .state_fusion import build_market_state


def refresh_market_dataset(settings: Settings, root: str | Path = ".", exchange=None) -> pd.DataFrame:
    root = Path(root)
    cache = root / "data" / f"{settings.symbol.replace('/','_')}_{settings.timeframe}.parquet"
    ex = exchange or exchange_client("binance", sandbox=False)
    return fetch_ohlcv_incremental(ex, settings.symbol, settings.timeframe, cache, settings.lookback_bars)


def autonomous_cycle(settings: Settings, query: str, root: str | Path = ".") -> dict:
    root = Path(root)
    ex = exchange_client("binance", sandbox=False)
    df = refresh_market_dataset(settings, root, ex)
    quality = audit_market_data(df, settings.timeframe)
    if not quality.passed:
        raise RuntimeError(f"Market data quality gate failed: {quality.to_dict()}")

    cognition = CognitionEngine(settings, root)
    research = cognition.run_research(df, query)
    growth = cognition.run_growth(df, query, research.get("strategy_candidates", []))

    try:
        macro_vals = {k: float(v["last"]) for k, v in growth.get("macro", {}).get("series", {}).items() if isinstance(v, dict) and v.get("last") is not None}
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
        state = build_market_state(
            ex, settings.symbol, settings.timeframe, float(df["close"].iloc[-1]),
            macro=macro_vals, external=ext_health,
            cross_symbols=list(getattr(settings, "cross_asset_symbols", ()) or ()),
            orderbook_levels=getattr(settings, "orderbook_levels", 50),
            impact_notional=getattr(settings, "orderbook_impact_notional", 10000.0),
        )
        cognition.registry.add_market_state(settings.symbol, state.timestamp, state.price, state.flatten())
        growth["market_state"] = state.flatten()
    except Exception as exc:
        growth["market_state_error"] = str(exc)

    promotion = auto_update(df, settings, model_dir=root / "models")
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
