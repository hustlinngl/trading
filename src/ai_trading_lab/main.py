from __future__ import annotations

import argparse
import json
import time
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import load_settings
from .data import cache_ohlcv, exchange_client, fetch_ohlcv, load_cached, timeframe_offset
from .evaluation import directional_validation_diagnostics, run_configured_backtest
from .policy import make_actions
from .engine import AdaptiveEngine, execution_aligned_targets
from .research import walk_forward, strategy_discovery
from .paper import one_iteration
from .autolearn import auto_update
from .optimizer import optimize_policy
from .objectives import robust_performance_utility
from .fingerprint import strong_dataset_fingerprint
from .autonomous import autonomous_cycle, daemon
from .deployment import model_semantics_fingerprint, deployment_semantics_fingerprint, MANAGED_BUNDLE_FILES, refresh_deployment_manifest as _refresh_deployment_manifest


def synthetic_data(s, n=5000):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(s.seed)
    pandas_freq = {'1m':'1min','3m':'3min','5m':'5min','15m':'15min','30m':'30min','1h':'1h','4h':'4h','1d':'1D'}.get(s.timeframe, s.timeframe)
    idx = pd.date_range('2024-01-01', periods=n, freq=pandas_freq, tz='UTC')
    regimes = np.repeat([0, 1, 2, 3], n // 4 + 1)[:n]
    drift = np.choose(regimes, [0.00002, 0.00010, -0.00008, 0.0])
    vol = np.choose(regimes, [0.0025, 0.0035, 0.0045, 0.0060])
    r = drift + rng.normal(0, vol, n)
    close = 100 * np.exp(np.cumsum(r))
    open_ = close * (1 + rng.normal(0, 0.0008, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.0025, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.0025, n))
    volume = rng.lognormal(10, 0.40, n) * (1 + (regimes == 2) * 0.6)
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=idx)


def asset_model_dir(symbol: str, root: str | Path = ".") -> Path:
    return Path(root) / 'models' / 'assets' / symbol.replace('/', '_').replace(':', '_')


def validate_research_data(df, settings):
    """Fail fast on data that is structurally unsafe for time-series research."""
    from .data_quality import audit_market_data
    report = audit_market_data(df, settings.timeframe)
    if not report.passed:
        issues = ", ".join(report.reasons) if report.reasons else "quality_check_failed"
        raise ValueError(f"Research data rejected: {issues}. Expected timeframe={settings.timeframe}, inferred={report.inferred_timeframe}.")
    return report


def asset_deployment_manifest(settings, root: str | Path = ".") -> Path:
    return asset_model_dir(settings.symbol, root) / "deployment_manifest.json"


def refresh_deployment_manifest(settings, root: str | Path = ".") -> dict:
    return _refresh_deployment_manifest(settings, root)


def _promote_asset_bundle(asset_dir: str | Path, champion_dir: str | Path = "models/champion") -> None:
    """Copy only deployable engine artifacts into the global champion bundle."""
    import shutil
    src = Path(asset_dir)
    dst = Path(champion_dir)
    dst.mkdir(parents=True, exist_ok=True)
    managed = MANAGED_BUNDLE_FILES
    for name in managed:
        path = src / name
        target = dst / name
        if path.exists():
            shutil.copy2(path, target)
        elif target.exists():
            target.unlink()


def train_base_asset(df, settings, *, promote_champion: bool = False):
    """Train and persist one asset's base bundle in the canonical location."""
    eng = AdaptiveEngine(settings)
    art = eng.fit(df)
    asset_dir = asset_model_dir(settings.symbol)
    asset_dir.mkdir(parents=True, exist_ok=True)
    eng.save(asset_dir)
    (asset_dir / "base_training_meta.json").write_text(json.dumps({
        "symbol": settings.symbol, "rows": int(len(df)),
        "start": str(df.index.min()), "end": str(df.index.max()),
        "timeframe": str(settings.timeframe),
        "data_fingerprint": strong_dataset_fingerprint(df),
        "model_semantics_fingerprint": model_semantics_fingerprint(settings),
        "deployment_semantics_fingerprint": deployment_semantics_fingerprint(settings),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")
    deployment = refresh_deployment_manifest(settings)
    if promote_champion:
        if not bool(deployment.get("ready")):
            failed = [name for name, ok in deployment.get("checks", {}).items() if not bool(ok)]
            raise RuntimeError(
                f"Refusing champion promotion for {settings.symbol}: deployment evidence is not ready"
                + (f" ({', '.join(failed)})" if failed else "")
            )
        _promote_asset_bundle(asset_dir)
    return asset_dir, art


def train_duration_asset(df, settings, *, holdout_frac: float, copy_legacy: bool = True):
    """Train and persist one asset's adaptive-duration specialist."""
    from .trade_window import train_trade_window_backbone
    asset_dir = asset_model_dir(settings.symbol)
    asset_dir.mkdir(parents=True, exist_ok=True)
    asset_model = asset_dir / 'trade_window_specialist.joblib'
    report = train_trade_window_backbone(df, settings, holdout_frac=holdout_frac, save_path=asset_model)
    (asset_dir / "trade_window_training_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    Path("logs").mkdir(exist_ok=True)
    Path("logs/trade_window_training_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    refresh_deployment_manifest(settings)
    if copy_legacy and asset_model.exists() and bool(report.get("production_ready", False)):
        import shutil
        global_model = Path(getattr(settings, 'trade_window_model_path', 'models/champion/trade_window_specialist.joblib'))
        global_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(asset_model, global_model)
    return asset_model, report


def read_asset_dataframe(meta: dict):
    from .dataset import read_market_file
    path = meta.get("path")
    if not path:
        raise ValueError("Imported asset has no registered path")
    return read_market_file(path)


def test_base_holdout(df, settings, holdout_frac: float) -> dict:
    import pandas as pd

    split = int(len(df) * (1.0 - holdout_frac))
    if split < max(500, settings.min_train_rows) or len(df) - split < 100:
        raise ValueError("Not enough rows for requested train/holdout split")
    train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()
    eng = AdaptiveEngine(settings)
    eng.fit(train_df)
    from .features import make_oos_features
    feat = make_oos_features(
        train_df, test_df, settings.horizon_bars,
        external_feature_lag_bars=getattr(settings, "external_feature_lag_bars", 1),
    )
    actions = make_actions(eng, feat, settings)
    # Evaluate direction against untouched, realized next-open outcomes. This
    # is intentionally separate from the policy's predicted-vs-predicted checks.
    holdout_predictions = eng.model.predict(feat)
    labelled_market = pd.concat([train_df, test_df])
    # Score probability against the exact executable barrier/time-stop
    # returns used to train the base model, including ambiguous-OHLC exclusions.
    _, realized_returns, _ = execution_aligned_targets(
        labelled_market,
        settings.horizon_bars,
        settings.pt_atr,
        settings.sl_atr,
    )
    realized_returns = realized_returns.reindex(test_df.index)
    realized_directional_validation = directional_validation_diagnostics(
        holdout_predictions["p_up"].reindex(test_df.index).to_numpy(),
        realized_returns.to_numpy(),
    )
    action_diagnostics = {
        "actions_by_side": {
            str(side): int(count)
            for side, count in actions.astype(str).value_counts(dropna=False).items()
        },
        "policy_gates": actions.attrs.get("diagnostics", {}),
    }
    bt = test_df.copy()
    bt["atr_14"] = feat["atr_14"].reindex(test_df.index).ffill()
    result = run_configured_backtest(bt, actions, settings)
    report = {
        "symbol": settings.symbol,
        "timeframe": str(settings.timeframe),
        "rows": len(df),
        "train_rows": len(train_df),
        "holdout_rows": len(test_df),
        # The final production bundle is intentionally refit on the full dataset
        # after this untouched temporal validation. Keep both fingerprints so the
        # deployment manifest can distinguish validation provenance from refit data.
        "data_fingerprint": strong_dataset_fingerprint(df),
        "validation_train_data_fingerprint": strong_dataset_fingerprint(train_df),
        "validation_holdout_data_fingerprint": strong_dataset_fingerprint(test_df),
        "validation_holdout_start": str(test_df.index.min()),
        "validation_holdout_end": str(test_df.index.max()),
        "validation_holdout_frac": float(holdout_frac),
        "action_diagnostics": action_diagnostics,
        "realized_directional_validation": realized_directional_validation,
        "model_semantics_fingerprint": model_semantics_fingerprint(settings),
        "deployment_semantics_fingerprint": deployment_semantics_fingerprint(settings),
        "holdout": result.stats,
    }
    asset_dir = asset_model_dir(settings.symbol)
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "base_holdout_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["deployment"] = refresh_deployment_manifest(settings)
    return report


def _emit_training_progress(symbol: str, stage: str, *, asset_started_at=None, stage_started_at=None, **details) -> None:
    """Emit line-buffered JSON progress events for long multi-asset training runs."""
    event = {
        "event": "training_progress",
        "symbol": str(symbol),
        "stage": str(stage),
    }
    if asset_started_at is not None:
        event["asset_elapsed_seconds"] = round(max(0.0, time.perf_counter() - asset_started_at), 2)
    if stage_started_at is not None:
        event["stage_elapsed_seconds"] = round(max(0.0, time.perf_counter() - stage_started_at), 2)
    event.update(details)
    print(json.dumps(event, sort_keys=True, default=str), file=sys.stderr, flush=True)


def train_complete_asset(df, settings, *, holdout_frac: float, promote_champion: bool = True):
    """One-click training with observable fit, holdout, specialist and readiness stages."""
    symbol = str(settings.symbol)
    asset_started_at = time.perf_counter()
    _emit_training_progress(
        symbol, "asset_training_started", asset_started_at=asset_started_at,
        rows=len(df), holdout_frac=float(holdout_frac),
    )

    stage_started_at = time.perf_counter()
    _emit_training_progress(symbol, "base_bundle_fit_started", asset_started_at=asset_started_at, rows=len(df))
    asset_dir, _ = train_base_asset(df, settings)
    _emit_training_progress(
        symbol, "base_bundle_fit_completed", asset_started_at=asset_started_at,
        stage_started_at=stage_started_at, bundle=str(asset_dir),
    )

    stage_started_at = time.perf_counter()
    _emit_training_progress(
        symbol, "base_holdout_validation_started", asset_started_at=asset_started_at,
        holdout_frac=float(holdout_frac),
    )
    base_holdout = test_base_holdout(df, settings, holdout_frac)
    holdout_stats = base_holdout.get("holdout", {}) if isinstance(base_holdout, dict) else {}
    _emit_training_progress(
        symbol, "base_holdout_validation_completed", asset_started_at=asset_started_at,
        stage_started_at=stage_started_at,
        holdout_rows=base_holdout.get("holdout_rows") if isinstance(base_holdout, dict) else None,
        holdout_trades=holdout_stats.get("trades_taken", holdout_stats.get("trades")),
    )
    Path("logs").mkdir(exist_ok=True)
    Path("logs/test_report.json").write_text(json.dumps(base_holdout, indent=2, default=str), encoding="utf-8")

    stage_started_at = time.perf_counter()
    _emit_training_progress(
        symbol, "trade_window_fit_started", asset_started_at=asset_started_at,
        rows=len(df),
    )
    asset_model, duration_report = train_duration_asset(
        df,
        settings,
        holdout_frac=holdout_frac,
        copy_legacy=promote_champion,
    )
    _emit_training_progress(
        symbol, "trade_window_fit_completed", asset_started_at=asset_started_at,
        stage_started_at=stage_started_at,
        production_ready=bool(duration_report.get("production_ready", False)),
        readiness_reason=duration_report.get("reason"),
        artifact=str(asset_model),
    )

    stage_started_at = time.perf_counter()
    _emit_training_progress(symbol, "deployment_readiness_started", asset_started_at=asset_started_at)
    deployment = refresh_deployment_manifest(settings)
    _emit_training_progress(
        symbol, "deployment_readiness_completed", asset_started_at=asset_started_at,
        stage_started_at=stage_started_at, production_ready=bool(deployment.get("ready")),
        failed_checks=sorted(
            name for name, ok in (deployment.get("checks", {}) or {}).items() if not bool(ok)
        ),
    )
    if deployment.get("ready") and promote_champion:
        _promote_asset_bundle(asset_dir)

    _emit_training_progress(
        symbol, "asset_training_completed", asset_started_at=asset_started_at,
        production_ready=bool(deployment.get("ready")), bundle=str(asset_dir),
    )
    return {
        "symbol": settings.symbol, "rows": len(df), "status": "trained",
        "bundle": str(asset_dir), "duration_bundle": str(asset_model),
        "production_ready": bool(deployment.get("ready")),
        "deployment": deployment,
        "base_holdout": base_holdout,
        "duration_report": duration_report,
    }

def system_doctor(settings, root: str | Path = ".") -> dict:
    import importlib.util, platform
    root = Path(root)
    deps = {name: importlib.util.find_spec(name) is not None for name in [
        "numpy", "pandas", "sklearn", "yaml", "joblib",
        "optuna", "scipy", "pyarrow",
        "ccxt", "requests",
        "plotly", "streamlit",
        "xgboost", "lightgbm",
    ]}
    directories = {name: (root / name).exists() for name in ["data", "models", "logs"]}
    core_names = ["numpy", "pandas", "sklearn", "yaml", "joblib"]
    research_names = ["numpy", "pandas", "sklearn", "yaml", "joblib", "optuna", "scipy", "pyarrow"]
    live_names = ["ccxt", "requests"]
    ui_names = ["plotly", "streamlit"]
    return {
        "version": (Path(root) / "VERSION").read_text(encoding="utf-8").strip() if (Path(root) / "VERSION").exists() else "unknown",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "python_64bit": platform.architecture()[0] == "64bit",
        "dependencies": deps,
        "directories": directories,
        "paper_only": bool(getattr(settings, "paper_only", True)),
        "sandbox": bool(getattr(settings, "sandbox", True)),
        "live_symbols": list(getattr(settings, "live_symbols", ()) or ()),
        "core_ready": all(deps.get(x, False) for x in core_names),
        "research_ready": all(deps.get(x, False) for x in research_names),
        "live_data_ready": all(deps.get(x, False) for x in live_names),
        "ui_live_ready": all(deps.get(x, False) for x in ui_names),
        "all_required_dependencies": all(deps.get(x, False) for x in research_names + live_names + ui_names),
        "optional_ml": {"xgboost": deps.get("xgboost", False), "lightgbm": deps.get("lightgbm", False)},
    }


def main():
    parser = argparse.ArgumentParser(description='Adaptive AI Trading Lab')
    parser.add_argument('command', choices=['download', 'bulk-download', 'import-data', 'list-data', 'train', 'train-all', 'train-complete', 'train-complete-all', 'train-window', 'train-window-all', 'test', 'test-window', 'test-all', 'live-scan', 'live-outcomes', 'research', 'discover', 'optimize', 'master-tune', 'benchmark', 'paper', 'paper-daemon', 'demo', 'auto-update', 'autonomous', 'grow', 'evolve', 'daemon', 'state', 'stream', 'real-history', 'real-ticker', 'bootstrap-live-data', 'doctor', 'rl-help'])
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--query', default=None, help='External-intelligence query for autonomous research')
    parser.add_argument('--cycles', type=int, default=0)
    parser.add_argument('--sleep-seconds', type=int, default=None)
    parser.add_argument('--start', default=None, help='Bulk data start date YYYY-MM-DD')
    parser.add_argument('--end', default=None, help='Bulk data end date YYYY-MM-DD')
    parser.add_argument('--market', default='spot', choices=['spot','futures-um'])
    parser.add_argument('--symbol', default=None)
    parser.add_argument('--timeframe', default=None)
    parser.add_argument('--trials', type=int, default=40)
    parser.add_argument('--data-path', default=None, help='Imported CSV/Parquet/JSON path')
    parser.add_argument('--data-dir', default='data/imported')
    parser.add_argument('--symbols', default=None, help='Comma-separated symbols for live scan')
    parser.add_argument('--holdout-frac', type=float, default=0.15)
    parser.add_argument('--limit', type=int, default=700, help='Maximum bars for bounded real-data adapters such as Kraken')
    parser.add_argument('--all-symbols', action='store_true', help='Discover every active exchange market eligible for OHLCV bootstrap')
    parser.add_argument('--market-types', default=None, help='Comma-separated market types for --all-symbols (default: spot,swap,future)')
    parser.add_argument('--live-bars', type=int, default=None, help='Closed OHLCV bars to cache for the live dashboard')
    parser.add_argument('--workers', type=int, default=None, help='Bounded parallel public-data workers for bootstrap-live-data (default 4, max 8)')
    parser.add_argument('--benchmark-bars', type=int, default=3000, help='Bars used by the diagnostic benchmark suite')
    args = parser.parse_args()
    s = load_settings(args.config)
    if args.symbol:
        s.symbol = args.symbol
    if args.timeframe:
        s.timeframe = args.timeframe
    for p in ['data', 'models', 'logs']: Path(p).mkdir(exist_ok=True)

    if args.command == 'import-data':
        if not args.data_path:
            raise SystemExit('import-data requires --data-path FILE_OR_DIRECTORY')
        from .dataset import import_market_path
        result = import_market_path(args.data_path, data_dir=args.data_dir, symbol=args.symbol)
        print(json.dumps(result, indent=2, default=str)); return

    if args.command == 'list-data':
        from .dataset import imported_registry
        print(json.dumps(imported_registry(args.data_dir), indent=2, default=str)); return

    if args.command == 'demo':
        df = synthetic_data(s, n=3000)
        split = int(len(df) * 0.70)
        train, test = df.iloc[:split], df.iloc[split:]
        engine = AdaptiveEngine(s)
        engine.fit(train)
        from .features import make_oos_features
        feat = make_oos_features(train, test, s.horizon_bars, external_feature_lag_bars=getattr(s, 'external_feature_lag_bars', 1))
        actions = make_actions(engine, feat, s)
        bt = test.copy(); bt['atr_14'] = feat['atr_14']
        out = run_configured_backtest(bt, actions, s)
        print(json.dumps({'holdout': 'last 30%', **out.stats}, indent=2))
        return

    if args.command == 'doctor':
        print(json.dumps(system_doctor(s, '.'), indent=2, default=str)); return

    if args.command in {'autonomous','grow','evolve'}:
        q = args.query or s.autonomous_query
        result = autonomous_cycle(s, q, '.')
        print(json.dumps(result, indent=2, default=str))
        return
    if args.command == 'daemon':
        q = args.query or s.autonomous_query
        daemon(s, q, '.', cycles=(args.cycles if args.cycles > 0 else None), sleep_seconds=(args.sleep_seconds or s.autonomous_interval_seconds))
        return

    if args.command == 'rl-help':
        print('Optional: pip install -e .[full], then use ai_trading_lab.experimental.rl_agent.train_ppo for execution/sizing research.')
        return

    if args.command == 'paper-daemon':
        from .paper import paper_daemon
        paper_daemon(s, cycles=(args.cycles if args.cycles > 0 else None), sleep_seconds=(args.sleep_seconds or getattr(s, 'poll_seconds', 60)))
        return

    if args.command == 'stream':
        import asyncio
        from .streaming import BinancePublicStream, collect_to_jsonl
        collector = BinancePublicStream(s.symbol, streams=('bookTicker','aggTrade'))
        path = Path('data') / f"{s.symbol.replace('/','_').replace(':','_')}_stream.jsonl"
        print(f'collecting public market stream -> {path}; stop with Ctrl+C')
        asyncio.run(collect_to_jsonl(collector, path, max_events=None))
        return

    if args.command == 'state':
        ex = exchange_client(s.__dict__.get('exchange', 'binance'), sandbox=False)
        from .state_fusion import build_market_state
        st = build_market_state(ex, s.symbol, s.timeframe, float(fetch_ohlcv(ex, s.symbol, s.timeframe, 2)['close'].iloc[-1]), cross_symbols=list(getattr(s, 'cross_asset_symbols', ()) or ()), orderbook_levels=getattr(s, 'orderbook_levels', 50), impact_notional=getattr(s, 'orderbook_impact_notional', 10000.0))
        print(json.dumps(st.flatten(), indent=2, default=str))
        return

    if args.command == 'live-scan':
        from .live import append_live_signal_history, scan_top5, write_live_snapshot
        symbols = [x.strip() for x in args.symbols.split(',') if x.strip()] if args.symbols else None
        results = scan_top5(s, '.', symbols=symbols)
        write_live_snapshot(results, '.')
        append_live_signal_history(results, '.')
        print(json.dumps([x.to_dict() for x in results], indent=2, default=str)); return
    if args.command == 'live-outcomes':
        from .live_tracker import update_live_signal_outcomes
        result = update_live_signal_outcomes(s, '.')
        print(json.dumps(result, indent=2, default=str)); return

    if args.command == 'bootstrap-live-data':
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import pandas as pd

        requested_symbols = [
            x.strip() for x in args.symbols.split(',') if x.strip()
        ] if args.symbols else []
        bars = max(80, int(args.live_bars or getattr(s, 'live_lookback_bars', 600)))
        out_dir = Path('data') / 'historical'
        out_dir.mkdir(parents=True, exist_ok=True)

        # Discovery uses one client. Each concurrent worker creates its own CCXT
        # client because exchange instances are not assumed thread-safe.
        discovery_exchange = exchange_client(
            getattr(s, 'exchange', 'binance'), sandbox=False
        )
        if args.all_symbols or not requested_symbols:
            allowed_types = {
                x.strip().lower()
                for x in (
                    args.market_types.split(',')
                    if args.market_types
                    else ('spot', 'swap', 'future')
                )
                if x.strip()
            }
            discovered = []
            markets = getattr(discovery_exchange, 'markets', {}) or {}
            for market in markets.values():
                if not isinstance(market, dict) or market.get('active') is False:
                    continue
                symbol = str(market.get('symbol') or '').strip()
                if not symbol:
                    continue
                market_type = str(
                    market.get('type')
                    or ('swap' if market.get('swap')
                        else 'future' if market.get('future')
                        else 'spot')
                ).strip().lower()
                if allowed_types and market_type not in allowed_types:
                    continue
                if market_type in {'swap', 'future'} and not bool(market.get('contract')):
                    continue
                capabilities = getattr(discovery_exchange, 'has', None)
                if isinstance(capabilities, dict) and capabilities.get('fetchOHLCV') is False:
                    continue
                discovered.append(symbol)
            symbols = sorted(dict.fromkeys(discovered))
        else:
            symbols = list(dict.fromkeys(requested_symbols))

        total = len(symbols)
        configured_workers = max(1, int(getattr(s, 'max_parallel_downloads', 4)))
        requested_workers = args.workers if args.workers is not None else min(4, configured_workers)
        workers = max(1, min(8, int(requested_workers)))
        if total:
            workers = min(workers, total)

        try:
            tf_delta = pd.Timedelta(timeframe_offset(str(getattr(s, 'timeframe', '15m'))))
            if not tf_delta > pd.Timedelta(0):
                raise ValueError('non-positive timeframe')
        except Exception:
            tf_delta = pd.Timedelta(minutes=15)
        max_age = max(pd.Timedelta(minutes=30), tf_delta * 2.5)

        print(json.dumps({
            'stage': 'discover',
            'exchange': getattr(s, 'exchange', 'binance'),
            'symbols': total,
            'bars_per_symbol': bars,
            'workers': workers,
            'market_types': (
                [x.strip() for x in args.market_types.split(',') if x.strip()]
                if args.market_types else ['spot', 'swap', 'future']
            ),
            'output_dir': str(out_dir),
        }), flush=True)

        worker_local = threading.local()

        def worker_exchange():
            instance = getattr(worker_local, 'exchange', None)
            if instance is None:
                instance = exchange_client(
                    getattr(s, 'exchange', 'binance'), sandbox=False
                )
                worker_local.exchange = instance
            return instance

        def bootstrap_one(symbol):
            path = out_dir / (
                f"{symbol.replace('/', '_').replace(':', '_')}_{s.timeframe}.csv"
            )
            tmp = path.with_suffix(path.suffix + '.tmp')
            try:
                # Resume only from a sufficiently complete, ordered, fresh timestamp
                # series. File mtime is not trusted because caches can be copied.
                if path.exists():
                    try:
                        cached = pd.read_csv(path, usecols=['timestamp'])
                        cached['timestamp'] = pd.to_datetime(
                            cached['timestamp'], utc=True, errors='coerce'
                        )
                        cached = cached.dropna(subset=['timestamp'])
                        timestamps = cached['timestamp']
                        if (
                            len(cached) >= bars
                            and not timestamps.empty
                            and timestamps.is_monotonic_increasing
                            and not timestamps.duplicated().any()
                        ):
                            last_stamp = timestamps.iloc[-1]
                            age = pd.Timestamp.now(tz='UTC') - last_stamp
                            if pd.Timedelta(0) <= age <= max_age:
                                return {
                                    'symbol': symbol,
                                    'timeframe': s.timeframe,
                                    'bars': int(len(cached)),
                                    'output': str(path),
                                    'source': 'cache',
                                    'end': last_stamp.isoformat(),
                                }
                    except Exception:
                        # Corrupt/stale files are refreshed below; failures are recorded
                        # only if the new download also fails.
                        pass

                frame = fetch_ohlcv(
                    worker_exchange(),
                    symbol,
                    s.timeframe,
                    limit=bars,
                    include_unclosed=False,
                )
                if frame is None or frame.empty:
                    raise RuntimeError('empty_ohlcv')
                # Same-directory replace is atomic on supported filesystems, so a
                # terminated process cannot leave a half-written final CSV.
                frame.to_csv(tmp, index_label='timestamp')
                tmp.replace(path)
                return {
                    'symbol': symbol,
                    'timeframe': s.timeframe,
                    'bars': int(len(frame)),
                    'start': str(frame.index.min()) if len(frame) else None,
                    'end': str(frame.index.max()) if len(frame) else None,
                    'output': str(path),
                    'source': 'network',
                }
            except Exception as exc:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                return {
                    'symbol': symbol,
                    'error': f'{type(exc).__name__}:{exc}',
                }

        results = []
        failures = []
        completed = 0
        # Workers own exchange clients; the caller records results deterministically.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='bootstrap') as pool:
            futures = {pool.submit(bootstrap_one, symbol): symbol for symbol in symbols}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    item = future.result()
                except Exception as exc:
                    item = {'symbol': symbol, 'error': f'{type(exc).__name__}:{exc}'}
                completed += 1
                if item.get('error'):
                    failures.append(item)
                    print(f'[{completed}/{total}] {symbol} ERROR {item["error"]}', flush=True)
                else:
                    results.append(item)
                    print(
                        f'[{completed}/{total}] {symbol} {item.get("bars", 0)} bars {item.get("source", "network")}',
                        flush=True,
                    )

        results.sort(key=lambda row: str(row.get('symbol', '')))
        failures.sort(key=lambda row: str(row.get('symbol', '')))
        summary = {
            'exchange': getattr(s, 'exchange', 'binance'),
            'timeframe': s.timeframe,
            'requested_symbols': total,
            'workers': workers,
            'completed': len(results),
            'failed': len(failures),
            'results': results,
            'failures': failures,
            'output_dir': str(out_dir),
            'generated_at': datetime.now(timezone.utc).isoformat(),
        }
        logs_dir = Path('logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
        report_path = logs_dir / 'bootstrap_live_data.json'
        report_tmp = report_path.with_suffix(report_path.suffix + '.tmp')
        report_tmp.write_text(
            json.dumps(summary, indent=2, default=str),
            encoding='utf-8',
        )
        report_tmp.replace(report_path)
        print(json.dumps(summary, indent=2, default=str))
        return

    if args.command == 'real-ticker':
        from .kraken_data import fetch_ticker
        symbols = [x.strip() for x in (args.symbols.split(',') if args.symbols else [s.symbol]) if x.strip()]
        snapshots = [fetch_ticker(symbol) for symbol in dict.fromkeys(symbols)]
        Path('logs').mkdir(exist_ok=True)
        Path('logs/real_ticker_snapshot.json').write_text(json.dumps({'snapshots': snapshots}, indent=2), encoding='utf-8')
        print(json.dumps(snapshots, indent=2))
        return

    if args.command == 'real-history':
        from .kraken_data import fetch_ohlcv as fetch_kraken_ohlcv, fingerprint_frame, save_provenance
        df, provenance = fetch_kraken_ohlcv(s.symbol, s.timeframe, limit=args.limit)
        out_dir = Path(args.data_dir) / 'kraken'
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{s.symbol.replace('/', '_')}_{s.timeframe}_kraken"
        csv_path = out_dir / f"{stem}.csv"
        df.to_csv(csv_path, index=True)
        save_provenance(provenance, out_dir / f"{stem}.provenance.json")
        print(json.dumps({**provenance.to_dict(), 'fingerprint': fingerprint_frame(df), 'output': str(csv_path)}, indent=2))
        return

    if args.command == 'benchmark':
        from .benchmarks import evaluate_suite, save_suite_report
        result = evaluate_suite(s, n=max(1200, int(args.benchmark_bars)))
        save_suite_report(result, 'logs/benchmark_suite.json')
        print(result.to_string(index=False))
        return

    cache_path = Path(getattr(s, 'cache_dir', 'data')) / f"{s.symbol.replace('/','_')}_{s.timeframe}.parquet"
    if args.command == 'bulk-download':
        from .binance_vision import download_range, merge_archives
        if not args.start or not args.end: raise SystemExit('bulk-download requires --start and --end')
        paths=download_range(s.symbol,s.timeframe,args.start,args.end,args.market)
        out=Path('data')/f"{s.symbol.replace('/','_')}_{s.timeframe}_{args.start}_{args.end}_{args.market}.parquet"
        merged=merge_archives(paths,out)
        print(json.dumps({'files':len(paths),'rows':len(merged),'output':str(out)},indent=2,default=str)); return
    if args.command == 'download':
        ex = exchange_client(getattr(s, 'exchange', 'binance'), sandbox=False)
        df=fetch_ohlcv(ex,s.symbol,s.timeframe,s.lookback_bars); cache_ohlcv(df,cache_path); print(f'saved {len(df)} bars -> {cache_path}'); return

    # Load a single command dataset lazily. Multi-asset commands below operate only
    # on their registered/local files and must not touch the exchange just because the
    # default symbol has no cache; this is critical for reproducible/offline training.
    needs_single_dataset = {
        'train', 'train-complete', 'test', 'test-window',
        'train-window', 'research', 'discover', 'optimize',
        'master-tune', 'auto-update',
    }
    df = None
    if args.command in needs_single_dataset:
        if args.data_path:
            from .dataset import read_market_file
            df = read_market_file(args.data_path)
        elif cache_path.exists():
            df = load_cached(cache_path, timeframe=s.timeframe)
        else:
            # Prefer deterministic bundled history before attempting a live exchange.
            bundled_path = Path('data/historical') / (
                f"{s.symbol.replace('/','_').replace(':','_')}_{s.timeframe}.csv"
            )
            if bundled_path.exists():
                from .dataset import read_market_file
                df = read_market_file(bundled_path)
            else:
                ex = exchange_client(getattr(s, 'exchange', 'binance'), sandbox=False)
                df=fetch_ohlcv(ex,s.symbol,s.timeframe,s.lookback_bars); cache_ohlcv(df,cache_path)
    if args.command == 'train':
        validate_research_data(df, s)
        asset_dir, art = train_base_asset(df, s)
        print(json.dumps({'symbol': s.symbol, 'rows': len(df), 'bundle': str(asset_dir), 'latest': art.predictions.iloc[-1].to_dict()}, default=str, indent=2)); return

    if args.command == 'train-complete':
        validate_research_data(df, s)
        report = train_complete_asset(df, s, holdout_frac=args.holdout_frac)
        print(json.dumps(report, default=str, indent=2)); return
    if args.command == 'train-all':
        # Train every locally available asset, not only explicitly imported files.
        # This makes bundled historical data (data/historical/*) first-class training
        # input while preserving the imported registry as the primary override.
        from .dataset import imported_registry, read_market_file, infer_symbol
        reports = []
        sources = {}
        for sym, meta in imported_registry(args.data_dir).items():
            imported_path = meta.get('path')
            if imported_path and Path(imported_path).exists():
                sources[str(sym)] = ('imported', imported_path)
        historical_root = Path('data/historical')
        if historical_root.exists():
            for path in sorted(historical_root.iterdir()):
                if path.suffix.lower() not in {'.csv', '.parquet', '.pq', '.json'}:
                    continue
                try:
                    sym = infer_symbol(path)
                    sources.setdefault(str(sym), ('historical', str(path)))
                except Exception:
                    continue

        for sym, (source, path) in sources.items():
            try:
                ss = load_settings(args.config)
                ss.symbol = sym
                data = read_asset_dataframe({'path': path})
                validate_research_data(data, ss)
                asset_dir, _ = train_base_asset(data, ss)
                reports.append({
                    'symbol': sym,
                    'rows': len(data),
                    'source': source,
                    'status': 'trained',
                    'bundle': str(asset_dir),
                })
            except Exception as exc:
                reports.append({
                    'symbol': sym,
                    'source': source,
                    'status': 'error',
                    'error': f'{type(exc).__name__}: {exc}',
                })
        Path('logs/train_all_report.json').write_text(
            json.dumps(reports, indent=2, default=str), encoding='utf-8'
        )
        print(json.dumps(reports, indent=2, default=str)); return
    if args.command == 'train-complete-all':
        from .dataset import imported_registry, infer_symbol
        reports = []
        sources = {}
        for sym, meta in imported_registry(args.data_dir).items():
            imported_path = meta.get('path')
            if imported_path and Path(imported_path).exists():
                sources[str(sym)] = ('imported', imported_path)
        historical_root = Path('data/historical')
        if historical_root.exists():
            for path in sorted(historical_root.iterdir()):
                if path.suffix.lower() not in {'.csv', '.parquet', '.pq', '.json'}:
                    continue
                try:
                    sources.setdefault(str(infer_symbol(path)), ('historical', str(path)))
                except Exception:
                    continue
        for asset_index, (sym, (source, path)) in enumerate(sources.items(), start=1):
            try:
                ss = load_settings(args.config); ss.symbol = sym
                data = read_asset_dataframe({'path': path})
                validate_research_data(data, ss)
                _emit_training_progress(
                    sym, "asset_data_ready", asset_index=asset_index, asset_count=len(sources),
                    source=source, rows=len(data), path=str(path),
                )
                out = train_complete_asset(
                    data, ss, holdout_frac=args.holdout_frac, promote_champion=False
                )
                out['source'] = source
                reports.append(out)
            except Exception as exc:
                _emit_training_progress(
                    sym, "asset_training_failed", asset_index=asset_index, asset_count=len(sources),
                    source=source, error=f'{type(exc).__name__}: {exc}',
                )
                reports.append({
                    'symbol': sym, 'source': source, 'status': 'error',
                    'error': f'{type(exc).__name__}: {exc}',
                })
        # Multi-asset training must never leave the global champion bound to whichever
        # asset happened to finish last. Promote only the configured primary symbol.
        primary = str(s.symbol)
        primary_row = next(
            (
                row for row in reports
                if str(row.get('symbol')) == primary and bool(row.get('production_ready'))
            ),
            None,
        )
        if primary_row:
            _promote_asset_bundle(asset_model_dir(primary))
        Path('logs/train_complete_all_report.json').write_text(
            json.dumps(reports, indent=2, default=str), encoding='utf-8'
        )
        print(json.dumps(reports, indent=2, default=str)); return
    if args.command == 'test':
        validate_research_data(df, s)
        report = test_base_holdout(df, s, args.holdout_frac)
        Path('logs/test_report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
        print(json.dumps(report, indent=2, default=str)); return
    if args.command == 'test-window':
        validate_research_data(df, s)
        from .trade_window import train_trade_window_backbone
        report = train_trade_window_backbone(df, s, holdout_frac=args.holdout_frac, save_path=None)
        Path('logs/test_window_report.json').write_text(json.dumps({'symbol': s.symbol, **report}, indent=2, default=str), encoding='utf-8')
        print(json.dumps({'symbol': s.symbol, **report}, indent=2, default=str)); return

    if args.command == 'train-window-all':
        from .dataset import imported_registry, read_market_file
        from .trade_window import train_trade_window_backbone
        reports = []
        for sym, meta in imported_registry(args.data_dir).items():
            try:
                ss = load_settings(args.config); ss.symbol = sym
                data = read_market_file(meta['path'])
                validate_research_data(data, ss)
                asset_model, out = train_duration_asset(data, ss, holdout_frac=args.holdout_frac, copy_legacy=False)
                reports.append({'symbol': sym, 'status': 'trained', 'production_ready': bool(out.get('production_ready')), 'artifact': str(asset_model), 'report': out})
            except Exception as exc:
                reports.append({'symbol': sym, 'status': 'error', 'error': str(exc)})
        Path('logs/train_window_all_report.json').write_text(json.dumps(reports, indent=2, default=str), encoding='utf-8')
        print(json.dumps(reports, indent=2, default=str)); return
    if args.command == 'test-all':
        from .dataset import imported_registry
        reports = []
        for sym, meta in imported_registry(args.data_dir).items():
            try:
                ss = load_settings(args.config); ss.symbol = sym
                data = read_asset_dataframe(meta)
                validate_research_data(data, ss)
                reports.append({'symbol': sym, 'status': 'tested', **test_base_holdout(data, ss, args.holdout_frac)})
            except Exception as exc:
                reports.append({'symbol': sym, 'status': 'error', 'error': str(exc)})
        Path('logs/test_all_report.json').write_text(json.dumps(reports, indent=2, default=str), encoding='utf-8')
        print(json.dumps(reports, indent=2, default=str)); return
    if args.command == 'train-window':
        validate_research_data(df, s)
        asset_model, out = train_duration_asset(df, s, holdout_frac=args.holdout_frac)
        Path('logs/trade_window_training_report.json').write_text(json.dumps(out, indent=2, default=str), encoding='utf-8')
        print(json.dumps({'artifact': str(asset_model), **out}, indent=2, default=str)); return
    if args.command == 'research':
        validate_research_data(df, s)
        result=walk_forward(df,s); result.to_csv('logs/walk_forward.csv',index=False); print(result.to_string(index=False)); return
    if args.command == 'discover':
        result=strategy_discovery(df,s); Path('logs/strategy_candidates.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result[:5],indent=2)); return
    if args.command == 'optimize':
        holdout_frac=max(0.05,min(0.30,float(args.holdout_frac)))
        holdout_cut=int(len(df)*(1.0-holdout_frac))
        tuning_df=df.iloc[:holdout_cut].copy()
        holdout_df=df.iloc[holdout_cut:].copy()
        validation_cut=int(len(tuning_df)*0.70)
        train_df=tuning_df.iloc[:validation_cut].copy()
        validation_df=tuning_df.iloc[validation_cut:].copy()
        if len(train_df)<max(200,int(getattr(s,'min_train_rows',1500))) or len(holdout_df)<100:
            raise SystemExit('optimize requires enough rows for train, validation and final holdout')
        result=optimize_policy(train_df,validation_df,s,trials=args.trials)
        best=result.get('best_params',{})
        tuned=load_settings(args.config)
        tuned.symbol=s.symbol; tuned.timeframe=s.timeframe
        for key,value in best.items():
            if hasattr(tuned,key): setattr(tuned,key,value)
        eng=AdaptiveEngine(tuned); eng.fit(tuning_df)
        feat=__import__('ai_trading_lab.features',fromlist=['make_oos_features']).make_oos_features(tuning_df,holdout_df,tuned.horizon_bars,external_feature_lag_bars=getattr(tuned,'external_feature_lag_bars',1))
        actions=make_actions(eng,feat,tuned)
        bt=holdout_df.copy(); bt['atr_14']=feat['atr_14']
        final=run_configured_backtest(bt,actions,tuned,stop_atr_mult=float(best.get('stop_atr_mult',tuned.stop_atr_mult)),take_profit_rr=float(best.get('take_profit_rr',tuned.take_profit_rr)))
        result['data_split']={'train_rows':len(train_df),'validation_rows':len(validation_df),'final_holdout_rows':len(holdout_df),'final_holdout_start':str(holdout_df.index[0]),'final_holdout_end':str(holdout_df.index[-1])}
        result['final_holdout']=final.stats
        result['warning']='Final holdout was used only once after tuning; it is not part of the optimization objective.'
        Path('logs/policy_optimization.json').write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
        print(json.dumps(result, indent=2, default=str)); return
    if args.command == 'master-tune':
        from .master_tuner import master_tune
        result = master_tune(df, s, trials=args.trials, final_holdout_frac=args.holdout_frac, save_path='logs/master_tuning_report.json')
        print(json.dumps(result, indent=2, default=str)); return
    if args.command == 'auto-update':
        bundle_dir=asset_model_dir(s.symbol)
        print(json.dumps(auto_update(df,s,model_dir=bundle_dir),indent=2,default=str)); return
    for _ in range(max(1,args.iterations)): print(json.dumps(one_iteration(s),indent=2))


if __name__ == '__main__': main()
