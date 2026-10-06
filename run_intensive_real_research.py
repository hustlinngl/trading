from __future__ import annotations

"""Intensive, research-first real-market training harness.

Downloads immutable Binance Vision monthly archives, validates/fingerprints data,
runs purged walk-forward research, bounded master tuning, a pristine chronological
holdout and 1x/1.5x/2x/3x cost stress. It never promotes a model.
"""

import argparse
import copy
import sys
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from ai_trading_lab.binance_vision import download_range, merge_archives
from ai_trading_lab.config import load_settings
from ai_trading_lab.data_quality import audit_market_data
from ai_trading_lab.engine import AdaptiveEngine
from ai_trading_lab.features import make_oos_features
from ai_trading_lab.fingerprint import strong_dataset_fingerprint
from ai_trading_lab.master_tuner import master_tune
from ai_trading_lab.policy import make_actions
from ai_trading_lab.evaluation import run_configured_backtest
from ai_trading_lab.objectives import robust_performance_utility
from ai_trading_lab.research_gates import score_asset_evidence


def _clone_settings(settings):
    return copy.deepcopy(settings)


def _stress(settings, multiplier: float):
    s = _clone_settings(settings)
    for name in ("fee_bps", "fee_bps_roundtrip", "slippage_bps", "impact_bps",
                 "market_impact_bps", "min_edge_after_cost_bps",
                 "borrow_bps_per_bar", "short_borrow_bps_per_bar"):
        if hasattr(s, name):
            try:
                setattr(s, name, float(getattr(s, name)) * float(multiplier))
            except (TypeError, ValueError):
                pass
    return s


def _holdout_eval(df: pd.DataFrame, settings, holdout_frac: float):
    split = int(len(df) * (1.0 - holdout_frac))
    train_df, holdout = df.iloc[:split].copy(), df.iloc[split:].copy()
    engine = AdaptiveEngine(settings)
    engine.fit(train_df)
    feat = make_oos_features(
        train_df, holdout, settings.horizon_bars,
        external_feature_lag_bars=getattr(settings, "external_feature_lag_bars", 1),
    )
    actions = make_actions(engine, feat, settings)
    bt = holdout.copy()
    if "atr_14" in feat:
        bt["atr_14"] = feat["atr_14"].reindex(holdout.index).ffill()
    result = run_configured_backtest(bt, actions, settings)
    return dict(result.stats), actions, bt


def _aggregate_bootstrap(fold_returns, fold_dds, seed):
    r = np.asarray([x for x in fold_returns if np.isfinite(x)], dtype=float)
    dd = np.asarray([x for x in fold_dds if np.isfinite(x)], dtype=float)
    out = {
        "fold_return_count": int(len(r)),
        "mean_fold_return": float(r.mean()) if len(r) else 0.0,
        "mean_fold_return_ci95": [0.0, 0.0],
        "prob_mean_fold_return_gt_zero": float(np.mean(r > 0.0)) if len(r) else 0.0,
    }
    if len(r) >= 10:
        rng = np.random.default_rng(seed)
        means = np.empty(3000)
        for i in range(len(means)):
            idx = rng.integers(0, len(r), len(r))
            means[i] = float(r[idx].mean())
        out["mean_fold_return_ci95"] = [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ]
    if len(dd) >= 10:
        out["fold_drawdown_mean"] = float(dd.mean())
    return out


def run_symbol(symbol, args, base_settings):
    settings = _clone_settings(base_settings)
    settings.symbol = symbol
    settings.timeframe = args.timeframe
    if args.horizon_bars is not None:
        settings.horizon_bars = int(args.horizon_bars)
    if args.min_train_rows is not None:
        settings.min_train_rows = int(args.min_train_rows)

    raw_dir = Path(args.data_dir) / "raw" / symbol.replace("/", "_") / args.timeframe
    merged_path = Path(args.data_dir) / "merged" / (
        f"{symbol.replace('/', '_')}_{args.timeframe}_{args.start}_{args.end}_{args.market}.parquet"
    )
    paths = download_range(
        symbol, args.timeframe, args.start, args.end, args.market, raw_dir,
        timeout=args.timeout, verify_checksum=not args.skip_checksum,
    )
    df = merge_archives(paths, None)
    if df.empty:
        raise RuntimeError(f"No real market data retrieved for {symbol}")
    if not isinstance(df.index, pd.DatetimeIndex) and "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df = clip_history(df, args.start, args.end)
    if df.empty:
        raise RuntimeError(f"Requested date range returned no completed rows for {symbol}")
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(merged_path, index=True)
    except Exception:
        df.to_csv(merged_path.with_suffix('.csv'))

    quality = audit_market_data(df, settings.timeframe)
    if not quality.passed:
        raise RuntimeError(f"Data quality failed for {symbol}: {quality.reasons}")

    tuning = master_tune(
        df, settings, trials=args.trials, final_holdout_frac=args.holdout_frac,
        save_path=Path(args.log_dir) /
        f"master_tuning_{symbol.replace('/', '_')}_{args.timeframe}.json",
    )

    # The tuner owns challenger selection and evaluates the tuned challenger on the
    # pristine final holdout. Keep a separate baseline holdout only for comparison.
    baseline_holdout_stats, _, _ = _holdout_eval(df, settings, args.holdout_frac)
    tuned_holdout_rows = tuning.get("final_holdout_summary") or []
    tuned_holdout_stats = {}
    for row in tuned_holdout_rows:
        if abs(float(row.get("cost_multiplier", 1.0)) - 1.0) < 1e-9:
            tuned_holdout_stats = dict(row)
            break
    if not tuned_holdout_stats:
        tuned_holdout_stats = dict(tuning.get("final_holdout", {}))
    stress = {
        str(row.get("cost_multiplier")): row
        for row in tuned_holdout_rows
        if "cost_multiplier" in row
    }

    wf_records = [x for x in (tuning.get("full_tuning_fold_rows") or []) if abs(float(x.get("cost_multiplier", 1.0)) - 1.0) < 1e-9]
    fold_returns = [float(x.get("total_return", 0.0)) for x in wf_records]
    fold_dds = [float(x.get("max_drawdown", 0.0)) for x in wf_records]
    evidence = _aggregate_bootstrap(fold_returns, fold_dds, int(settings.seed))
    evidence["holdout_utility"] = robust_performance_utility(
        tuned_holdout_stats,
        tuned_holdout_stats,
        min_trades=int(getattr(settings, "base_min_holdout_trades", 20)),
        max_drawdown=float(getattr(settings, "base_max_holdout_drawdown", -0.25)),
    )

    result = {
        "symbol": symbol,
        "timeframe": args.timeframe,
        "market": args.market,
        "rows": int(len(df)),
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "data_fingerprint": strong_dataset_fingerprint(df),
        "archive_provenance": archive_provenance(paths),
        "quality": asdict(quality),
        "walk_forward": wf_records,
        "walk_forward_summary": {
            "folds": int(len(wf_records)),
            "median_return": float(np.median([float(x.get("total_return", 0.0)) for x in wf_records])) if wf_records else 0.0,
            "positive_folds": int(sum(float(x.get("total_return", 0.0)) > 0 for x in wf_records)),
            "median_robust_score": float(np.median([float(x.get("robust_score", 0.0)) for x in wf_records])) if wf_records else 0.0,
            "worst_drawdown": float(min([float(x.get("max_drawdown", 0.0)) for x in wf_records])) if wf_records else 0.0,
            "trades": int(sum(int(x.get("trades", 0)) for x in wf_records)),
        },
        "master_tuning": {
            "best_params": tuning.get("best_params"),
            "study_best_value": tuning.get("study_best_value"),
            "stability": tuning.get("stability"),
            "negative_control_placebo": tuning.get("negative_control_placebo"),
            "full_tuning_set_verification": tuning.get("full_tuning_set_verification"),
            "final_statistical_evidence": tuning.get("final_statistical_evidence"),
        },
        "holdout_baseline": baseline_holdout_stats,
        "holdout_tuned": tuned_holdout_stats,
        "holdout_data_rows": int(tuning.get("data", {}).get("rows_final_holdout", 0)),
        "cost_stress_tuned": stress,
        "bootstrap_evidence": evidence,
        "promotion": "NOT_PERFORMED",
    }
    result["evidence"] = score_asset_evidence(result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="BTC/USDT,ETH/USDT,SOL/USDT")
    p.add_argument("--timeframe", default="15m")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat())
    p.add_argument("--market", default="spot", choices=["spot", "futures-um"])
    p.add_argument("--trials", type=int, default=60)
    p.add_argument("--holdout-frac", type=float, default=0.15)
    p.add_argument("--horizon-bars", type=int, default=None)
    p.add_argument("--min-train-rows", type=int, default=None)
    p.add_argument("--data-dir", default="data/intensive_real")
    p.add_argument("--log-dir", default="logs/intensive_real")
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--skip-checksum", action="store_true")
    p.add_argument("--cost-multipliers", default="1,1.5,2,3")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--require-all-assets", action="store_true")
    args = p.parse_args()
    args.cost_multipliers = [float(x.strip()) for x in args.cost_multipliers.split(",") if x.strip()]
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    settings = load_settings(args.config)
    reports = []
    for symbol in dict.fromkeys(x.strip() for x in args.symbols.split(",") if x.strip()):
        try:
            print(f"=== REAL RESEARCH {symbol} {args.timeframe} ===", flush=True)
            reports.append(run_symbol(symbol, args, settings))
        except Exception as exc:
            reports.append({"symbol": symbol, "status": "ERROR", "error": str(exc)})

    ok = [x for x in reports if x.get("rows", 0) > 0 and x.get("status") != "ERROR"]
    summary = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "symbols": [x.strip() for x in args.symbols.split(",") if x.strip()],
            "timeframe": args.timeframe,
            "start": args.start,
            "end": args.end,
            "market": args.market,
            "trials": args.trials,
            "holdout_frac": args.holdout_frac,
            "cost_multipliers": args.cost_multipliers,
        },
        "assets_completed": len(ok),
        "assets_failed": len(reports) - len(ok),
        "reports": reports,
        "deployment": "DISABLED",
    }
    out = Path(args.log_dir) / "intensive_real_research_summary.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    if args.require_all_assets and len(ok) != len(reports):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
