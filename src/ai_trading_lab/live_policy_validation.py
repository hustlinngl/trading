from __future__ import annotations

"""Out-of-sample evaluation of the exact combined live signal gate.

This module deliberately evaluates only fixed, predeclared gate variants. It never
selects a threshold on the final holdout and never turns missing evidence into a pass.
"""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .deployment import deployment_semantics_fingerprint, model_semantics_fingerprint
from .evaluation import run_configured_backtest
from .fingerprint import strong_dataset_fingerprint
from .policy import live_signal_gate
from .statistical_evidence import combinatorial_pbo, deflated_sharpe_ratio


def _actions_for_gate(predictions: pd.DataFrame, specialist: pd.DataFrame, settings) -> pd.Series:
    rows = predictions.join(specialist, how="left", rsuffix="_specialist")
    actions = []
    market_type = str(getattr(settings, "backtest_market_type", "spot") or "spot").lower()
    derivative = market_type in {"swap", "future", "perpetual"}
    for _, row in rows.iterrows():
        item = row.to_dict()
        item.setdefault("market_type", market_type)
        # Historical funding is part of the execution contract for derivatives.
        # Do not treat absent funding evidence as a zero-cost trade.
        if derivative and bool(getattr(settings, "require_funding_data_for_derivatives", True)):
            funding_col = str(getattr(settings, "funding_rate_column", "funding_rate"))
            if funding_col not in row.index or not np.isfinite(pd.to_numeric(row.get(funding_col), errors="coerce")):
                item["funding_data_missing"] = True
        item.setdefault("funding_cost_return", 0.0)
        action, _ = live_signal_gate(item, settings)
        actions.append(action)
    return pd.Series(actions, index=rows.index, dtype="object", name="live_policy_action")


def _stats_for_actions(market: pd.DataFrame, actions: pd.Series, settings) -> dict:
    return run_configured_backtest(market, actions, settings).stats


def build_live_policy_holdout_report(
    *,
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    holdout_predictions: pd.DataFrame,
    settings,
    specialist_path: str | Path,
    output_path: str | Path,
) -> dict:
    """Score the combined live gate on a final temporal holdout and persist provenance."""
    from .trade_window import score_trade_window_frame

    full_df = full_df.sort_index()
    train_df = train_df.sort_index()
    holdout_df = holdout_df.sort_index()
    # atr_14 is attached only for execution simulation; the persisted dataset
    # fingerprints must match the raw OHLCV provenance used by the base report.
    full_fingerprint_df = full_df.drop(columns=["atr_14"], errors="ignore")
    holdout_fingerprint_df = holdout_df.drop(columns=["atr_14"], errors="ignore")
    if holdout_df.empty or not holdout_predictions.index.equals(holdout_df.index):
        raise ValueError("live_policy_holdout_prediction_index_mismatch")
    if not train_df.index.max() < holdout_df.index.min():
        raise ValueError("live_policy_holdout_not_temporally_pristine")

    specialist_all = score_trade_window_frame(
        full_df, settings, model_path=specialist_path, symbol=str(settings.symbol)
    )
    specialist_holdout = specialist_all.reindex(holdout_df.index)
    market = holdout_df.copy()
    if "atr_14" not in market:
        raise ValueError("live_policy_holdout_missing_atr")
    market_type = str(getattr(settings, "backtest_market_type", "spot") or "spot").lower()
    funding_col = str(getattr(settings, "funding_rate_column", "funding_rate"))
    if market_type in {"swap", "future", "perpetual"} and bool(getattr(settings, "require_funding_data_for_derivatives", True)):
        if funding_col not in market or not pd.to_numeric(market[funding_col], errors="coerce").notna().any():
            raise ValueError("live_policy_holdout_missing_historical_funding")

    # Predeclared strictness variants are used only for robustness diagnostics;
    # the configured policy (variant 0) is never selected using holdout performance.
    variants = []
    for confidence_floor, edge_multiplier in (
        (None, 1.0),
        (0.86, 1.25),
        (0.90, 1.50),
    ):
        variant = copy.copy(settings)
        if confidence_floor is not None:
            variant.signal_confidence_threshold = max(
                float(getattr(settings, "signal_confidence_threshold", 0.82)),
                confidence_floor,
            )
            variant.signal_probability_threshold = max(
                float(getattr(settings, "signal_probability_threshold", 0.72)),
                confidence_floor,
            )
        variant.signal_min_expected_return = (
            float(getattr(settings, "signal_min_expected_return", 0.003)) * edge_multiplier
        )
        variants.append(variant)

    actions_by_variant = [
        _actions_for_gate(holdout_predictions, specialist_holdout, variant)
        for variant in variants
    ]
    backtest_results = [
        run_configured_backtest(market, actions, settings)
        for actions in actions_by_variant
    ]
    results = [result.stats for result in backtest_results]
    primary_result = backtest_results[0]
    primary = results[0]
    primary_actions = actions_by_variant[0]
    min_trades = int(getattr(settings, "base_min_holdout_trades", 20))
    primary_trade_count = int(primary.get("trades", primary.get("trades_taken", 0)) or 0)

    # Fold diagnostics use contiguous, non-overlapping segments of the untouched
    # holdout. A fold without trades is retained as a zero-return observation.
    fold_returns = []
    fold_trades = []
    score_matrix = []
    fold_indices = np.array_split(np.arange(len(market)), 6)
    for indices in fold_indices:
        if not len(indices):
            continue
        segment = market.iloc[indices]
        row_stats = []
        for actions in actions_by_variant:
            segment_actions = actions.reindex(segment.index).fillna("FLAT")
            row_stats.append(_stats_for_actions(segment, segment_actions, settings))
        score_matrix.append([
            float(stats.get("total_return", 0.0)) for stats in row_stats
        ])
        primary_stats = row_stats[0]
        fold_returns.append(float(primary_stats.get("total_return", 0.0)))
        fold_trades.append(int(primary_stats.get("trades", primary_stats.get("trades_taken", 0)) or 0))

    trade_returns = np.asarray([], dtype=float)
    # Per-trade net return is normalized by entry equity, keeping the DSR input
    # in consistent per-observation units rather than mixing annualized Sharpe.
    if not primary_result.trades.empty:
        trade_returns = (
            pd.to_numeric(primary_result.trades["pnl_net"], errors="coerce")
            / pd.to_numeric(primary_result.trades["entry_equity"], errors="coerce").replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)

    active_folds = sum(1 for count in fold_trades if count > 0)
    positive_fold_fraction = (
        float(np.mean(np.asarray(fold_returns) > 0.0)) if fold_returns else 0.0
    )
    stability_available = len(fold_returns) >= 6 and active_folds >= int(
        getattr(settings, "master_optimization_folds", 6)
    )
    stability = {
        "available": bool(stability_available),
        "stable": bool(
            stability_available
            and positive_fold_fraction >= 2.0 / 3.0
            and all(np.isfinite(value) for value in fold_returns)
        ),
        "positive_fold_fraction": positive_fold_fraction,
        "active_folds": int(active_folds),
        "fold_returns": fold_returns,
        "fold_trades": fold_trades,
    }

    pbo = combinatorial_pbo(np.asarray(score_matrix, dtype=float), partitions=6, seed=int(settings.seed))
    from .research_ledger import total_trials
    ledger_path = Path(getattr(settings, "research_ledger_path", "data/research_ledger.json"))
    prior_trial_count = total_trials(ledger_path)
    effective_trial_count = max(
        3,
        int(getattr(settings, "master_trials", 3)),
        prior_trial_count + len(variants),
    )
    dsr_available = bool(
        len(trade_returns) >= 30
        and np.isfinite(trade_returns).all()
        and float(np.std(trade_returns, ddof=1)) > 1e-12
    )
    dsr = (
        deflated_sharpe_ratio(
            trade_returns,
            trials=effective_trial_count,
            periods_per_year=365.0,
        )
        if dsr_available
        else None
    )

    # Negative control: circularly shift the already-gated signals by random,
    # non-zero offsets and rerun the same costed execution engine. This preserves
    # signal count while breaking its timestamp alignment to subsequent prices.
    placebo_returns = []
    if len(primary_actions) >= 30 and primary_trade_count >= min_trades:
        rng = np.random.default_rng(int(settings.seed))
        values = primary_actions.to_numpy(dtype=object)
        for _ in range(100):
            offset = int(rng.integers(1, len(values)))
            placebo_actions = pd.Series(np.roll(values, offset), index=primary_actions.index)
            placebo_stats = _stats_for_actions(market, placebo_actions, settings)
            placebo_returns.append(float(placebo_stats.get("total_return", 0.0)))
    placebo = {
        "available": bool(len(placebo_returns) >= 100 and primary_trade_count >= min_trades),
        "placebo_samples": int(len(placebo_returns)),
        "observed_return": float(primary.get("total_return", -1.0)),
        "placebo_p95_return": float(np.quantile(placebo_returns, 0.95)) if placebo_returns else None,
        "passes_95pct_negative_control": bool(
            len(placebo_returns) >= 100
            and primary_trade_count >= min_trades
            and float(primary.get("total_return", -1.0)) > float(np.quantile(placebo_returns, 0.95))
        ),
    }

    base_report_path = Path(output_path).with_name("base_holdout_report.json")
    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    # Reuse the same overlap-aware governance ledger as research/tuning. A changed
    # full-data fingerprint does not make an overlapping holdout pristine again.
    from .research_ledger import register_holdout_access
    scope = (
        f"{getattr(settings, 'symbol', 'UNKNOWN')}|"
        f"{getattr(settings, 'timeframe', 'UNKNOWN')}|"
        f"{getattr(settings, 'model_name', 'adaptive_signal')}"
    )
    holdout_access = register_holdout_access(
        ledger_path,
        dataset_fingerprint=strong_dataset_fingerprint(full_fingerprint_df),
        holdout_start=str(holdout_df.index.min()),
        holdout_end=str(holdout_df.index.max()),
        holdout_frac=float(len(holdout_df) / max(1, len(full_df))),
        purpose="combined_live_policy_holdout_validation",
        scope=scope,
        metadata={
            "symbol": str(settings.symbol),
            "timeframe": str(settings.timeframe),
            "holdout_fingerprint": strong_dataset_fingerprint(holdout_fingerprint_df),
        },
    )
    action_counts = {
        action: int((primary_actions == action).sum())
        for action in ("LONG", "SHORT", "FLAT")
    }
    action_total = max(1, int(len(primary_actions)))
    report = {
        "schema_version": 1,
        "symbol": str(settings.symbol),
        "signal_gate_diagnostics": {
            "rows": int(len(primary_actions)),
            "action_counts": action_counts,
            "action_fractions": {
                action: float(count / action_total)
                for action, count in action_counts.items()
            },
            "configured_policy_trade_count": primary_trade_count,
            "minimum_required_trades": min_trades,
        },
        "timeframe": str(settings.timeframe),
        "data_fingerprint": strong_dataset_fingerprint(full_fingerprint_df),
        "validation_train_data_fingerprint": strong_dataset_fingerprint(train_df),
        "validation_holdout_data_fingerprint": strong_dataset_fingerprint(holdout_fingerprint_df),
        "holdout_start": str(holdout_df.index.min()),
        "holdout_end": str(holdout_df.index.max()),
        "model_semantics_fingerprint": model_semantics_fingerprint(settings),
        "deployment_semantics_fingerprint": deployment_semantics_fingerprint(settings),
        "live_signal_gate_applied": True,
        "specialist_gate_applied": bool(getattr(settings, "trade_window_required_for_signal", True)),
        "holdout_governance": {
            "pristine": bool(holdout_access.get("pristine", False)),
            "holdout_access_key": holdout_access.get("holdout_key"),
            "overlapping_prior_runs": holdout_access.get("overlapping_prior_runs", []),
            "temporal_order_verified": True,
            "thresholds_selected_on_holdout": False,
            "execution_costs_included": True,
            "execution_engine": "run_configured_backtest",
            "predeclared_gate_variants": 3,
        },
        "holdout": primary,
        "statistical_evidence": {
            "deflated_sharpe_ratio": float(dsr) if dsr is not None else None,
            "deflated_sharpe_ratio_available": dsr_available,
            "deflated_sharpe_trial_count": int(effective_trial_count),
            "prior_research_trial_count": int(prior_trial_count),
            "probability_of_backtest_overfitting": pbo,
            "stability": stability,
            "negative_control": placebo,
            "trade_return_observations": int(len(trade_returns)),
            "predeclared_candidate_count": len(variants),
        },
        "candidate_fold_count": int(active_folds),
        "variant_diagnostics": [
            {"variant": i, "stats": stats}
            for i, stats in enumerate(results)
        ],
        "base_holdout_data_fingerprint": str(base_report.get("validation_holdout_data_fingerprint", "")),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    temporary.replace(target)
    return report
