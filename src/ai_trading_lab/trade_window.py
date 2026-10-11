from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from .deployment import deployment_semantics_fingerprint, model_semantics_fingerprint
from .features import make_features
from .fingerprint import strong_dataset_fingerprint


def timeframe_minutes(timeframe):
    tf = str(timeframe).lower()
    return {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}.get(tf, 15)


def _atr(df, period=14):
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _barrier_labels(df, horizon_bars=96, min_bars=12, pt_atr=1.25, sl_atr=0.90):
    """Return event labels and the matching executable long-side price path.

    Direction is +1/-1 only for a first barrier event after the configured
    minimum duration; earlier exits and time-stops are class 0. Gross return,
    exit_position and holding_bars describe the actual executable path whether
    or not that exit qualifies as a 3–24h event. OHLC collisions with unknown
    intrabar order remain non-executable and have no fabricated return.

    The barrier scan is vectorized across decision rows for each holding bar.
    This preserves the per-row event priority while avoiding the previous
    Python-level loop over every row *and* every bar (O(rows * horizon) Python
    iterations), which made full-universe historical training unnecessarily slow.
    """
    horizon = int(horizon_bars)
    minimum = int(min_bars)
    pt = float(pt_atr)
    sl = float(sl_atr)
    if horizon < 1:
        raise ValueError("horizon_bars must be >= 1")
    if minimum < 1 or minimum > horizon:
        raise ValueError("min_bars must be between 1 and horizon_bars")
    if not np.isfinite(pt) or pt <= 0 or not np.isfinite(sl) or sl <= 0:
        raise ValueError("barrier ATR multipliers must be finite and > 0")

    x = df.sort_index().copy()
    opens = pd.to_numeric(x["open"], errors="coerce").to_numpy(float)
    highs = pd.to_numeric(x["high"], errors="coerce").to_numpy(float)
    lows = pd.to_numeric(x["low"], errors="coerce").to_numpy(float)
    atr = _atr(x).to_numpy(float)
    size = len(x)
    directions = np.full(size, np.nan, dtype=float)
    gross_returns = np.full(size, np.nan, dtype=float)
    exit_positions = np.full(size, np.nan, dtype=float)
    holding_bars = np.full(size, np.nan, dtype=float)
    execution_status = np.full(size, "unresolved", dtype=object)

    # The time-stop exits at open[i + horizon + 1]. Inspect only the horizon
    # holding candles: i+1 through i+horizon, not the exit candle itself.
    if size <= horizon + 1:
        return pd.DataFrame(
            {
                "direction": directions,
                "margin": np.abs(directions),
                "gross_return": gross_returns,
                "exit_position": exit_positions,
                "holding_bars": holding_bars,
                "execution_status": execution_status,
            },
            index=x.index,
        )

    decisions = np.arange(size - horizon - 1, dtype=int)
    entry_positions = decisions + 1
    time_stop_positions = decisions + horizon + 1
    entries = opens[entry_positions]
    time_stop_prices = opens[time_stop_positions]
    valid_initial = (
        np.isfinite(atr[decisions])
        & (atr[decisions] > 0)
        & np.isfinite(entries)
        & (entries > 0)
        & np.isfinite(time_stop_prices)
        & (time_stop_prices > 0)
    )
    active = decisions[valid_initial]
    if active.size:
        active_entries = opens[active + 1]
        upper = active_entries + pt * atr[active]
        lower = active_entries - sl * atr[active]
        directions[active] = 0.0
        gross_returns[active] = opens[active + horizon + 1] / active_entries - 1.0
        exit_positions[active] = (active + horizon + 1).astype(float)
        holding_bars[active] = float(horizon)
        execution_status[active] = "time_stop"

        for held in range(horizon):
            if active.size == 0:
                break
            bar_positions = active + 1 + held
            bar_open = opens[bar_positions]

            invalid_open = ~np.isfinite(bar_open) | (bar_open <= 0)
            if invalid_open.any():
                bad = active[invalid_open]
                directions[bad] = np.nan
                gross_returns[bad] = np.nan
                exit_positions[bad] = np.nan
                holding_bars[bad] = np.nan
                execution_status[bad] = "invalid_ohlc"
            keep = ~invalid_open
            active, bar_positions = active[keep], bar_positions[keep]
            if active.size == 0:
                break
            bar_open = opens[bar_positions]
            active_upper, active_lower = upper[keep], lower[keep]
            active_entries = opens[active + 1]

            # Opening gaps take precedence over intrabar high/low checks.
            gap_up = bar_open >= active_upper
            gap_down = (~gap_up) & (bar_open <= active_lower)
            gap = gap_up | gap_down
            if gap.any():
                idx = active[gap]
                directions[idx] = np.where(held >= minimum, np.where(gap_up[gap], 1.0, -1.0), 0.0)
                gross_returns[idx] = bar_open[gap] / active_entries[gap] - 1.0
                exit_positions[idx] = bar_positions[gap].astype(float)
                holding_bars[idx] = float(max(1, held + 1))
                execution_status[idx] = np.where(gap_up[gap], "opening_gap_take_profit", "opening_gap_stop_loss")
            keep = ~gap
            active, bar_positions = active[keep], bar_positions[keep]
            active_upper, active_lower = active_upper[keep], active_lower[keep]
            active_entries = active_entries[keep]
            if active.size == 0:
                break

            bar_high = highs[bar_positions]
            bar_low = lows[bar_positions]
            invalid_range = ~np.isfinite(bar_high) | ~np.isfinite(bar_low)
            if invalid_range.any():
                bad = active[invalid_range]
                directions[bad] = np.nan
                gross_returns[bad] = np.nan
                exit_positions[bad] = np.nan
                holding_bars[bad] = np.nan
                execution_status[bad] = "invalid_ohlc"
            keep = ~invalid_range
            active, bar_positions = active[keep], bar_positions[keep]
            active_upper, active_lower = active_upper[keep], active_lower[keep]
            active_entries = active_entries[keep]
            bar_high, bar_low = bar_high[keep], bar_low[keep]
            if active.size == 0:
                break

            hit_up = bar_high >= active_upper
            hit_down = bar_low <= active_lower
            ambiguous = hit_up & hit_down
            if ambiguous.any():
                idx = active[ambiguous]
                directions[idx] = 0.0
                gross_returns[idx] = np.nan
                exit_positions[idx] = bar_positions[ambiguous].astype(float)
                holding_bars[idx] = float(max(1, held + 1))
                execution_status[idx] = "ambiguous_intrabar"

            resolved = ambiguous | hit_up | hit_down
            unambiguous_up = hit_up & ~hit_down
            unambiguous_down = hit_down & ~hit_up
            if unambiguous_up.any():
                idx = active[unambiguous_up]
                directions[idx] = 1.0 if held >= minimum else 0.0
                gross_returns[idx] = active_upper[unambiguous_up] / active_entries[unambiguous_up] - 1.0
                exit_positions[idx] = bar_positions[unambiguous_up].astype(float)
                holding_bars[idx] = float(max(1, held + 1))
                execution_status[idx] = "take_profit"
            if unambiguous_down.any():
                idx = active[unambiguous_down]
                directions[idx] = -1.0 if held >= minimum else 0.0
                gross_returns[idx] = active_lower[unambiguous_down] / active_entries[unambiguous_down] - 1.0
                exit_positions[idx] = bar_positions[unambiguous_down].astype(float)
                holding_bars[idx] = float(max(1, held + 1))
                execution_status[idx] = "stop_loss"
            active = active[~resolved]

    return pd.DataFrame(
        {
            "direction": directions,
            "margin": np.abs(directions),
            "gross_return": gross_returns,
            "exit_position": exit_positions,
            "holding_bars": holding_bars,
            "execution_status": execution_status,
        },
        index=x.index,
    )

def _fill_feature_frame(frame, feature_columns, fill_values):
    numeric = frame.reindex(columns=feature_columns).replace([np.inf, -np.inf], np.nan)
    return numeric.fillna(fill_values.reindex(feature_columns)).fillna(0.0)


def _wilson_lower_bound(successes, total, z=1.6448536269514722):
    n = int(total)
    if n <= 0:
        return 0.0
    phat = float(successes) / n
    denom = 1.0 + (z * z) / n
    centre = phat + (z * z) / (2 * n)
    radius = z * np.sqrt(max(0.0, phat * (1.0 - phat) / n + (z * z) / (4 * n * n)))
    return float((centre - radius) / denom)


def _event_score_arrays(probabilities, classes):
    """Map a 3-class (-1, 0, +1) model into event and conditional direction scores."""
    proba = np.asarray(probabilities, dtype=float)
    if proba.ndim != 2:
        raise ValueError("trade_window_probability_shape_mismatch")
    class_values = [int(value) for value in np.asarray(classes).tolist()]
    positions = {value: index for index, value in enumerate(class_values)}
    if not {-1, 0, 1}.issubset(positions):
        raise ValueError("trade_window_class_contract_mismatch")
    if proba.shape[1] != len(class_values) or not np.isfinite(proba).all() or (proba < 0).any():
        raise ValueError("trade_window_invalid_probabilities")

    p_short = proba[:, positions[-1]]
    p_neutral = proba[:, positions[0]]
    p_long = proba[:, positions[1]]
    p_event = p_short + p_long
    p_direction = np.divide(
        np.maximum(p_short, p_long),
        p_event,
        out=np.zeros_like(p_event),
        where=p_event > 1e-12,
    )
    direction = np.where(p_long >= p_short, 1, -1)
    return p_short, p_neutral, p_long, p_event, p_direction, direction



BALANCED_PROBABILITY_SEMANTICS = "balanced_class_weight_prior_corrected_v1"


def _correct_balanced_class_probabilities(probabilities, classes, class_counts):
    """Undo inverse-frequency class weighting before treating scores as event probabilities.

    class_weight='balanced' changes the class prior seen by the estimator. For
    probabilities in the original population, reweight each class score by its
    observed training count and renormalize. This prior correction is not a
    substitute for out-of-sample calibration; it prevents weighted scores from
    being mislabeled as natural event probabilities.
    """
    proba = np.asarray(probabilities, dtype=float)
    labels = [int(value) for value in np.asarray(classes).tolist()]
    if proba.ndim != 2 or proba.shape[1] != len(labels):
        raise ValueError("trade_window_probability_shape_mismatch")
    if not np.isfinite(proba).all() or (proba < 0).any():
        raise ValueError("trade_window_invalid_probabilities")

    if not isinstance(class_counts, dict):
        raise ValueError("trade_window_class_prior_metadata_missing")
    counts = []
    for label in labels:
        value = class_counts.get(str(label), class_counts.get(label))
        try:
            count = float(value)
        except (TypeError, ValueError):
            raise ValueError("trade_window_class_prior_metadata_missing") from None
        if not np.isfinite(count) or count <= 0:
            raise ValueError("trade_window_class_prior_metadata_invalid")
        counts.append(count)

    totals = proba.sum(axis=1, keepdims=True)
    if np.any(totals <= 1e-12):
        raise ValueError("trade_window_invalid_probabilities")
    normalized = proba / totals
    corrected = normalized * np.asarray(counts, dtype=float)[None, :]
    corrected_totals = corrected.sum(axis=1, keepdims=True)
    if np.any(corrected_totals <= 1e-12):
        raise ValueError("trade_window_prior_correction_failed")
    return corrected / corrected_totals

def directional_event_probabilities(probabilities, classes, min_event_probability=0.60):
    """Return conditional direction only when the qualifying event is sufficiently likely."""
    threshold = float(min_event_probability)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("trade_window_min_event_probability_invalid")
    short, neutral, long, event, confidence, direction = _event_score_arrays(
        np.asarray(probabilities, dtype=float).reshape(1, -1),
        classes,
    )
    p_event = float(event[0])
    predicted_direction = (
        ("LONG" if int(direction[0]) > 0 else "SHORT")
        if p_event >= threshold and p_event > 1e-12
        else "FLAT"
    )
    return {
        "p_short": float(short[0]),
        "p_neutral": float(neutral[0]),
        "p_long": float(long[0]),
        "event_probability": p_event,
        "direction": predicted_direction,
        "direction_confidence": float(confidence[0]),
    }


def _not_ready_report(reason, *, save_path=None, **details):
    # Remove a prior valid-looking artifact if the current training run cannot
    # produce a compatible model. A stale .joblib must never outlive its report.
    if save_path is not None:
        try:
            Path(save_path).unlink(missing_ok=True)
        except OSError:
            pass
    return {"production_ready": False, "reason": str(reason), **details}


def train_trade_window_backbone(df, settings, holdout_frac=0.15, save_path=None):
    df = df.sort_index().copy()
    if df.index.has_duplicates:
        return _not_ready_report("duplicate_timestamps", save_path=save_path)
    base_minutes = float(
        getattr(settings, "base_bar_minutes", timeframe_minutes(getattr(settings, "timeframe", "15m")))
    )
    min_hours = float(getattr(settings, "trade_window_min_hours", 3.0))
    max_hours = float(getattr(settings, "trade_window_max_hours", 24.0))
    min_bars = max(1, int(round(min_hours * 60 / base_minutes)))
    max_bars = max(min_bars, int(round(max_hours * 60 / base_minutes)))
    if not 0.05 <= float(holdout_frac) <= 0.50:
        return _not_ready_report("invalid_holdout_fraction", save_path=save_path, holdout_frac=float(holdout_frac))

    x, _, _ = make_features(
        df,
        max_bars,
        external_feature_lag_bars=getattr(settings, "external_feature_lag_bars", 1),
    )
    labels = _barrier_labels(
        df,
        max_bars,
        min_bars,
        pt_atr=float(getattr(settings, "trade_window_pt_atr", 1.25)),
        sl_atr=float(getattr(settings, "trade_window_sl_atr", 0.90)),
    )
    common = x.index.intersection(labels.index)
    x = x.loc[common]
    labels = labels.loc[common]
    feature_columns = [
        column for column in x.columns
        if pd.api.types.is_numeric_dtype(x[column])
    ]
    x = x.reindex(columns=feature_columns).replace([np.inf, -np.inf], np.nan)
    valid = labels["direction"].notna()
    x = x.loc[valid]
    y = labels.loc[valid, "direction"].astype(int)
    if len(x) < 80:
        return _not_ready_report("insufficient_labeled_rows", save_path=save_path, rows=int(len(x)))

    original_positions = df.index.get_indexer(x.index)
    split_pos = int(len(df) * (1.0 - float(holdout_frac)))
    # Label lookahead reaches open[pos + max_bars + 1], which must remain strictly
    # before the first holdout row. Purge that boundary bar as well.
    train_mask = (original_positions >= 0) & ((original_positions + max_bars + 1) < split_pos)
    holdout_mask = (original_positions >= split_pos)
    train_x_raw = x.loc[train_mask]
    train_y = y.loc[train_mask]
    holdout_x_raw = x.loc[holdout_mask]
    holdout_y = y.loc[holdout_mask]
    if len(train_x_raw) < 40 or len(holdout_x_raw) < 20:
        return _not_ready_report(
            "insufficient_purged_split",
            save_path=save_path,
            rows=int(len(x)),
            train_rows=int(len(train_x_raw)),
            holdout_rows=int(len(holdout_x_raw)),
        )

    # Impute with statistics from the chronological training subset only and
    # persist the exact transform used by live inference.
    fill_values = (
        train_x_raw.median(numeric_only=True)
        .reindex(feature_columns)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    train_x = _fill_feature_frame(train_x_raw, feature_columns, fill_values)
    holdout_x = _fill_feature_frame(holdout_x_raw, feature_columns, fill_values)

    class_counts = {str(label): int((train_y == label).sum()) for label in (-1, 0, 1)}
    if any(count == 0 for count in class_counts.values()):
        return _not_ready_report(
            "insufficient_three_class_training_support",
            save_path=save_path,
            rows=int(len(x)),
            train_rows=int(len(train_x)),
            holdout_rows=int(len(holdout_x)),
            train_class_counts=class_counts,
        )

    model = ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=12,
        max_features="sqrt",
        random_state=int(settings.seed),
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(train_x, train_y)
    weighted_proba = model.predict_proba(holdout_x)
    try:
        proba = _correct_balanced_class_probabilities(
            weighted_proba, model.classes_, class_counts
        )
        p_short, p_neutral, p_long, p_event, p_direction, pred_direction = _event_score_arrays(
            proba, model.classes_
        )
    except ValueError as exc:
        return _not_ready_report(
            str(exc),
            save_path=save_path,
            rows=int(len(x)),
            train_rows=int(len(train_x)),
            holdout_rows=int(len(holdout_x)),
            train_class_counts=class_counts,
        )

    min_event_probability = float(getattr(settings, "trade_window_min_event_probability", 0.60))
    min_confidence = float(getattr(settings, "trade_window_min_confidence", 0.80))
    candidates = (p_event >= min_event_probability) & (p_direction >= min_confidence)
    truth = holdout_y.to_numpy(dtype=int)
    positions = df.index.get_indexer(holdout_x.index)

    # Use non-overlapping full-horizon outcomes for statistical and economic
    # readiness. Otherwise adjacent 24h assessments masquerade as independent trades.
    accepted = []
    last_exit_pos = -1
    raw_cost_bps = 2.0 * (
        float(getattr(settings, "fee_bps", 0.0)) + float(getattr(settings, "slippage_bps", 0.0))
    )
    raw_cost_bps += 2.0 * float(getattr(settings, "impact_bps_per_sqrt", 0.0)) * np.sqrt(
        float(np.clip(getattr(settings, "max_participation_pct", 0.10), 0.0, 1.0))
    )
    borrow_bps_per_bar = max(0.0, float(getattr(settings, "short_borrow_bps_per_bar", 0.0)))
    require_short_borrow_cost = bool(getattr(settings, "require_short_borrow_cost", True))
    unexecutable_short_candidates = 0

    open_prices = pd.to_numeric(df["open"], errors="coerce").to_numpy(float)
    ambiguous_candidate_exclusions = 0
    unresolved_candidate_exclusions = 0
    for local_i, candidate in enumerate(candidates):
        if not bool(candidate):
            continue
        pos = int(positions[local_i])
        entry_pos = pos + 1
        label_row = labels.loc[holdout_x.index[local_i]]
        execution_status = str(label_row.get("execution_status", "missing_execution_path"))
        try:
            gross_market_return = float(label_row.get("gross_return", np.nan))
            exit_pos_float = float(label_row.get("exit_position", np.nan))
            held_bars = float(label_row.get("holding_bars", np.nan))
        except (TypeError, ValueError, OverflowError):
            gross_market_return = float("nan")
            exit_pos_float = float("nan")
            held_bars = float("nan")
        if execution_status == "ambiguous_intrabar":
            ambiguous_candidate_exclusions += 1
            continue
        if (
            not np.isfinite(gross_market_return)
            or not np.isfinite(exit_pos_float)
            or not np.isfinite(held_bars)
        ):
            unresolved_candidate_exclusions += 1
            continue
        exit_pos = int(exit_pos_float)
        if (
            pos < 0
            or entry_pos >= len(df)
            or exit_pos >= len(df)
            or exit_pos < entry_pos
            or pos <= last_exit_pos
        ):
            continue
        entry = float(open_prices[entry_pos])
        if not np.isfinite(entry) or entry <= 0:
            unresolved_candidate_exclusions += 1
            continue
        side = int(pred_direction[local_i])
        if side < 0 and require_short_borrow_cost and borrow_bps_per_bar <= 0.0:
            # The configured execution model cannot open this short, so do not
            # count it as evidence that the deployable strategy is profitable.
            unexecutable_short_candidates += 1
            continue
        # Use the actual first barrier/gap exit or time-stop return, not a
        # fictive full-horizon open when the position would already be liquidated.
        cost_bps = raw_cost_bps + (
            borrow_bps_per_bar * max(1.0, held_bars) if side < 0 else 0.0
        )
        gross = float(side) * gross_market_return
        net_return = gross - cost_bps / 10_000.0
        accepted.append(
            {
                "position": pos,
                "entry_position": entry_pos,
                "exit_position": exit_pos,
                "holding_bars": float(held_bars),
                "execution_status": execution_status,
                "truth": int(truth[local_i]),
                "prediction": side,
                "event_probability": float(p_event[local_i]),
                "direction_confidence": float(p_direction[local_i]),
                "gross_return": float(gross),
                "estimated_cost_bps": float(cost_bps),
                "net_return": float(net_return),
            }
        )
        last_exit_pos = exit_pos

    support = len(accepted)
    successes = sum(row["prediction"] == row["truth"] for row in accepted)
    precision = float(successes / support) if support else 0.0
    wilson = _wilson_lower_bound(successes, support)
    net_returns = np.asarray([row["net_return"] for row in accepted], dtype=float)
    mean_net = float(net_returns.mean()) if len(net_returns) else 0.0
    compounded_net = (
        float(np.prod(1.0 + net_returns) - 1.0)
        if len(net_returns) and np.all(net_returns > -1.0)
        else -1.0
    )

    target_precision = float(getattr(settings, "trade_window_target_precision", 0.80))
    min_wilson = float(getattr(settings, "trade_window_min_holdout_wilson", 0.60))
    min_net = float(getattr(settings, "trade_window_min_net_return", 0.0005))
    min_trades = int(getattr(settings, "trade_window_min_holdout_trades", 12))
    ready = (
        precision >= target_precision
        and wilson >= min_wilson
        and support >= min_trades
        and mean_net >= min_net
        and (
            compounded_net > 0.0
            if bool(getattr(settings, "trade_window_require_positive_holdout_backtest", True))
            else True
        )
    )
    def _score_distribution(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            return {"n": 0}
        return {
            "n": int(len(values)),
            "min": float(np.min(values)),
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
            "max": float(np.max(values)),
        }

    event_pass = p_event >= min_event_probability
    confidence_pass = p_direction >= min_confidence
    report = {
        "production_ready": bool(ready),
        "holdout_precision": precision,
        "holdout_wilson_lower": wilson,
        "holdout_signals": int(support),
        "holdout_candidates_before_nonoverlap": int(candidates.sum()),
        "holdout_probability_diagnostics": {
            "event_probability": _score_distribution(p_event),
            "conditional_direction_confidence": _score_distribution(p_direction),
            "min_event_probability": min_event_probability,
            "min_direction_confidence": min_confidence,
            "event_probability_gate_passes": int(event_pass.sum()),
            "direction_confidence_gate_passes": int(confidence_pass.sum()),
            "both_probability_gates_pass": int((event_pass & confidence_pass).sum()),
        },
        "holdout_unexecutable_short_candidates": int(unexecutable_short_candidates),
        "holdout_ambiguous_candidate_exclusions": int(ambiguous_candidate_exclusions),
        "holdout_unresolved_candidate_exclusions": int(unresolved_candidate_exclusions),
        "holdout_event_probability_mean": float(p_event.mean()) if len(p_event) else 0.0,
        "holdout_neutral_rate": float(np.mean(truth == 0)) if len(truth) else 0.0,
        "holdout_directional_confidence_mean": float(p_direction.mean()) if len(p_direction) else 0.0,
        "holdout_net_return_mean": mean_net,
        "holdout_net_return_compounded": compounded_net,
        "holdout_economic_observations": int(len(net_returns)),
        "min_event_probability": min_event_probability,
        "min_direction_confidence": min_confidence,
        "target_precision": target_precision,
        "min_wilson": min_wilson,
        "min_net_return": min_net,
        "min_holdout_trades": min_trades,
        "rows": int(len(x)),
        "train_rows": int(len(train_x)),
        "holdout_rows": int(len(holdout_x)),
        "train_class_counts": class_counts,
        "probability_semantics": BALANCED_PROBABILITY_SEMANTICS,
        "holdout_class_counts": {str(label): int((holdout_y == label).sum()) for label in (-1, 0, 1)},
        "min_hours": min_hours,
        "max_hours": max_hours,
        "symbol": str(getattr(settings, "symbol", "")),
        "timeframe": str(getattr(settings, "timeframe", "15m")),
        "data_fingerprint": strong_dataset_fingerprint(df),
        "model_semantics_fingerprint": model_semantics_fingerprint(settings),
        "deployment_semantics_fingerprint": deployment_semantics_fingerprint(settings),
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "feature_columns": list(feature_columns),
                "fill_values": fill_values,
                "report": report,
            },
            path,
        )
    return report


def score_trade_window_frame(
    df, settings, model_path=None, symbol=None, *, allow_unready_for_validation=False
):
    """Score the specialist row-by-row using the same checks as live inference.

    This validation-only override lets the isolated research evaluator measure an
    unpromoted candidate without weakening the default live-inference contract.

    This is used by live inference and chronological policy validation. It does not
    use future labels; features are causal, class-prior correction is identical to
    the live path, and missing/untrained features are rejected before imputation.
    """
    frame = df.sort_index().copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    index = frame.index
    columns = {
        "trade_window_available": False,
        "trade_window_ready": False,
        "trade_window_direction": "FLAT",
        "trade_window_confidence": 0.0,
        "trade_window_event_probability": 0.0,
        "trade_window_reason": "unavailable",
        "trade_window_report_precision": np.nan,
        "trade_window_feature_missing_fraction": 1.0,
    }

    def failed(reason, *, available=False, precision=np.nan):
        result = pd.DataFrame(index=index)
        result["trade_window_available"] = bool(available)
        result["trade_window_ready"] = False
        result["trade_window_direction"] = "FLAT"
        result["trade_window_confidence"] = 0.0
        result["trade_window_event_probability"] = 0.0
        result["trade_window_reason"] = str(reason)
        result["trade_window_report_precision"] = float(precision) if np.isfinite(precision) else np.nan
        result["trade_window_feature_missing_fraction"] = 1.0
        return result

    if not bool(getattr(settings, "trade_window_enabled", True)):
        return failed("disabled")
    path = Path(
        model_path
        or getattr(settings, "trade_window_model_path", "models/champion/trade_window_specialist.joblib")
    )
    if not path.exists():
        return failed("artifact_missing")
    if frame.empty:
        return failed("empty_data")
    if frame.index.has_duplicates:
        return failed("duplicate_timestamps")

    try:
        artifact = joblib.load(path)
        model = artifact["model"]
        feature_columns = list(artifact.get("feature_columns", []))
        fill_values = artifact.get("fill_values")
        report = artifact.get("report", {}) or {}
        if not feature_columns or not isinstance(fill_values, pd.Series):
            return failed("feature_imputation_metadata_missing")
        expected_symbol = str(symbol or getattr(settings, "symbol", ""))
        expected_timeframe = str(getattr(settings, "timeframe", "15m"))
        if str(report.get("symbol", "")) != expected_symbol:
            return failed("symbol_mismatch")
        if str(report.get("timeframe", "")) != expected_timeframe:
            return failed("timeframe_mismatch")
        if str(report.get("model_semantics_fingerprint", "")) != model_semantics_fingerprint(settings):
            return failed("model_semantics_mismatch")
        if str(report.get("deployment_semantics_fingerprint", "")) != deployment_semantics_fingerprint(settings):
            return failed("deployment_semantics_mismatch")
        precision = float(report.get("holdout_precision", 0.0) or 0.0)
        if report.get("probability_semantics") != BALANCED_PROBABILITY_SEMANTICS:
            return failed("probability_semantics_missing", available=True, precision=precision)
        if not bool(report.get("production_ready", False)) and not allow_unready_for_validation:
            return failed("not_production_ready", available=True, precision=precision)

        base_minutes = float(
            getattr(settings, "base_bar_minutes", timeframe_minutes(getattr(settings, "timeframe", "15m")))
        )
        max_bars = max(
            1,
            int(round(float(getattr(settings, "trade_window_max_hours", 24.0)) * 60.0 / base_minutes)),
        )
        features, _, _ = make_features(
            frame,
            max_bars,
            external_feature_lag_bars=getattr(settings, "external_feature_lag_bars", 1),
        )
        absent = [column for column in feature_columns if column not in features.columns]
        if absent:
            return failed("missing_feature_columns:" + ",".join(absent[:32]), available=True, precision=precision)
        raw = features.reindex(columns=feature_columns).replace([np.inf, -np.inf], np.nan)
        numeric = raw.apply(pd.to_numeric, errors="coerce")
        missing_fraction = numeric.isna().mean(axis=1).astype(float)
        try:
            allowed_missing_fraction = float(getattr(settings, "live_max_missing_feature_fraction", 0.0))
        except (TypeError, ValueError, OverflowError):
            allowed_missing_fraction = -1.0
        if not np.isfinite(allowed_missing_fraction) or not 0.0 <= allowed_missing_fraction <= 1.0:
            return failed("invalid_live_max_missing_feature_fraction", available=True, precision=precision)
        valid_features = missing_fraction.le(allowed_missing_fraction + 1e-12)
        x = _fill_feature_frame(numeric, feature_columns, fill_values)
        if x.empty:
            return failed("empty_features", available=True, precision=precision)

        weighted_probability = model.predict_proba(x)
        probability = _correct_balanced_class_probabilities(
            weighted_probability, model.classes_, report.get("train_class_counts")
        )
        p_short, p_neutral, p_long, p_event, p_direction, pred_direction = _event_score_arrays(
            probability, model.classes_
        )
        min_confidence = float(getattr(settings, "trade_window_min_confidence", 0.80))
        min_event_probability = float(getattr(settings, "trade_window_min_event_probability", 0.60))
        active = (
            valid_features.to_numpy(dtype=bool)
            & (p_event >= min_event_probability)
            & (p_direction >= min_confidence)
            & (p_event > 1e-12)
        )
        directions = np.where(pred_direction > 0, "LONG", "SHORT")
        directions = np.where(active, directions, "FLAT")
        reasons = np.where(
            ~valid_features.to_numpy(dtype=bool),
            "live_feature_missing_fraction",
            np.where(active, "ok", "below_event_or_direction_threshold"),
        )
        result = pd.DataFrame(
            {
                "trade_window_available": True,
                "trade_window_ready": active,
                "trade_window_direction": directions,
                "trade_window_confidence": p_direction.astype(float),
                "trade_window_event_probability": p_event.astype(float),
                "trade_window_reason": reasons,
                "trade_window_report_precision": precision,
                "trade_window_feature_missing_fraction": missing_fraction.astype(float),
            },
            index=features.index,
        )
        return result.reindex(index)
    except Exception as exc:
        return failed(f"{type(exc).__name__}:{exc}")


def assess_trade_window(df, settings, model_path=None, symbol=None):
    """Return the final row from the shared strict specialist inference path."""
    fallback = {
        "trade_window_available": False,
        "trade_window_ready": False,
        "trade_window_direction": "FLAT",
        "trade_window_confidence": 0.0,
        "trade_window_event_probability": 0.0,
        "trade_window_reason": "unavailable",
        "trade_window_report_precision": 0.0,
        "trade_window_feature_missing_fraction": 1.0,
    }
    scores = score_trade_window_frame(df, settings, model_path=model_path, symbol=symbol)
    if scores.empty:
        return fallback
    return scores.iloc[-1].to_dict()
