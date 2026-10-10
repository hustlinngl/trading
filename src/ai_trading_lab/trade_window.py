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
    """Label the first executable barrier event, retaining no-event outcomes.

    A label is +1/-1 only when that barrier is first reached after the configured
    minimum holding period. No qualifying event (including an ambiguous OHLC
    collision) is class 0. Rows without enough past/future data remain NaN and are
    excluded. Opening gaps are evaluated before the candle's intrabar high/low.
    The time-stop candle's high/low is not inspected: liquidation occurs at its open.
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
    directions = np.full(len(x), np.nan, dtype=float)

    # The time-stop exits at open[i + horizon + 1]. Inspect only the horizon
    # holding candles: i+1 through i+horizon, not the exit candle itself.
    for i in range(max(0, len(x) - horizon - 1)):
        atr_i = atr[i]
        entry_i = i + 1
        exit_open_i = entry_i + horizon
        entry = opens[entry_i]
        if not np.isfinite(atr_i) or atr_i <= 0 or not np.isfinite(entry) or entry <= 0:
            continue
        if not np.isfinite(opens[exit_open_i]) or opens[exit_open_i] <= 0:
            continue

        upper = entry + pt * atr_i
        lower = entry - sl * atr_i
        outcome = 0.0

        for j in range(entry_i, exit_open_i):
            bar_open = opens[j]
            if not np.isfinite(bar_open) or bar_open <= 0:
                outcome = np.nan
                break

            held = j - i
            # The opening print is the earliest executable event in this candle.
            if bar_open >= upper:
                outcome = 1.0 if held >= minimum else 0.0
                break
            if bar_open <= lower:
                outcome = -1.0 if held >= minimum else 0.0
                break

            if not np.isfinite(highs[j]) or not np.isfinite(lows[j]):
                outcome = np.nan
                break
            hit_up = highs[j] >= upper
            hit_down = lows[j] <= lower
            if hit_up and hit_down:
                # Intrabar order is unknowable from OHLC; do not invent a side.
                outcome = 0.0
                break
            if hit_up:
                outcome = 1.0 if held >= minimum else 0.0
                break
            if hit_down:
                outcome = -1.0 if held >= minimum else 0.0
                break

        directions[i] = outcome

    return pd.DataFrame(
        {"direction": directions, "margin": np.abs(directions)},
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


def directional_event_probabilities(probabilities, classes):
    """Publicly testable scalar view of the event-gated direction prediction."""
    short, neutral, long, event, confidence, direction = _event_score_arrays(
        np.asarray(probabilities, dtype=float).reshape(1, -1),
        classes,
    )
    p_event = float(event[0])
    return {
        "p_short": float(short[0]),
        "p_neutral": float(neutral[0]),
        "p_long": float(long[0]),
        "event_probability": p_event,
        "direction": ("LONG" if int(direction[0]) > 0 else "SHORT") if p_event > 1e-12 else "FLAT",
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
    proba = model.predict_proba(holdout_x)
    try:
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

    open_prices = pd.to_numeric(df["open"], errors="coerce").to_numpy(float)
    for local_i, candidate in enumerate(candidates):
        if not bool(candidate):
            continue
        pos = int(positions[local_i])
        entry_pos = pos + 1
        exit_pos = pos + max_bars + 1
        if pos < 0 or entry_pos >= len(df) or exit_pos >= len(df) or pos <= last_exit_pos:
            continue
        entry = float(open_prices[entry_pos])
        exit_price = float(open_prices[exit_pos])
        if not np.isfinite(entry) or entry <= 0 or not np.isfinite(exit_price) or exit_price <= 0:
            continue
        side = int(pred_direction[local_i])
        cost_bps = raw_cost_bps + (borrow_bps_per_bar * max_bars if side < 0 else 0.0)
        gross = float(side) * (exit_price / entry - 1.0)
        net_return = gross - cost_bps / 10_000.0
        accepted.append(
            {
                "position": pos,
                "truth": int(truth[local_i]),
                "prediction": side,
                "event_probability": float(p_event[local_i]),
                "direction_confidence": float(p_direction[local_i]),
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
    report = {
        "production_ready": bool(ready),
        "holdout_precision": precision,
        "holdout_wilson_lower": wilson,
        "holdout_signals": int(support),
        "holdout_candidates_before_nonoverlap": int(candidates.sum()),
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


def assess_trade_window(df, settings, model_path=None, symbol=None):
    """Evaluate the persisted multi-class 3–24h specialist as an event/direction verifier."""
    path = Path(
        model_path
        or getattr(settings, "trade_window_model_path", "models/champion/trade_window_specialist.joblib")
    )
    out = {
        "trade_window_available": False,
        "trade_window_ready": False,
        "trade_window_direction": "FLAT",
        "trade_window_confidence": 0.0,
        "trade_window_event_probability": 0.0,
        "trade_window_reason": "disabled",
    }
    if not bool(getattr(settings, "trade_window_enabled", True)):
        return out
    if not path.exists():
        out["trade_window_reason"] = "artifact_missing"
        return out

    try:
        artifact = joblib.load(path)
        model = artifact["model"]
        feature_columns = list(artifact.get("feature_columns", []))
        fill_values = artifact.get("fill_values")
        report = artifact.get("report", {}) or {}
        if not feature_columns or not isinstance(fill_values, pd.Series):
            out["trade_window_reason"] = "feature_imputation_metadata_missing"
            return out
        expected_symbol = str(symbol or getattr(settings, "symbol", ""))
        expected_timeframe = str(getattr(settings, "timeframe", "15m"))
        if str(report.get("symbol", "")) != expected_symbol:
            out["trade_window_reason"] = "symbol_mismatch"
            return out
        if str(report.get("timeframe", "")) != expected_timeframe:
            out["trade_window_reason"] = "timeframe_mismatch"
            return out
        if str(report.get("model_semantics_fingerprint", "")) != model_semantics_fingerprint(settings):
            out["trade_window_reason"] = "model_semantics_mismatch"
            return out
        if str(report.get("deployment_semantics_fingerprint", "")) != deployment_semantics_fingerprint(settings):
            out["trade_window_reason"] = "deployment_semantics_mismatch"
            return out
        if not bool(report.get("production_ready", False)):
            out.update(
                {
                    "trade_window_available": True,
                    "trade_window_reason": "not_production_ready",
                    "trade_window_report_precision": float(report.get("holdout_precision", 0.0) or 0.0),
                }
            )
            return out

        df = df.sort_index().copy()
        if df.index.has_duplicates:
            out["trade_window_reason"] = "duplicate_timestamps"
            return out

        base_minutes = float(
            getattr(settings, "base_bar_minutes", timeframe_minutes(getattr(settings, "timeframe", "15m")))
        )
        max_bars = max(
            1, int(round(float(getattr(settings, "trade_window_max_hours", 24.0)) * 60.0 / base_minutes))
        )
        features, _, _ = make_features(
            df,
            max_bars,
            external_feature_lag_bars=getattr(settings, "external_feature_lag_bars", 1),
        )
        x = _fill_feature_frame(features, feature_columns, fill_values)
        if x.empty:
            out["trade_window_reason"] = "empty_features"
            return out

        probability = model.predict_proba(x.iloc[[-1]])[0]
        scores = directional_event_probabilities(probability, model.classes_)
        direction = scores["direction"]
        confidence = scores["direction_confidence"]
        event_probability = scores["event_probability"]
        min_confidence = float(getattr(settings, "trade_window_min_confidence", 0.80))
        min_event_probability = float(getattr(settings, "trade_window_min_event_probability", 0.60))
        active = direction in {"LONG", "SHORT"} and event_probability >= min_event_probability and confidence >= min_confidence

        out.update(
            {
                "trade_window_available": True,
                "trade_window_ready": bool(active),
                "trade_window_direction": direction if active else "FLAT",
                "trade_window_confidence": confidence,
                "trade_window_event_probability": event_probability,
                "trade_window_report_precision": float(report.get("holdout_precision", 0.0) or 0.0),
                "trade_window_reason": "ok" if active else "below_event_or_direction_threshold",
            }
        )
        return out
    except Exception as exc:
        out["trade_window_reason"] = f"{type(exc).__name__}:{exc}"
        return out
