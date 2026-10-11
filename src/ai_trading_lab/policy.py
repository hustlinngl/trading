from __future__ import annotations

import numpy as np
import pandas as pd

from .meta import MetaPolicy


def _column(frame, name, default=0.0):
    if name in frame:
        return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    return np.full(len(frame), float(default), dtype=float)


def _base_rate_adjusted_conviction(probability, baseline):
    """Measure probability lift over the side's empirical training prior."""
    denominator = np.maximum(1.0 - baseline, 1e-6)
    return np.clip((probability - baseline) / denominator, -1.0, 1.0)


def decide_actions(
    pred,
    regime,
    analog,
    meta_p,
    settings,
    *,
    regime_persistence=None,
    regime_probs=None,
    short_pred=None,
    short_meta_p=None,
    short_analog=None,
    **overrides,
):
    """Score long and short heads independently; never infer short odds as 1-p_up."""
    def val(name, default):
        value = overrides.get(name, getattr(settings, name, default))
        return float(value) if isinstance(default, float) else value

    probability_threshold = float(val("probability_threshold", 0.57))
    minimum_probability_edge = float(val("minimum_probability_edge", 0.05))
    min_expected_return = float(val("min_expected_return", 0.0015))
    decision_threshold = float(val("decision_threshold", 0.16))
    meta_threshold = float(val("meta_threshold", 0.53))
    conformal_blend = float(np.clip(val("conformal_blend", 0.60), 0.0, 1.0))

    fee_bps = float(getattr(settings, "fee_bps", 0.0))
    slippage_bps = float(getattr(settings, "slippage_bps", 0.0))
    impact_bps = float(getattr(settings, "impact_bps_per_sqrt", 0.0))
    participation = float(np.clip(getattr(settings, "max_participation_pct", 0.10), 0.0, 1.0))
    borrow_bps_per_bar = float(max(0.0, getattr(settings, "short_borrow_bps_per_bar", 0.0)))
    max_holding = int(max(1, getattr(settings, "max_holding_bars", 96)))
    base_hurdle_bps = (
        2.0 * (fee_bps + slippage_bps)
        + 2.0 * impact_bps * np.sqrt(participation)
        + max(0.0, float(getattr(settings, "min_edge_after_cost_bps", 5.0)))
    )
    short_hurdle_bps = base_hurdle_bps + borrow_bps_per_bar * max_holding
    long_return_floor = max(min_expected_return, base_hurdle_bps / 10000.0)
    short_return_floor = max(min_expected_return, short_hurdle_bps / 10000.0)

    p_long = _column(pred, "p_up", np.nan)
    prior_long = np.clip(_column(pred, "probability_baseline", 0.5), 0.0, 1.0)
    er_long = _column(pred, "expected_return", np.nan)
    lcb_long = _column(pred, "expected_return_lcb", np.nan)
    dis_long = _column(pred, "model_disagreement", np.nan)
    meta_long = np.asarray(meta_p, dtype=float).reshape(-1)
    if len(meta_long) != len(pred):
        raise ValueError("long_meta_prediction_length_mismatch")
    long_robust_return = er_long - conformal_blend * (er_long - lcb_long)

    short_ready = (
        short_pred is not None
        and short_meta_p is not None
        and short_analog is not None
    )
    if short_ready:
        p_short = _column(short_pred, "p_up", np.nan)
        prior_short = np.clip(_column(short_pred, "probability_baseline", 0.5), 0.0, 1.0)
        er_short = _column(short_pred, "expected_return", np.nan)
        lcb_short = _column(short_pred, "expected_return_lcb", np.nan)
        dis_short = _column(short_pred, "model_disagreement", np.nan)
        meta_short = np.asarray(short_meta_p, dtype=float).reshape(-1)
        if len(meta_short) != len(pred):
            raise ValueError("short_meta_prediction_length_mismatch")
        edge_short = _column(short_analog, "edge", np.nan)
        agreement_short = _column(short_analog, "agreement", np.nan)
        short_robust_return = er_short - conformal_blend * (er_short - lcb_short)
    else:
        p_short = prior_short = er_short = lcb_short = dis_short = np.full(len(pred), np.nan)
        meta_short = edge_short = agreement_short = short_robust_return = np.full(len(pred), np.nan)

    edge_long = _column(analog, "edge", np.nan)
    agreement_long = _column(analog, "agreement", np.nan)

    regime_vector = regime.map(
        {
            "trend_up": 1.0,
            "high_vol_up": 0.45,
            "range_or_down": -0.15,
            "high_vol_down": -1.0,
            "transition": 0.0,
            "unknown": 0.0,
        }
    ).fillna(0.0).to_numpy(dtype=float)

    weights = np.clip(
        np.asarray(
            [
                val("conviction_weight", 0.34),
                val("edge_weight", 0.30),
                val("memory_weight", 0.16),
                val("meta_weight", 0.18),
                val("regime_weight", 0.08),
            ],
            dtype=float,
        ),
        0.0,
        None,
    )
    weights /= max(float(weights.sum()), 1e-12)
    conviction_weight, edge_weight, memory_weight, meta_weight, regime_weight = weights
    uncertainty_mult = float(val("uncertainty_penalty_mult", 2.0))

    long_score = (
        conviction_weight * _base_rate_adjusted_conviction(p_long, prior_long)
        + edge_weight * np.tanh(long_robust_return * 50.0)
        + memory_weight * np.tanh(edge_long * 50.0)
        + meta_weight * ((meta_long - 0.5) * 2.0)
        + regime_weight * regime_vector
        + (agreement_long - 0.5) * 0.20
        - np.minimum(0.35, np.maximum(0.0, dis_long * uncertainty_mult))
    )
    short_score = (
        conviction_weight * _base_rate_adjusted_conviction(p_short, prior_short)
        + edge_weight * np.tanh(short_robust_return * 50.0)
        + memory_weight * np.tanh(edge_short * 50.0)
        + meta_weight * ((meta_short - 0.5) * 2.0)
        - regime_weight * regime_vector
        + (agreement_short - 0.5) * 0.20
        - np.minimum(0.35, np.maximum(0.0, dis_short * uncertainty_mult))
    )

    long_probability_floor = np.maximum(
        probability_threshold,
        prior_long + minimum_probability_edge,
    )
    short_probability_floor = np.maximum(
        probability_threshold,
        prior_short + minimum_probability_edge,
    )
    long_ok = (
        np.isfinite(p_long)
        & np.isfinite(prior_long)
        & np.isfinite(long_robust_return)
        & np.isfinite(meta_long)
        & np.isfinite(long_score)
        & (p_long >= long_probability_floor)
        & (long_robust_return >= long_return_floor)
        & (meta_long >= meta_threshold)
        & (long_score >= decision_threshold)
    )
    short_ok = (
        bool(getattr(settings, "short_signals_enabled", False))
        & np.isfinite(p_short)
        & np.isfinite(prior_short)
        & np.isfinite(short_robust_return)
        & np.isfinite(meta_short)
        & np.isfinite(short_score)
        & (p_short >= short_probability_floor)
        & (short_robust_return >= short_return_floor)
        & (meta_short >= meta_threshold)
        & (short_score >= decision_threshold)
    )

    actions = np.full(len(pred), "FLAT", dtype=object)
    choose_long = long_ok & (~short_ok | (long_score >= short_score))
    choose_short = short_ok & ~choose_long
    actions[choose_long] = "LONG"
    actions[choose_short] = "SHORT"

    # Expose the strongest side's score for diagnostics even when no side passes
    # all gates. Side-specific scores are retained separately for the live gate.
    best_score = np.fmax(long_score, short_score)
    best_score[~np.isfinite(best_score)] = np.nan
    score_series = pd.Series(best_score, index=pred.index, name="score")
    score_series.attrs["long_scores"] = long_score.tolist()
    score_series.attrs["short_scores"] = short_score.tolist()
    score_series.attrs["effective_min_expected_return"] = float(max(long_return_floor, short_return_floor))
    score_series.attrs["economic_hurdle_bps"] = float(short_hurdle_bps)

    def summarize(values):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if not len(arr):
            return {"n": 0}
        return {
            "n": int(len(arr)),
            "min": float(np.min(arr)),
            "p50": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
            "p99": float(np.quantile(arr, 0.99)),
            "max": float(np.max(arr)),
        }

    score_series.attrs["diagnostics"] = {
        "rows": int(len(pred)),
        "pass_probability": int(np.sum((p_long >= long_probability_floor) | (p_short >= short_probability_floor))),
        "pass_robust_return": int(np.sum((long_robust_return >= long_return_floor) | (short_robust_return >= short_return_floor))),
        "pass_meta": int(np.sum((meta_long >= meta_threshold) | (meta_short >= meta_threshold))),
        "pass_score": int(np.sum((long_score >= decision_threshold) | (short_score >= decision_threshold))),
        "pass_all_policy_gates": int(np.sum(long_ok | short_ok)),
        "pass_long_probability": int(np.sum(p_long >= long_probability_floor)),
        "pass_short_probability": int(np.sum(p_short >= short_probability_floor)),
        "pass_long_policy": int(np.sum(long_ok)),
        "pass_short_policy": int(np.sum(short_ok)),
        "thresholds": {
            "directional_probability": float(probability_threshold),
            "minimum_probability_edge": float(minimum_probability_edge),
            "long_expected_return": float(long_return_floor),
            "short_expected_return": float(short_return_floor),
            "meta_success": float(meta_threshold),
            "score": float(decision_threshold),
            "long_probability_p50": float(np.nanquantile(p_long, 0.50)) if np.isfinite(p_long).any() else None,
            "short_probability_p50": float(np.nanquantile(p_short, 0.50)) if np.isfinite(p_short).any() else None,
        },
        "distributions": {
            "long_probability": summarize(p_long),
            "long_probability_baseline": summarize(prior_long),
            "short_probability": summarize(p_short),
            "short_probability_baseline": summarize(prior_short),
            "long_expected_return": summarize(er_long),
            "short_expected_return": summarize(er_short),
            "long_robust_expected_return": summarize(long_robust_return),
            "short_robust_expected_return": summarize(short_robust_return),
            "long_meta_success": summarize(meta_long),
            "short_meta_success": summarize(meta_short),
            "long_score": summarize(long_score),
            "short_score": summarize(short_score),
            "long_model_disagreement": summarize(dis_long),
            "short_model_disagreement": summarize(dis_short),
        },
    }
    return pd.Series(actions, index=pred.index, name="action"), score_series


def live_signal_gate(row, settings, *, profile="strict"):
    """Apply selected-side gates; the research profile is paper-only and explicitly looser."""
    profile = str(profile).strip().lower()
    if profile not in {"strict", "research"}:
        raise ValueError(f"unknown_signal_profile:{profile}")
    research = profile == "research"
    action = str(row.get("action", "FLAT") or "FLAT").upper()
    is_short = action == "SHORT"
    reasons = []

    if action not in {"LONG", "SHORT"}:
        reasons.append("base_policy")

    if is_short:
        if not bool(getattr(settings, "short_signals_enabled", False)):
            reasons.append("short_signals_disabled")
        market_type = str(row.get("market_type", "unknown") or "unknown").strip().lower()
        if market_type not in {"margin", "swap", "future", "perpetual"}:
            reasons.append("short_instrument_not_supported")
        if bool(getattr(settings, "require_short_borrow_cost", True)) and float(
            getattr(settings, "short_borrow_bps_per_bar", 0.0)
        ) <= 0.0:
            reasons.append("short_borrow_cost_missing")

    if bool(row.get("funding_data_missing", False)):
        reasons.append("funding_data_missing")

    if is_short:
        fields = {
            "p_short": row.get("p_short"),
            "short_expected_return": row.get("short_expected_return"),
            "short_expected_return_lcb": row.get("short_expected_return_lcb"),
            "short_probability_baseline": row.get("short_probability_baseline"),
            "short_meta_success": row.get("short_meta_success"),
            "short_score": row.get("short_score"),
            "short_model_disagreement": row.get("short_model_disagreement"),
            "short_analog_n": row.get("short_analog_n"),
            "short_analog_agreement": row.get("short_analog_agreement"),
        }
    else:
        fields = {
            "p_up": row.get("p_up"),
            "probability_baseline": row.get("probability_baseline"),
            "expected_return": row.get("expected_return"),
            "expected_return_lcb": row.get("expected_return_lcb"),
            "meta_success": row.get("meta_success"),
            "score": row.get("score"),
            "model_disagreement": row.get("model_disagreement"),
            "analog_n": row.get("analog_n"),
            "analog_agreement": row.get("analog_agreement"),
        }

    missing = []
    for key, value in fields.items():
        if value is None:
            missing.append(key)
            continue
        try:
            if not np.isfinite(float(value)):
                missing.append(key)
        except (TypeError, ValueError, OverflowError):
            missing.append(key)
    if missing:
        reasons.append("missing_prediction:" + ",".join(missing))
        return "FLAT", reasons

    if is_short:
        probability = float(row["p_short"])
        baseline = float(row["short_probability_baseline"])
        expected_return_lcb = float(row["short_expected_return_lcb"])
        meta_success = float(row["short_meta_success"])
        score = float(row["short_score"])
        disagreement = float(row["short_model_disagreement"])
        analog_n = int(float(row["short_analog_n"]))
        analog_agreement = float(row["short_analog_agreement"])
    else:
        probability = float(row["p_up"])
        baseline = float(row.get("probability_baseline", 0.5))
        expected_return_lcb = float(row["expected_return_lcb"])
        meta_success = float(row["meta_success"])
        score = float(row["score"])
        disagreement = float(row["model_disagreement"])
        analog_n = int(float(row["analog_n"]))
        analog_agreement = float(row["analog_agreement"])

    funding_drag = max(0.0, float(row.get("funding_cost_return", 0.0) or 0.0))
    robust_expected_return = expected_return_lcb - funding_drag

    if bool(getattr(settings, "signal_only_mode", True)):
        if research:
            confidence_gate = float(getattr(settings, "research_signal_confidence_threshold", 0.60))
            probability_gate = float(getattr(settings, "research_signal_probability_threshold", 0.55))
            edge_gate = float(getattr(settings, "research_minimum_probability_edge", 0.02))
            return_gate = float(getattr(settings, "research_signal_min_expected_return", 0.001))
            meta_gate = float(getattr(settings, "research_signal_meta_threshold", 0.52))
            score_gate = float(getattr(settings, "research_signal_min_score", 0.04))
            disagreement_gate = float(getattr(settings, "research_signal_max_disagreement", 0.20))
            min_neighbors = int(getattr(settings, "research_signal_memory_min_neighbors", 8))
            min_agreement = float(getattr(settings, "research_signal_memory_min_agreement", 0.55))
        else:
            confidence_gate = float(getattr(settings, "signal_confidence_threshold", 0.82))
            probability_gate = float(getattr(settings, "signal_probability_threshold", 0.72))
            edge_gate = float(getattr(settings, "minimum_probability_edge", 0.05))
            return_gate = float(getattr(settings, "signal_min_expected_return", 0.003))
            meta_gate = float(getattr(settings, "signal_meta_threshold", 0.62))
            score_gate = float(getattr(settings, "signal_min_score", 0.22))
            disagreement_gate = float(getattr(settings, "signal_max_disagreement", 0.05))
            min_neighbors = int(getattr(settings, "signal_memory_min_neighbors", 16))
            min_agreement = float(getattr(settings, "signal_memory_min_agreement", 0.70))
        probability_floor = max(confidence_gate, probability_gate, baseline + edge_gate)
        if probability < probability_floor:
            reasons.append("signal_probability")
        if robust_expected_return < return_gate:
            reasons.append("signal_expected_return")
        if meta_success < meta_gate:
            reasons.append("signal_meta")
        if score < score_gate:
            reasons.append("signal_score")
        if disagreement > disagreement_gate:
            reasons.append("model_disagreement")
        if analog_n < min_neighbors:
            reasons.append("memory_neighbors")
        if analog_agreement < min_agreement:
            reasons.append("memory_agreement")

    specialist_required = (
        bool(getattr(settings, "research_trade_window_required_for_signal", False))
        if research
        else bool(getattr(settings, "trade_window_required_for_signal", False))
    )
    if specialist_required:
        tw_available = bool(row.get("trade_window_available", False))
        tw_ready = bool(row.get("trade_window_ready", False))
        tw_direction = str(row.get("trade_window_direction", "FLAT"))
        tw_confidence = float(row.get("trade_window_confidence", 0.0))
        if not tw_available:
            reasons.append("trade_window_missing")
        elif not tw_ready:
            reasons.append("trade_window_not_ready")
        elif tw_direction != action:
            reasons.append("trade_window_disagreement")
        elif tw_confidence < float(getattr(settings, "trade_window_min_confidence", 0.80)):
            reasons.append("trade_window_confidence")

    if reasons:
        return "FLAT", reasons
    return action, []


def make_actions(
    engine,
    features,
    settings,
    probability_threshold=None,
    min_expected_return=None,
    decision_threshold=None,
    meta_threshold=None,
):
    pred = engine.model.predict(features)
    short_pred = engine.short_model.predict(features)
    regime = engine.regimes.transform(features)
    regime_persistence = engine.regimes.persistence(features)
    regime_probs = engine.regimes.semantic_probabilities(features)
    analog = engine.memory.query_many(features)
    short_analog = engine.short_memory.query_many(features)

    meta_detector = engine.meta_regimes if engine.meta_regimes is not None else engine.regimes
    meta_regime = meta_detector.transform(features)
    meta_persistence = meta_detector.persistence(features)
    meta_probs = meta_detector.semantic_probabilities(features)
    meta_x = MetaPolicy.frame(
        pred, features, meta_regime, analog,
        regime_persistence=meta_persistence, regime_probs=meta_probs,
    )
    short_meta_x = MetaPolicy.frame(
        short_pred, features, meta_regime, short_analog,
        regime_persistence=meta_persistence, regime_probs=meta_probs,
    )
    meta_p = engine.meta.predict_proba(meta_x)
    short_meta_p = engine.short_meta.predict_proba(short_meta_x)
    overrides = {
        key: value
        for key, value in {
            "probability_threshold": probability_threshold,
            "min_expected_return": min_expected_return,
            "decision_threshold": decision_threshold,
            "meta_threshold": meta_threshold,
        }.items()
        if value is not None
    }
    actions, scores = decide_actions(
        pred,
        regime,
        analog,
        meta_p,
        settings,
        regime_persistence=regime_persistence,
        regime_probs=regime_probs,
        short_pred=short_pred,
        short_meta_p=short_meta_p,
        short_analog=short_analog,
        **overrides,
    )
    actions.attrs["score"] = scores
    actions.attrs["diagnostics"] = scores.attrs.get("diagnostics", {})
    return actions
