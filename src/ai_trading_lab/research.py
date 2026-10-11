from __future__ import annotations

import numpy as np
import pandas as pd

from .engine import AdaptiveEngine
from .evaluation import run_configured_backtest
from .features import make_features, make_oos_features
from .labels import triple_barrier_labels
from .meta import MetaPolicy
from .policy import decide_actions
from .research_ledger import register_trials
from .strategy_lab import evolve, score_candidate
from .validation import aggregate_fold_stats, robust_score


def _fold_prediction(engine, train, test, settings):
    """Produce train-fitted side-specific predictions for one untouched fold."""
    features = make_oos_features(
        train,
        test,
        settings.horizon_bars,
        external_feature_lag_bars=getattr(settings, "external_feature_lag_bars", 1),
    )
    long_pred = engine.model.predict(features)
    short_pred = engine.short_model.predict(features)
    regime = engine.regimes.transform(features)
    persistence = engine.regimes.persistence(features)
    regime_probs = engine.regimes.semantic_probabilities(features)
    long_memory = engine.memory.query_many(features)
    short_memory = engine.short_memory.query_many(features)

    detector = engine.meta_regimes if engine.meta_regimes is not None else engine.regimes
    meta_regime = detector.transform(features)
    meta_persistence = detector.persistence(features)
    meta_probs = detector.semantic_probabilities(features)
    long_meta_x = MetaPolicy.frame(
        long_pred, features, meta_regime, long_memory,
        regime_persistence=meta_persistence, regime_probs=meta_probs,
    )
    short_meta_x = MetaPolicy.frame(
        short_pred, features, meta_regime, short_memory,
        regime_persistence=meta_persistence, regime_probs=meta_probs,
    )
    long_meta = engine.meta.predict_proba(long_meta_x)
    short_meta = engine.short_meta.predict_proba(short_meta_x)
    actions, scores = decide_actions(
        long_pred,
        regime,
        long_memory,
        long_meta,
        settings,
        regime_persistence=persistence,
        regime_probs=regime_probs,
        short_pred=short_pred,
        short_meta_p=short_meta,
        short_analog=short_memory,
    )
    predictions = long_pred.copy()
    predictions["p_short"] = short_pred["p_up"]
    predictions["short_probability_baseline"] = short_pred["probability_baseline"]
    predictions["short_expected_return"] = short_pred["expected_return"]
    predictions["short_expected_return_lcb"] = short_pred["expected_return_lcb"]
    predictions["short_expected_return_ucb"] = short_pred["expected_return_ucb"]
    predictions["short_model_disagreement"] = short_pred["model_disagreement"]
    predictions["short_meta_success"] = short_meta
    predictions["short_analog_edge"] = short_memory["edge"]
    predictions["short_analog_agreement"] = short_memory["agreement"]
    predictions["short_analog_n"] = short_memory["n"]
    predictions["long_score"] = scores.attrs.get("long_scores", np.nan)
    predictions["short_score"] = scores.attrs.get("short_scores", np.nan)
    predictions["regime"] = regime
    predictions["action"] = actions
    predictions.attrs["policy_diagnostics"] = dict(scores.attrs.get("diagnostics", {}))
    return features, predictions, actions


def walk_forward(df: pd.DataFrame, settings, independent_test=False):
    rows = []
    train_bars = int(settings.walk_forward_train_bars)
    test_bars = int(settings.walk_forward_test_bars)
    step = max(
        int(settings.walk_forward_step),
        test_bars if independent_test else int(settings.walk_forward_step),
    )
    purge = int(
        getattr(
            settings,
            "validation_purge_bars",
            int(settings.horizon_bars) + max(1, int(settings.horizon_bars) // 2),
        )
    )
    cursor = train_bars + purge
    while cursor + test_bars <= len(df):
        test_start = cursor
        train_end = test_start - purge
        train_start = max(0, train_end - train_bars)
        train = df.iloc[train_start:train_end].copy()
        test = df.iloc[test_start:test_start + test_bars].copy()
        if len(train) < int(getattr(settings, "min_train_rows", 1500)):
            cursor += step
            continue

        engine = AdaptiveEngine(settings)
        engine.fit(train)
        test_features, predictions, actions = _fold_prediction(
            engine, train, test, settings
        )
        bt = test.copy()
        bt["atr_14"] = test_features["atr_14"]
        result = run_configured_backtest(
            bt, pd.Series(actions, index=test.index), settings
        )

        # Randomize only the timing of this fold's model actions, preserving the
        # exact LONG/SHORT/FLAT counts and using the same execution/cost engine.
        rng = np.random.default_rng(int(settings.seed) + len(rows))
        randomized_actions = pd.Series(
            rng.permutation(np.asarray(actions, dtype=object)),
            index=test.index,
            name="randomized_model_activity",
        )
        random_result = run_configured_backtest(bt, randomized_actions, settings)

        buy_hold_gross_return = (
            float(test["close"].iloc[-1] / test["open"].iloc[0] - 1.0)
            if len(test) and float(test["open"].iloc[0]) > 0.0
            else float("nan")
        )
        gate_diagnostics = predictions.attrs.get("policy_diagnostics", {})
        rows.append(
            {
                "train_start": str(train.index[0]),
                "train_end": str(train.index[-1]),
                "test_start": str(test.index[0]),
                "test_end": str(test.index[-1]),
                "purge_bars": purge,
                "median_regime": engine.regimes.transform(test_features).mode().iloc[0],
                **result.stats,
                "robust_score": robust_score(result.stats),
                "baseline_buy_hold_gross_return": buy_hold_gross_return,
                "baseline_random_entries_net_return": float(
                    random_result.stats.get("total_return", 0.0)
                ),
                "baseline_random_entries_trades": int(
                    random_result.stats.get("trades", 0)
                ),
                "baseline_random_entries_action_counts": {
                    str(key): int(value)
                    for key, value in randomized_actions.value_counts().items()
                },
                "policy_gate_diagnostics": gate_diagnostics,
                "policy_action_counts": {
                    str(key): int(value) for key, value in actions.value_counts().items()
                },
            }
        )
        cursor += step
    return pd.DataFrame(rows)


def strategy_discovery(df, settings):
    split = int(len(df) * 0.75)
    train_df, valid_df = df.iloc[:split], df.iloc[split:]
    train_x, _, _ = make_features(train_df, settings.horizon_bars)
    train_tb = triple_barrier_labels(
        train_df, settings.horizon_bars, settings.pt_atr, settings.sl_atr
    )
    valid_x = make_oos_features(
        train_df,
        valid_df,
        settings.horizon_bars,
        external_feature_lag_bars=getattr(settings, "external_feature_lag_bars", 1),
    )
    valid_tb = triple_barrier_labels(
        valid_df, settings.horizon_bars, settings.pt_atr, settings.sl_atr
    )
    train_ret = train_tb["tb_return"]
    valid_ret = valid_tb["tb_return"]

    impact = float(getattr(settings, "impact_bps_per_sqrt", 0.0)) * np.sqrt(
        max(0.0, float(getattr(settings, "max_participation_pct", 0.10)))
    )
    research_cost_bps = (
        float(getattr(settings, "fee_bps", 0.0))
        + float(getattr(settings, "slippage_bps", 0.0))
        + impact
    )
    best = evolve(
        train_x,
        train_ret,
        settings.population_size,
        settings.generations,
        settings.seed,
        cost_bps=research_cost_bps,
    )
    ledger_path = getattr(settings, "research_ledger_path", "data/research_ledger.json")
    register_trials(
        ledger_path,
        int(settings.population_size) * int(settings.generations),
        kind="strategy_evolution",
        metadata={"split_ratio": 0.75, "cost_bps": research_cost_bps},
    )
    serial = [
        {
            "train_score": float(score),
            "validation_score": float(
                score_candidate(candidate, valid_x, valid_ret, cost_bps=research_cost_bps)
            ),
            "direction": candidate.direction,
            "rules": [rule.__dict__ for rule in candidate.rules],
        }
        for score, candidate in best
    ]
    serial.sort(key=lambda row: row["validation_score"], reverse=True)
    return serial


def research_report(df, settings):
    folds = walk_forward(df, settings)
    return {
        "folds": folds.to_dict("records"),
        "aggregate": aggregate_fold_stats(folds),
    }
