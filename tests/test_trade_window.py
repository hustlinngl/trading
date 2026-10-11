from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from ai_trading_lab.config import load_settings
from ai_trading_lab.deployment import deployment_semantics_fingerprint, model_semantics_fingerprint
from ai_trading_lab.trade_window import (
    BALANCED_PROBABILITY_SEMANTICS,
    _barrier_labels,
    _correct_balanced_class_probabilities,
    _fill_feature_frame,
    assess_trade_window,
    directional_event_probabilities,
    score_trade_window_frame,
    train_trade_window_backbone,
)


def _flat_market(rows=40):
    index = pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": np.full(rows, 100.0),
            "high": np.full(rows, 100.1),
            "low": np.full(rows, 99.9),
            "close": np.full(rows, 100.0),
            "volume": np.full(rows, 100_000.0),
        },
        index=index,
    )
    return frame


def test_trade_window_retains_neutral_no_event_labels():
    frame = _flat_market()
    labels = _barrier_labels(frame, horizon_bars=3, min_bars=1, pt_atr=10.0, sl_atr=10.0)

    assert labels.loc[frame.index[14], "direction"] == 0.0
    assert labels.loc[frame.index[14], "margin"] == 0.0
    assert labels["direction"].iloc[-4:].isna().all()


def test_trade_window_records_barrier_exit_return_not_later_time_stop_return():
    frame = _flat_market(rows=40)
    decision = frame.index[14]
    # For decision 14, entry is open[15], the upper barrier is crossed in bar 16,
    # then price gaps down at the later time-stop open[18]. Realized PnL is the
    # barrier exit (+0.25%), not the fictive time-stop return (-1%).
    frame.loc[frame.index[16], ["open", "high", "low", "close"]] = [
        100.0, 100.6, 99.9, 100.2
    ]
    frame.loc[frame.index[18], ["open", "high", "low", "close"]] = [
        99.0, 99.1, 98.9, 99.0
    ]

    labels = _barrier_labels(frame, horizon_bars=3, min_bars=1, pt_atr=1.25, sl_atr=0.90)

    # ATR is 0.2 on the decision bar: the take-profit is at 100.25.
    assert labels.loc[decision, "direction"] == 1.0
    assert labels.loc[decision, "execution_status"] == "take_profit"
    assert labels.loc[decision, "exit_position"] == 16.0
    assert labels.loc[decision, "holding_bars"] == 2.0
    assert np.isclose(labels.loc[decision, "gross_return"], 0.0025)
    assert np.isclose(frame.loc[frame.index[18], "open"] / frame.loc[frame.index[15], "open"] - 1.0, -0.01)


def test_trade_window_ambiguous_barrier_collision_has_unknown_pnl():
    frame = _flat_market(rows=40)
    decision = frame.index[14]
    frame.loc[frame.index[16], ["open", "high", "low", "close"]] = [
        100.0, 100.6, 99.5, 100.0
    ]

    labels = _barrier_labels(frame, horizon_bars=3, min_bars=1, pt_atr=1.25, sl_atr=0.90)

    assert labels.loc[decision, "direction"] == 0.0
    assert labels.loc[decision, "execution_status"] == "ambiguous_intrabar"
    assert np.isnan(labels.loc[decision, "gross_return"])


def test_trade_window_opening_gap_is_classified_before_later_intrabar_collision():
    frame = _flat_market()
    # At decision bar 14, entry is open[15]. This candle gaps below the lower
    # barrier, then trades across both barrier levels. The opening gap is first.
    frame.loc[frame.index[16], ["open", "high", "low", "close"]] = [
        99.7, 100.6, 99.5, 100.0
    ]

    labels = _barrier_labels(frame, horizon_bars=3, min_bars=1, pt_atr=1.25, sl_atr=0.90)

    assert labels.loc[frame.index[14], "direction"] == -1.0


def test_trade_window_time_stop_candle_is_not_misread_as_barrier_event():
    frame = _flat_market()
    # horizon=3 enters on bar 15 and time-stops at open[18]. A huge high on the
    # exit candle must not create a LONG event because liquidation is at its open.
    frame.loc[frame.index[18], ["open", "high", "low", "close"]] = [
        100.5, 110.0, 100.4, 100.5
    ]

    labels = _barrier_labels(frame, horizon_bars=3, min_bars=1, pt_atr=1.25, sl_atr=0.90)

    assert labels.loc[frame.index[14], "direction"] == 0.0
    assert labels.loc[frame.index[14], "execution_status"] == "time_stop"
    assert labels.loc[frame.index[14], "exit_position"] == 18.0
    assert np.isclose(labels.loc[frame.index[14], "gross_return"], 0.005)



def test_trade_window_minimum_holding_duration_is_measured_from_entry_open():
    # At 15m bars, the event in bar i+12 occurs only 11 intervals after the
    # executable entry open[i+1] (2h45m). It must not qualify for a 3h minimum.
    early = _flat_market(rows=40)
    early.loc[early.index[26], ["open", "high", "low", "close"]] = [
        100.0, 100.7, 99.9, 100.5
    ]
    early_labels = _barrier_labels(
        early, horizon_bars=16, min_bars=12, pt_atr=1.25, sl_atr=0.90
    )
    assert early_labels.loc[early.index[14], "direction"] == 0.0

    # The same barrier in bar i+13 is 12 intervals after entry and qualifies.
    on_time = _flat_market(rows=40)
    on_time.loc[on_time.index[27], ["open", "high", "low", "close"]] = [
        100.0, 100.7, 99.9, 100.5
    ]
    on_time_labels = _barrier_labels(
        on_time, horizon_bars=16, min_bars=12, pt_atr=1.25, sl_atr=0.90
    )
    assert on_time_labels.loc[on_time.index[14], "direction"] == 1.0


def test_event_probability_is_separate_from_conditional_direction_confidence():
    long = directional_event_probabilities([0.10, 0.20, 0.70], [-1, 0, 1])
    assert long["direction"] == "LONG"
    assert np.isclose(long["event_probability"], 0.80)
    assert np.isclose(long["direction_confidence"], 0.875)

    neutral = directional_event_probabilities([0.01, 0.98, 0.01], [-1, 0, 1])
    assert neutral["direction"] == "FLAT"
    assert np.isclose(neutral["event_probability"], 0.02)
    assert neutral["direction_confidence"] < 0.80



def test_class_balanced_scores_are_prior_corrected_before_event_thresholding():
    # Weighted training can make rare event classes appear much more likely
    # than they are in the original 96%-neutral population.
    corrected = _correct_balanced_class_probabilities(
        [[0.30, 0.40, 0.30]],
        classes=[-1, 0, 1],
        class_counts={"-1": 200, "0": 9600, "1": 200},
    )

    assert np.isclose(corrected.sum(), 1.0)
    assert corrected[0, 1] > 0.95
    assert corrected[0, 0] + corrected[0, 2] < 0.05


def test_class_prior_correction_fails_closed_when_training_counts_are_missing():
    try:
        _correct_balanced_class_probabilities([[0.2, 0.6, 0.2]], [-1, 0, 1], {})
    except ValueError as exc:
        assert str(exc) == "trade_window_class_prior_metadata_missing"
    else:
        raise AssertionError("missing class priors must be rejected")

def test_trade_window_train_and_serve_share_training_median_imputation():
    raw = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 4.0, np.nan]})
    medians = pd.Series({"a": 2.0, "b": 4.0})

    prepared = _fill_feature_frame(raw, ["a", "b"], medians)

    assert prepared.to_numpy().tolist() == [[1.0, 4.0], [2.0, 4.0], [3.0, 4.0]]


def test_trade_window_holdout_scores_actual_barrier_exit_not_time_stop_open(monkeypatch):
    import ai_trading_lab.trade_window as trade_window

    frame = _flat_market(rows=600)
    settings = load_settings("config.yaml")
    settings.symbol = "BTC/USDT"
    settings.timeframe = "15m"
    settings.trade_window_min_hours = 0.25
    settings.trade_window_max_hours = 1.0
    settings.trade_window_min_event_probability = 0.60
    settings.trade_window_min_confidence = 0.80
    settings.trade_window_target_precision = 0.80
    settings.trade_window_min_holdout_wilson = 0.0
    settings.trade_window_min_holdout_trades = 1
    settings.trade_window_min_net_return = 0.0
    settings.trade_window_require_positive_holdout_backtest = True
    settings.fee_bps = 1.0
    settings.slippage_bps = 1.0
    settings.impact_bps_per_sqrt = 0.0
    settings.require_short_borrow_cost = True
    settings.short_borrow_bps_per_bar = 0.0

    # Put a few executable up/down events in the training segment so the
    # three-class model has valid training support.
    for pos in range(30, 450, 20):
        if (pos // 20) % 2:
            frame.loc[frame.index[pos], ["open", "high", "low", "close"]] = [
                100.0, 100.6, 99.9, 100.2
            ]
        else:
            frame.loc[frame.index[pos], ["open", "high", "low", "close"]] = [
                100.0, 100.1, 99.5, 99.8
            ]

    candidate_pos = 485
    candidate_time = frame.index[candidate_pos]
    # Its take-profit triggers after entry. The later time-stop open is -5%,
    # but a real barrier-driven position has already exited at +0.25%.
    frame.loc[frame.index[candidate_pos + 2], ["open", "high", "low", "close"]] = [
        100.0, 100.6, 99.9, 100.2
    ]
    frame.loc[frame.index[candidate_pos + 5], ["open", "high", "low", "close"]] = [
        95.0, 95.1, 94.9, 95.0
    ]

    class FakeExtraTrees:
        def __init__(self, **kwargs):
            self.classes_ = np.asarray([-1, 0, 1])

        def fit(self, X, y):
            assert set(np.asarray(y, dtype=int)) == {-1, 0, 1}
            return self

        def predict_proba(self, X):
            result = np.tile([0.01, 0.98, 0.01], (len(X), 1))
            result[X.index == candidate_time] = [0.01, 0.03, 0.96]
            return result

    monkeypatch.setattr(trade_window, "ExtraTreesClassifier", FakeExtraTrees)
    report = train_trade_window_backbone(frame, settings, holdout_frac=0.20)

    assert report["holdout_candidates_before_nonoverlap"] == 1
    probability_diagnostics = report["holdout_probability_diagnostics"]
    assert probability_diagnostics["both_probability_gates_pass"] == 1
    assert probability_diagnostics["event_probability"]["n"] > 0
    assert probability_diagnostics["event_probability"]["max"] >= settings.trade_window_min_event_probability
    assert report["holdout_signals"] == 1
    assert report["holdout_economic_observations"] == 1
    assert np.isclose(report["holdout_net_return_mean"], 0.0025 - 0.0004)
    assert report["holdout_net_return_mean"] > 0.0
    assert report["holdout_net_return_compounded"] > 0.0


def test_unready_specialist_can_be_scored_only_by_explicit_validation_override(tmp_path):
    settings = load_settings("config.yaml")
    settings.symbol = "BTC/USDT"
    settings.live_max_missing_feature_fraction = 1.0
    frame = _flat_market(rows=40)
    model = DummyClassifier(strategy="prior")
    model.fit(np.asarray([[0.0], [1.0], [2.0]]), np.asarray([-1, 0, 1]))
    artifact_path = tmp_path / "candidate.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_columns": ["ret_1"],
            "fill_values": pd.Series({"ret_1": 0.0}),
            "report": {
                "production_ready": False,
                "symbol": settings.symbol,
                "timeframe": settings.timeframe,
                "model_semantics_fingerprint": model_semantics_fingerprint(settings),
                "deployment_semantics_fingerprint": deployment_semantics_fingerprint(settings),
                "holdout_precision": 0.0,
                "train_class_counts": {"-1": 1, "0": 1, "1": 1},
                "probability_semantics": BALANCED_PROBABILITY_SEMANTICS,
            },
        },
        artifact_path,
    )

    live_scores = score_trade_window_frame(
        frame, settings, artifact_path, symbol=settings.symbol
    )
    research_scores = score_trade_window_frame(
        frame,
        settings,
        artifact_path,
        symbol=settings.symbol,
        allow_unready_for_validation=True,
    )

    assert set(live_scores["trade_window_reason"]) == {"not_production_ready"}
    assert not live_scores["trade_window_ready"].any()
    assert research_scores["trade_window_available"].all()
    assert set(research_scores["trade_window_reason"]) == {
        "below_event_or_direction_threshold"
    }


def test_trade_window_refuses_binary_only_training_and_removes_stale_artifact(tmp_path, monkeypatch):
    import ai_trading_lab.trade_window as trade_window

    frame = _flat_market(rows=360)
    settings = load_settings("config.yaml")
    settings.trade_window_min_hours = 0.25
    settings.trade_window_max_hours = 1.0
    settings.timeframe = "15m"
    path = tmp_path / "trade_window_specialist.joblib"
    path.write_bytes(b"old-compatible-looking-artifact")

    def neutral_only_labels(df, horizon_bars=4, min_bars=1, pt_atr=1.25, sl_atr=0.90):
        values = np.zeros(len(df), dtype=float)
        values[-int(horizon_bars) - 1:] = np.nan
        return pd.DataFrame(
            {"direction": values, "margin": np.abs(values)},
            index=df.index,
        )

    monkeypatch.setattr(trade_window, "_barrier_labels", neutral_only_labels)

    report = train_trade_window_backbone(frame, settings, holdout_frac=0.20, save_path=path)

    assert report["production_ready"] is False
    assert report["reason"] == "insufficient_three_class_training_support"
    assert not path.exists()



def test_assess_trade_window_requires_event_probability_and_uses_training_imputation(tmp_path):
    settings = load_settings("config.yaml")
    settings.symbol = "BTC/USDT"
    settings.trade_window_min_event_probability = 0.60
    settings.trade_window_min_confidence = 0.80
    # This test exercises the persisted training-median imputer itself. The
    # production default remains fail-closed (0.0 missing-feature fraction).
    settings.live_max_missing_feature_fraction = 1.0
    frame = _flat_market(rows=40)

    def save_prior(path, classes):
        model = DummyClassifier(strategy="prior")
        x = np.arange(len(classes), dtype=float).reshape(-1, 1)
        model.fit(x, np.asarray(classes, dtype=int))
        class_counts = {
            str(label): int(np.sum(np.asarray(classes, dtype=int) == label))
            for label in (-1, 0, 1)
        }
        report = {
            "production_ready": True,
            "symbol": settings.symbol,
            "timeframe": settings.timeframe,
            "model_semantics_fingerprint": model_semantics_fingerprint(settings),
            "deployment_semantics_fingerprint": deployment_semantics_fingerprint(settings),
            "holdout_precision": 0.85,
            "train_class_counts": class_counts,
            "probability_semantics": BALANCED_PROBABILITY_SEMANTICS,
        }
        joblib.dump(
            {
                "model": model,
                "feature_columns": ["ret_1"],
                "fill_values": pd.Series({"ret_1": 0.25}),
                "report": report,
            },
            path,
        )

    strong_path = tmp_path / "strong.joblib"
    save_prior(strong_path, [-1, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    result = assess_trade_window(frame, settings, strong_path, symbol=settings.symbol)

    assert result["trade_window_available"] is True
    assert result["trade_window_ready"] is True
    assert result["trade_window_direction"] == "LONG"
    assert result["trade_window_event_probability"] >= settings.trade_window_min_event_probability
    assert result["trade_window_confidence"] >= settings.trade_window_min_confidence

    neutral_path = tmp_path / "neutral.joblib"
    save_prior(neutral_path, [-1, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    neutral = assess_trade_window(frame, settings, neutral_path, symbol=settings.symbol)

    assert neutral["trade_window_available"] is True
    assert neutral["trade_window_ready"] is False
    assert neutral["trade_window_direction"] == "FLAT"
    assert neutral["trade_window_event_probability"] < settings.trade_window_min_event_probability


def test_vectorized_barrier_labels_match_reference_scan_on_mixed_ohlc_paths():
    # Keep a deliberately simple row-by-row oracle in tests: the production
    # implementation is vectorized, but must preserve exact event priority,
    # gap handling, ambiguous-candle semantics, and time-stop economics.
    rng = np.random.default_rng(731)
    rows = 180
    returns = rng.normal(0.0, 0.002, rows)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]] * (1.0 + rng.normal(0.0, 0.0005, rows))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0001, 0.004, rows))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0001, 0.004, rows))
    # Include both-sided hits, opening gaps, and invalid rows so the test covers
    # resolution ordering as well as ordinary take-profit/stop-loss paths.
    high[35] = max(high[35], open_[35] * 1.03)
    low[35] = min(low[35], open_[35] * 0.97)
    open_[52] = high[52] * 1.01
    high[77] = np.nan
    frame = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.uniform(10.0, 1000.0, rows)},
        index=pd.date_range("2026-01-01", periods=rows, freq="15min", tz="UTC"),
    )
    horizon, minimum, pt, sl = 9, 3, 1.25, 0.90
    actual = _barrier_labels(frame, horizon, minimum, pt, sl)

    sorted_frame = frame.sort_index()
    opens = sorted_frame["open"].to_numpy(float)
    highs = sorted_frame["high"].to_numpy(float)
    lows = sorted_frame["low"].to_numpy(float)
    prev = sorted_frame["close"].shift(1)
    atr = pd.concat(
        [
            sorted_frame["high"] - sorted_frame["low"],
            (sorted_frame["high"] - prev).abs(),
            (sorted_frame["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1).rolling(14, min_periods=14).mean().to_numpy(float)
    expected = {
        "direction": np.full(rows, np.nan),
        "margin": np.full(rows, np.nan),
        "gross_return": np.full(rows, np.nan),
        "exit_position": np.full(rows, np.nan),
        "holding_bars": np.full(rows, np.nan),
    }
    statuses = np.full(rows, "unresolved", dtype=object)
    for i in range(max(0, rows - horizon - 1)):
        entry_i, exit_i = i + 1, i + horizon + 1
        entry = opens[entry_i]
        if not np.isfinite(atr[i]) or atr[i] <= 0 or not np.isfinite(entry) or entry <= 0 or not np.isfinite(opens[exit_i]) or opens[exit_i] <= 0:
            continue
        upper, lower = entry + pt * atr[i], entry - sl * atr[i]
        outcome = 0.0
        gross = opens[exit_i] / entry - 1.0
        exit_pos, held_bars, status = float(exit_i), float(horizon), "time_stop"
        for j in range(entry_i, exit_i):
            bar_open = opens[j]
            held = j - entry_i
            if not np.isfinite(bar_open) or bar_open <= 0:
                outcome, gross, exit_pos, held_bars, status = np.nan, np.nan, np.nan, np.nan, "invalid_ohlc"
                break
            if bar_open >= upper:
                outcome = 1.0 if held >= minimum else 0.0
                gross, exit_pos, held_bars, status = bar_open / entry - 1.0, float(j), float(max(1, held + 1)), "opening_gap_take_profit"
                break
            if bar_open <= lower:
                outcome = -1.0 if held >= minimum else 0.0
                gross, exit_pos, held_bars, status = bar_open / entry - 1.0, float(j), float(max(1, held + 1)), "opening_gap_stop_loss"
                break
            if not np.isfinite(highs[j]) or not np.isfinite(lows[j]):
                outcome, gross, exit_pos, held_bars, status = np.nan, np.nan, np.nan, np.nan, "invalid_ohlc"
                break
            hit_up, hit_down = highs[j] >= upper, lows[j] <= lower
            if hit_up and hit_down:
                outcome, gross, exit_pos, held_bars, status = 0.0, np.nan, float(j), float(max(1, held + 1)), "ambiguous_intrabar"
                break
            if hit_up:
                outcome = 1.0 if held >= minimum else 0.0
                gross, exit_pos, held_bars, status = upper / entry - 1.0, float(j), float(max(1, held + 1)), "take_profit"
                break
            if hit_down:
                outcome = -1.0 if held >= minimum else 0.0
                gross, exit_pos, held_bars, status = lower / entry - 1.0, float(j), float(max(1, held + 1)), "stop_loss"
                break
        expected["direction"][i] = outcome
        expected["margin"][i] = abs(outcome)
        expected["gross_return"][i] = gross
        expected["exit_position"][i] = exit_pos
        expected["holding_bars"][i] = held_bars
        statuses[i] = status

    for column, values in expected.items():
        np.testing.assert_allclose(
            actual[column].to_numpy(float), values, rtol=1e-12, atol=1e-12,
            equal_nan=True, err_msg=f"vectorized mismatch in {column}",
        )
    np.testing.assert_array_equal(actual["execution_status"].to_numpy(), statuses)
