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
    assert report["holdout_signals"] == 1
    assert report["holdout_economic_observations"] == 1
    assert np.isclose(report["holdout_net_return_mean"], 0.0025 - 0.0004)
    assert report["holdout_net_return_mean"] > 0.0
    assert report["holdout_net_return_compounded"] > 0.0


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
                "feature_columns": ["synthetic_feature"],
                "fill_values": pd.Series({"synthetic_feature": 0.25}),
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
