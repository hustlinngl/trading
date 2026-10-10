from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from ai_trading_lab.config import load_settings
from ai_trading_lab.deployment import deployment_semantics_fingerprint, model_semantics_fingerprint
from ai_trading_lab.trade_window import (
    _barrier_labels,
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
        100.0, 110.0, 99.9, 100.0
    ]

    labels = _barrier_labels(frame, horizon_bars=3, min_bars=1, pt_atr=1.25, sl_atr=0.90)

    assert labels.loc[frame.index[14], "direction"] == 0.0


def test_event_probability_is_separate_from_conditional_direction_confidence():
    long = directional_event_probabilities([0.10, 0.20, 0.70], [-1, 0, 1])
    assert long["direction"] == "LONG"
    assert np.isclose(long["event_probability"], 0.80)
    assert np.isclose(long["direction_confidence"], 0.875)

    neutral = directional_event_probabilities([0.01, 0.98, 0.01], [-1, 0, 1])
    assert neutral["direction"] == "FLAT"
    assert np.isclose(neutral["event_probability"], 0.02)
    assert neutral["direction_confidence"] < 0.80


def test_trade_window_train_and_serve_share_training_median_imputation():
    raw = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 4.0, np.nan]})
    medians = pd.Series({"a": 2.0, "b": 4.0})

    prepared = _fill_feature_frame(raw, ["a", "b"], medians)

    assert prepared.to_numpy().tolist() == [[1.0, 4.0], [2.0, 4.0], [3.0, 4.0]]


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
        report = {
            "production_ready": True,
            "symbol": settings.symbol,
            "timeframe": settings.timeframe,
            "model_semantics_fingerprint": model_semantics_fingerprint(settings),
            "deployment_semantics_fingerprint": deployment_semantics_fingerprint(settings),
            "holdout_precision": 0.85,
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
