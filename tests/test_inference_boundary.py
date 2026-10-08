from __future__ import annotations

import inspect

import pandas as pd
import pytest

from ai_trading_lab.config import load_settings
from ai_trading_lab.inference import InferenceBundle


def test_live_and_paper_paths_do_not_import_training_engine():
    import ai_trading_lab.live as live_mod
    import ai_trading_lab.paper as paper_mod
    import ai_trading_lab.inference as inference_mod

    assert "AdaptiveEngine" not in inspect.getsource(live_mod)
    assert "AdaptiveEngine" not in inspect.getsource(paper_mod)
    assert "AdaptiveEngine" not in inspect.getsource(inference_mod)


def test_inference_features_are_target_free():
    settings = load_settings("config.yaml")
    bundle = object.__new__(InferenceBundle)
    bundle.settings = settings

    index = pd.date_range("2026-01-01", periods=320, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1000.0,
        },
        index=index,
    )

    features = bundle.features(frame)

    for name in ("future_ret", "tb_label", "tb_return", "target", "y", "_future_ret"):
        assert name not in features.columns


def test_strict_inference_rejects_missing_model_features():
    settings = load_settings("config.yaml")
    bundle = object.__new__(InferenceBundle)
    bundle.settings = settings

    class FakeModel:
        feature_cols = ["required_feature"]

    bundle.model = FakeModel()

    frame = pd.DataFrame(
        {"other_feature": [1.0]},
        index=pd.date_range("2026-01-01", periods=1, tz="UTC"),
    )

    with pytest.raises(ValueError, match=r"missing_live_features:required_feature"):
        bundle.predict_frame(frame, strict=True)
