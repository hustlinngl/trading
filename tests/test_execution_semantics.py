from dataclasses import replace
import pandas as pd
import numpy as np

from ai_trading_lab.config import load_settings
from ai_trading_lab.execution_semantics import validate_execution_alignment
from ai_trading_lab.memory import AnalogMemory


def test_execution_semantics_match_default_training_geometry():
    settings = load_settings("config.yaml")
    semantics = validate_execution_alignment(settings)
    assert semantics.horizon_bars == settings.horizon_bars == 8
    assert semantics.stop_atr_mult == settings.sl_atr == 1.0
    assert semantics.take_profit_rr == settings.pt_atr / settings.sl_atr == 1.6


def test_analog_memory_excludes_overlapping_neighbors():
    idx = pd.date_range("2026-01-01", periods=80, freq="15min", tz="UTC")
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(80, 3)), index=idx, columns=["a", "b", "c"])
    y = pd.Series(np.linspace(-1, 1, 80), index=idx)
    memory = AnalogMemory(k=8, exclusion_bars=7).fit(x, y)
    out = memory.query_many(x.iloc[[40]], exclude_self=True, exclusion_bars=7)
    assert out.iloc[0]["n"] >= 1
    ts = idx[40]
    positions = np.arange(len(memory.timestamps))
    distances = np.abs(positions - 40)
    # The directly queried point and its 7 overlapping neighbors must not be used.
    assert not np.any(distances[np.isfinite(memory.outcomes)] <= 7)
