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
    x = pd.DataFrame(0.0, index=idx, columns=["a", "b", "c"])
    y = pd.Series(0.0, index=idx)
    y.iloc[33:48] = 10.0
    memory = AnalogMemory(k=4, exclusion_bars=7).fit(x, y)
    query = x.iloc[[40]]
    leaked = memory.query_many(query, exclude_self=True, exclusion_bars=0)
    guarded = memory.query_many(query, exclude_self=True, exclusion_bars=7)
    assert leaked.iloc[0]["edge"] > 9.0
    assert guarded.iloc[0]["edge"] < 1.0
