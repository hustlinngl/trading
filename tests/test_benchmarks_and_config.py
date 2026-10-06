from __future__ import annotations

import pandas as pd

from ai_trading_lab.benchmarks import evaluate_suite
from ai_trading_lab.config import load_settings


def test_config_reads_economic_edge_hurdle():
    settings = load_settings("config.yaml")
    assert settings.min_edge_after_cost_bps == 5.0


def test_benchmark_suite_returns_controls():
    settings = load_settings("config.yaml")
    result = evaluate_suite(settings, n=1200)
    assert {"flat", "buy_hold", "trend", "mean_reversion"}.issubset(set(result["strategy"]))
    assert "adaptive_engine" in set(result["strategy"])
