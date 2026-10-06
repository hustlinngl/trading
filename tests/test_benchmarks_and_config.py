from __future__ import annotations

from ai_trading_lab.benchmarks import evaluate_suite
from ai_trading_lab.config import load_settings


def test_config_reads_and_overrides_economic_edge_hurdle(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        """
model:
  min_expected_return: 0.001
  min_edge_after_cost_bps: 12.5
""",
        encoding="utf-8",
    )
    settings = load_settings(config)
    assert settings.min_expected_return == 0.001
    assert settings.min_edge_after_cost_bps == 12.5


def test_benchmark_suite_returns_controls():
    settings = load_settings("config.yaml")
    result = evaluate_suite(settings, n=1200)
    assert {"flat", "buy_hold", "trend", "mean_reversion"}.issubset(set(result["strategy"]))
    assert "adaptive_engine" in set(result["strategy"])
