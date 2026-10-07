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


def test_promotion_gate_has_absolute_floor_without_stale_score_dependency():
    from ai_trading_lab.promotion import promotion_gate
    stats = {
        "folds": 8,
        "total_trades": 80,
        "score": 0.15,
        "positive_fold_ratio": 0.75,
        "worst_drawdown": -0.10,
        "bootstrap_superiority_prob": 0.80,
    }
    gate = promotion_gate(
        stats,
        champion_score=0.90,
        min_folds=4,
        min_trades=50,
        max_dd=-0.20,
        min_positive_fold_ratio=0.65,
        min_bootstrap_prob=0.58,
        min_score=0.0,
        require_score_improvement=False,
    )
    assert gate["approved"]
    assert gate["checks"]["score_improves"] is True
