from __future__ import annotations

import json

from ai_trading_lab.benchmarks import evaluate_suite, save_suite_report
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


def test_benchmark_suite_returns_controls_and_signal_gate_diagnostics(tmp_path):
    settings = load_settings("config.yaml")
    result = evaluate_suite(settings, n=1200)
    assert {"flat", "buy_hold", "trend", "mean_reversion"}.issubset(set(result["strategy"]))
    assert "adaptive_engine" in set(result["strategy"])

    required = {
        "data_source", "data_rows", "train_rows", "test_rows", "test_start", "test_end",
        "action_long_count", "action_short_count", "action_flat_count",
        "policy_pass_all_gates", "core_live_gate_candidates",
        "core_live_gate_approved", "core_live_gate_rejected",
        "core_live_gate_rejections",
    }
    assert required.issubset(result.columns)
    assert set(result["data_source"]).issubset({"cached_dataset", "synthetic_fallback"})
    assert (result["data_rows"] == 1200).all()
    assert (result["train_rows"] + result["test_rows"] == result["data_rows"]).all()
    assert (result["test_rows"] > 0).all()
    assert (
        result["action_long_count"]
        + result["action_short_count"]
        + result["action_flat_count"]
        == result["test_rows"]
    ).all()

    adaptive = result.loc[result["strategy"] == "adaptive_engine"].iloc[0]
    assert adaptive["core_live_gate_approved"] <= adaptive["core_live_gate_candidates"]
    assert adaptive["core_live_gate_rejected"] == (
        adaptive["core_live_gate_candidates"] - adaptive["core_live_gate_approved"]
    )
    assert isinstance(json.loads(adaptive["core_live_gate_rejections"]), dict)

    report_path = save_suite_report(result, tmp_path / "benchmark_suite.json")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["metadata"]["data_source"] == adaptive["data_source"]
    assert "trade-window specialist" in payload["diagnostic_scope"]


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


def test_settings_default_to_strict_deployment():
    from ai_trading_lab.config import Settings
    settings = Settings()
    assert settings.trade_window_required_for_signal is True
    assert settings.require_deployment_manifest_for_signal is True


def test_release_versions_are_aligned():
    from pathlib import Path
    import re
    from ai_trading_lab import __version__
    root = Path(__file__).resolve().parents[1]
    version_file = (root / "VERSION").read_text(encoding="utf-8").strip()
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert match is not None
    readme = (root / "README.md").read_text(encoding="utf-8")
    release_match = re.search(r"^Release [*]{2}([^*]+)[*]{2}", readme, re.MULTILINE)
    assert release_match is not None
    assert __version__ == version_file == match.group(1) == release_match.group(1)
