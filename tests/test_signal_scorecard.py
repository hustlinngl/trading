import json

import numpy as np

from ai_trading_lab.signal_scorecard import (
    EDGE_SUPPORTED,
    INCONCLUSIVE,
    INSUFFICIENT_EVIDENCE,
    NEGATIVE_EDGE,
    bootstrap_mean_interval,
    build_scorecard,
    format_scorecard,
    run_scorecard,
    score_group,
    wilson_interval,
)


def _row(ret, symbol="BTC/USDT", side="LONG", outcome=None):
    if outcome is None:
        outcome = "WIN" if ret is not None and ret > 0 else "LOSS"
    return {
        "status": "SIGNAL",
        "symbol": symbol,
        "signal": side,
        "outcome": outcome,
        "realized_return": ret,
    }


def _noisy(mean, n, sd=0.01, seed=1):
    rng = np.random.default_rng(seed)
    return [float(x) for x in rng.normal(mean, sd, n)]


def test_small_sample_never_gets_a_verdict():
    rows = [_row(0.02) for _ in range(10)]
    g = score_group(rows, min_resolved=30)
    assert g["verdict"] == INSUFFICIENT_EVIDENCE
    assert g["with_return"] == 10


def test_empty_journal_is_insufficient():
    assert score_group([])["verdict"] == INSUFFICIENT_EVIDENCE


def test_clear_positive_edge_is_supported():
    rows = [_row(r) for r in _noisy(0.01, 200, sd=0.01)]
    g = score_group(rows, min_resolved=30)
    assert g["verdict"] == EDGE_SUPPORTED
    assert g["mean_ci"][0] > 0


def test_clear_negative_edge_is_flagged():
    rows = [_row(r) for r in _noisy(-0.01, 200, sd=0.01)]
    assert score_group(rows, min_resolved=30)["verdict"] == NEGATIVE_EDGE


def test_pure_noise_is_inconclusive():
    rows = [_row(r) for r in _noisy(0.0, 120, sd=0.02, seed=7)]
    assert score_group(rows, min_resolved=30)["verdict"] == INCONCLUSIVE


def test_ambiguous_signals_are_not_imputed_and_block_positive_verdict():
    good = [_row(r) for r in _noisy(0.01, 100, sd=0.01)]
    ambiguous = [_row(None, outcome="AMBIGUOUS") for _ in range(60)]
    g = score_group(good + ambiguous, min_resolved=30)
    assert g["with_return"] == 100
    assert g["ambiguous"] == 60
    assert g["verdict"] == INCONCLUSIVE


def test_open_signals_are_counted_but_not_scored():
    rows = [_row(0.01)] * 5 + [
        {"status": "SIGNAL", "symbol": "BTC/USDT", "signal": "LONG"} for _ in range(3)
    ]
    g = score_group(rows, min_resolved=2)
    assert g["open"] == 3
    assert g["with_return"] == 5


def test_non_finite_and_bool_returns_are_ignored():
    rows = [_row(float("nan")), _row(float("inf")), _row(True), _row(0.01)]
    assert score_group(rows, min_resolved=1)["with_return"] == 1


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    values = np.asarray(_noisy(0.005, 80))
    a = bootstrap_mean_interval(values, 0.95)
    b = bootstrap_mean_interval(values, 0.95)
    assert a == b
    assert a[0] < values.mean() < a[1]


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(0, 10, 0.95)
    assert abs(lo) < 1e-12 and 0.0 < hi < 0.5
    lo, hi = wilson_interval(10, 10, 0.95)
    assert abs(hi - 1.0) < 1e-12 and lo > 0.5
    assert wilson_interval(0, 0, 0.95) == (0.0, 1.0)


def test_breakdowns_use_bonferroni_adjusted_confidence():
    rows = [_row(r, symbol="BTC/USDT") for r in _noisy(0.0, 60, seed=2)]
    rows += [_row(r, symbol="ETH/USDT") for r in _noisy(0.0, 60, seed=3)]
    card = build_scorecard(rows, min_resolved=30, confidence=0.95)
    assert card["overall"]["confidence"] == 0.95
    assert abs(card["by_symbol"]["BTC/USDT"]["confidence"] - 0.975) < 1e-12


def test_non_signal_rows_are_ignored():
    rows = [_row(0.01), {"status": "WAIT", "symbol": "BTC/USDT", "realized_return": 5.0}]
    assert score_group(rows, min_resolved=1)["with_return"] == 1


def test_run_scorecard_reads_journal_and_writes_report(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    lines = [json.dumps(_row(r)) for r in _noisy(0.01, 60)]
    (logs / "live_signal_history.jsonl").write_text("\n".join(lines) + "\nnot json\n", encoding="utf-8")
    card = run_scorecard(tmp_path, min_resolved=30)
    assert card["overall"]["with_return"] == 60
    saved = json.loads((logs / "signal_scorecard.json").read_text(encoding="utf-8"))
    assert saved["overall"]["verdict"] == card["overall"]["verdict"]
    text = format_scorecard(card)
    assert "OVERALL" in text and "SYMBOL BTC/USDT" in text


def test_missing_journal_gives_insufficient_evidence(tmp_path):
    card = run_scorecard(tmp_path, write=False)
    assert card["overall"]["verdict"] == INSUFFICIENT_EVIDENCE
    assert "OVERALL" in format_scorecard(card)
