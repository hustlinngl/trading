"""No-network regression checks for the standalone SPY research path."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ai_trading_lab import spy_research as spy


def sample_history(rows=1000, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2017-01-02", periods=rows, tz="America/New_York")
    overnight = rng.normal(0.0003, 0.004, rows)
    opening = 200 * np.exp(np.cumsum(overnight))
    closing = opening * (1 + rng.normal(0, 0.003, rows))
    return pd.DataFrame({
        "open": opening,
        "high": np.maximum(opening, closing) * 1.002,
        "low": np.minimum(opening, closing) * 0.998,
        "close": closing,
        "volume": rng.integers(100_000, 1_000_000, rows),
    }, index=idx)


def test_adjusted_history_rejects_duplicates_invalid_bars_and_short_series():
    raw = sample_history()
    bad = raw.copy()
    bad.iloc[4, bad.columns.get_loc("high")] = 1
    with pytest.raises(ValueError, match="highs"):
        spy.validate_adjusted_history(bad)
    with pytest.raises(ValueError, match="600"):
        spy.validate_adjusted_history(raw.iloc[:500])
    with pytest.raises(ValueError, match="unique"):
        spy.validate_adjusted_history(pd.concat([raw.iloc[:2], raw.iloc[:2]]))
    with pytest.raises(ValueError, match="timezone"):
        spy.validate_adjusted_history(raw.tz_localize(None))


def test_features_end_at_decision_and_label_uses_next_two_opens():
    data = spy.validate_adjusted_history(sample_history())
    x, y = spy.make_observations(data)
    assert x.index.is_monotonic_increasing
    i = data.index.get_loc(x.index[0])
    expected = data["open"].iloc[i + 2] / data["open"].iloc[i + 1] - 1
    assert y.iloc[0] == pytest.approx(expected)
    assert x.iloc[0]["return_1"] == pytest.approx(data["close"].iloc[i] / data["close"].iloc[i - 1] - 1)
    assert x.index[-1] == data.index[-3]  # last two labels unavailable


def test_temporal_split_keeps_test_outside_training():
    data = spy.validate_adjusted_history(sample_history())
    x, _ = spy.make_observations(data)
    a, b, c = spy.chronological_slices(len(x))
    assert a.stop == b.start and b.stop == c.start and c.stop == len(x)
    assert x.index[a.stop - 1] < x.index[b.start] < x.index[c.start]


def test_economic_cost_and_no_trades():
    trades = spy.evaluate_one_day(np.array([0.80, 0.20]), np.array([0.05, -0.10]), side_cost_bps=10)
    assert trades["trades"] == 1
    assert trades["compounded_net_return"] == pytest.approx(0.048)
    assert trades["win_rate_after_cost"] == 1
    nothing = spy.evaluate_one_day(np.array([0.10, 0.20]), np.array([0.05, -0.1]))
    assert nothing["trades"] == 0
    assert nothing["win_rate_after_cost"] is None
    assert nothing["compounded_net_return"] == pytest.approx(0)
    with pytest.raises(ValueError, match="Invalid"):
        spy.evaluate_one_day(np.array([1.2]), np.array([0.01]))


def test_analyze_stores_auditable_research_version_and_no_live_signal(tmp_path):
    result = spy.analyze_spy(tmp_path, adjusted_history=sample_history())
    assert result["asset"] == "SPY"
    assert result["status"] == "RESEARCH_ONLY"
    assert result["production_ready"] is False
    assert result["allows_orders"] is False
    assert result["price_adjustment"] == "dividend_and_split_adjusted_ohlcv"
    assert result["split"]["train_end"] < result["split"]["test_start"]
    folder = tmp_path / "models" / "spy_research" / result["run_id"]
    model = folder / "research_model.joblib"
    assert model.exists() and (folder / "adjusted_spy_sessions.csv").exists()
    assert hashlib.sha256(model.read_bytes()).hexdigest() == result["model_sha256"]
    assert json.loads((folder / "report.json").read_text()) == result
    other = spy.analyze_spy(tmp_path, adjusted_history=sample_history())
    assert other["run_id"] != result["run_id"]
    assert len(spy.list_studies(tmp_path)) == 2


def test_training_does_not_access_future_test_labels(tmp_path):
    original = sample_history()
    modified = original.copy()
    # Alter final candles only; model was already fitted on older sessions.
    modified.loc[modified.index[-30:], ["open", "close", "high", "low"]] *= 1.20
    old = spy.analyze_spy(tmp_path / "a", adjusted_history=original)
    new = spy.analyze_spy(tmp_path / "b", adjusted_history=modified)
    assert old["validation"] == new["validation"]
    assert old["test"] != new["test"]


def test_list_studies_skips_invalid_json(tmp_path):
    p = tmp_path / "models" / "spy_research" / "bad"
    p.mkdir(parents=True)
    (p / "report.json").write_text("{ broken", encoding="utf-8")
    assert spy.list_studies(tmp_path) == []
