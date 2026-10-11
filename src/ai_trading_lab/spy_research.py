"""SPY-only research experiment. No trade execution or production signals.

Data comes from Yahoo Finance as dividend/split-adjusted OHLCV. A decision is
made after close(t); the hypothetical trade enters at open(t+1), exits
open(t+2), and pays estimated entry and exit costs. All fits happen before the
chronological test interval. Saved bundles are local, research-only artifacts.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .spy_evidence import study_evidence

FEATURES = (
    "return_1", "return_5", "return_20", "ma_gap_10", "ma_gap_50",
    "volatility_20", "daily_range", "relative_volume",
)
ENTRY_PROBABILITY = 0.57
SIDE_COST_BPS = 7.0
MIN_OBSERVATIONS = 600


def fetch_adjusted_spy() -> pd.DataFrame:
    """Fetch SPY *adjusted* prices. Never silently substitute unadjusted prices."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("SPY requires yfinance. Launch with start_atlas.bat or install requirements-atlas.txt.") from exc
    frame = yf.Ticker("SPY").history(
        period="max", interval="1d", auto_adjust=True, actions=True
    )
    if frame is None or frame.empty:
        raise RuntimeError("Yahoo returned no adjusted SPY history. No experiment was run.")
    frame.columns = [str(x).strip().lower().replace(" ", "_") for x in frame.columns]
    frame = frame.rename(columns={"stock_splits": "splits"})
    return validate_adjusted_history(frame)


def validate_adjusted_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate externally adjusted OHLCV (callers must establish adjustment)."""
    required = ("open", "high", "low", "close", "volume")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("SPY history must have a datetime index")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("SPY dates must be ordered and unique")
    if not set(required).issubset(frame.columns):
        raise ValueError("SPY adjusted OHLCV columns are missing")
    out = frame.loc[:, required].copy()
    idx = frame.index
    if idx.tz is None:
        raise ValueError("SPY history must be timezone aware")
    # Interpret bars as US exchange sessions, not UTC calendar dates.
    sessions = idx.tz_convert("America/New_York").date
    out.index = pd.DatetimeIndex(pd.to_datetime(sessions, utc=True), name="session")
    if out.index.has_duplicates:
        raise ValueError("Multiple bars for the same SPY session")
    out = out.apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError("SPY history contains missing or nonfinite adjusted prices")
    if (out.loc[:, ("open", "high", "low", "close")] <= 0).any().any():
        raise ValueError("SPY prices must be strictly positive")
    if (out["volume"] < 0).any():
        raise ValueError("SPY volume is negative")
    if (out["high"] < out[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("SPY highs are inconsistent with adjusted prices")
    if (out["low"] > out[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("SPY lows are inconsistent with adjusted prices")
    if len(out) < MIN_OBSERVATIONS:
        raise ValueError(f"SPY requires at least {MIN_OBSERVATIONS} adjusted daily sessions")
    if (out.index[-1] - out.index[0]).days < 730:
        raise ValueError("SPY history must span multiple calendar years")
    return out


def make_observations(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Decision features at close(t), executable future open(t+1)->open(t+2)."""
    close = frame["close"]
    open_ = frame["open"]
    change = close.pct_change(fill_method=None)
    volume_median = frame["volume"].rolling(20).median().replace(0, np.nan)
    x = pd.DataFrame({
        "return_1": change,
        "return_5": close.pct_change(5, fill_method=None),
        "return_20": close.pct_change(20, fill_method=None),
        "ma_gap_10": close / close.rolling(10).mean() - 1,
        "ma_gap_50": close / close.rolling(50).mean() - 1,
        "volatility_20": change.rolling(20).std(),
        "daily_range": (frame["high"] - frame["low"]) / close,
        "relative_volume": frame["volume"] / volume_median - 1,
    }, index=frame.index)
    future = (open_.shift(-2) / open_.shift(-1) - 1).rename("future_return")
    combined = pd.concat([x, future], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(combined) < MIN_OBSERVATIONS - 60:
        raise ValueError("Not enough complete, labelled SPY sessions")
    return combined.loc[:, FEATURES], combined["future_return"]


def chronological_slices(n: int) -> tuple[slice, slice, slice]:
    """A fixed, deterministic 60/20/20 split; never randomize market time."""
    if n < MIN_OBSERVATIONS - 60:
        raise ValueError("Not enough observations for chronological evaluation")
    first, second = int(n * 0.60), int(n * 0.80)
    if first < 200 or second - first < 80 or n - second < 80:
        raise ValueError("SPY split has insufficient support")
    return slice(0, first), slice(first, second), slice(second, n)


def _compounded(returns: np.ndarray) -> float:
    values = np.asarray(returns, dtype=float)
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Nonfinite or impossible strategy returns")
    return float(np.expm1(np.log1p(values).sum()))


def _drawdown(returns: np.ndarray) -> float:
    equity = np.r_[1.0, np.cumprod(1.0 + np.asarray(returns, dtype=float))]
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def evaluate_one_day(
    probability: np.ndarray,
    future_return: np.ndarray,
    *,
    entry_probability: float = ENTRY_PROBABILITY,
    side_cost_bps: float = SIDE_COST_BPS,
) -> dict:
    """Flat or one-day LONG. Charge both sides on each selected trade."""
    p = np.asarray(probability, dtype=float)
    r = np.asarray(future_return, dtype=float)
    if p.shape != r.shape or p.ndim != 1 or not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError("Evaluation inputs must be finite, aligned 1-D arrays")
    if (p < 0).any() or (p > 1).any() or (r <= -1).any():
        raise ValueError("Invalid probability or one-day return")
    if not 0 < entry_probability < 1 or side_cost_bps < 0:
        raise ValueError("Invalid pre-registered probability/cost assumption")
    traded = p >= entry_probability
    net = np.where(traded, r - 2 * side_cost_bps / 10000.0, 0.0)
    wins = int(np.count_nonzero(net[traded] > 0))
    return {
        "sessions": int(len(r)),
        "trades": int(traded.sum()),
        "win_rate_after_cost": float(wins / traded.sum()) if traded.any() else None,
        "compounded_net_return": _compounded(net),
        "max_drawdown": _drawdown(net),
        "average_daily_net_return": float(net.mean()) if len(net) else None,
    }


def _history_digest(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def _write_report(run_dir: Path, report: dict) -> None:
    (run_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


def analyze_spy(
    project_root: str | Path = ".",
    *,
    adjusted_history: pd.DataFrame | None = None,
    side_cost_bps: float = SIDE_COST_BPS,
) -> dict:
    """One honest research run, saved as a distinct immutable local experiment.

    adjusted_history is for controlled tests or trusted already-adjusted data.
    Never submit raw/unadjusted SPY candles through this argument.
    """
    if side_cost_bps < 0:
        raise ValueError("Cost cannot be negative")
    history = (
        fetch_adjusted_spy()
        if adjusted_history is None
        else validate_adjusted_history(adjusted_history)
    )
    x, future = make_observations(history)
    training, validation, testing = chronological_slices(len(x))
    # The label at t consumes the next TWO opens. Purge the last two
    # decisions before each split boundary to avoid label-window leakage.
    fit = slice(training.start, training.stop - 2)
    val_check = slice(validation.start, validation.stop - 2)
    train_y = (future.iloc[fit] > 0).astype(int)
    if train_y.nunique() != 2:
        raise ValueError("SPY training labels contain only one class")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))
    model.fit(x.iloc[fit], train_y)
    pv = model.predict_proba(x.iloc[val_check])[:, 1]
    pt = model.predict_proba(x.iloc[testing])[:, 1]

    val = evaluate_one_day(pv, future.iloc[val_check].to_numpy(), side_cost_bps=side_cost_bps)
    test = evaluate_one_day(pt, future.iloc[testing].to_numpy(), side_cost_bps=side_cost_bps)
    evidence = study_evidence(
        pt, future.iloc[testing].to_numpy(),
        probability_gate=ENTRY_PROBABILITY, cost_bps_each_side=side_cost_bps,
    )
    # A separate always-invested, dividend-aware benchmark. Buy at the same
    # first eligible next-open and sell at the last eligible following open;
    # unlike a daily flip policy, buy & hold pays round-trip costs only once.
    test_dates = future.iloc[testing].index
    loc_first = history.index.get_loc(test_dates[0])
    loc_last = history.index.get_loc(test_dates[-1])
    first_open = float(history["open"].iloc[loc_first + 1])
    final_open = float(history["open"].iloc[loc_last + 2])
    cost = side_cost_bps / 10000.0
    benchmark_return = float((final_open / first_open) * (1 - cost) ** 2 - 1)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
    run_dir = Path(project_root) / "models" / "spy_research" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    history.to_csv(run_dir / "adjusted_spy_sessions.csv", index_label="session")
    joblib.dump(model, run_dir / "research_model.joblib")
    model_hash = hashlib.sha256((run_dir / "research_model.joblib").read_bytes()).hexdigest()
    report = {
        "run_id": run_id,
        "asset": "SPY",
        "status": "RESEARCH_ONLY",
        "production_ready": False,
        "allows_orders": False,
        "data_source": "Yahoo Finance" if adjusted_history is None else "caller_supplied_adjusted",
        "price_adjustment": "dividend_and_split_adjusted_ohlcv",
        "history_sha256": _history_digest(history),
        "model_sha256": model_hash,
        "history_sessions": len(history),
        "features": list(FEATURES),
        "execution": "features close(t); enter adjusted open(t+1), exit adjusted open(t+2)",
        "split": {
            "train_end": str(x.index[fit.stop - 1].date()),
            "validation_start": str(x.index[validation.start].date()),
            "validation_end": str(x.index[val_check.stop - 1].date()),
            "test_start": str(x.index[testing.start].date()),
            "test_end": str(x.index[testing.stop - 1].date()),
        },
        "purge_bars_at_train_and_test_boundaries": 2,
        "entry_probability": ENTRY_PROBABILITY,
        "per_side_cost_bps": side_cost_bps,
        "validation": val,
        "test": test,
        "test_evidence": evidence,
        "test_buy_and_hold_net_return": benchmark_return,
        "test_excess_vs_buy_and_hold": test["compounded_net_return"] - benchmark_return,
        "test_outperformed_buy_and_hold": test["compounded_net_return"] > benchmark_return,
        "caveat": (
            "Exploratory holdout, not a pristine independent final test. "
            "No edge or deployability is established by this experiment. "
            "Dividend-adjusted OHLC simulations are not executable fills; "
            "costs are estimates. Saved joblib models must never be loaded from untrusted sources."
        ),
    }
    _write_report(run_dir, report)
    return report


def list_studies(project_root: str | Path = ".") -> list[dict]:
    """Read JSON reports only; never deserialize untrusted model artifacts."""
    root = Path(project_root) / "models" / "spy_research"
    if not root.is_dir():
        return []
    result = []
    for path in sorted(root.glob("*/report.json"), reverse=True):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(item, dict) and item.get("asset") == "SPY":
                result.append(item)
        except (OSError, ValueError):
            continue
    return result
