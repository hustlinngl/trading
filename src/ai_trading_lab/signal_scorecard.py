"""Honest statistical scorecard for the live signal journal.

The terminal journals every published signal and later resolves it against
public OHLC candles. Individual outcomes are noisy, so this module answers the
question that actually matters: *do the resolved signals, after estimated
costs, show a statistically supported edge?*

Design rules (same spirit as the rest of the lab):

* Only resolved signals with a known net return enter the return statistics.
  ``AMBIGUOUS`` outcomes have an unknown return and are never imputed; their
  share is reported and a high share blocks a positive verdict.
* A verdict needs a minimum sample. Below it the answer is
  ``INSUFFICIENT_EVIDENCE`` -- never a guess.
* ``EDGE_SUPPORTED`` requires the bootstrap confidence interval of the mean net
  return to sit strictly above zero *and* a high probabilistic Sharpe ratio.
* Per-symbol and per-side breakdowns use a Bonferroni-adjusted confidence
  level, so slicing the journal into many groups does not manufacture a
  "winner" by chance.
* Returns are OHLC-based estimates, not exchange fills or account PnL.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .statistical_evidence import probabilistic_sharpe_ratio

HISTORY_NAME = "live_signal_history.jsonl"
SCORECARD_NAME = "signal_scorecard.json"

EDGE_SUPPORTED = "EDGE_SUPPORTED"
NEGATIVE_EDGE = "NEGATIVE_EDGE"
INCONCLUSIVE = "INCONCLUSIVE"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

DEFAULT_MIN_RESOLVED = 30
DEFAULT_CONFIDENCE = 0.95
DEFAULT_PSR_THRESHOLD = 0.95
DEFAULT_MAX_AMBIGUOUS_SHARE = 0.20
DEFAULT_BOOTSTRAP = 4000
_SEED = 20260101


def load_journal(path: str | Path) -> list[dict[str, Any]]:
    """Read the JSONL signal journal, skipping unreadable lines."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _finite_return(row: dict[str, Any]) -> float | None:
    value = row.get("realized_return")
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def wilson_interval(successes: int, trials: int, confidence: float) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if trials <= 0:
        return (0.0, 1.0)
    from scipy.stats import norm

    z = float(norm.ppf(0.5 + confidence / 2.0))
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2.0 * trials)) / denom
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean_interval(
    returns: np.ndarray,
    confidence: float,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = _SEED,
) -> tuple[float, float]:
    """Deterministic percentile bootstrap interval for the mean."""
    values = np.asarray(returns, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(int(n_boot), len(values)))
    means = values[idx].mean(axis=1)
    alpha = 1.0 - confidence
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def _profit_factor(returns: np.ndarray) -> float | None:
    gains = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    if losses <= 0:
        return None if gains <= 0 else float("inf")
    return gains / losses


def _max_drawdown(returns: np.ndarray) -> float:
    """Max drawdown of the compounded equity of signals taken in order (1 unit each)."""
    if len(returns) == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    return float(np.max(1.0 - equity / peak))


def score_group(
    rows: Iterable[dict[str, Any]],
    *,
    min_resolved: int = DEFAULT_MIN_RESOLVED,
    confidence: float = DEFAULT_CONFIDENCE,
    psr_threshold: float = DEFAULT_PSR_THRESHOLD,
    max_ambiguous_share: float = DEFAULT_MAX_AMBIGUOUS_SHARE,
    n_boot: int = DEFAULT_BOOTSTRAP,
) -> dict[str, Any]:
    """Score one group of journal rows and return counts, statistics and verdict."""
    rows = [r for r in rows if r.get("status", "SIGNAL") == "SIGNAL"]
    total = len(rows)
    outcomes = [str(r.get("outcome") or "").upper() for r in rows]
    open_count = sum(1 for o in outcomes if not o)
    ambiguous = sum(1 for o in outcomes if o == "AMBIGUOUS")
    returns_list = [x for x in (_finite_return(r) for r in rows) if x is not None]
    returns = np.asarray(returns_list, dtype=float)
    n = int(len(returns))
    closed = total - open_count
    ambiguous_share = ambiguous / closed if closed else 0.0

    result: dict[str, Any] = {
        "signals": total,
        "open": open_count,
        "closed": closed,
        "ambiguous": ambiguous,
        "ambiguous_share": round(ambiguous_share, 4),
        "with_return": n,
        "confidence": confidence,
    }

    if n == 0:
        result.update(
            verdict=INSUFFICIENT_EVIDENCE,
            reason="no resolved signal has a known net return yet",
        )
        return result

    mean = float(returns.mean())
    sd = float(returns.std(ddof=1)) if n > 1 else 0.0
    t_stat = mean / (sd / math.sqrt(n)) if sd > 1e-12 and n > 1 else 0.0
    wins = int((returns > 0).sum())
    hit_lo, hit_hi = wilson_interval(wins, n, confidence)
    ci_lo, ci_hi = bootstrap_mean_interval(returns, confidence, n_boot=n_boot)
    psr = float(probabilistic_sharpe_ratio(returns, periods_per_year=1.0, benchmark_sr=0.0)) if n > 2 else 0.0
    pf = _profit_factor(returns)

    result.update(
        mean_net_return=mean,
        median_net_return=float(np.median(returns)),
        std_net_return=sd,
        total_net_return=float(returns.sum()),
        t_stat=float(t_stat),
        hit_rate=wins / n,
        hit_rate_ci=[hit_lo, hit_hi],
        mean_ci=[ci_lo, ci_hi],
        profit_factor=pf if pf is None or math.isfinite(pf) else None,
        psr=psr,
        max_drawdown=_max_drawdown(returns),
        best=float(returns.max()),
        worst=float(returns.min()),
    )

    if n < min_resolved:
        result.update(
            verdict=INSUFFICIENT_EVIDENCE,
            reason=f"{n} resolved signals with a return; at least {min_resolved} are required",
        )
    elif ci_hi < 0.0:
        result.update(
            verdict=NEGATIVE_EDGE,
            reason="the confidence interval of the mean net return is entirely below zero",
        )
    elif ci_lo > 0.0 and psr >= psr_threshold:
        if ambiguous_share > max_ambiguous_share:
            result.update(
                verdict=INCONCLUSIVE,
                reason=(
                    f"{ambiguous_share:.0%} of closed signals are ambiguous with unknown return; "
                    "the positive result may be biased"
                ),
            )
        else:
            result.update(
                verdict=EDGE_SUPPORTED,
                reason="mean net return is above zero with the required confidence and PSR",
            )
    else:
        result.update(
            verdict=INCONCLUSIVE,
            reason="the confidence interval of the mean net return includes zero",
        )
    return result


def build_scorecard(
    rows: list[dict[str, Any]],
    *,
    min_resolved: int = DEFAULT_MIN_RESOLVED,
    confidence: float = DEFAULT_CONFIDENCE,
    psr_threshold: float = DEFAULT_PSR_THRESHOLD,
    max_ambiguous_share: float = DEFAULT_MAX_AMBIGUOUS_SHARE,
    n_boot: int = DEFAULT_BOOTSTRAP,
) -> dict[str, Any]:
    """Build the overall scorecard plus per-side and per-symbol breakdowns."""
    signals = [r for r in rows if r.get("status", "SIGNAL") == "SIGNAL"]

    def sub_confidence(groups: int) -> float:
        # Bonferroni: keep the family-wise error at (1 - confidence).
        return 1.0 - (1.0 - confidence) / max(1, groups)

    kwargs = dict(
        min_resolved=min_resolved,
        psr_threshold=psr_threshold,
        max_ambiguous_share=max_ambiguous_share,
        n_boot=n_boot,
    )

    by_side: dict[str, list[dict[str, Any]]] = {}
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    for r in signals:
        side = str(r.get("signal", "")).upper()
        if side in {"LONG", "SHORT"}:
            by_side.setdefault(side, []).append(r)
        symbol = str(r.get("symbol", "")).strip()
        if symbol:
            by_symbol.setdefault(symbol, []).append(r)

    side_conf = sub_confidence(len(by_side))
    symbol_conf = sub_confidence(len(by_symbol))
    return {
        "disclaimer": (
            "Returns are OHLC-derived estimates after configured costs, not exchange fills or "
            "account PnL. Signals taken on the same bar across assets are correlated, so the "
            "effective sample can be smaller than the signal count."
        ),
        "overall": score_group(signals, confidence=confidence, **kwargs),
        "by_side": {
            k: score_group(v, confidence=side_conf, **kwargs) for k, v in sorted(by_side.items())
        },
        "by_symbol": {
            k: score_group(v, confidence=symbol_conf, **kwargs) for k, v in sorted(by_symbol.items())
        },
    }


def format_scorecard(card: dict[str, Any]) -> str:
    """Render the scorecard as a readable text report."""
    lines: list[str] = []

    def pct(x: Any, digits: int = 2) -> str:
        return "n/a" if x is None or not math.isfinite(float(x)) else f"{float(x) * 100:.{digits}f}%"

    def block(title: str, g: dict[str, Any]) -> None:
        lines.append(f"{title}: {g['verdict']}")
        lines.append(
            f"  signals {g['signals']} | closed {g['closed']} | open {g['open']} | "
            f"ambiguous {g['ambiguous']} | with return {g['with_return']}"
        )
        if g["with_return"]:
            lo, hi = g["mean_ci"]
            hlo, hhi = g["hit_rate_ci"]
            lines.append(
                f"  mean net {pct(g['mean_net_return'])} "
                f"(CI {g['confidence']:.1%}: {pct(lo)} .. {pct(hi)}) | t {g['t_stat']:.2f} | PSR {g['psr']:.2f}"
            )
            pf = g["profit_factor"]
            lines.append(
                f"  hit rate {pct(g['hit_rate'], 1)} ({pct(hlo, 1)} .. {pct(hhi, 1)}) | "
                f"profit factor {'n/a' if pf is None else f'{pf:.2f}'} | max drawdown {pct(g['max_drawdown'], 1)}"
            )
        lines.append(f"  -> {g['reason']}")

    block("OVERALL", card["overall"])
    for name, g in card["by_side"].items():
        lines.append("")
        block(f"SIDE {name}", g)
    for name, g in card["by_symbol"].items():
        lines.append("")
        block(f"SYMBOL {name}", g)
    lines.append("")
    lines.append(card["disclaimer"])
    return "\n".join(lines)


def run_scorecard(
    root: str | Path = ".",
    *,
    min_resolved: int = DEFAULT_MIN_RESOLVED,
    confidence: float = DEFAULT_CONFIDENCE,
    write: bool = True,
) -> dict[str, Any]:
    """Load the journal under ``root/logs`` and build (and optionally persist) the scorecard."""
    root = Path(root)
    rows = load_journal(root / "logs" / HISTORY_NAME)
    card = build_scorecard(rows, min_resolved=min_resolved, confidence=confidence)
    if write:
        out = root / "logs" / SCORECARD_NAME
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
        tmp.replace(out)
    return card
