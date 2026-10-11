"""Forward-only SPY shadow observations.

The journal is NOT an order system and never promotes a model. The newest
saved research model is chosen by creation time, not by highest viewed
holdout return. Each calendar session may be recorded once; old forecasts
are never backfilled or replaced. Outcomes mature only on later closed bars.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .spy_research import (
    ENTRY_PROBABILITY, FEATURES, SIDE_COST_BPS,
    fetch_adjusted_spy, list_studies, make_features_only,
    only_closed_sessions, validate_adjusted_history,
)


def _journal_path(root: Path) -> Path:
    return root / "logs" / "spy_shadow_observations.json"


def read_shadow_journal(root: str | Path = ".") -> list[dict]:
    path = _journal_path(Path(root))
    if not path.is_file():
        return []
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, list):
            raise ValueError("Shadow journal has an unexpected schema")
        if not all(isinstance(record, dict) for record in result):
            raise ValueError("Shadow journal contains invalid records")
        return result
    except json.JSONDecodeError as exc:
        raise ValueError("Shadow journal is corrupted; refusing to discard or overwrite it") from exc


def _save(root: Path, journal: list[dict]) -> None:
    path = _journal_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as fh:
            json.dump(journal, fh, indent=2, sort_keys=True, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_trusted_latest_model(root: Path):
    """Verify a saved local artifact against its adjacent audit report.

    The entire local folder, including the report, is a trusted boundary.
    Never import/download an untrusted .joblib or move this to a remote API.
    """
    studies = list_studies(root)
    if not studies:
        raise RuntimeError("No SPY research model. Run Analyze SPY first.")
    study = studies[0]
    run_id = str(study.get("run_id", ""))
    if not run_id or not run_id.replace("_", "").isalnum():
        raise ValueError("SPY study identifier is invalid")
    folder = root / "models" / "spy_research" / run_id
    report_file = folder / "report.json"
    if not report_file.is_file():
        raise RuntimeError("Missing research evidence manifest")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if report != study or report.get("production_ready") is not False or report.get("allows_orders") is not False:
        raise RuntimeError("Model report is not an authorized research-only artifact")
    if tuple(report.get("features", [])) != FEATURES:
        raise RuntimeError("The model was trained with different features")
    path = folder / "research_model.joblib"
    if not path.is_file():
        raise RuntimeError("The selected local model file is missing")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != report.get("model_sha256"):
        raise RuntimeError("The model fingerprint changed since training")
    model = joblib.load(path)  # trusted local artifact ONLY
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("The research model has no probability interface")
    return report, model


def _resolve(records: list[dict], history: pd.DataFrame, side_cost_bps: float) -> int:
    resolved = 0
    for record in records:
        if record.get("outcome") is not None:
            continue
        date = str(record.get("session", ""))
        try:
            loc = int(history.index.get_loc(pd.Timestamp(date, tz="UTC")))
        except (KeyError, ValueError, TypeError):
            continue
        if loc + 2 >= len(history):
            continue  # mature only after *closed* t+2 is available
        entry = float(history["open"].iloc[loc + 1])
        exit_ = float(history["open"].iloc[loc + 2])
        gross = exit_ / entry - 1.0
        acted = record.get("shadow_side") == "LONG"
        record["outcome"] = {
            "entry_session": str(history.index[loc + 1].date()),
            "exit_session": str(history.index[loc + 2].date()),
            "market_open_to_open_return": float(gross),
            "shadow_net_return": float(gross - 2.0 * side_cost_bps / 10000.0) if acted else None,
            "paper_only": True,
        }
        resolved += 1
    return resolved


def track_spy(
    root: str | Path = ".",
    *,
    adjusted_history: pd.DataFrame | None = None,
    as_of=None,
) -> dict:
    """Settle matured observations and optionally add *today's* new shadow decision.

    Works once per recently closed NYSE session. Older missed sessions are
    never silently reconstructed with hindsight.
    """
    root = Path(root)
    now = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    if now.tzinfo is None:
        raise ValueError("as_of must be timezone aware")
    current = (
        fetch_adjusted_spy() if adjusted_history is None
        else validate_adjusted_history(only_closed_sessions(adjusted_history, as_of=now))
    )
    current = validate_adjusted_history(current)
    report, model = _read_trusted_latest_model(root)
    journal = read_shadow_journal(root)
    if len({str(x.get("session")) for x in journal}) != len(journal):
        raise ValueError("Duplicate shadow session records; refusing to rewrite")
    cost = float(report["per_side_cost_bps"])
    resolved = _resolve(journal, current, cost)
    day = str(current.index[-1].date())
    now_ny = now.tz_convert("America/New_York")
    stale_days = (now_ny.date() - current.index[-1].date()).days
    if stale_days < 0 or stale_days > 7:
        raise RuntimeError("SPY latest completed daily candle is stale or future-dated. No forecast recorded.")
    added = False
    if not any(x.get("session") == day for x in journal):
        if day <= str(report["split"]["test_end"]):
            raise RuntimeError("New shadow observation must occur AFTER the saved historical test interval")
        frame = make_features_only(current)
        features = frame.iloc[[-1]]
        if not np.isfinite(features.to_numpy()).all():
            raise RuntimeError("SPY latest closed features are invalid; no forecast recorded.")
        probabilities = model.predict_proba(features)
        probability = float(probabilities[0, 1])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Research model returned an invalid probability")
        journal.append({
            "session": day,
            "observed_at_utc": now.tz_convert("UTC").isoformat(),
            "model_run_id": report["run_id"],
            "model_sha256": report["model_sha256"],
            "probability_up": probability,
            "probability_gate": ENTRY_PROBABILITY,
            "shadow_side": "LONG" if probability >= ENTRY_PROBABILITY else "WAIT",
            "paper_only": True,
            "outcome": None,
        })
        added = True
    if resolved or added:
        _save(root, journal)
    resolved_long = [
        x["outcome"]["shadow_net_return"] for x in journal
        if x.get("shadow_side") == "LONG" and isinstance(x.get("outcome"), dict)
        and x["outcome"].get("shadow_net_return") is not None
    ]
    return {
        "status": "SHADOW_ONLY",
        "production_ready": False,
        "run_id": report["run_id"],
        "as_of_session": day,
        "recorded_new_observation": added,
        "new_outcomes_resolved": resolved,
        "observations": len(journal),
        "resolved_shadow_trades": len(resolved_long),
        "net_return_on_resolved_shadow_trades": (
            float(np.expm1(np.log1p(resolved_long).sum())) if resolved_long else None
        ),
        "latest": journal[-1] if journal else None,
        "note": (
            "Local one-session-at-a-time prospective observations, never hindsight "
            "backfilled. Paper estimates, not executable orders, profits, or model qualification."
        ),
    }
