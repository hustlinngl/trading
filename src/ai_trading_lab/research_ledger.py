from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
from typing import Any


SCHEMA_VERSION = 2


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now(),
        "updated_at": _now(),
        "total_trials": 0,
        "trial_runs": [],
        "holdout_accesses": [],
    }


def _path(path: str | Path) -> Path:
    return Path(path)


def load_ledger(path: str | Path) -> dict[str, Any]:
    p = _path(path)
    if not p.exists():
        return _default()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        # A corrupt ledger cannot be interpreted as a pristine holdout.
        return _default()
    if not isinstance(data, dict):
        return _default()
    data.setdefault("schema_version", SCHEMA_VERSION)
    data.setdefault("created_at", _now())
    data.setdefault("updated_at", _now())
    data.setdefault("total_trials", 0)
    data.setdefault("trial_runs", [])
    data.setdefault("holdout_accesses", [])
    return data


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def total_trials(path: str | Path) -> int:
    return max(0, int(load_ledger(path).get("total_trials", 0)))


def register_trials(
    path: str | Path,
    count: int,
    *,
    kind: str,
    dataset_fingerprint: str | None = None,
    run_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist global multiple-testing exposure for a completed experiment run."""
    n = max(0, int(count))
    ledger = load_ledger(path)
    before = max(0, int(ledger.get("total_trials", 0)))
    after = before + n
    entry = {
        "run_id": str(run_id or f"{kind}:{_now()}"),
        "kind": str(kind),
        "count": n,
        "cumulative_trials_before": before,
        "cumulative_trials_after": after,
        "dataset_fingerprint": str(dataset_fingerprint) if dataset_fingerprint else None,
        "recorded_at": _now(),
        "metadata": metadata or {},
    }
    ledger["total_trials"] = after
    ledger["trial_runs"] = [*ledger.get("trial_runs", []), entry][-5000:]
    ledger["updated_at"] = _now()
    _write_atomic(_path(path), ledger)
    return {
        "recorded": n,
        "cumulative_trials_before": before,
        "cumulative_trials_after": after,
        "run": entry,
    }


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _overlaps(start_a: Any, end_a: Any, start_b: Any, end_b: Any) -> bool:
    a0, a1 = _parse_timestamp(start_a), _parse_timestamp(end_a)
    b0, b1 = _parse_timestamp(start_b), _parse_timestamp(end_b)
    # Missing/unparseable historical bounds are uncertainty, not proof of novelty.
    if None in (a0, a1, b0, b1) or a0 > a1 or b0 > b1:
        return True
    return max(a0, b0) <= min(a1, b1)


def holdout_key(
    *,
    dataset_fingerprint: str,
    holdout_start: str,
    holdout_end: str,
    holdout_frac: float,
    scope: str = "global",
) -> str:
    """Exact id for an interval, distinct from the overlap-based pristine test."""
    payload = json.dumps(
        {
            "scope": str(scope or "global"),
            "dataset_fingerprint": str(dataset_fingerprint),
            "holdout_start": str(holdout_start),
            "holdout_end": str(holdout_end),
            "holdout_frac": float(holdout_frac),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def register_holdout_access(
    path: str | Path,
    *,
    dataset_fingerprint: str,
    holdout_start: str,
    holdout_end: str,
    holdout_frac: float,
    purpose: str,
    run_id: str | None = None,
    scope: str = "global",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a holdout read and reject overlapping windows across dataset fingerprints.

    Pristine status is based on temporal overlap within a data/model scope, not on
    an exact dataset fingerprint. Thus daily rolling windows cannot become pristine
    just because the full dataset hash changes. Legacy entries without scope are
    treated as global and conservatively block overlapping scoped windows.
    """
    ledger = load_ledger(path)
    scope = str(scope or "global")
    key = holdout_key(
        dataset_fingerprint=dataset_fingerprint,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        holdout_frac=holdout_frac,
        scope=scope,
    )
    previous = [
        x for x in ledger.get("holdout_accesses", [])
        if isinstance(x, dict) and str(x.get("holdout_key")) == key
    ]
    current_run_id = str(run_id or f"{purpose}:{_now()}")
    prior_overlapping_runs: set[str] = set()
    for prior in ledger.get("holdout_accesses", []):
        if not isinstance(prior, dict):
            continue
        prior_scope = str(prior.get("scope") or "global")
        if prior_scope not in {"global", scope} and scope != "global":
            continue
        if not _overlaps(
            holdout_start, holdout_end,
            prior.get("holdout_start"), prior.get("holdout_end"),
        ):
            continue
        prior_run = str(prior.get("run_id") or "")
        if prior_run and prior_run != current_run_id:
            prior_overlapping_runs.add(prior_run)
    access = {
        "access_id": f"{key}:{len(previous) + 1}",
        "holdout_key": key,
        "scope": scope,
        "dataset_fingerprint": str(dataset_fingerprint),
        "holdout_start": str(holdout_start),
        "holdout_end": str(holdout_end),
        "holdout_frac": float(holdout_frac),
        "purpose": str(purpose),
        "run_id": current_run_id,
        "accessed_at": _now(),
        "metadata": metadata or {},
        "overlapping_prior_run_count": len(prior_overlapping_runs),
    }
    ledger["holdout_accesses"] = [*ledger.get("holdout_accesses", []), access][-5000:]
    ledger["updated_at"] = _now()
    _write_atomic(_path(path), ledger)
    return {
        "holdout_key": key,
        "scope": scope,
        "pristine": not prior_overlapping_runs,
        "overlapping_prior_runs": sorted(prior_overlapping_runs),
        "access_count": len(previous) + 1,
        "previous_access_count": len(previous),
        "same_run_access_count": sum(1 for x in previous if str(x.get("run_id")) == current_run_id),
        "access": access,
    }
