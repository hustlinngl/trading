from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
from typing import Any

SCHEMA_VERSION = 1


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


def holdout_key(
    *,
    dataset_fingerprint: str,
    holdout_start: str,
    holdout_end: str,
    holdout_frac: float,
) -> str:
    payload = json.dumps(
        {
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
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record a holdout read and explicitly classify whether that holdout is still pristine."""
    ledger = load_ledger(path)
    key = holdout_key(
        dataset_fingerprint=dataset_fingerprint,
        holdout_start=holdout_start,
        holdout_end=holdout_end,
        holdout_frac=holdout_frac,
    )
    previous = [
        x for x in ledger.get("holdout_accesses", [])
        if isinstance(x, dict) and str(x.get("holdout_key")) == key
    ]
    access = {
        "access_id": f"{key}:{len(previous) + 1}",
        "holdout_key": key,
        "dataset_fingerprint": str(dataset_fingerprint),
        "holdout_start": str(holdout_start),
        "holdout_end": str(holdout_end),
        "holdout_frac": float(holdout_frac),
        "purpose": str(purpose),
        "run_id": str(run_id or f"{purpose}:{_now()}"),
        "accessed_at": _now(),
        "metadata": metadata or {},
    }
    ledger["holdout_accesses"] = [*ledger.get("holdout_accesses", []), access][-5000:]
    ledger["updated_at"] = _now()
    _write_atomic(_path(path), ledger)
    return {
        "holdout_key": key,
        "pristine": not previous,
        "access_count": len(previous) + 1,
        "previous_access_count": len(previous),
        "access": access,
    }
