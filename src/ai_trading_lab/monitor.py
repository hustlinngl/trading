from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

def load_decisions(path: str | Path = 'logs/paper_decisions.jsonl') -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    rows = []
    for line in p.read_text(encoding='utf-8', errors='replace').splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return pd.DataFrame(rows)

def health_report(decisions: pd.DataFrame) -> dict:
    if decisions.empty:
        return {'status': 'NO_DATA'}
    recent = decisions.tail(96)
    scores = recent.get('score', pd.Series(dtype=float)).astype(float)
    dd = recent.get('analog_dispersion', pd.Series(dtype=float)).astype(float)
    return {
        'status': 'OK',
        'decisions': int(len(decisions)),
        'long_pct': float((recent['action'] == 'LONG').mean()) if 'action' in recent else 0.0,
        'short_pct': float((recent['action'] == 'SHORT').mean()) if 'action' in recent else 0.0,
        'flat_pct': float((recent['action'] == 'FLAT').mean()) if 'action' in recent else 0.0,
        'mean_score': float(scores.mean()) if len(scores) else 0.0,
        'mean_memory_dispersion': float(dd.mean()) if len(dd) else 0.0,
        'high_uncertainty_pct': float((recent.get('model_disagreement', pd.Series(0, index=recent.index)) > 0.08).mean()),
    }
