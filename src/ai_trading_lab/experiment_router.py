from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import math
import numpy as np

@dataclass
class ResearchTask:
    task_id: str
    kind: str
    hypothesis: str
    priority: float
    expected_information: float
    expected_edge: float
    compute_cost: float
    novelty: float
    parent_id: str | None = None
    def to_dict(self):
        return asdict(self)

class ResearchRouter:
    """Allocates limited research compute using an information-per-cost score."""
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
    def rank(self, tasks: list[ResearchTask], budget: float = 10.0, top_k: int = 12) -> list[ResearchTask]:
        scored = []
        for t in tasks:
            evidence = max(0.0, min(1.0, t.expected_information))
            novelty = max(0.0, min(1.0, t.novelty))
            edge = max(-1.0, min(1.0, t.expected_edge))
            value = (0.55 * evidence + 0.30 * novelty + 0.15 * max(edge, 0.0)) / max(0.25, t.compute_cost)
            value += float(self.rng.uniform(0, 1e-5))
            scored.append((value, t))
        scored.sort(key=lambda z: z[0], reverse=True)
        picked, spent = [], 0.0
        for value, t in scored:
            if len(picked) >= top_k:
                break
            if spent + t.compute_cost <= budget or not picked:
                t.priority = float(value)
                picked.append(t); spent += t.compute_cost
        return picked
    @staticmethod
    def from_hypotheses(hypotheses: list[dict[str, Any]]) -> list[ResearchTask]:
        out = []
        for i, h in enumerate(hypotheses):
            evidence = min(1.0, len(h.get("evidence", [])) / 5.0)
            complexity = max(1.0, float(h.get("complexity", 1)))
            out.append(ResearchTask(
                task_id=str(h.get("key") or f"hyp-{i}"), kind="hypothesis",
                hypothesis=str(h.get("thesis") or h.get("hypothesis") or ""), priority=0.0,
                expected_information=0.6 + 0.4 * evidence,
                expected_edge=0.15 if h.get("feature_family") in {"trend", "breakout"} else 0.05,
                compute_cost=1.0 + 0.6 * math.log1p(complexity), novelty=0.7 if evidence < 0.5 else 0.4,
            ))
        return out
