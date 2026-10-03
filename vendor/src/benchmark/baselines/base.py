"""Common interface for scoring baselines compared against EquiCEval."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol


@dataclass
class BaselineResult:
    verdict: str            # "certified" | "disproved" | "unresolved" | "unsupported"
    score: float            # continuous anomaly score (higher = more likely faulty)
    details: Dict[str, Any] = field(default_factory=dict)
    scope: str = "whole_instance"


class Baseline(Protocol):
    name: str

    def evaluate(self, ref_ir, cand_ir) -> BaselineResult: ...
