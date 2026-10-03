"""Adapter exposing the reference-form baseline behind the Baseline interface."""
from __future__ import annotations

from src.benchmark.baselines.base import BaselineResult
from src.benchmark.reference_form_baseline import ReferenceFormBaseline

_FAULT_TOL = 1e-9


class ReferenceFormAdapter:
    """Refai & Ahmed reference-form reimplementation (modes ``refform``/``strong``)."""

    def __init__(self, mode: str = "refform"):
        self.name = mode
        self._impl = ReferenceFormBaseline(mode)

    def evaluate(self, ref_ir, cand_ir) -> BaselineResult:
        details = self._impl.evaluate(ref_ir, cand_ir)
        verdict = "disproved" if details["score"] > _FAULT_TOL else "certified"
        return BaselineResult(verdict=verdict, score=float(details["score"]),
                              details=details, scope="whole_instance")
