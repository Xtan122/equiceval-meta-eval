"""Adapter for the canonicalized reference-form baseline (``refform`` / ``strong``).

Used to meta-evaluate EquiCEval on the **self-generated labelled benchmark**
(no EquivaMap published label exists there). The baseline returns an anomaly
``score`` (larger = more likely faulty); it is thresholded into a binary decision.
"""
from __future__ import annotations

from time import perf_counter

from src.benchmark.reference_form_baseline import ReferenceFormBaseline
from src.benchmark.verified_evaluation import ir_from_dict

from expeval.adapters.common import Decision, Pair

FAULT_TOL = 1e-9


class ReferenceFormAdapter:
    def __init__(self, mode: str = "refform", tolerance: float = FAULT_TOL):
        if mode not in ("refform", "strong"):
            raise ValueError("mode must be 'refform' or 'strong'")
        self.mode = mode
        self.tolerance = tolerance
        self.name = mode

    def score(self, pair: Pair) -> Decision:
        reference = ir_from_dict(pair.reference_ir)
        candidate = ir_from_dict(pair.candidate_ir)
        started = perf_counter()
        result = ReferenceFormBaseline(self.mode).evaluate(reference, candidate)
        elapsed = perf_counter() - started
        score = float(result["score"])
        return Decision(
            pair_id=pair.pair_id,
            method=self.name,
            outcome="faulty" if score > self.tolerance else "equivalent",
            diagnostics={
                "score": score,
                "cons_rmse_exceeds": result.get("cons_rmse_exceeds"),
                "cons_precision": result.get("cons_precision"),
                "cons_recall": result.get("cons_recall"),
            },
            cost={"elapsed_seconds": elapsed},
        )
