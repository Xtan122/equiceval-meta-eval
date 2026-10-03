"""Adapter for the EquiCEval engine (vendored under ``vendor/src``)."""
from __future__ import annotations

from time import perf_counter

from src.benchmark.verified_evaluation import ir_from_dict
from src.equiceval.evaluator import EquiCEvalEvaluator

from expeval.adapters.common import Decision, Pair

VERDICT_TO_OUTCOME = {
    "certified": "equivalent",
    "within_tolerance": "tolerance",
    "within_scope": "equivalent",
    "disproved": "faulty",
    "unresolved": "unresolved",
    "unsupported": "unsupported",
}


class EquiCEvalAdapter:
    name = "equiceval"

    def __init__(self, seconds: float = 1.0, mode: str = "certificates"):
        if mode not in {"direct_only", "certificates", "early_stop"}:
            raise ValueError(f"unknown mode {mode!r}")
        self.seconds = seconds
        self.mode = mode

    def score(self, pair: Pair) -> Decision:
        reference = ir_from_dict(pair.reference_ir)
        candidate = ir_from_dict(pair.candidate_ir)
        started = perf_counter()
        result = EquiCEvalEvaluator(
            solver_time_limit=self.seconds,
            contract=pair.contract,
            use_certificates=self.mode != "direct_only",
            stop_on_witness=self.mode == "early_stop",
        ).evaluate(reference, candidate,
                   provided_substitutions=pair.provided_substitutions or None)
        diagnostic = result.to_diagnostic_vector()
        elapsed = perf_counter() - started
        return Decision(
            pair_id=pair.pair_id,
            method=self.name,
            outcome=VERDICT_TO_OUTCOME[diagnostic["verdict"]],
            diagnostics={
                "verdict": diagnostic["verdict"],
                "Delta_right": diagnostic.get("Delta_right"),
                "Delta_left": diagnostic.get("Delta_left"),
                "MatchCoverage": diagnostic.get("MatchCoverage"),
                "MapCoverage": diagnostic.get("MapCoverage"),
            },
            cost={
                "elapsed_seconds": elapsed,
                "solver_calls": (diagnostic.get("budget") or {}).get("solver_calls", 0),
            },
        )
