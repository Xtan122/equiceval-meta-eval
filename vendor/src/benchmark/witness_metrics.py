"""Witness validity and coverage metrics (paper §8.2).

An exported counterexample is *valid* only if it passes an independent re-check:
the point belongs to the source model (exact, integrality respected) and it
violates the target model. Validity is the fraction of exported witnesses that
pass; coverage is the fraction of labelled feasible-set faults that carry at
least one valid witness. Both are ``None`` when their denominator is zero.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.benchmark.verified_evaluation import ir_from_dict
from src.equiceval.evidence import point_is_feasible, point_violates, violating_row_names


def localization_check(ref_ir, cand_ir, label_evidence, diagnostic):
    """Top-1 localization pairs for one record.

    The independent counterexample is feasible in one model and violates the
    other; ``true`` is the set of rows it violates there, ``predicted`` is the
    ordered list EquiCEval reports for the matching direction. Returns None when
    there is no counterexample (e.g. a certified pair).
    """
    point = (((label_evidence or {}).get("counterexample")) or {}).get("point")
    if not point:
        return None
    if not point_is_feasible(cand_ir, point):
        true_names = violating_row_names(cand_ir, point)
        predicted = (diagnostic.get("left_evidence") or {}).get("violated_constraints") or []
    elif not point_is_feasible(ref_ir, point):
        true_names = violating_row_names(ref_ir, point)
        predicted = (diagnostic.get("right_evidence") or {}).get("violated_constraints") or []
    else:
        return None
    if not true_names:
        return None
    return {"true": true_names, "predicted": list(predicted)}


def witness_check(ref_ir, cand_ir, diagnostic: Dict[str, Any]) -> Tuple[bool, bool]:
    """Return ``(present, valid)`` for one diagnostic vector.

    ``right_evidence`` (under-constraining) witnesses live in the candidate and
    must violate the reference; ``left_evidence`` (over-constraining) witnesses
    live in the reference and must violate the candidate.
    """
    present = valid = False
    for evidence, source, target in (
        (diagnostic.get("right_evidence"), cand_ir, ref_ir),
        (diagnostic.get("left_evidence"), ref_ir, cand_ir),
    ):
        if not evidence or evidence.get("status") != "disproved":
            continue
        point = evidence.get("variable_assignment") or {}
        if not point:
            continue
        present = True
        try:
            if point_is_feasible(source, point) and point_violates(target, point):
                valid = True
        except (ValueError, KeyError, TypeError):
            continue
    return present, valid


def _evaluate_affine(spec: Dict[str, Any], point: Dict[str, float]) -> float:
    coeffs = (spec or {}).get("coeffs") or {}
    return float((spec or {}).get("constant", 0.0)) + sum(
        float(coeff) * float(point[name]) for name, coeff in coeffs.items())


def _replay_points_on_raw_models(
    record: Dict[str, Any], diagnostic: Dict[str, Any], point: Dict[str, float]
):
    """Lift a reference-coordinate witness back to both raw formulations.

    The evaluator emits witnesses after projecting the candidate into reference
    coordinates. Dataset adapters may additionally project the reference or
    eliminate candidate auxiliaries. This function reverses both operations so
    the witness is checked against the LPs that were actually read from disk.
    """
    raw_ref_data = record.get("raw_reference_ir")
    raw_cand_data = record.get("raw_candidate_ir")
    if not raw_ref_data or not raw_cand_data:
        return None
    raw_ref = ir_from_dict(raw_ref_data)
    raw_cand = ir_from_dict(raw_cand_data)

    projection = record.get("reference_projection_map") or {}
    if projection:
        if any(any(name not in point for name in expr) for expr in projection.values()):
            return None
        ref_point = {
            root: sum(float(coeff) * float(point[name]) for name, coeff in expr.items())
            for root, expr in projection.items()
        }
    else:
        if any(name not in point for name in raw_ref.variables):
            return None
        ref_point = {name: float(point[name]) for name in raw_ref.variables}

    mapping = ((diagnostic.get("mapping") or {}).get("affine_substitutions")
               or record.get("provided_substitutions") or {})
    cand_point: Dict[str, float] = {}
    for name in raw_cand.variables:
        if name in mapping:
            try:
                cand_point[name] = _evaluate_affine(mapping[name], point)
            except (KeyError, TypeError, ValueError):
                return None
        elif name in point:
            cand_point[name] = float(point[name])

    pending = dict(record.get("auxiliary_eliminations") or {})
    while pending:
        progress = False
        for name, spec in list(pending.items()):
            coeffs = (spec or {}).get("coeffs") or {}
            if all(source in cand_point for source in coeffs):
                cand_point[name] = _evaluate_affine(spec, cand_point)
                del pending[name]
                progress = True
        if not progress:
            return None
    if any(name not in cand_point for name in raw_cand.variables):
        return None
    return raw_ref, raw_cand, ref_point, cand_point


def witness_check_record(record: Dict[str, Any], diagnostic: Dict[str, Any]) -> Tuple[bool, bool]:
    """Validate exported feasible-set witnesses on the raw source models."""
    present = valid = False
    for key, source_is_candidate in (("right_evidence", True), ("left_evidence", False)):
        evidence = diagnostic.get(key)
        if not evidence or evidence.get("status") != "disproved":
            continue
        point = evidence.get("variable_assignment") or {}
        if not point:
            continue
        present = True
        replay = _replay_points_on_raw_models(record, diagnostic, point)
        if replay is None:
            continue
        raw_ref, raw_cand, ref_point, cand_point = replay
        try:
            if source_is_candidate:
                valid |= (point_is_feasible(raw_cand, cand_point)
                          and point_violates(raw_ref, ref_point))
            else:
                valid |= (point_is_feasible(raw_ref, ref_point)
                          and point_violates(raw_cand, cand_point))
        except (ValueError, KeyError, TypeError):
            continue
    return present, valid


def _rate(numerator: int, denominator: int) -> Dict[str, Any]:
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def witness_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    present = [r for r in rows if (r.get("witness") or {}).get("present")]
    valid = [r for r in present if r["witness"].get("valid")]
    faults = [r for r in rows if r.get("label") is False and r.get("fault_kind") == "feasible_set"]
    faults_with_witness = [r for r in faults if (r.get("witness") or {}).get("valid")]
    return {
        "validity_rate": _rate(len(valid), len(present)),
        "coverage": _rate(len(faults_with_witness), len(faults)),
        "exported_witnesses": len(present),
    }
