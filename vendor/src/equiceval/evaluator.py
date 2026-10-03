"""Reference-based evaluation under an explicit semantic contract."""
from dataclasses import dataclass, field, replace
from math import isfinite
from fractions import Fraction
from typing import Optional, Any, List
from src.equiceval.canonical_ir import CanonicalIR, CanonicalConstraint, build_projection_certificate, _exact_decimal_float
from src.equiceval.contracts import (
    ARGMIN,
    FEASIBLE_SET,
    OBJECTIVE_AFFINE,
    OBJECTIVE_VALUE,
    PRIMARY_EQUIVAFORMULATION_CONTRACT,
    SUPPORTED_CONTRACTS,
)
from src.equiceval.matching import EquivalenceMatcher, ConstraintMatchResult
from src.equiceval.directed_query import DirectedDiscrepancyEvaluator, CounterexampleWitness
from src.equiceval.behavioral_discrepancy import BehavioralDiscrepancyEvaluator
from src.equiceval.objective_discrepancy import (
    ObjectiveDiscrepancyEvaluator,
    detect_positive_affine_alignment,
)
from src.equiceval.precheck import ModelPreChecker
from src.utils.solver_wrapper import QueryBudget

DIAGNOSTIC_SCHEMA_VERSION = "2.1"


@dataclass
class EquiCEvalResult:
    problem_name: str
    var_match_score: float
    con_match_score: float
    d_sat: float
    d_margin: float
    delta_right: float
    delta_left: float
    delta_f: float
    optimality_gap: Optional[float]
    witness_right: CounterexampleWitness
    witness_left: CounterexampleWitness
    matched_details: List[ConstraintMatchResult]
    status: Optional[Any] = None
    verdict: str = "unresolved"
    contract: str = PRIMARY_EQUIVAFORMULATION_CONTRACT
    feasibility_verdict: str = "unresolved"
    objective_verdict: str = "unresolved"
    objective_evidence: Optional[Any] = None
    budget: dict = field(default_factory=dict)
    mapping: dict = field(default_factory=dict)
    reference_status: Optional[Any] = None
    notes: List[str] = field(default_factory=list)
    component_report: dict = field(default_factory=dict)
    sampling: dict = field(default_factory=dict)
    absolute_optimality_gap: Optional[float] = None
    objective_coefficients: dict = field(default_factory=dict)
    delta_value: Optional[float] = None
    cross_regret_ref_given_cand: Optional[float] = None
    cross_regret_cand_given_ref: Optional[float] = None
    objective5_status: str = "not_requested"
    optimizer_sets_equal: bool = False
    behavioral_solver_calls: int = 0
    objective5_result: Optional[Any] = None

    def to_diagnostic_vector(self):
        def safe(x):
            return float(x) if x is not None and isfinite(x) else None
        return {
            "schema_version": DIAGNOSTIC_SCHEMA_VERSION, "contract": self.contract, "verdict": self.verdict,
            "feasibility_verdict": self.feasibility_verdict, "objective_verdict": self.objective_verdict,
            "VarMatch": safe(self.var_match_score), "ConMatch": safe(self.con_match_score),
            "MapCoverage": safe(self.var_match_score),
            "MatchCoverage": self.component_report.get("certified_coverage"),
            "NumericalMatchCoverage": self.component_report.get("accepted_coverage"),
            "D_sat": safe(self.d_sat), "D_margin": safe(self.d_margin),
            "Delta_right": safe(self.delta_right), "Delta_left": safe(self.delta_left),
            "Delta_f": safe(self.delta_f), "Delta_value": safe(self.delta_value),
            "OptimalityGap": safe(self.optimality_gap),
            "Gap_abs": safe(self.absolute_optimality_gap), "Gap_sym": safe(self.optimality_gap),
            "ObjectiveStatus": self.objective5_status,
            "R_ref_given_cand": safe(self.cross_regret_ref_given_cand),
            "R_cand_given_ref": safe(self.cross_regret_cand_given_ref),
            "optimizer_sets_equal": self.optimizer_sets_equal,
            "cross_regret": {"status": self.objective5_status,
                             "reference_from_candidate": safe(self.cross_regret_ref_given_cand),
                             "candidate_from_reference": safe(self.cross_regret_cand_given_ref)},
            "SolverCalls": self.budget.get("solver_calls"),
            "BehavioralSolverCalls": self.behavioral_solver_calls,
            "Status": self.status.to_dict() if self.status else None,
            "ReferenceStatus": self.reference_status.to_dict() if self.reference_status else None,
            "right_evidence": self.witness_right.to_dict(),
            "left_evidence": self.witness_left.to_dict(),
            "objective_evidence": self.objective_evidence.to_dict() if self.objective_evidence else None,
            "budget": self.budget, "mapping": self.mapping, "notes": self.notes,
            "components": self.component_report, "sampling": self.sampling,
            "objective_coefficients": self.objective_coefficients,
            "capabilities": {
                "affine_mapping": "invertible_diagonal_only",
                "context_matching": "bounded_search",
                "group_matching": "optional_one_vs_two_residual_rows",
                "objective_unit_conversion": "via_projection_affine_substitutions",
                "positive_affine_objective_contract": "reference = a*candidate+b; a>0",
                "argmin_contract": "level_set_proportionality"},
        }


def combine_directions(right, left):
    if "disproved" in (right.status, left.status):
        return "disproved"
    if right.status == left.status == "certified":
        return "certified"
    if all(x.status in ("certified", "within_tolerance") for x in (right, left)):
        return "within_tolerance"
    return "unresolved"


class EquiCEvalEvaluator:
    def __init__(self, solver_tolerance=1e-5, solver_time_limit=30.0,
                 safety_box_bound=None, max_solver_calls=None,
                 use_certificates=True, stop_on_witness=False,
                 contract=PRIMARY_EQUIVAFORMULATION_CONTRACT, context_matching=True,
                 group_matching=False, max_context_pairs=32, max_group_trials=16):
        if contract not in SUPPORTED_CONTRACTS:
            raise ValueError(f"Supported contracts: {', '.join(SUPPORTED_CONTRACTS)}")
        self.tolerance = solver_tolerance
        self.solver_time_limit = solver_time_limit
        self.safety_box_bound = safety_box_bound
        self.max_solver_calls = max_solver_calls
        self.contract = contract
        for limit in (max_context_pairs, max_group_trials):
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
                raise ValueError("Component search limits must be nonnegative integers")
        self.component_options = dict(context_matching=context_matching, group_matching=group_matching,
                                      max_context_pairs=max_context_pairs, max_group_trials=max_group_trials)
        self.matcher = EquivalenceMatcher(solver_tolerance)
        self.directed_evaluator = DirectedDiscrepancyEvaluator(
            solver_tolerance, solver_time_limit, use_certificates, stop_on_witness)

    @staticmethod
    def _blank(direction):
        return CounterexampleWitness(direction, float("nan"), {}, [])

    def _run_objective5(self, out, gt_ir, projected, gt_optimal_val,
                        cand_optimal_val, budget):
        """Module 5: Δ_value + bidirectional cross-regret over P* ∩ P̂.

        Descriptive/diagnostic here; the argmin contract reads its
        ``optimizer_sets_equal`` flag. Never flips a timeout into zero.
        """
        evaluator = ObjectiveDiscrepancyEvaluator(
            self.tolerance, self.solver_time_limit, budget=budget)
        res = evaluator.evaluate(gt_ir, projected, gt_optimal_val, cand_optimal_val)
        out.objective5_result = res
        out.objective5_status = res.status
        def finite_or_none(x):
            return float(x) if x is not None and isfinite(x) else None
        out.delta_value = finite_or_none(res.delta_value)
        out.cross_regret_ref_given_cand = finite_or_none(res.cross_regret_ref_given_cand)
        out.cross_regret_cand_given_ref = finite_or_none(res.cross_regret_cand_given_ref)
        out.optimizer_sets_equal = bool(res.optimizer_sets_equal)
        return res

    def _argmin_verdict_from_module(self, o5, fallback):
        """Prefer solver-certified cross-regrets; fall back to the symbolic test."""
        r1, r2 = o5.cross_regret_ref_given_cand, o5.cross_regret_cand_given_ref
        if all(x is not None and isfinite(x) for x in (r1, r2)):
            if r1 <= self.tolerance and r2 <= self.tolerance:
                return "certified"
            if r1 > self.tolerance or r2 > self.tolerance:
                return "disproved"
        return fallback

    @staticmethod
    def _argmin_verdict(gt_ir, projected):
        """Same optimal decision set: signed objective vectors are positive multiples.

        Additive constants do not change the argmin, so they are ignored. On a
        bounded common feasible set, proportional gradients coincide; otherwise
        the objective directions differ and the argmin is reported as disproved.
        """
        ref_sign = 1 if gt_ir.objective_sense == "minimize" else -1
        cand_sign = 1 if projected.objective_sense == "minimize" else -1
        names = set(gt_ir.objective_coeffs) | set(projected.objective_coeffs)
        ratio = None
        for v in names:
            ar = ref_sign * Fraction(str(gt_ir.objective_coeffs.get(v, 0)))
            ac = cand_sign * Fraction(str(projected.objective_coeffs.get(v, 0)))
            if ar == 0 and ac == 0:
                continue
            if ar == 0 or ac == 0:
                return "disproved"
            current = ar / ac
            if current <= 0:
                return "disproved"
            if ratio is None:
                ratio = current
            elif current != ratio:
                return "disproved"
        return "certified"

    def evaluate(self, gt_ir, cand_ir, gt_optimal_val=None, cand_optimal_val=None,
                 num_samples=100, random_seed=42, run_precheck=True,
                 provided_substitutions=None):
        budget = QueryBudget(self.solver_time_limit, self.max_solver_calls)
        out = EquiCEvalResult(
            getattr(gt_ir, "problem_name", "unknown"), float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), None,
            self._blank("under_constraining"), self._blank("over_constraining"), [],
            contract=self.contract)
        projected, explain = None, False

        def finish():
            # M2/M3 may explain a result but must not starve decisive M4/M5.
            if explain:
                self._describe_components(out, gt_ir, projected, budget, num_samples, random_seed)
            out.budget = budget.to_dict()
            return out

        if gt_ir is None or cand_ir is None:
            out.notes.append("Missing input representation.")
            return finish()
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples < 0:
            raise ValueError("num_samples must be a nonnegative integer")
        # Validate expressions before normalization can hide malformed inputs.
        for ir in (gt_ir, cand_ir):
            if ir.unsupported_scope_features():
                out.verdict = "unsupported"
                return finish()
            for variable in ir.variables.values():
                if (variable.lower_bound != variable.lower_bound or
                        variable.upper_bound != variable.upper_bound or
                        variable.lower_bound == float("inf") or
                        variable.upper_bound == -float("inf")):
                    out.notes.append("Invalid input variable bound.")
                    return finish()
                if variable.var_type.lower() not in ("continuous", "cont", "real", "integer", "int", "binary", "bin"):
                    out.verdict = "unsupported"
                    out.notes.append("Unknown input variable type.")
                    return finish()
            names = set(ir.variables)
            expressions = [(ir.objective_coeffs, ir.objective_constant)] + [
                (c.coeffs, c.constant) for c in ir.constraints]
            if (ir.objective_sense not in ("minimize", "maximize") or
                any(not set(a) <= names or not isfinite(b) or
                    any(not isfinite(v) for v in a.values()) for a, b in expressions)):
                out.notes.append("Invalid expression, objective sense, or unknown variable.")
                return finish()
        cert = build_projection_certificate(gt_ir, cand_ir, provided_substitutions)
        out.mapping = cert.to_dict()
        # Every witness emitted after projection is expressed in reference
        # coordinates.  External validators must lift it back to the raw
        # candidate coordinates before checking the original candidate model.
        out.mapping["diagnostic_coordinate_space"] = "reference"
        out.mapping["coverage_semantics"] = "locally verified pairs retained before failure; partial coverage is a lower bound"
        out.var_match_score = (len(set(cert.variable_map.values())) / len(gt_ir.variables)
                               if gt_ir.variables else float("nan"))
        if not cert.is_verified:
            out.verdict = "unsupported" if cert.unsupported_features else "unresolved"
            out.notes.append(cert.reason)
            return finish()
        try:
            projected = cert.project_ir(cand_ir, gt_ir)
        except ValueError as exc:
            out.verdict = "unsupported"
            out.notes.append(str(exc))
            return finish()
        out.notes.extend(cert.warnings)
        ref_sign = 1 if gt_ir.objective_sense == "minimize" else -1
        cand_sign = 1 if projected.objective_sense == "minimize" else -1
        exact_objective = (
            all(ref_sign*Fraction(str(gt_ir.objective_coeffs.get(v, 0))) ==
                cand_sign*Fraction(str(projected.objective_coeffs.get(v, 0)))
                for v in gt_ir.variables)
            and ref_sign*Fraction(str(gt_ir.objective_constant)) ==
            cand_sign*Fraction(str(projected.objective_constant)))
        positive_affine = detect_positive_affine_alignment(
            gt_ir, projected, tolerance=max(1e-12, self.tolerance * 1e-3))
        out.objective_coefficients = {
            "status": "exact" if exact_objective else "different",
            "positive_affine_alignment": positive_affine.to_dict(),
            "reference": {"signed_coeffs": {v: ref_sign*a for v, a in gt_ir.objective_coeffs.items()},
                          "signed_constant": ref_sign*gt_ir.objective_constant},
            "candidate": {"signed_coeffs": {v: cand_sign*a for v, a in projected.objective_coeffs.items()},
                          "signed_constant": cand_sign*projected.objective_constant},
            "unit_conversion": "none; only recorded input units are accepted"}
        if run_precheck:
            checker = ModelPreChecker(solver_time_limit=self.solver_time_limit, budget=budget)
            ref_status, cand_status = checker.run(gt_ir, cand_ir, provided_substitutions=provided_substitutions)
            out.status, out.reference_status = cand_status, ref_status
            # Invalid references are dataset errors, not generator mistakes.
            if ref_status.s_feas.value == "infeasible" or ref_status.scope_issues:
                out.notes.append("Invalid reference; no correctness conclusion.")
                return finish()
        right = self.directed_evaluator.evaluate_delta_right(
            gt_ir, projected, self.safety_box_bound, budget)
        left = self.directed_evaluator.evaluate_delta_left(
            gt_ir, projected, self.safety_box_bound, budget)
        out.witness_right, out.witness_left = right, left
        out.delta_right, out.delta_left = right.discrepancy_value, left.discrepancy_value
        if left.source_status == "Infeasible":
            out.notes.append("Reference domain is empty in the declared scope.")
            return finish()
        explain = True
        out.feasibility_verdict = combine_directions(right, left)
        if right.source_status == "Infeasible" and left.is_feasible_candidate:
            out.feasibility_verdict = "disproved"
            out.notes.append("Candidate infeasibility is a numerical solver conclusion.")

        if self.contract == FEASIBLE_SET:
            out.objective_verdict = "not_requested"
            out.verdict = out.feasibility_verdict
        elif self.contract == ARGMIN:
            # Module 5 decision fidelity: two solver-certified cross-regrets.
            o5 = self._run_objective5(out, gt_ir, projected, gt_optimal_val,
                                      cand_optimal_val, budget)
            arg_verdict = self._argmin_verdict_from_module(
                o5, self._argmin_verdict(gt_ir, projected))
            out.objective_verdict = arg_verdict
            if out.feasibility_verdict == "disproved" or arg_verdict == "disproved":
                out.verdict = "disproved"
            elif out.feasibility_verdict == arg_verdict == "certified":
                out.verdict = "certified"
            else:
                out.verdict = "unresolved"
        elif self.contract == OBJECTIVE_AFFINE:
            if (out.feasibility_verdict == "disproved"
                    and self.directed_evaluator.stop_on_witness):
                out.verdict = "disproved"
                out.objective_verdict = "not_checked"
                out.objective5_status = "skipped_early_stop"
                out.notes.append(
                    "Positive-affine objective check skipped after a feasible-set witness.")
            else:
                out.objective_verdict = (
                    "certified" if positive_affine.is_resolved else "disproved")
                if (out.feasibility_verdict == "disproved"
                        or out.objective_verdict == "disproved"):
                    out.verdict = "disproved"
                elif out.feasibility_verdict == "certified":
                    out.verdict = "certified"
                elif out.feasibility_verdict == "within_tolerance":
                    out.verdict = "within_tolerance"
                else:
                    out.verdict = "unresolved"
                out.notes.append(
                    "Objective affine check: " + positive_affine.reason)
                # Decision/value diagnostics remain descriptive; they cannot
                # override the algebraic contract verdict.
                self._run_objective5(out, gt_ir, projected, gt_optimal_val,
                                     cand_optimal_val, budget)
        elif out.feasibility_verdict == "disproved" and self.directed_evaluator.stop_on_witness:
            out.verdict = "disproved"
            out.objective5_status = "skipped_early_stop"
            out.notes.append("Objective check skipped after a decisive feasible-set counterexample.")
        elif self.contract == OBJECTIVE_VALUE:
            # Check numerical objective agreement on the intersection, not sampled maxima.
            intersection = replace(
                gt_ir, constraints=list(gt_ir.constraints) + list(projected.constraints))
            scale = max(1, sum(abs(v) for v in gt_ir.objective_coeffs.values()), abs(gt_ir.objective_constant))
            try:
                def difference(a, b):
                    return _exact_decimal_float(ref_sign*Fraction(str(a)) - cand_sign*Fraction(str(b)))
                diff = CanonicalConstraint("objective_difference", {
                    v: difference(gt_ir.objective_coeffs.get(v, 0), projected.objective_coeffs.get(v, 0))
                    for v in gt_ir.variables},
                    difference(gt_ir.objective_constant, projected.objective_constant), "==")
            except ValueError as exc:
                out.objective_verdict = "unsupported"
                out.verdict = "disproved" if out.feasibility_verdict == "disproved" else "unsupported"
                out.notes.append(str(exc))
                return finish()
            objective_target = replace(intersection, constraints=[diff])
            obj = self.directed_evaluator._solve_max_violation(
                intersection, objective_target, "objective_value", self.safety_box_bound, budget,
                normalization_scales={"objective_difference": scale}, phase="objective")
            out.objective_evidence = obj
            out.delta_f = obj.discrepancy_value
            out.objective_verdict = "not_applicable" if obj.status == "source_infeasible" else obj.status
            if out.feasibility_verdict == "disproved" or obj.status == "disproved":
                out.verdict = "disproved"
            elif out.feasibility_verdict == obj.status == "certified":
                out.verdict = "certified"
            elif out.feasibility_verdict in ("certified", "within_tolerance") and obj.status in ("certified", "within_tolerance"):
                out.verdict = "within_tolerance"
            # Module 5 diagnostics only for the value contract (verdict already
            # decided above): Delta_value + bidirectional cross-regret.
            self._run_objective5(out, gt_ir, projected, gt_optimal_val,
                                 cand_optimal_val, budget)
        if self.safety_box_bound is not None and out.verdict in ("certified", "within_tolerance"):
            out.verdict = "within_scope"
            out.notes.append("This conclusion applies only inside the declared box, not the whole instance.")
        # Caller-provided optima are descriptive only; never affect the verdict.
        if gt_optimal_val is not None and cand_optimal_val is not None and all(isfinite(x) for x in (gt_optimal_val, cand_optimal_val)):
            denominator = abs(gt_optimal_val) + abs(cand_optimal_val)
            out.absolute_optimality_gap = abs(gt_optimal_val-cand_optimal_val)
            out.optimality_gap = 2*abs(gt_optimal_val-cand_optimal_val)/denominator if denominator else 0.0
            out.notes.append("OptimalityGap uses caller-supplied values; provenance must be verified separately.")
        return finish()

    def _describe_components(self, out, reference, candidate, budget, num_samples, random_seed):
        ref_rows = reference.merge_bounds_into_constraints()
        cand_rows = candidate.merge_bounds_into_constraints()
        report = self.matcher.evaluate_components(
            ref_rows, cand_rows, budget=budget, directed=self.directed_evaluator,
            safety_box_bound=self.safety_box_bound, **self.component_options)
        out.matched_details = report.pop("matches")
        report["matches"] = [m.to_dict() for m in out.matched_details]
        report["trials"] = [m.to_dict() for m in report["trials"]]
        nr = report["reference_row_count"]
        report["certified_coverage"] = report["certified_reference_rows"] / nr if nr else None
        report["accepted_coverage"] = report["matched_reference_rows"] / nr if nr else None
        report["coverage_scope"] = "declared_box" if self.safety_box_bound is not None else "whole_instance"
        report["unmatched_rows"] = {
            "reference": [{"index": i, "name": ref_rows.constraints[i].name,
                           "label": "unresolved", "reason": "No selected certified/numerical correspondence"}
                          for i in report["unmatched_reference_indices"]],
            "candidate": [{"index": j, "name": cand_rows.constraints[j].name,
                           "label": "unresolved", "reason": "No selected certified/numerical correspondence"}
                          for j in report["unmatched_candidate_indices"]]}
        out.component_report = report
        out.con_match_score = report["accepted_coverage"] if nr else float("nan")
        # M3 — targeted behavioral discrepancy on the selected one-to-one pairs
        # (feasible / boundary / thin-margin / parameter-perturbed strata), per
        # the manuscript. Descriptive only; never a certificate.
        pairs = [m for m in out.matched_details if len(m.gt_indices or (m.gt_index,)) ==
                 len(m.candidate_indices or (m.candidate_index,)) == 1]
        behavioral = BehavioralDiscrepancyEvaluator(
            solver_tolerance=self.tolerance, solver_time_limit=self.solver_time_limit,
            budget=budget)
        bres = behavioral.evaluate(ref_rows, cand_rows, pairs, random_seed=random_seed)
        out.behavioral_solver_calls = bres.num_solver_calls
        out.d_sat, out.d_margin = bres.d_sat, bres.d_margin
        out.sampling = {
            "status": "computed" if bres.total_samples else "not_applicable",
            "reason": None if bres.total_samples else
                      "No usable target samples or selected one-to-one pairs",
            "requested_samples": num_samples,
            "pair_count": len(pairs), "total_samples": bres.total_samples,
            "sample_counts": dict(bres.sample_counts),
            "unmatched_reference_rows": len(report["unmatched_reference_indices"]),
            "unmatched_candidate_rows": len(report["unmatched_candidate_indices"]),
            "group_matches_excluded": sum(m.label.value == "group-equivalent" for m in out.matched_details),
            "seed": random_seed, "tolerance": self.tolerance,
            "targeting": "feasible/boundary/thin-margin/perturbed strata",
            "interpretation": "descriptive only; targeted solver samples, not a certificate"}
