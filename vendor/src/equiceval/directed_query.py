"""Budgeted directional verification with per-row bounds and reusable witnesses."""
from dataclasses import dataclass, field, asdict, replace
from math import isfinite
from typing import Dict, List, Optional, Any
from src.equiceval.canonical_ir import CanonicalIR, CanonicalConstraint
from src.equiceval.evidence import point_is_feasible, find_implication_certificate, rational
from src.utils.solver_wrapper import QueryBudget


@dataclass
class CounterexampleWitness:
    direction: str
    discrepancy_value: float
    variable_assignment: Dict[str, float]
    violated_constraints: List[str]
    is_feasible_candidate: bool = False  # Historical name; means feasible in SOURCE.
    description: str = ""
    status: str = "unresolved"
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    obligations: List[Dict[str, Any]] = field(default_factory=list)
    evidence_level: str = "none"
    scope: str = "whole_instance"
    source_status: str = "unknown"

    def to_dict(self):
        def clean(v):
            if isinstance(v, float) and not isfinite(v):
                return None
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            if isinstance(v, list):
                return [clean(x) for x in v]
            return v
        return clean(asdict(self))


class DirectedDiscrepancyEvaluator:
    def __init__(self, solver_tolerance=1e-5, solver_time_limit=30.0,
                 use_certificates=True, stop_on_witness=False):
        if not isfinite(solver_tolerance) or solver_tolerance < 0:
            raise ValueError("Tolerance must be finite and nonnegative")
        self.tolerance = solver_tolerance
        self.time_limit = solver_time_limit
        self.use_certificates = use_certificates
        self.stop_on_witness = stop_on_witness

    def _solve_max_violation(self, domain_ir, target_ir, direction,
                             safety_box_bound=None, budget=None, normalization_scales=None,
                             phase="direction"):
        budget = budget if budget is not None else QueryBudget(self.time_limit)
        out = CounterexampleWitness(direction, float("nan"), {}, [],
                                    scope="box" if safety_box_bound is not None else "whole_instance")
        if domain_ir.unsupported_scope_features() or target_ir.unsupported_scope_features():
            out.status = "unsupported"
            return out
        if set(domain_ir.variables) != set(target_ir.variables):
            out.status = "unsupported"
            return out
        def kind(v):
            return {"bin": "binary", "int": "integer", "cont": "continuous", "real": "continuous"}.get(v.var_type.lower(), v.var_type.lower())
        if any(kind(v) != kind(target_ir.variables[n]) for n, v in domain_ir.variables.items()):
            out.status = "unsupported"
            out.description = "Different variable types require a separate semantic check."
            return out
        if safety_box_bound is not None:
            if not isfinite(safety_box_bound) or safety_box_bound <= 0:
                raise ValueError("Box bound must be finite and positive")
            domain_ir = replace(domain_ir, variables={
                n: replace(v, lower_bound=max(v.lower_bound, -safety_box_bound),
                           upper_bound=min(v.upper_bound, safety_box_bound))
                for n, v in domain_ir.variables.items()})
        specs = {n: (v.var_type, v.lower_bound, v.upper_bound) for n, v in domain_ir.variables.items()}
        constraints = [(c.coeffs, c.sense, -c.constant) for c in domain_ir.constraints]
        feasibility = budget.solve(f"{phase}_feasibility", var_specs=specs,
                                   objective_coeffs={}, objective_sense="minimize",
                                   constraints_list=constraints)
        out.source_status = feasibility.status
        if feasibility.status == "Infeasible":
            out.status, out.description = "source_infeasible", "Empty source domain; maximum is not applicable."
            return out
        if not point_is_feasible(domain_ir, feasibility.variable_values):
            out.description = "Source feasibility has not been independently established."
            return out
        out.is_feasible_candidate = True
        targets = target_ir.merge_bounds_into_constraints().constraints
        signed = []
        sides = []
        for index, target in enumerate(targets):
            c = target.flip_to_leq()
            for sign in ((1, -1) if c.sense == "==" else (1,)):
                q = (normalization_scales or {}).get(target.name, c.normalization_scale)
                signed.append((index, target.name, CanonicalConstraint(
                    target.name, {v: sign*a for v, a in c.coeffs.items()},
                    sign*c.constant, "<="), q))
                sides.append("+" if sign == 1 else "-")
        out.obligations = [
            {"target_index": i, "target_name": name, "side": j,
             "signed_side": sides[j],
             "target_row": {"coeffs": dict(c.coeffs), "constant": c.constant, "sense": c.sense},
             "normalization_scale": q,
             "status": "unresolved", "lower_bound": 0.0, "upper_bound": None,
             "certificate": None}
            for j, (i, name, c, q) in enumerate(signed)]
        best_point = dict(feasibility.variable_values)
        best_violation = 0.0

        def reuse(point):
            nonlocal best_point, best_violation
            if not point_is_feasible(domain_ir, point):
                return
            maximum = 0.0
            for ob, (_, _, c, q) in zip(out.obligations, signed):
                value = sum(rational(a)*rational(point[v]) for v, a in c.coeffs.items()) + rational(c.constant)
                value = max(rational(0), value) / rational(q) if q else rational(0)
                ob["lower_bound"] = max(ob["lower_bound"], float(value))
                if value > rational(self.tolerance):
                    ob["status"] = "disproved"
                    ob["witness"] = dict(point)
                maximum = max(maximum, float(value))
            if maximum > best_violation:
                best_point, best_violation = dict(point), maximum

        reuse(best_point)
        for ob, (_, _, c, q) in zip(out.obligations, signed):
            if self.stop_on_witness and best_violation > self.tolerance:
                break
            if q == 0:
                ob.update(status="certified", upper_bound=0.0)
                continue
            # An existing witness settles detection for this obligation.
            if ob["status"] == "disproved" and self.stop_on_witness:
                continue
            if self.use_certificates and ob["status"] != "disproved" and budget.remaining > 0:
                cert = find_implication_certificate(domain_ir, c, budget,
                                                    phase=f"{phase}_certificate")
                if cert is not None:
                    ob.update(status="certified", upper_bound=0.0, certificate=cert)
                    continue
            result = budget.solve(f"{phase}_max_violation", var_specs=specs,
                                  objective_coeffs=c.coeffs, objective_sense="maximize",
                                  constraints_list=constraints)
            ob["solver_status"] = result.status
            if result.status in ("Optimal", "Timeout"):
                reuse(result.variable_values)
            if result.status == "Optimal" and result.objective_bound is not None:
                upper = max(0.0, (result.objective_bound + c.constant) / q)
                # Never report a numerical upper bound below an exact primal value.
                if upper + 1e-10 >= ob["lower_bound"]:
                    ob["upper_bound"] = max(upper, ob["lower_bound"])
                    if upper <= self.tolerance and ob["status"] != "disproved":
                        ob["status"] = "within_tolerance"
            if result.status == "Error":
                ob["error"] = result.message
        out.lower_bound = best_violation
        uppers = [ob["upper_bound"] for ob in out.obligations]
        out.upper_bound = max(uppers, default=0.0) if all(x is not None for x in uppers) else None
        out.variable_assignment = best_point
        # Only rows violated at THIS reported point belong in this list.
        out.violated_constraints = sorted(set(
            name for (_, name, c, q) in signed
            if q and (sum(rational(a)*rational(best_point[v]) for v, a in c.coeffs.items())
                      + rational(c.constant)) / rational(q) > rational(self.tolerance)))
        if best_violation > self.tolerance:
            out.status, out.evidence_level = "disproved", "exact_decimal_witness"
        elif all(ob["status"] == "certified" for ob in out.obligations):
            out.status, out.evidence_level = "certified", "rational_row_certificates"
        elif out.upper_bound is not None and out.upper_bound <= self.tolerance:
            out.status, out.evidence_level = "within_tolerance", "numerical_solver_bounds"
        out.discrepancy_value = best_violation if out.upper_bound is not None or best_violation > self.tolerance else float("nan")
        out.description = ("Reported value is a lower bound; consult upper_bound and status. "
                           "Certificates are conditional on the mapping and declared scope.")
        return out

    def evaluate_delta_right(self, gt_ir, cand_ir, safety_box_bound=None, budget=None):
        return self._solve_max_violation(cand_ir, gt_ir, "under_constraining", safety_box_bound, budget)

    def evaluate_delta_left(self, gt_ir, cand_ir, safety_box_bound=None, budget=None):
        return self._solve_max_violation(gt_ir, cand_ir, "over_constraining", safety_box_bound, budget)

    def evaluate_directed_discrepancies(self, gt_ir, cand_ir, budget=None):
        budget = budget if budget is not None else QueryBudget(self.time_limit)
        return (self.evaluate_delta_right(gt_ir, cand_ir, budget=budget),
                self.evaluate_delta_left(gt_ir, cand_ir, budget=budget))
