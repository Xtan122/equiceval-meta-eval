"""Reproducible verifier experiments; labels never manufacture solver outputs."""
from dataclasses import asdict
from fractions import Fraction
from itertools import product
from math import ceil, floor, isfinite, prod
from src.equiceval.canonical_ir import CanonicalIR, CanonicalVariable, CanonicalConstraint
from src.equiceval.contracts import (
    ARGMIN,
    OBJECTIVE_AFFINE,
    OBJECTIVE_VALUE,
    SUPPORTED_CONTRACTS,
)


def ir_from_dict(data):
    variables = {}
    for name, value in data["variables"].items():
        spec = dict(value)
        spec.setdefault("name", name)
        for key in ("lower_bound", "upper_bound"):
            if key in spec:
                spec[key] = float(spec[key])
        variables[name] = CanonicalVariable(**spec)
    return CanonicalIR(
        data["problem_name"], variables,
        [CanonicalConstraint(**c) for c in data["constraints"]],
        data.get("objective_sense", "minimize"), data.get("objective_coeffs", {}),
        data.get("objective_constant", 0), data.get("metadata", {}))


def ir_to_dict(ir):
    def clean(value):
        if isinstance(value, float) and not isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    return clean(asdict(ir))


def enumerate_oracle(reference, candidate, contract, max_points=100000):
    """Independent exhaustive oracle for small bounded integer identity maps.

    Deliberately does not call normalization, mapping, solver, or verifier helpers.
    Non-enumerable inputs return an unknown label; metadata flags are not proof.
    """
    if contract not in SUPPORTED_CONTRACTS:
        raise ValueError("Unknown contract")
    for ir in (reference, candidate):
        if ir.metadata.get("nonlinear") or ir.metadata.get("has_nonlinear_terms") or ir.metadata.get("unsupported_features"):
            return None, {"method": "unavailable", "reason": "unsupported input semantics"}
    names = sorted(reference.variables)
    if names != sorted(candidate.variables):
        return None, {"method": "unavailable", "reason": "identity coordinates required"}
    domains = []
    for name in names:
        ref, cand = reference.variables[name], candidate.variables[name]
        if any(v.var_type.lower() not in ("integer", "int", "binary", "bin") for v in (ref, cand)):
            return None, {"method": "unavailable", "reason": "integer variables required"}
        bounds = []
        for v in (ref, cand):
            low, high = v.lower_bound, v.upper_bound
            if v.var_type.lower() in ("binary", "bin"):
                low, high = max(0, low), min(1, high)
            if not isfinite(low) or not isfinite(high):
                return None, {"method": "unavailable", "reason": "finite bounds required"}
            bounds.append((ceil(low), floor(high)))
        lo, hi = min(x[0] for x in bounds), max(x[1] for x in bounds)
        domains.append(range(lo, hi + 1))
    count = prod(len(d) for d in domains)
    if count > max_points:
        return None, {"method": "unavailable", "reason": "enumeration limit exceeded", "points": count}
    def frac(x):
        return Fraction(str(x))
    def feasible(ir, p):
        for name, v in ir.variables.items():
            lo, hi = v.lower_bound, v.upper_bound
            if v.var_type.lower() in ("binary", "bin"):
                lo, hi = max(0, lo), min(1, hi)
            if not frac(lo) <= p[name] <= frac(hi):
                return False
        for c in ir.constraints:
            value = sum(frac(a)*p[n] for n, a in c.coeffs.items()) + frac(c.constant)
            if c.sense not in ("<=", ">=", "=="):
                raise ValueError("Unsupported constraint sense")
            if (c.sense == "<=" and value > 0 or c.sense == ">=" and value < 0 or
                    c.sense == "==" and value != 0):
                return False
        return True
    def objective(ir, p):
        sign = 1 if ir.objective_sense == "minimize" else -1
        return sign*(sum(frac(a)*p[n] for n, a in ir.objective_coeffs.items()) + frac(ir.objective_constant))
    ref_count, first_failure = 0, None
    ref_objectives: Dict[tuple, Fraction] = {}
    cand_objectives: Dict[tuple, Fraction] = {}
    for coords in product(*domains):
        p = dict(zip(names, coords))
        r, c = feasible(reference, p), feasible(candidate, p)
        ref_count += r
        if r != c and first_failure is None:
            first_failure = {"kind": "feasible_set", "point": p}
        if r:
            ref_objectives[coords] = objective(reference, p)
        if c:
            cand_objectives[coords] = objective(candidate, p)
        if r and c and contract == OBJECTIVE_VALUE and objective(reference, p) != objective(candidate, p) and first_failure is None:
            first_failure = {"kind": "objective_value", "point": p}
    affine_alignment = None
    if first_failure is None and contract == OBJECTIVE_AFFINE:
        ref_sign = 1 if reference.objective_sense == "minimize" else -1
        cand_sign = 1 if candidate.objective_sense == "minimize" else -1
        ref_coeffs = {name: ref_sign * frac(reference.objective_coeffs.get(name, 0))
                      for name in names}
        cand_coeffs = {name: cand_sign * frac(candidate.objective_coeffs.get(name, 0))
                       for name in names}
        active_ref = {name: value for name, value in ref_coeffs.items() if value != 0}
        active_cand = {name: value for name, value in cand_coeffs.items() if value != 0}
        scale = None
        if set(active_ref) == set(active_cand):
            if not active_ref:
                scale = Fraction(1)
            else:
                first = next(iter(sorted(active_ref)))
                proposed = active_ref[first] / active_cand[first]
                if proposed > 0 and all(
                        active_ref[name] == proposed * active_cand[name]
                        for name in active_ref):
                    scale = proposed
        if scale is None:
            first_failure = {"kind": "objective_affine",
                             "reason": "objectives are not positive affine"}
        else:
            ref_const = ref_sign * frac(reference.objective_constant)
            cand_const = cand_sign * frac(candidate.objective_constant)
            affine_alignment = {
                "a": float(scale), "b": float(ref_const - scale * cand_const),
                "equation": "reference = a * candidate + b",
            }
    evidence = {"method": "independent_integer_enumeration", "points": count,
                "contract": contract, "reference_feasible_points": ref_count,
                "counterexample": first_failure}
    if affine_alignment is not None:
        evidence["alignment"] = affine_alignment
    if ref_count == 0:
        evidence["reason"] = "Invalid/empty reference"
        return None, evidence
    if first_failure is None and contract == ARGMIN and ref_objectives:
        min_ref = min(ref_objectives.values())
        arg_ref = {coords for coords, value in ref_objectives.items() if value == min_ref}
        min_cand = min(cand_objectives.values()) if cand_objectives else None
        arg_cand = {coords for coords, value in cand_objectives.items() if value == min_cand}
        if arg_ref != arg_cand:
            shared = min(arg_ref ^ arg_cand)
            evidence["counterexample"] = {"kind": "argmin",
                                           "point": dict(zip(names, shared))}
            return False, evidence
    return first_failure is None, evidence


def demo_records():
    """Small diagnostic fixtures, not a publication benchmark."""
    from dataclasses import replace
    V, C = CanonicalVariable, CanonicalConstraint
    ref = CanonicalIR("interval_integer", {"x": V("x", "integer", 0, 8)},
                      [C("cap", {"x": 1}, -8, "<=")], "minimize", {"x": 1})
    variants = [
        ("identity", ref),
        ("scaled_row", replace(ref, constraints=[C("scaled", {"x": 0.1}, -0.8, "<=")])),
        ("redundant", replace(ref, constraints=ref.constraints+[C("redundant", {"x": 1}, -10, "<=")])),
        ("missing_cap", replace(ref, variables={"x": V("x", "integer", 0, 10)}, constraints=[])),
        ("too_tight", replace(ref, variables={"x": V("x", "integer", 0, 5)})),
        ("missing_lower_bound", replace(ref, variables={"x": V("x", "integer", -1, 8)})),
        ("objective_scale", replace(ref, objective_coeffs={"x": 2})),
        ("wrong_sense", replace(ref, objective_sense="maximize")),
        ("infeasible", replace(ref, constraints=[C("impossible", {"x": 1}, -9, ">=")])),
    ]
    records = [(name, ref, cand) for name, cand in variants]
    two = CanonicalIR("two_integer", {"x": V("x", "integer", 0, 1),
                                     "y": V("y", "integer", 0, 2)}, [], "minimize", {"x": 1, "y": 1})
    records.append(("group_implication", two, replace(two, constraints=[C("total", {"x": 1, "y": 1}, -3, "<=")])))
    eq = CanonicalIR("equality_integer", {"x": V("x", "integer", -1, 0)},
                     [C("zero", {"x": 1}, 0, "==")], "minimize", {})
    records.append(("missing_equality", eq, replace(eq, constraints=[])))
    records.append(("zero_objective", replace(ref, objective_coeffs={}), replace(ref, objective_coeffs={})))
    return [{"record_id": name, "family": r.problem_name,
             "reference_ir": ir_to_dict(r), "candidate_ir": ir_to_dict(c)}
            for name, r, c in records]
