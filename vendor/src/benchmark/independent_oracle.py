"""Independent equivalence labels beyond the integer-identity oracle.

``verified_evaluation.enumerate_oracle`` only handles integer variables with
identical names. Many benchmark pairs are continuous (Diet) or rename variables,
so they were left unlabelled. This module adds two independent checks:

1. **Exact bijection match** for pure variable renaming (any variable type):
   tries every reference/candidate name bijection and compares bounds,
   constraints (up to positive scaling) and the objective exactly.
2. **LP containment** for continuous/mixed pairs with identical names: proves
   ``P_ref ⊆ P_cand`` and ``P_cand ⊆ P_ref`` by maximizing each row of one model
   over the other (scipy/HiGHS). Equivalent labels are sound because both
   polyhedra are contained in each other.

Not-equivalent labels always carry a witness that is re-verified with exact
rational arithmetic (and integrality), so a numerical solver verdict alone never
becomes a label. These checks do not use any EquiCEval mapping, normalization or
verifier helper.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import permutations
from math import isfinite
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linprog

from src.benchmark.verified_evaluation import enumerate_oracle
from src.equiceval.contracts import ARGMIN, FEASIBLE_SET, OBJECTIVE_AFFINE, OBJECTIVE_VALUE

DEFAULT_RELATIVE_TOLERANCE = 1e-7
_EXACT_SLACK = Fraction(1, 10 ** 9)


def _frac(x) -> Fraction:
    return Fraction(str(x))


# ── exact canonical rows ────────────────────────────────────────────────────

def _leq_rows(ir) -> List[Tuple[Dict[str, Fraction], Fraction]]:
    """Every constraint normalized to ``a·x + b <= 0``; equalities become two rows."""
    rows: List[Tuple[Dict[str, Fraction], Fraction]] = []
    for c in ir.constraints:
        coeffs = {v: _frac(a) for v, a in c.coeffs.items()}
        b = _frac(c.constant)
        sense = c.sense
        if sense == "<=":
            rows.append((coeffs, b))
        elif sense == ">=":
            rows.append(({v: -a for v, a in coeffs.items()}, -b))
        elif sense == "==":
            rows.append((coeffs, b))
            rows.append(({v: -a for v, a in coeffs.items()}, -b))
        else:
            raise ValueError(f"Unsupported constraint sense: {sense}")
    return rows


def _row_key(coeffs: Dict[str, Fraction], b: Fraction):
    scale = sum(abs(a) for a in coeffs.values()) + abs(b)
    if scale == 0:
        return ("trivial",)
    return tuple(sorted((v, a / scale) for v, a in coeffs.items())) + (b / scale,)


def _bound_key(value: float):
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return _frac(value)


def _var_key(var):
    return (var.var_type.lower(), _bound_key(var.lower_bound), _bound_key(var.upper_bound))


def _signed_objective(ir):
    sign = 1 if ir.objective_sense == "minimize" else -1
    coeffs = {v: sign * _frac(a) for v, a in ir.objective_coeffs.items()}
    return coeffs, sign * _frac(ir.objective_constant)


def _proportional(a: Dict[str, Fraction], b: Dict[str, Fraction]) -> bool:
    """True when ``a`` and ``b`` are positive multiples (same argmin direction)."""
    ratio = None
    for v in set(a) | set(b):
        x, y = a.get(v, Fraction(0)), b.get(v, Fraction(0))
        if x == 0 and y == 0:
            continue
        if x == 0 or y == 0:
            return False
        current = x / y
        if current <= 0:
            return False
        if ratio is None:
            ratio = current
        elif current != ratio:
            return False
    return True


def _positive_affine_details(a: Dict[str, Fraction], a_const: Fraction,
                             b: Dict[str, Fraction], b_const: Fraction):
    """Exact check for ``a_objective = scale*b_objective + shift``."""
    active_a = {name: value for name, value in a.items() if value != 0}
    active_b = {name: value for name, value in b.items() if value != 0}
    if set(active_a) != set(active_b):
        return None
    if not active_a:
        scale = Fraction(1)
    else:
        first = next(iter(sorted(active_a)))
        scale = active_a[first] / active_b[first]
        if scale <= 0 or any(active_a[name] != scale * active_b[name]
                             for name in active_a):
            return None
    return {"a": float(scale), "b": float(a_const - scale * b_const),
            "equation": "reference = a * candidate + b"}


# ── 1. exact renaming bijection ─────────────────────────────────────────────

def _renaming_label(reference, candidate, contract):
    rnames = sorted(reference.variables)
    cnames = sorted(candidate.variables)
    if len(rnames) != len(cnames):
        return None
    if set(rnames) == set(cnames):
        # Same names: only the identity mapping can be a rename; this still
        # catches pure positive row scaling without any factorial search.
        candidates = [tuple(rnames)]
    else:
        # A full bijection search is exponential; keep it bounded.
        if len(cnames) > 8:
            return None
        candidates = permutations(cnames)
    ref_rows = sorted(_row_key(c, b) for c, b in _leq_rows(reference))
    ref_obj, ref_const = _signed_objective(reference)
    for perm in candidates:
        pi = dict(zip(perm, rnames))  # candidate variable -> reference variable
        if any(_var_key(candidate.variables[c]) != _var_key(reference.variables[r])
               for c, r in pi.items()):
            continue
        cand_rows = sorted(
            _row_key({pi[v]: a for v, a in c.items()}, b) for c, b in _leq_rows(candidate))
        if cand_rows != ref_rows:
            continue
        if contract == OBJECTIVE_VALUE:
            cand_obj, cand_const = _signed_objective(candidate)
            if cand_const != ref_const:
                continue
            remapped = {pi.get(v, v): a for v, a in cand_obj.items()}
            if remapped != ref_obj:
                continue
        elif contract in (OBJECTIVE_AFFINE, ARGMIN):
            cand_obj, _ = _signed_objective(candidate)
            remapped = {pi.get(v, v): a for v, a in cand_obj.items()}
            if not _proportional(ref_obj, remapped):
                continue
        is_identity = all(pi[c] == c for c in pi)
        method = ("independent_scaled_row_match" if is_identity
                  else "independent_renaming_bijection")
        return True, {"method": method, "mapping": pi, "contract": contract}
    return None


# ── 2. LP containment (identical names) ─────────────────────────────────────

def _bounds(ir, names):
    out = []
    for v in names:
        var = ir.variables[v]
        lo, hi = var.lower_bound, var.upper_bound
        if var.var_type.lower() in ("binary", "bin"):
            lo, hi = 0.0, 1.0
        out.append((None if not isfinite(lo) else float(lo),
                    None if not isfinite(hi) else float(hi)))
    return out


def _build_lp(ir, names):
    A, b = [], []
    for coeffs, k in _leq_rows(ir):
        row = [float(coeffs.get(v, 0)) for v in names]
        if all(x == 0.0 for x in row):
            if k > 0:  # 0 <= -k is impossible
                A.append([0.0] * len(names))
                b.append(-1.0)
            continue
        A.append(row)
        b.append(float(-k))
    matrix = np.array(A, dtype=float).reshape(-1, len(names)) if A else np.zeros((0, len(names)))
    return matrix, np.array(b, dtype=float), _bounds(ir, names)


def _bound_exact_rows(ir, names):
    rows = []
    for v in names:
        var = ir.variables[v]
        lo, hi = var.lower_bound, var.upper_bound
        if var.var_type.lower() in ("binary", "bin"):
            lo, hi = 0.0, 1.0
        if isfinite(lo):
            rows.append(({v: _frac(-1)}, _frac(lo)))
        if isfinite(hi):
            rows.append(({v: _frac(1)}, -_frac(hi)))
    return rows


def _normalized_rows(ir, names):
    rows = list(_leq_rows(ir)) + _bound_exact_rows(ir, names)
    out = []
    for coeffs, k in rows:
        scale = sum(abs(a) for a in coeffs.values()) + abs(k)
        if scale == 0:
            continue
        out.append((np.array([float(coeffs.get(v, 0) / scale) for v in names]),
                    float(k / scale)))
    return out


def _max_row(source, a, b):
    matrix, rhs, bounds = source
    res = linprog(c=[-x for x in a], A_ub=matrix if len(matrix) else None,
                  b_ub=rhs if len(rhs) else None, bounds=bounds, method="highs")
    if res.status == 2:
        return "unbounded", None, None
    if res.status != 0 or res.x is None:
        return "unknown", None, None
    return "ok", float(np.dot(a, res.x) + b), res.x


def _exact_feasible(ir, point, names, tolerance=_EXACT_SLACK) -> bool:
    p = {}
    for v in names:
        x = point[v]
        if not isfinite(x):
            return False
        p[v] = _frac(x)
    for v in names:
        var = ir.variables[v]
        lo, hi = var.lower_bound, var.upper_bound
        if var.var_type.lower() in ("binary", "bin"):
            lo, hi = 0.0, 1.0
        if var.var_type.lower() in ("integer", "int", "binary", "bin") and p[v].denominator != 1:
            return False
        if isfinite(lo) and p[v] < _frac(lo):
            return False
        if isfinite(hi) and p[v] > _frac(hi):
            return False
    for coeffs, b in _leq_rows(ir):
        value = sum(a * p[v] for v, a in coeffs.items()) + b
        if value <= 0:
            continue
        scale = sum(abs(a) for a in coeffs.values()) + abs(b)
        if value <= tolerance * max(scale, Fraction(1)):
            continue
        return False
    return True


def _exact_violates(ir, point, names, tolerance=_EXACT_SLACK) -> bool:
    p = {v: _frac(point[v]) for v in names}
    for coeffs, b in list(_leq_rows(ir)) + _bound_exact_rows(ir, names):
        value = sum(a * p[v] for v, a in coeffs.items()) + b
        scale = sum(abs(a) for a in coeffs.values()) + abs(b)
        if value > tolerance * max(scale, Fraction(1)):
            return True
    return False


def _contains(source_ir, source_lp, target_ir, target_rows, names, tol):
    for a, b in target_rows:
        status, value, x = _max_row(source_lp, a, b)
        if status != "ok":
            return status, None
        if value <= tol:
            continue
        point = {v: float(x[i]) for i, v in enumerate(names)}
        if _exact_feasible(source_ir, point, names) and _exact_violates(target_ir, point, names):
            return "violated", point
        return "unknown", None
    return "contained", None


def _objective_label(reference, candidate, names, ref_lp, tol):
    ref_obj, ref_const = _signed_objective(reference)
    cand_obj, cand_const = _signed_objective(candidate)
    diff = {v: ref_obj.get(v, 0) - cand_obj.get(v, 0) for v in names}
    dconst = ref_const - cand_const
    if all(a == 0 for a in diff.values()) and dconst == 0:
        return True, {"method": "independent_lp_containment",
                      "objective": "identical", "contract": "feasible_set_and_objective_value"}
    for sign in (1.0, -1.0):
        a = np.array([sign * float(diff[v]) for v in names])
        if not np.any(a) and sign * float(dconst) == 0.0:
            continue
        status, value, x = _max_row(ref_lp, a, sign * float(dconst))
        if status != "ok":
            return None, None
        if abs(value) <= tol:
            continue
        point = {v: float(x[i]) for i, v in enumerate(names)}
        if _exact_feasible(reference, point, names) and abs(
                sum(diff[v] * _frac(point[v]) for v in names) + dconst) > _EXACT_SLACK:
            return False, {"method": "independent_lp_containment",
                           "counterexample": {"kind": "objective_value", "point": point},
                           "contract": "feasible_set_and_objective_value"}
        return None, None
    return True, {"method": "independent_lp_containment",
                  "objective": "equal_on_feasible_set", "contract": "feasible_set_and_objective_value"}


def _lp_label(reference, candidate, contract, tolerance=DEFAULT_RELATIVE_TOLERANCE):
    names = sorted(reference.variables)
    if names != sorted(candidate.variables):
        return None
    ref_lp = _build_lp(reference, names)
    cand_lp = _build_lp(candidate, names)
    ref_rows = _normalized_rows(reference, names)
    cand_rows = _normalized_rows(candidate, names)

    status, point = _contains(reference, ref_lp, candidate, cand_rows, names, tolerance)
    if status == "violated":
        return False, {"method": "independent_lp_containment",
                       "counterexample": {"kind": "feasible_set", "point": point},
                       "contract": contract}
    if status != "contained":
        return None

    status, point = _contains(candidate, cand_lp, reference, ref_rows, names, tolerance)
    if status == "violated":
        return False, {"method": "independent_lp_containment",
                       "counterexample": {"kind": "feasible_set", "point": point},
                       "contract": contract}
    if status != "contained":
        return None

    if contract == OBJECTIVE_VALUE:
        label, evidence = _objective_label(reference, candidate, names, ref_lp, tolerance)
        if label is None:
            return None
        return label, evidence
    if contract in (OBJECTIVE_AFFINE, ARGMIN):
        ref_obj, ref_const = _signed_objective(reference)
        cand_obj, cand_const = _signed_objective(candidate)
        alignment = _positive_affine_details(
            ref_obj, ref_const, cand_obj, cand_const)
        if alignment is not None:
            return True, {"method": "independent_lp_containment",
                          "objective": ("positive_affine" if contract == OBJECTIVE_AFFINE
                                        else "same_argmin_certified_by_positive_affine"),
                          "alignment": alignment, "contract": contract}
        if contract == OBJECTIVE_AFFINE:
            return False, {"method": "independent_lp_containment",
                           "counterexample": {"kind": "objective_affine"},
                           "contract": contract}
        # Non-proportionality alone does not prove different optimizer sets.
        return None
    return True, {"method": "independent_lp_containment", "kind": "feasible_set",
                  "contract": contract}


# ── entry point ─────────────────────────────────────────────────────────────

def independent_label(reference, candidate, contract,
                       tolerance=DEFAULT_RELATIVE_TOLERANCE):
    """Label a pair independently; ``label`` is True/False/None (unknown)."""
    label, evidence = enumerate_oracle(reference, candidate, contract)
    if label is not None:
        return label, evidence
    renamed = _renaming_label(reference, candidate, contract)
    if renamed is not None:
        return renamed
    lp = _lp_label(reference, candidate, contract, tolerance)
    if lp is not None:
        return lp
    return None, evidence
