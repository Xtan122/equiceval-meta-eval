"""Relabel độc lập cho EquivaFormulation: chiếu map công bố + kiểm MILP.

Mục tiêu: gán nhãn ``True``/``False`` chỉ khi có đủ bằng chứng cho từng cặp
``(O, candidate)`` theo một contract, không phụ thuộc EquiCEval. Cặp không
đủ điều kiện chứng minh phải giữ nhãn ``None`` (chưa kết luận).

Quy trình cho mỗi cặp:

1. Dựng tương ứng biến ``root_var = affine(candidate_vars)`` từ
   ``variable_mappings.json`` do dataset công bố (hỗ trợ mảng/chỉ số và nhiều
   số hạng: ``PackageCount[i] = b1[i] + b2[i]``).
2. Chỉ chiếu ánh xạ một--một an toàn sang không gian candidate; khử biến phụ
   liên tục/fixed bằng đại số để hai bên cùng tập biến. Ánh xạ nhiều--một,
   một--nhiều và khử biến nguyên được để lại là ``None``.
3. Kiểm tương đương bằng MILP (scipy ``milp``), có kiểm chứng nghiệm phản ví dụ
   bằng số hữu tỉ; xử lý cả mô hình nguyên và vô biên.
4. Fallback không cần tương ứng: objective candidate hằng vs reference không
   hằng, và map công bố không đơn ánh (``_j``/``_k``).
"""
from __future__ import annotations

import json
import math
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from src.benchmark.independent_oracle import (
    _bound_exact_rows,
    _exact_feasible,
    _exact_violates,
    _leq_rows,
    _row_key as _oracle_row_key,
)
from src.equiceval.canonical_ir import (
    CanonicalConstraint,
    CanonicalIR,
    CanonicalVariable,
)
from src.equiceval.contracts import ARGMIN, FEASIBLE_SET, OBJECTIVE_AFFINE, OBJECTIVE_VALUE

DEFAULT_TOL = 1e-7
_INT_TYPES = ("integer", "int", "binary", "bin")


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


_INDEX_RE = re.compile(r"^(?P<base>.*?)(?P<idx>(?:\[[^\]]*\]|_\d+)+)$")


def _split_base_index(name: str) -> Tuple[str, str]:
    match = _INDEX_RE.match(name)
    if match:
        return match.group("base"), match.group("idx")
    return name, ""


def _indexed(base: str, variables) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name in variables:
        candidate_base, index = _split_base_index(name)
        if candidate_base == base:
            out[index] = name
    return out


# ── published mapping ───────────────────────────────────────────────────────

def read_published_map(path: Path) -> Dict[str, List[Tuple[float, str]]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    result: Dict[str, List[Tuple[float, str]]] = {}
    for original, entries in raw.items():
        if not isinstance(entries, list):
            continue
        parsed = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            var = entry.get("variable")
            coeff = entry.get("constant")
            if var is None or coeff is None:
                continue
            try:
                parsed.append((float(coeff), str(var)))
            except (TypeError, ValueError):
                continue
        if parsed:
            result[str(original)] = parsed
    return result


def build_root_to_candidate(
    published: Dict[str, List[Tuple[float, str]]],
    root_vars,
    cand_vars,
) -> Optional[Dict[str, Dict[str, float]]]:
    """``root_var -> {candidate_var: coeff}`` với mở rộng chỉ số.

    Trả ``None`` nếu có root_var không map được, hoặc map trùng đích
    (không đơn ánh) — dấu hiệu cặp không có tương ứng hợp lệ.
    """
    mapping: Dict[str, Dict[str, float]] = {}
    for root_name in root_vars:
        base, index = _split_base_index(root_name)
        terms = published.get(base)
        if not terms:
            return None
        expr: Dict[str, float] = {}
        for coeff, cand_base in terms:
            cbase = _sanitize(cand_base)
            cand_full = None
            if index:
                index_map = _indexed(cbase, cand_vars)
                cand_full = index_map.get(index)
                if cand_full is None and f"{cbase}{index}" in cand_vars:
                    cand_full = f"{cbase}{index}"
            elif cbase in cand_vars:
                cand_full = cbase
            if cand_full is None:
                return None
            expr[cand_full] = expr.get(cand_full, 0.0) + coeff
        expr = {v: c for v, c in expr.items() if c != 0}
        if not expr:
            return None
        mapping[root_name] = expr
    return mapping


def _rewrite_linear(coeffs: Dict[str, float], mapping: Dict[str, Dict[str, float]]):
    out: Dict[str, float] = {}
    for name, coeff in coeffs.items():
        if name in mapping:
            for target, factor in mapping[name].items():
                out[target] = out.get(target, 0.0) + coeff * factor
        else:
            out[name] = out.get(name, 0.0) + coeff
    return {v: c for v, c in out.items() if c != 0}


def _project_reference_grouped(
    reference: CanonicalIR,
    mapping: Dict[str, Dict[str, float]],
    candidate: CanonicalIR,
) -> Optional[CanonicalIR]:
    """Chiếu reference qua substitution nhóm ``root = sum a_i * cand_i``.

    Khác ánh xạ một--một, ở đây ``root`` là tổ hợp tuyến tính của nhiều biến
    candidate (``_d`` base-10, ``_h`` phép thay tuyến tính). Ta **không** gán
    miền của root cho từng biến candidate; thay vào đó:

    1. Miền/kiểu của các biến candidate lấy từ chính candidate (vì phép so là
       trên không gian candidate, và containment MILP sẽ kiểm cả hai chiều).
    2. Cận hữu hạn của root được mang theo dưới dạng **ràng buộc tuyến tính**
       trên biểu thức (đúng như nhánh một--một).
    3. Bảo toàn lattice: nếu root là biến nguyên/nhị phân thì mọi hạng tử phải
       nguyên và mọi biến candidate tham gia phải nguyên/nhị phân, nếu không trả
       ``None`` (ánh xạ ``x = Σ a_i y_i`` với ``y_i`` liên tục có thể phá lattice).

    Mọi phán quyết sau đó vẫn phải qua ``check_equivalence`` (containment), nên
    substitution sai không thể tự động tạo ra tương đương giả.
    """
    image = {v for expr in mapping.values() for v in expr}
    if not image:
        return None
    if any(v not in candidate.variables for v in image):
        return None
    for root_name, expr in mapping.items():
        root = reference.variables[root_name]
        if root.var_type.lower() not in _INT_TYPES:
            continue
        for cand_name, coeff in expr.items():
            cand_var = candidate.variables[cand_name]
            if cand_var.var_type.lower() not in _INT_TYPES:
                return None
            if not math.isclose(coeff, round(coeff), abs_tol=1e-9):
                return None

    variables = {name: candidate.variables[name] for name in sorted(image)}
    constraints = [
        CanonicalConstraint(c.name, _rewrite_linear(c.coeffs, mapping),
                            c.constant, c.sense)
        for c in reference.constraints
    ]
    for name, var in reference.variables.items():
        if math.isfinite(var.lower_bound):
            constraints.append(CanonicalConstraint(
                f"bound_lo_{name}", _rewrite_linear({name: -1.0}, mapping),
                var.lower_bound, "<="))
        if math.isfinite(var.upper_bound):
            constraints.append(CanonicalConstraint(
                f"bound_hi_{name}", _rewrite_linear({name: 1.0}, mapping),
                -var.upper_bound, "<="))
    return CanonicalIR(
        reference.problem_name, variables, constraints,
        reference.objective_sense,
        _rewrite_linear(reference.objective_coeffs, mapping),
        reference.objective_constant, metadata=dict(reference.metadata),
    )


def project_reference(
    reference: CanonicalIR,
    mapping: Dict[str, Dict[str, float]],
    candidate: CanonicalIR,
) -> Optional[CanonicalIR]:
    """Thay ``root_var = sum coeff * cand_var`` vào reference.

    Reference sau chiếu dùng không gian biến candidate. Với ánh xạ một--một,
    miền và kiểu của reference được chuyển theo ánh xạ, tuyệt đối không sao chép
    miền/kiểu của candidate. Với substitution nhóm (``x = x0 + 10 x1``,
    ``x = y1 + y2``) thì không thể chuyển miền theo từng biến, nên đi theo nhánh
    ``_project_reference_grouped`` và để phần containment MILP quyết định. Nếu
    không biểu diễn được miền nguyên một cách đúng, trả ``None`` thay vì mở rộng
    hay thu hẹp miền.
    """
    if set(mapping) != set(reference.variables):
        return None
    if any(not expr for expr in mapping.values()):
        return None
    if any(len(expr) != 1 for expr in mapping.values()):
        return _project_reference_grouped(reference, mapping, candidate)
    targets = [next(iter(expr)) for expr in mapping.values()]
    if len(set(targets)) != len(targets):
        return None

    image = sorted({v for expr in mapping.values() for v in expr})
    variables = {}
    for name in image:
        root_name = next(root for root, expr in mapping.items() if name in expr)
        root = reference.variables[root_name]
        scale = mapping[root_name][name]
        if scale == 0:
            return None
        # An integer/binary coordinate cannot in general be rescaled while
        # preserving its lattice. Accept only sign flips for such variables.
        if root.var_type.lower() in _INT_TYPES and not math.isclose(abs(scale), 1.0):
            return None
        lower, upper = root.lower_bound / scale, root.upper_bound / scale
        if scale < 0:
            lower, upper = upper, lower
        variables[name] = CanonicalVariable(
            name=name, var_type=root.var_type,
            lower_bound=lower, upper_bound=upper,
            provenance=root.provenance, semantic_role=root.semantic_role,
            requirement_id=root.requirement_id, unit=root.unit,
            is_auxiliary=root.is_auxiliary, index_sets=list(root.index_sets))
    constraints = [
        CanonicalConstraint(c.name, _rewrite_linear(c.coeffs, mapping),
                            c.constant, c.sense)
        for c in reference.constraints
    ]
    for name, var in reference.variables.items():
        if name not in mapping:
            continue
        expr = mapping[name]
        if math.isfinite(var.lower_bound):
            neg = _rewrite_linear({name: -1.0}, mapping)
            constraints.append(CanonicalConstraint(
                f"bound_lo_{name}", neg, var.lower_bound, "<="))
        if math.isfinite(var.upper_bound):
            pos = _rewrite_linear({name: 1.0}, mapping)
            constraints.append(CanonicalConstraint(
                f"bound_hi_{name}", pos, -var.upper_bound, "<="))
    return CanonicalIR(
        reference.problem_name, variables, constraints,
        reference.objective_sense,
        _rewrite_linear(reference.objective_coeffs, mapping),
        reference.objective_constant, metadata=dict(reference.metadata),
    )


# ── aux elimination (algebraic, exact) ──────────────────────────────────────

def _substitute(ir: CanonicalIR, var: str, expr: Dict[str, float], const: float) -> CanonicalIR:
    removed = ir.variables[var]
    bounds_rows = []
    if math.isfinite(removed.lower_bound):  # var >= lo  =>  -expr·y <= lo - const
        bounds_rows.append(CanonicalConstraint(
            f"bound_lo_{var}", {v: -f for v, f in expr.items()},
            removed.lower_bound - const, "<="))
    if math.isfinite(removed.upper_bound):  # var <= hi  =>  expr·y <= hi - const
        bounds_rows.append(CanonicalConstraint(
            f"bound_hi_{var}", dict(expr), const - removed.upper_bound, "<="))

    def rewrite(coeffs: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, coeff in coeffs.items():
            if name == var:
                for t, f in expr.items():
                    out[t] = out.get(t, 0.0) + coeff * f
            else:
                out[name] = out.get(name, 0.0) + coeff
        return {v: c for v, c in out.items() if c != 0}

    new_constraints = []
    for c in ir.constraints:
        constant = c.constant + c.coeffs.get(var, 0.0) * const
        new_constraints.append(CanonicalConstraint(c.name, rewrite(c.coeffs),
                                                    constant, c.sense))
    new_constraints.extend(bounds_rows)
    variables = {n: v for n, v in ir.variables.items() if n != var}
    return CanonicalIR(
        ir.problem_name, variables, new_constraints, ir.objective_sense,
        rewrite(ir.objective_coeffs),
        ir.objective_constant + ir.objective_coeffs.get(var, 0.0) * const,
        metadata=dict(ir.metadata),
    )


def _record_elimination(eliminations, var, expr, const):
    """Track ``var = expr·kept + const`` and compose it into earlier records."""
    for key, (prev_expr, prev_const) in list(eliminations.items()):
        if var in prev_expr:
            factor = prev_expr[var]
            merged = {k: v for k, v in prev_expr.items() if k != var}
            for target, coeff in expr.items():
                merged[target] = merged.get(target, 0.0) + factor * coeff
            eliminations[key] = ({k: v for k, v in merged.items() if v != 0},
                                 prev_const + factor * const)
    eliminations[var] = (dict(expr), const)


def eliminate_to_keep(ir: CanonicalIR, keep: set):
    """Khử biến không thuộc ``keep`` nếu fixed hoặc định nghĩa bởi một equality.

    Returns ``(ir, eliminations)`` where ``eliminations[var] = (expr, const)``
    reconstructs an eliminated variable as ``var = sum(expr·kept) + const``;
    this is what lets a witness be re-verified on the *original* model
    (upgrade paper §1151-1152).
    """
    eliminations: Dict[str, Tuple[Dict[str, float], float]] = {}
    changed = True
    while changed:
        changed = False
        for name, var in list(ir.variables.items()):
            if name in keep:
                continue
            if math.isclose(var.lower_bound, var.upper_bound, rel_tol=0, abs_tol=1e-12):
                ir = _substitute(ir, name, {}, var.lower_bound)
                _record_elimination(eliminations, name, {}, var.lower_bound)
                changed = True
                break
        if changed:
            continue
        for c in ir.constraints:
            if c.sense != "==":
                continue
            terms = {v: a for v, a in c.coeffs.items() if abs(a) > 1e-12}
            free = [v for v in terms if v not in keep and v in ir.variables]
            if len(free) == 1:
                var = free[0]
                # Substituting an integer/binary variable through an equality
                # can erase a lattice restriction. That requires a separate
                # integer-preservation proof, so leave it unresolved here.
                if ir.variables[var].var_type.lower() in _INT_TYPES:
                    continue
                coeff = terms[var]
                expr = {v: -a / coeff for v, a in terms.items() if v != var}
                const = -c.constant / coeff
                ir = _substitute(ir, var, expr, const)
                _record_elimination(eliminations, var, expr, const)
                changed = True
                break
        if changed:
            continue
        # A constraint that became trivially true/false is dropped / flagged.
        kept = []
        for c in ir.constraints:
            scale = sum(abs(a) for a in c.coeffs.values()) + abs(c.constant)
            if scale <= 1e-12 and c.sense == "==":
                continue
            kept.append(c)
        if len(kept) != len(ir.constraints):
            ir = CanonicalIR(ir.problem_name, dict(ir.variables), kept,
                             ir.objective_sense, dict(ir.objective_coeffs),
                             ir.objective_constant, metadata=dict(ir.metadata))
            changed = True
    return ir, eliminations


# ── MILP equivalence ────────────────────────────────────────────────────────

def _names_union(a: CanonicalIR, b: CanonicalIR) -> List[str]:
    return sorted(set(a.variables) | set(b.variables))


def rows_reproduce(reference: CanonicalIR, candidate: CanonicalIR) -> bool:
    """Second, solver-free certificate: the two row sets are identical.

    Normalizes every row (including variable bounds) to ``a·x + b <= 0`` and
    compares the sign/scale-invariant keys. Used as an *independent second
    opinion* for ``True`` labels produced by the solver: if both the MILP
    containment and the normalized row sets agree, a systematic bug in one path
    would have to reproduce itself exactly in the other.
    """
    def keys(ir):
        merged = ir.merge_bounds_into_constraints()
        return {_oracle_row_key(coeffs, b) for coeffs, b in _leq_rows(merged)}
    return keys(reference) == keys(candidate)


def _bounds_and_integrality(ir: CanonicalIR, names: List[str]):
    bounds = []
    integrality = []
    for name in names:
        var = ir.variables.get(name)
        if var is None:
            bounds.append((-np.inf, np.inf))
            integrality.append(0)
            continue
        low, high = var.lower_bound, var.upper_bound
        if var.var_type.lower() in ("binary", "bin"):
            low, high = 0.0, 1.0
        lo = -np.inf if not math.isfinite(low) else low
        hi = np.inf if not math.isfinite(high) else high
        bounds.append((lo, hi))
        integrality.append(1 if var.var_type.lower() in _INT_TYPES else 0)
    return bounds, integrality


def _max_linear(ir, coeffs: Dict[str, float], names: List[str], time_limit: float = 5.0,
                box_bound: Optional[float] = None):
    """max ``sum coeff*x`` trên feasible set của ``ir`` (MILP).

    ``box_bound`` chỉ dùng để **tìm phản ví dụ** khi miền không bị chặn (§M0:
    truy vấn trong hộp B). Nó không bao giờ được dùng để chứng minh ``True``.
    """
    rows = _leq_rows(ir)
    A, ub = [], []
    for row, b in rows:
        vec = np.array([float(row.get(n, 0.0)) for n in names])
        if not np.any(vec):
            if float(b) > DEFAULT_TOL:
                return "infeasible", None, None
            continue
        A.append(vec)
        ub.append(float(-b))
    bounds, integrality = _bounds_and_integrality(ir, names)
    if box_bound is not None:
        bounds = [
            (-box_bound if lo == -np.inf else lo, box_bound if hi == np.inf else hi)
            for lo, hi in bounds
        ]
    c = np.array([-float(coeffs.get(n, 0.0)) for n in names])
    if not np.any(c):
        return "ok", 0.0, {n: 0.0 for n in names}
    kwargs = dict(
        c=c,
        bounds=Bounds([b[0] for b in bounds], [b[1] for b in bounds]),
        integrality=np.array(integrality),
        options={"time_limit": time_limit},
    )
    if A:
        kwargs["constraints"] = LinearConstraint(np.array(A), -np.inf, np.array(ub))
    result = milp(**kwargs)
    if result.status == 2:
        return "infeasible", None, None
    if result.status == 3:
        return "unbounded", None, None
    if result.status != 0 or result.x is None:
        return "unknown", None, None
    point = {n: float(result.x[i]) for i, n in enumerate(names)}
    return "ok", float(-result.fun), point


def _target_rows(ir, names):
    return _leq_rows(ir) + _bound_exact_rows(ir, names)


def _round_point(point: Dict[str, float], source: CanonicalIR, names) -> Dict[str, float]:
    """Làm tròn biến nguyên về số nguyên gần nhất (nhiễu float của MILP)."""
    out: Dict[str, float] = {}
    for name in names:
        value = float(point.get(name, 0.0))
        var = source.variables.get(name)
        if var is not None and var.var_type.lower() in _INT_TYPES:
            value = float(round(value))
        out[name] = value
    return out


def _reconstruct_candidate(point, raw_candidate, eliminations):
    """Khôi phục điểm đầy đủ trên candidate gốc (kể cả biến phụ đã khử)."""
    full = dict(point)
    pending = {v: (dict(e), c) for v, (e, c) in eliminations.items()}
    progress = True
    while pending and progress:
        progress = False
        for var, (expr, const) in list(pending.items()):
            if all(k in full for k in expr):
                full[var] = sum(coef * full[k] for k, coef in expr.items()) + const
                del pending[var]
                progress = True
    if pending or any(v not in full for v in raw_candidate.variables):
        return None
    return {v: full[v] for v in raw_candidate.variables}


def _map_to_root(full_point, root_to_candidate):
    return {root: sum(coef * full_point.get(cand, 0.0) for cand, coef in expr.items())
            for root, expr in root_to_candidate.items()}


def _raw_witness_ok(point, source_is_reference, raw_reference, raw_candidate,
                    root_to_candidate, eliminations):
    """Xác minh phản ví dụ trên **mô hình gốc** (upgrade paper §1151-1152)."""
    full = _reconstruct_candidate(point, raw_candidate, eliminations)
    if full is None:
        return False
    root_point = _map_to_root(full, root_to_candidate)
    ref_names = list(raw_reference.variables)
    cand_names = list(raw_candidate.variables)
    if source_is_reference:
        return (_exact_feasible(raw_reference, root_point, ref_names)
                and _exact_violates(raw_candidate, full, cand_names))
    return (_exact_feasible(raw_candidate, full, cand_names)
            and _exact_violates(raw_reference, root_point, ref_names))


def _raw_objective_diff(point, raw_reference, raw_candidate, root_to_candidate,
                        eliminations):
    full = _reconstruct_candidate(point, raw_candidate, eliminations)
    if full is None:
        return None
    root_point = _map_to_root(full, root_to_candidate)
    if not (_exact_feasible(raw_reference, root_point, list(raw_reference.variables))
            and _exact_feasible(raw_candidate, full, list(raw_candidate.variables))):
        return None
    ref_sign = 1 if raw_reference.objective_sense == "minimize" else -1
    cand_sign = 1 if raw_candidate.objective_sense == "minimize" else -1
    ref_val = ref_sign * (sum(a * _exact_frac(root_point[v])
                              for v, a in raw_reference.objective_coeffs.items())
                          + _exact_frac(raw_reference.objective_constant))
    cand_val = cand_sign * (sum(a * _exact_frac(full[v])
                               for v, a in raw_candidate.objective_coeffs.items())
                           + _exact_frac(raw_candidate.objective_constant))
    return ref_val - cand_val


def _positive_affine_objectives(reference: CanonicalIR, candidate: CanonicalIR,
                                names: List[str]):
    """Oracle đại số độc lập: ``ref = a*candidate+b``, ``a>0``.

    Hàm này cố ý không gọi implementation trong EquiCEval.
    """
    ref_sign = 1 if reference.objective_sense == "minimize" else -1
    cand_sign = 1 if candidate.objective_sense == "minimize" else -1
    ref = {name: ref_sign * _exact_frac(reference.objective_coeffs.get(name, 0.0))
           for name in names}
    cand = {name: cand_sign * _exact_frac(candidate.objective_coeffs.get(name, 0.0))
            for name in names}
    active_ref = {name: value for name, value in ref.items() if value != 0}
    active_cand = {name: value for name, value in cand.items() if value != 0}
    if set(active_ref) != set(active_cand):
        return False, {"reason": "objective support differs"}
    if not active_ref:
        a = Fraction(1)
    else:
        first = next(iter(sorted(active_ref)))
        a = active_ref[first] / active_cand[first]
        if a <= 0 or any(active_ref[name] != a * active_cand[name]
                         for name in active_ref):
            return False, {"reason": "objective coefficients are not positive proportional"}
    ref_const = ref_sign * _exact_frac(reference.objective_constant)
    cand_const = cand_sign * _exact_frac(candidate.objective_constant)
    b = ref_const - a * cand_const
    return True, {
        "reason": "positive affine objective alignment",
        "a": float(a),
        "b": float(b),
        "equation": "reference = a * candidate + b",
    }


def check_equivalence(reference: CanonicalIR, candidate: CanonicalIR,
                      contract: str = OBJECTIVE_AFFINE, *,
                      raw_reference: Optional[CanonicalIR] = None,
                      raw_candidate: Optional[CanonicalIR] = None,
                      root_to_candidate: Optional[Dict[str, Dict[str, float]]] = None,
                      candidate_eliminations=None,
                      safety_box_bound: Optional[float] = None):
    """Nhãn độc lập bằng MILP; trả ``(label, evidence)``.

    Khi truyền ``raw_reference``/``raw_candidate``/``root_to_candidate``/
    ``candidate_eliminations``, mọi phản ví dụ phải được xác minh lại trên mô
    hình gốc; nếu không xác minh được thì trả ``None`` (§1151-1152, §1153).

    ``safety_box_bound`` chỉ dùng để lấy phản ví dụ khi miền không bị chặn
    (§M0: truy vấn trong hộp B). Nếu có hàng phải dùng hộp, ta **không** bao giờ
    trả ``True`` (chưa kiểm ngoài hộp) — chỉ trả False kèm ``scope=box``.
    """
    if set(reference.variables) != set(candidate.variables):
        return None, {"method": "milp_containment",
                      "reason": "variable sets differ after projection"}
    names = _names_union(reference, candidate)
    raw_available = all(x is not None for x in
                        (raw_reference, raw_candidate, root_to_candidate,
                         candidate_eliminations))
    box_used = False

    for source, target in ((reference, candidate), (candidate, reference)):
        source_is_reference = source is reference
        for row, b in _target_rows(target, names):
            status, raw, point = _max_linear(source, {**row}, names)
            boxed = False
            if status == "infeasible":
                # A feasibility query never diverges, but an unbounded integer
                # domain can make the MILP hit the time limit ("unknown"). Retry
                # inside the declared box so an unbounded instance is still
                # decided, without ever turning a box result into ``True``.
                if safety_box_bound is None:
                    continue
            if status in ("unbounded", "unknown") and safety_box_bound is not None:
                status, raw, point = _max_linear(
                    source, {**row}, names, box_bound=safety_box_bound)
                boxed = True
            if status == "unbounded":
                return None, {"method": "milp_containment",
                              "reason": "unbounded violation without witness"}
            if status == "infeasible":
                continue
            if status != "ok":
                # unparse the reason: box=None and status unknown -> unresolved
                return None, {"method": "milp_containment", "reason": status}
            value = raw + float(b)
            if value <= DEFAULT_TOL:
                box_used = box_used or boxed
                continue
            if point is not None:
                point = _round_point(point, source, names)
                if raw_available:
                    ok = _raw_witness_ok(point, source_is_reference, raw_reference,
                                         raw_candidate, root_to_candidate,
                                         candidate_eliminations)
                else:
                    ok = (_exact_feasible(source, point, names)
                          and _exact_violates(target, point, names))
                if ok:
                    evidence = {"method": "milp_containment",
                                "counterexample": {"kind": "feasible_set",
                                                   "point": point},
                                "source_is_reference": source_is_reference,
                                "verified_on": "raw" if raw_available else "projected"}
                    if boxed:
                        evidence["scope"] = (
                            f"box[-{safety_box_bound:g},{safety_box_bound:g}]")
                    return False, evidence
            reason = ("witness not exact on original model" if raw_available
                      else "witness not exact")
            return None, {"method": "milp_containment", "reason": reason}
        if box_used:
            return None, {"method": "milp_containment",
                          "reason": "only decided within safety box; "
                                    "unbounded outside box"}

    if contract == FEASIBLE_SET:
        return True, {"method": "milp_containment", "kind": "feasible_set",
                      "contract": contract,
                      "rows_reproduce": rows_reproduce(reference, candidate)}
    if contract in (OBJECTIVE_AFFINE, ARGMIN):
        aligned, alignment = _positive_affine_objectives(reference, candidate, names)
        if aligned:
            return True, {
                "method": "independent_positive_affine_alignment",
                "objective": ("positive_affine" if contract == OBJECTIVE_AFFINE
                              else "same_argmin_certified_by_positive_affine"),
                "alignment": alignment,
                "contract": contract,
                "rows_reproduce": rows_reproduce(reference, candidate),
            }
        if contract == OBJECTIVE_AFFINE:
            return False, {
                "method": "independent_positive_affine_alignment",
                "counterexample": {"kind": "objective_affine"},
                "alignment": alignment,
                "contract": contract,
            }
        # Không affine không đồng nghĩa chắc chắn khác argmin: hai hàm tuyến tính
        # khác nhau vẫn có thể tình cờ đạt cực tiểu tại cùng một đỉnh.
        return None, {
            "method": "unsupported",
            "reason": "non-affine objectives need an independent optimizer-set proof",
            "contract": contract,
        }
    if contract != OBJECTIVE_VALUE:
        return None, {"method": "milp_containment",
                      "reason": f"contract {contract} not supported"}

    # Normalize minimization/maximization to the common convention "minimize
    # signed objective". Thus ``min f`` and ``max -f`` are equivalent.
    ref_sign = 1.0 if reference.objective_sense == "minimize" else -1.0
    cand_sign = 1.0 if candidate.objective_sense == "minimize" else -1.0
    ref_obj = {n: ref_sign * a for n, a in reference.objective_coeffs.items()}
    cand_obj = {n: cand_sign * a for n, a in candidate.objective_coeffs.items()}
    diff = {n: ref_obj.get(n, 0.0) - cand_obj.get(n, 0.0) for n in names}
    dconst = (ref_sign * reference.objective_constant
              - cand_sign * candidate.objective_constant)
    for sign in (1.0, -1.0):
        coeffs = {n: sign * d for n, d in diff.items()}
        status, raw, point = _max_linear(reference, coeffs, names)
        if status == "infeasible":
            break
        if status == "unbounded":
            return None, {"method": "milp_containment",
                          "reason": "unbounded objective difference without witness"}
        if status != "ok":
            return None, {"method": "milp_containment", "reason": status}
        value = raw + sign * dconst
        if abs(value) <= DEFAULT_TOL:
            continue
        if point is not None:
            point = _round_point(point, reference, names)
            if raw_available:
                raw_diff = _raw_objective_diff(point, raw_reference, raw_candidate,
                                               root_to_candidate, candidate_eliminations)
                ok = raw_diff is not None and abs(raw_diff) > Fraction(1, 10 ** 9)
            else:
                exact = (sum(diff[n] * _exact_frac(point[n]) for n in names)
                         + _exact_frac(dconst))
                ok = _exact_feasible(reference, point, names) and abs(exact) > Fraction(1, 10 ** 9)
            if ok:
                return False, {"method": "milp_containment",
                               "counterexample": {"kind": "objective_value",
                                                  "point": point},
                               "verified_on": "raw" if raw_available else "projected"}
        reason = ("objective witness not exact on original model" if raw_available
                  else "objective witness not exact")
        return None, {"method": "milp_containment", "reason": reason}
    return True, {"method": "milp_containment", "objective": "equal",
                  "contract": contract,
                  "rows_reproduce": rows_reproduce(reference, candidate)}


def _exact_frac(x) -> Fraction:
    return Fraction(str(float(x)))


# ── fallback certificates without a correspondence ──────────────────────────

def constant_objective_non_equivalence(reference: CanonicalIR, candidate: CanonicalIR):
    """Candidate objective hằng (mọi hệ số trên biến tự do bằng 0) vs reference không hằng.

    Bổ sung guard: chỉ kết luận ``False`` khi mục tiêu reference **thật sự biến
    thiên** trên miền khả thi của nó (nếu miền chỉ có một điểm hoặc mục tiêu hằng
    thì hai hàm vẫn có thể affine tương đương). Đây là certificate độc lập với
    ánh xạ biến, dùng cho họ ``_k`` (feasibility problem).
    """
    fixed = {n for n, v in candidate.variables.items()
             if math.isclose(v.lower_bound, v.upper_bound, abs_tol=1e-12)}
    free_obj = {n: c for n, c in candidate.objective_coeffs.items()
                if abs(c) > 1e-12 and n not in fixed}
    if free_obj:
        return None
    if not any(abs(c) > 1e-12 for c in reference.objective_coeffs.values()):
        return None

    names = _names_union(reference, reference)
    sign = 1.0 if reference.objective_sense != "maximize" else -1.0
    coeffs = {n: sign * a for n, a in reference.objective_coeffs.items()}
    # ``_max_linear`` returns the MAX of the given linear form. ``max_neg`` is
    # the max of the negated form, so the true minimum is ``-max_neg`` and the
    # objective varies iff ``max - min = max + max_neg > 0``.
    status_hi, max_pos, _ = _max_linear(reference, coeffs, names)
    status_lo, max_neg, _ = _max_linear(
        reference, {n: -c for n, c in coeffs.items()}, names)
    if status_hi == "unbounded" or status_lo == "unbounded":
        varies = True
    elif status_hi == "ok" and status_lo == "ok":
        varies = (max_pos + max_neg) > 1e-9
    else:
        return None
    if not varies:
        return None
    return False, {"method": "constant_objective_certificate",
                   "counterexample": {"kind": "objective_value"},
                   "reason": "candidate objective constant, reference non-constant",
                   "contract": "feasible_set_and_objective_affine"}



def structural_signature(ir: CanonicalIR):
    """Chữ ký bất biến theo hoán vị biến (WL 3 vòng) của đồ thị biến--ràng buộc."""
    import hashlib

    vlab = {n: f"V:{v.var_type.lower()}:{round(v.lower_bound, 6)}:{round(v.upper_bound, 6)}"
            for n, v in ir.variables.items()}
    clab = {c.name: f"C:{c.sense}" for c in ir.constraints}
    edges = [(v, c.name, round(a, 6)) for c in ir.constraints
             for v, a in c.coeffs.items()]

    def digest(text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()[:10]

    for _ in range(3):
        new_v = {}
        for name in ir.variables:
            incident = sorted((a, clab[c]) for v, c, a in edges if v == name)
            new_v[name] = digest(f"{vlab[name]}|{incident}")
        new_c = {}
        for c in ir.constraints:
            incident = sorted((a, vlab[v]) for v, cn, a in edges if cn == c.name)
            new_c[c.name] = digest(f"{clab[c.name]}|{incident}")
        vlab, clab = new_v, new_c
    return tuple(sorted(vlab.values())), tuple(sorted(clab.values()))


def structural_mismatch_certificate(reference: CanonicalIR, candidate: CanonicalIR):
    if structural_signature(reference) == structural_signature(candidate):
        return None
    return False, {"method": "structural_signature",
                   "counterexample": {"kind": "structure"},
                   "reason": "no declared correspondence and non-isomorphic structure"}
