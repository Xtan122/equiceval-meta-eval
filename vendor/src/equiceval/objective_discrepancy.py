"""
Module 5 — Objective Discrepancy (3 Semantics) for EquiCEval.

Answers three separate questions about how the candidate objective f̂ relates to
the reference objective f* over the common feasible set P∩ = P* ∩ P̂:

  1. Symbolic / numerical fidelity  — is f* == a·f̂ + b (a > 0) symbolically?
  2. Value fidelity                — Δ_value = max_{y∈P∩} |f*(y) − f̃(y)| / (s_f + ε)
  3. Decision fidelity             — bidirectional worst-case cross-regret
                                     R_{*←M̂}, R_{M̂←*} over the other's optimizer set

plus the outcome gap (Gap_abs primary, Gap_sym symmetric).

All objectives are first reduced to a minimization framing so that the formulas
are sense-agnostic (a max problem's coefficients are negated); this also makes a
candidate that flipped min↔max surface as alignment-unresolved (a ≤ 0).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.equiceval.canonical_ir import CanonicalIR, _unit_conversion_factor
from src.utils.solver_wrapper import UnifiedSolver, SolverResult

_FLOAT_INF = float("inf")
_FLOAT_NAN = float("nan")


@dataclass
class ObjectiveAlignment:
    """Symbolic alignment f*(y) = a·f̂(y) + b, a > 0 (in reduced form)."""
    is_resolved: bool = False
    a: float = 1.0
    b: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_resolved": self.is_resolved,
            "a": self.a,
            "b": self.b,
            "reason": self.reason,
            "equation": "reference = a * candidate + b",
        }


@dataclass
class ObjectiveDiscrepancyResult:
    """Full Module 5 output for one (M*, M̂) pair."""
    delta_value: float                        # Δ_value; NaN if NA / unresolved
    cross_regret_ref_given_cand: float        # R_{*←M̂}
    cross_regret_cand_given_ref: float        # R_{M̂←*}
    gap_abs: float                            # |z* − ẑ|
    gap_sym: float                            # 2|z*−ẑ| / (|z*|+|ẑ|+ε)
    status: str = "ok"                        # 'ok' | 'empty-intersection' | 'alignment-unresolved' | 'timeout' | 'unbounded'
    alignment: ObjectiveAlignment = field(default_factory=ObjectiveAlignment)
    num_solver_calls: int = 0
    optimizer_sets_equal: bool = False        # paper §4.5: both cross-regrets certified at zero

    def to_dict(self) -> Dict[str, Any]:
        return {
            "delta_value": self.delta_value,
            "cross_regret_ref_given_cand": self.cross_regret_ref_given_cand,
            "cross_regret_cand_given_ref": self.cross_regret_cand_given_ref,
            "gap_abs": self.gap_abs,
            "gap_sym": self.gap_sym,
            "status": self.status,
            "optimizer_sets_equal": self.optimizer_sets_equal,
            "alignment": {
                "is_resolved": self.alignment.is_resolved,
                "a": self.alignment.a,
                "b": self.alignment.b,
                "reason": self.alignment.reason,
            },
            "num_solver_calls": self.num_solver_calls,
        }


def _reduce_objective(sense: str, coeffs: Dict[str, float], constant: float) -> Tuple[Dict[str, float], float]:
    """Converts any objective to a minimization framing (sense-agnostic)."""
    if sense.lower() == "minimize":
        return dict(coeffs), constant
    return {v: -c for v, c in coeffs.items()}, -constant


def detect_positive_affine_alignment(
    gt_ir: CanonicalIR,
    cand_ir: CanonicalIR,
    tolerance: float = 1e-9,
) -> ObjectiveAlignment:
    """Kiểm tra ``f_ref = a f_cand + b`` với ``a > 0`` sau chuẩn hóa min/max.

    Đây là chứng nhận đại số toàn cục, không chỉ kiểm hai mô hình tình cờ có
    cùng một nghiệm tối ưu. Hằng số được phép khác; hệ số của candidate được
    đổi về đơn vị của reference trước khi tìm ``a``.
    """
    ref_coeffs, ref_const = _reduce_objective(
        gt_ir.objective_sense, gt_ir.objective_coeffs, gt_ir.objective_constant)
    cand_coeffs, cand_const = _reduce_objective(
        cand_ir.objective_sense, cand_ir.objective_coeffs, cand_ir.objective_constant)
    converted = dict(cand_coeffs)
    for name, coeff in cand_coeffs.items():
        cand_var = cand_ir.variables.get(name)
        ref_var = gt_ir.variables.get(name)
        if cand_var is None or ref_var is None:
            continue
        factor = _unit_conversion_factor(cand_var.unit, ref_var.unit)
        if factor is not None:
            converted[name] = coeff * factor

    ref_active = {name: value for name, value in ref_coeffs.items()
                  if abs(value) > tolerance}
    cand_active = {name: value for name, value in converted.items()
                   if abs(value) > tolerance}
    if set(ref_active) != set(cand_active):
        return ObjectiveAlignment(False, reason="objective support differs")
    if not ref_active:
        return ObjectiveAlignment(
            True, a=1.0, b=ref_const - cand_const,
            reason="both objectives are constant")

    first = next(iter(sorted(ref_active)))
    a = ref_active[first] / cand_active[first]
    if a <= 0:
        return ObjectiveAlignment(
            False, reason="non-positive scaling after min/max normalization")
    for name in ref_active:
        residual = ref_active[name] - a * cand_active[name]
        scale = max(1.0, abs(ref_active[name]), abs(a * cand_active[name]))
        if abs(residual) > tolerance * scale:
            return ObjectiveAlignment(False, reason="coefficients are not proportional")
    return ObjectiveAlignment(
        True, a=float(a), b=float(ref_const - a * cand_const),
        reason="positive affine objective alignment")


def _var_specs(ir: CanonicalIR) -> Dict[str, Tuple[str, float, float]]:
    return {
        vname: (var.var_type, var.lower_bound, var.upper_bound)
        for vname, var in ir.variables.items()
    }


def _constraints_list(ir: CanonicalIR) -> List[Tuple[Dict[str, float], str, float]]:
    return [(c.coeffs, c.sense, -c.constant) for c in ir.constraints]


class ObjectiveDiscrepancyEvaluator:
    """
    Computes the three-semantics objective discrepancy between a canonicalized
    reference and a canonicalized, projected candidate.
    """

    def __init__(self, solver_tolerance: float = 1e-5, solver_time_limit: float = 30.0,
                 budget=None):
        self.tolerance = solver_tolerance
        self.solver_time_limit = solver_time_limit
        self.budget = budget
        self._num_calls = 0

    def evaluate(
        self,
        gt_ir: CanonicalIR,                 # canonicalized reference
        cand_ir: CanonicalIR,               # canonicalized candidate (projected into ref space)
        gt_optimal_val: float,              # z* (reference optimum)
        cand_optimal_val: Optional[float],  # ẑ (candidate optimum); None if infeasible/unknown
        eps: float = 1e-8,
    ) -> ObjectiveDiscrepancyResult:
        gap_abs, gap_sym = _gaps(gt_optimal_val, cand_optimal_val, eps)
        self._num_calls = 0

        # Guard: candidate must live in the reference variable space.
        cand_extra = set(cand_ir.variables) - set(gt_ir.variables)
        if cand_extra:
            return ObjectiveDiscrepancyResult(
                delta_value=_FLOAT_NAN,
                cross_regret_ref_given_cand=_FLOAT_NAN,
                cross_regret_cand_given_ref=_FLOAT_NAN,
                gap_abs=gap_abs, gap_sym=gap_sym,
                status="alignment-unresolved",
                alignment=ObjectiveAlignment(False, reason=f"candidate has variables outside reference space: {sorted(cand_extra)}"),
                num_solver_calls=self._num_calls,
            )

        var_specs = _var_specs(gt_ir)
        intersection = _constraints_list(gt_ir) + _constraints_list(cand_ir)

        # Reduced (minimization-framing) objectives.
        gt_coeffs, gt_const = _reduce_objective(gt_ir.objective_sense, gt_ir.objective_coeffs, gt_ir.objective_constant)
        cand_coeffs, cand_const = _reduce_objective(cand_ir.objective_sense, cand_ir.objective_coeffs, cand_ir.objective_constant)

        # Paper §4.5: unit conversion BEFORE alignment detection — bring the
        # candidate objective into the reference's unit space (e.g. kg → tonnes).
        cand_coeffs = self._apply_unit_conversion(gt_ir, cand_ir, cand_coeffs)

        # 1. Intersection feasibility.
        feas = self._solve(var_specs, {v: 0.0 for v in var_specs}, "minimize", intersection)
        if feas.status == "Infeasible":
            return ObjectiveDiscrepancyResult(
                delta_value=_FLOAT_NAN,
                cross_regret_ref_given_cand=_FLOAT_NAN,
                cross_regret_cand_given_ref=_FLOAT_NAN,
                gap_abs=gap_abs, gap_sym=gap_sym,
                status="empty-intersection",
                alignment=ObjectiveAlignment(False, reason="P* ∩ P̂ is empty"),
                num_solver_calls=self._num_calls,
            )
        if feas.status != "Optimal":
            return ObjectiveDiscrepancyResult(
                delta_value=_FLOAT_NAN,
                cross_regret_ref_given_cand=_FLOAT_NAN,
                cross_regret_cand_given_ref=_FLOAT_NAN,
                gap_abs=gap_abs, gap_sym=gap_sym,
                status="timeout",
                num_solver_calls=self._num_calls,
            )

        # 2. Symbolic alignment (on reduced forms → a ≤ 0 catches min↔max flips).
        align = self._detect_alignment(gt_coeffs, gt_const, cand_coeffs, cand_const)

        # 3. Value fidelity Δ_value.
        delta_value = _FLOAT_NAN
        if align.is_resolved:
            delta_value = self._delta_value(gt_coeffs, gt_const, cand_coeffs, cand_const,
                                            align, var_specs, intersection, gt_ir)

        # 4. Decision fidelity (cross-regret on the raw reduced objectives).
        regret_1, regret_2, r_status = self._cross_regret(
            gt_coeffs, gt_const, cand_coeffs, cand_const,
            var_specs, intersection, gt_ir, cand_ir,
        )

        # Status priority: alignment issue unless a harder solver failure surfaced.
        if not align.is_resolved:
            status = "alignment-unresolved"
        else:
            status = "ok"
        if delta_value == _FLOAT_INF or r_status == "unbounded":
            status = "unbounded"
        elif r_status == "timeout":
            status = "timeout"

        # Paper §4.5: two zero cross-regrets certify optimizer-set equality on P∩.
        optimizer_sets_equal = (
            regret_1 == regret_1 and regret_2 == regret_2
            and regret_1 <= self.tolerance and regret_2 <= self.tolerance
        )

        return ObjectiveDiscrepancyResult(
            delta_value=delta_value,
            cross_regret_ref_given_cand=regret_1,
            cross_regret_cand_given_ref=regret_2,
            gap_abs=gap_abs, gap_sym=gap_sym,
            status=status,
            alignment=align,
            num_solver_calls=self._num_calls,
            optimizer_sets_equal=optimizer_sets_equal,
        )

    def _apply_unit_conversion(
        self,
        gt_ir: CanonicalIR,
        cand_ir: CanonicalIR,
        cand_coeffs: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Paper §4.5: scales candidate objective coefficients into the reference's
        unit space. For a shared variable, value_in_cand_unit = factor ·
        value_in_ref_unit, so the candidate coefficient becomes coeff * factor.
        Unknown / missing units are left untouched.
        """
        out = dict(cand_coeffs)
        for vname, coeff in cand_coeffs.items():
            cand_var = cand_ir.variables.get(vname)
            ref_var = gt_ir.variables.get(vname)
            if cand_var is None or ref_var is None:
                continue
            factor = _unit_conversion_factor(cand_var.unit, ref_var.unit)
            if factor is not None:
                out[vname] = coeff * factor
        return out

    # ─────────────────────────────────────────────────────────────────────────
    # Alignment
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_alignment(
        self,
        gt_coeffs: Dict[str, float],
        gt_const: float,
        cand_coeffs: Dict[str, float],
        cand_const: float,
    ) -> ObjectiveAlignment:
        f = {v: c for v, c in gt_coeffs.items() if abs(c) > 1e-12}
        g = {v: c for v, c in cand_coeffs.items() if abs(c) > 1e-12}

        if set(f) != set(g):
            return ObjectiveAlignment(False, reason="objective support differs")
        if not f:
            # Both objectives are identically zero → trivially aligned.
            return ObjectiveAlignment(True, a=1.0, b=gt_const - cand_const,
                                      reason="both objectives are zero")

        a = None
        for v in f:
            if abs(f[v]) > 1e-12:
                a = f[v] / g[v]
                break
        if a is None or a <= 0:
            return ObjectiveAlignment(False, reason="non-positive or zero scaling (sense flip?)")

        for v in f:
            if abs(f[v] - a * g[v]) > 1e-9 * max(1.0, abs(f[v])):
                return ObjectiveAlignment(False, reason="coefficients not proportional")

        b = gt_const - a * cand_const
        return ObjectiveAlignment(True, a=a, b=b, reason="symbolic proportional alignment")

    # ─────────────────────────────────────────────────────────────────────────
    # Value fidelity
    # ─────────────────────────────────────────────────────────────────────────

    def _delta_value(
        self,
        gt_coeffs: Dict[str, float],
        gt_const: float,
        cand_coeffs: Dict[str, float],
        cand_const: float,
        align: ObjectiveAlignment,
        var_specs: Dict[str, Tuple[str, float, float]],
        intersection: List[Tuple[Dict[str, float], str, float]],
        gt_ir: CanonicalIR,
    ) -> float:
        # d(y) = f*_red(y) − (a·f̂_red(y) + b)
        diff_coeffs = {v: gt_coeffs.get(v, 0.0) - align.a * cand_coeffs.get(v, 0.0) for v in var_specs}
        diff_const = (gt_const - align.a * cand_const) - align.b

        r_max = self._solve(var_specs, diff_coeffs, "maximize", intersection)
        r_min = self._solve(var_specs, diff_coeffs, "minimize", intersection)

        if r_max.status == "Unbounded" or r_min.status == "Unbounded":
            return _FLOAT_INF
        if r_max.status != "Optimal" or r_min.status != "Optimal":
            return _FLOAT_NAN

        worst = max(abs(r_max.objective_value + diff_const), abs(r_min.objective_value + diff_const))
        s_f = max(1.0, sum(abs(c) for c in gt_ir.objective_coeffs.values()), abs(gt_ir.objective_constant))
        return worst / (s_f + 1e-9)

    # ─────────────────────────────────────────────────────────────────────────
    # Decision fidelity (cross-regret)
    # ─────────────────────────────────────────────────────────────────────────

    def _cross_regret(
        self,
        gt_coeffs: Dict[str, float],
        gt_const: float,
        cand_coeffs: Dict[str, float],
        cand_const: float,
        var_specs: Dict[str, Tuple[str, float, float]],
        intersection: List[Tuple[Dict[str, float], str, float]],
        gt_ir: CanonicalIR,
        cand_ir: CanonicalIR,
    ) -> Tuple[float, float, str]:
        # z*_∩ = min over P∩ of f*_red.
        res_star = self._solve(var_specs, gt_coeffs, "minimize", intersection)
        # ẑ_∩ = min over P∩ of f̂_red.
        res_hat = self._solve(var_specs, cand_coeffs, "minimize", intersection)

        if res_star.status not in ("Optimal",) or res_hat.status not in ("Optimal",):
            return _FLOAT_NAN, _FLOAT_NAN, "timeout"

        z_star_int = res_star.objective_value + gt_const
        z_hat_int = res_hat.objective_value + cand_const

        s_f = max(1.0, sum(abs(c) for c in gt_ir.objective_coeffs.values()), abs(gt_ir.objective_constant))
        s_hat = max(1.0, sum(abs(c) for c in cand_ir.objective_coeffs.values()), abs(cand_ir.objective_constant))

        # R_{*←M̂}: maximize f*_red over P∩ ∩ {f̂_red = ẑ_∩}.
        cand_eq = [({v: cand_coeffs.get(v, 0.0) for v in var_specs}, "==", res_hat.objective_value)]
        r1 = self._solve_max_with_equality(var_specs, gt_coeffs, intersection, cand_eq, cand_coeffs, res_hat.objective_value)
        regret_1 = _FLOAT_NAN
        if r1.status == "Unbounded":
            return _FLOAT_INF, _FLOAT_NAN, "unbounded"
        if r1.status == "Optimal":
            regret_1 = (r1.objective_value + gt_const - z_star_int) / (s_f + 1e-9)

        # R_{M̂←*}: maximize f̂_red over P∩ ∩ {f*_red = z*_∩}.
        gt_eq = [({v: gt_coeffs.get(v, 0.0) for v in var_specs}, "==", res_star.objective_value)]
        r2 = self._solve_max_with_equality(var_specs, cand_coeffs, intersection, gt_eq, gt_coeffs, res_star.objective_value)
        regret_2 = _FLOAT_NAN
        if r2.status == "Unbounded":
            return regret_1, _FLOAT_INF, "unbounded"
        if r2.status == "Optimal":
            regret_2 = (r2.objective_value + cand_const - z_hat_int) / (s_hat + 1e-9)

        return regret_1, regret_2, "ok"

    def _solve_max_with_equality(
        self,
        var_specs: Dict[str, Tuple[str, float, float]],
        objective_coeffs: Dict[str, float],
        intersection: List[Tuple[Dict[str, float], str, float]],
        equality: List[Tuple[Dict[str, float], str, float]],
        obj_coeffs_for_band: Dict[str, float],
        band_center: float,
    ) -> SolverResult:
        """Maximizes `objective_coeffs` over P∩ plus an exact-equality constraint,
        falling back to a ±τ band if the exact equality makes the subproblem
        infeasible (numeric tightness)."""
        res = self._solve(var_specs, objective_coeffs, "maximize", intersection + equality)
        if res.status == "Optimal":
            return res
        if res.status != "Infeasible":
            return res
        # Fallback: optimizer set within a tolerance band.
        band = [
            ({v: obj_coeffs_for_band.get(v, 0.0) for v in var_specs}, "<=", band_center + self.tolerance),
            ({v: obj_coeffs_for_band.get(v, 0.0) for v in var_specs}, ">=", band_center - self.tolerance),
        ]
        return self._solve(var_specs, objective_coeffs, "maximize", intersection + band)

    # ─────────────────────────────────────────────────────────────────────────
    # Solver plumbing
    # ─────────────────────────────────────────────────────────────────────────

    def _solve(
        self,
        var_specs: Dict[str, Tuple[str, float, float]],
        objective_coeffs: Dict[str, float],
        objective_sense: str,
        constraints_list: List[Tuple[Dict[str, float], str, float]],
    ) -> SolverResult:
        self._num_calls += 1
        if self.budget is not None:
            return self.budget.solve(
                "objective",
                var_specs=var_specs,
                objective_coeffs=objective_coeffs,
                objective_sense=objective_sense,
                constraints_list=constraints_list,
                time_limit=self.solver_time_limit,
            )
        return UnifiedSolver.solve_pulp_model(
            var_specs=var_specs,
            objective_coeffs=objective_coeffs,
            objective_sense=objective_sense,
            constraints_list=constraints_list,
            time_limit=self.solver_time_limit,
        )


def _gaps(gt_optimal_val: Optional[float], cand_optimal_val: Optional[float], eps: float) -> Tuple[float, float]:
    """Gap_abs (primary) + Gap_sym (symmetric). NA (NaN) when either optimum is missing/non-finite."""
    if (
        gt_optimal_val is None
        or cand_optimal_val is None
        or not np.isfinite(gt_optimal_val)
        or not np.isfinite(cand_optimal_val)
    ):
        return _FLOAT_NAN, _FLOAT_NAN
    gap_abs = abs(gt_optimal_val - cand_optimal_val)
    gap_sym = 2.0 * gap_abs / (abs(gt_optimal_val) + abs(cand_optimal_val) + eps)
    return gap_abs, gap_sym
