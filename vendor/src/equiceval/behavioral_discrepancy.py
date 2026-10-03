"""
Module 3 — Normalized Behavioral Discrepancy for EquiCEval.

For every matched constraint pair (c*, ĉ) between a reference model M* and a
candidate model M̂, this module computes two descriptive metrics:

    D_sat    — satisfaction disagreement rate
    D_margin — normalized margin RMSE

using *normalized* violations v_c(y) = CanonicalConstraint.evaluate_residual(y)
evaluated over *targeted* sample points (feasible, boundary, parameter-perturbed)
instead of uniform random samples.

D_sat / D_margin are descriptive metrics only — they are NOT certificates and
must never gate solver decisions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import random

from src.equiceval.canonical_ir import CanonicalIR, CanonicalConstraint
from src.equiceval.matching import ConstraintMatchResult
from src.utils.solver_wrapper import UnifiedSolver, SolverResult

_SAT_TOL = 1e-5
_SAFETY_BOX = 1e6


@dataclass
class PairDiscrepancy:
    """Per-pair discrepancy between one matched (c*, ĉ) pair."""
    gt_index: int
    candidate_index: int
    gt_name: str
    candidate_name: str
    d_sat: float
    d_margin: float
    n_samples: int


@dataclass
class BehavioralDiscrepancyResult:
    """Aggregate D_sat / D_margin plus per-pair detail and sample counts."""
    d_sat: float                                        # NaN if no samples
    d_margin: float                                     # NaN if no samples
    per_pair: List[PairDiscrepancy] = field(default_factory=list)
    sample_counts: Dict[str, int] = field(default_factory=dict)  # {'feasible': n, 'boundary': n, 'perturbed': n}
    total_samples: int = 0
    num_solver_calls: int = 0                           # targeted-sampling solver queries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "d_sat": self.d_sat,
            "d_margin": self.d_margin,
            "per_pair": [
                {
                    "gt_index": p.gt_index,
                    "candidate_index": p.candidate_index,
                    "gt_name": p.gt_name,
                    "candidate_name": p.candidate_name,
                    "d_sat": p.d_sat,
                    "d_margin": p.d_margin,
                    "n_samples": p.n_samples,
                }
                for p in self.per_pair
            ],
            "sample_counts": dict(self.sample_counts),
            "total_samples": self.total_samples,
            "num_solver_calls": self.num_solver_calls,
        }


class BehavioralDiscrepancyEvaluator:
    """
    Targeted-sample generation + D_sat / D_margin computation.

    Inputs (gt_ir, cand_ir) MUST already be canonicalized and variable-aligned
    (post Module 1 projection). match_results indices refer to
    gt_ir.constraints / cand_ir.constraints as passed in.
    """

    def __init__(
        self,
        solver_tolerance: float = _SAT_TOL,
        solver_time_limit: float = 30.0,
        budget=None,
    ):
        self.tolerance = solver_tolerance
        self.solver_time_limit = solver_time_limit
        self.budget = budget
        self._num_calls = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        gt_ir: CanonicalIR,                 # canonicalized reference
        cand_ir: CanonicalIR,               # canonicalized candidate (projected into ref space if applicable)
        match_results: List[ConstraintMatchResult],
        num_feasible: int = 10,
        num_boundary_per_pair: int = 2,
        num_perturbed_per_pair: int = 2,
        perturbation_scale: float = 0.01,
        random_seed: int = 42,
        thin_margin_threshold: float = 0.2,
    ) -> BehavioralDiscrepancyResult:
        self._num_calls = 0
        if not match_results:
            return BehavioralDiscrepancyResult(
                d_sat=float("nan"),
                d_margin=float("nan"),
                per_pair=[],
                sample_counts={},
                total_samples=0,
                num_solver_calls=0,
            )

        rng = random.Random(random_seed)

        gt_var_specs = _var_specs(gt_ir)
        cand_var_specs = _var_specs(cand_ir)
        gt_constraints = _constraints_list(gt_ir)
        cand_constraints = _constraints_list(cand_ir)

        feasible: List[Dict[str, float]] = []
        boundary: List[Dict[str, float]] = []
        perturbed: List[Dict[str, float]] = []

        # ── 6.1 Feasible stratum — solver-produced points (integrality-safe). ─
        feasible.extend(
            self._feasible_samples(gt_ir, gt_var_specs, gt_constraints, max(1, num_feasible), rng)
        )
        feasible.extend(
            self._feasible_samples(cand_ir, cand_var_specs, cand_constraints, max(1, num_feasible), rng)
        )

        # ── 6.2 Boundary stratum — solver-driven exploration toward each face ─
        boundary.extend(
            self._generate_boundary_samples(
                gt_ir, cand_ir,
                gt_var_specs, cand_var_specs,
                gt_constraints, cand_constraints,
                match_results, num_boundary_per_pair,
            )
        )

        # ── 6.3 Thin-margin stratum — constraints active on a thin slice ────
        _, gt_thin = self._generate_thin_margin_samples(
            gt_ir, gt_var_specs, gt_constraints, thin_margin_threshold,
        )
        _, cand_thin = self._generate_thin_margin_samples(
            cand_ir, cand_var_specs, cand_constraints, thin_margin_threshold,
        )
        thin_margin = gt_thin + cand_thin

        # ── 6.4 Parameter-perturbed stratum, per matched pair ────────────────
        for m in match_results:
            gt_c = gt_ir.constraints[m.gt_index]
            cand_c = cand_ir.constraints[m.candidate_index]

            for k in range(max(1, num_perturbed_per_pair)):
                sign = 1.0 if (k // 2) % 2 == 0 else -1.0
                if k % 2 == 0:
                    p = self._perturbed_tight_point(
                        gt_ir, gt_c, m.gt_index, gt_var_specs, gt_constraints,
                        sign, perturbation_scale,
                    )
                else:
                    p = self._perturbed_tight_point(
                        cand_ir, cand_c, m.candidate_index, cand_var_specs, cand_constraints,
                        sign, perturbation_scale,
                    )
                if p is not None:
                    perturbed.append(p)

        # Jitter collected feasible/boundary/thin-margin points (continuous vars only).
        for p in feasible + boundary + thin_margin:
            j = self._jitter(p, gt_var_specs, rng, perturbation_scale)
            if j is not None:
                perturbed.append(j)

        feasible = _dedup(feasible)
        boundary = _dedup(boundary)
        thin_margin = _dedup(thin_margin)
        perturbed = _dedup(perturbed)

        all_samples = feasible + boundary + thin_margin + perturbed
        sample_counts = {
            "feasible": len(feasible),
            "boundary": len(boundary),
            "thin_margin": len(thin_margin),
            "perturbed": len(perturbed),
        }
        total_samples = sum(sample_counts.values())

        if total_samples == 0:
            return BehavioralDiscrepancyResult(
                d_sat=float("nan"),
                d_margin=float("nan"),
                per_pair=[],
                sample_counts=sample_counts,
                total_samples=0,
                num_solver_calls=self._num_calls,
            )

        # ── 7. Compute D_sat and D_margin ────────────────────────────────────
        sat_sum = 0.0
        margin_sq_sum = 0.0
        n_total = 0
        per_pair: List[PairDiscrepancy] = []

        for m in match_results:
            gt_c = gt_ir.constraints[m.gt_index]
            cand_c = cand_ir.constraints[m.candidate_index]

            p_sat = 0.0
            p_margin_sq = 0.0
            p_n = 0
            for y in all_samples:
                v_gt = gt_c.evaluate_residual(y)
                v_cand = cand_c.evaluate_residual(y)

                sat_gt = v_gt <= self.tolerance
                sat_cand = v_cand <= self.tolerance
                if sat_gt != sat_cand:
                    p_sat += 1.0
                    sat_sum += 1.0

                margin_diff = v_gt - v_cand
                p_margin_sq += margin_diff * margin_diff
                margin_sq_sum += margin_diff * margin_diff
                p_n += 1
                n_total += 1

            if p_n > 0:
                per_pair.append(
                    PairDiscrepancy(
                        gt_index=m.gt_index,
                        candidate_index=m.candidate_index,
                        gt_name=m.gt_name,
                        candidate_name=m.candidate_name,
                        d_sat=p_sat / p_n,
                        d_margin=math.sqrt(p_margin_sq / p_n),
                        n_samples=p_n,
                    )
                )

        return BehavioralDiscrepancyResult(
            d_sat=sat_sum / n_total,
            d_margin=math.sqrt(margin_sq_sum / n_total),
            per_pair=per_pair,
            sample_counts=sample_counts,
            total_samples=total_samples,
            num_solver_calls=self._num_calls,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Stratum generators
    # ─────────────────────────────────────────────────────────────────────────

    def _feasible_samples(
        self,
        ir: CanonicalIR,
        var_specs: Dict[str, Tuple[str, float, float]],
        constraints_list: List[Tuple[Dict[str, float], str, float]],
        num: int,
        rng: random.Random,
    ) -> List[Dict[str, float]]:
        """Feasible points via minimize-0 + random linear objectives."""
        samples: List[Dict[str, float]] = []
        dummy = {v: 0.0 for v in var_specs}

        res = self._solve(var_specs, dummy, "minimize", constraints_list)
        if res.status == "Optimal" and res.variable_values:
            samples.append(res.variable_values)

        attempts = 0
        while len(samples) < num and attempts < max(1, num * 5):
            attempts += 1
            weights = {v: rng.uniform(-1.0, 1.0) for v in var_specs}
            sense = "minimize" if rng.random() < 0.5 else "maximize"
            res = self._solve(var_specs, weights, sense, constraints_list)
            if res.status == "Optimal" and res.variable_values:
                samples.append(res.variable_values)
        return samples

    def _tight_point(
        self,
        ir: CanonicalIR,
        var_specs: Dict[str, Tuple[str, float, float]],
        constraints_list: List[Tuple[Dict[str, float], str, float]],
        target: CanonicalConstraint,
    ) -> Optional[Dict[str, float]]:
        """
        Finds a point on the boundary of `target` within the given feasible set.
        For '<=' maximize g_target; for '==' any feasible point is already tight.
        """
        if target.sense == "==":
            res = self._solve(var_specs, {v: 0.0 for v in var_specs}, "minimize", constraints_list)
            if res.status == "Optimal" and res.variable_values:
                return res.variable_values
            return None

        res = self._solve(var_specs, dict(target.coeffs), "maximize", constraints_list)
        if res.status == "Optimal" and res.variable_values:
            return res.variable_values
        return None

    def _generate_boundary_samples(
        self,
        gt_ir: CanonicalIR,
        cand_ir: CanonicalIR,
        gt_var_specs: Dict[str, Tuple[str, float, float]],
        cand_var_specs: Dict[str, Tuple[str, float, float]],
        gt_constraints: List[Tuple[Dict[str, float], str, float]],
        cand_constraints: List[Tuple[Dict[str, float], str, float]],
        match_results: List[ConstraintMatchResult],
        num_per_pair: int,
    ) -> List[Dict[str, float]]:
        """
        Solver-driven constraint-boundary exploration (paper §4.3): for each
        matched constraint pair, optimize toward the constraint faces to obtain
        points that lie exactly on a face (violation ≈ 0).
        """
        samples: List[Dict[str, float]] = []
        for m in match_results:
            gt_c = gt_ir.constraints[m.gt_index]
            cand_c = cand_ir.constraints[m.candidate_index]
            for k in range(max(1, num_per_pair)):
                if k % 2 == 0:
                    p = self._tight_point(gt_ir, gt_var_specs, gt_constraints, gt_c)
                else:
                    p = self._tight_point(cand_ir, cand_var_specs, cand_constraints, cand_c)
                if p is not None:
                    samples.append(p)
        return samples

    def _generate_thin_margin_samples(
        self,
        ir: CanonicalIR,
        var_specs: Dict[str, Tuple[str, float, float]],
        constraints_list: List[Tuple[Dict[str, float], str, float]],
        threshold: float = 0.2,
    ) -> Tuple[List[int], List[Dict[str, float]]]:
        """
        Thin-margin targeting (paper §4.3): detect constraints that are active
        only on a small feasibility region (tight slack) and sample near their
        faces. A constraint is "thin" when the normalized residual range over
        the feasible set is ≤ threshold.

        Returns (tight_constraint_indices, boundary_samples_near_them).
        """
        tight: List[int] = []
        samples: List[Dict[str, float]] = []
        for idx, c in enumerate(ir.constraints):
            s_c = max(1.0, sum(abs(a) for a in c.coeffs.values()), abs(c.constant))
            res_max = self._solve(var_specs, dict(c.coeffs), "maximize", constraints_list)
            res_min = self._solve(var_specs, dict(c.coeffs), "minimize", constraints_list)
            if res_max.status != "Optimal" or res_min.status != "Optimal":
                continue
            g_max = res_max.objective_value + c.constant
            g_min = res_min.objective_value + c.constant
            margin = abs(g_max - g_min) / s_c
            # The feasible region must (a) actually reach the constraint face and
            # (b) be thin in that direction. This avoids flagging dominated
            # constraints whose residual range is small only because other
            # constraints already pin the variables.
            near_face = abs(g_max) <= threshold * s_c
            if near_face and margin <= threshold:
                tight.append(idx)
                if res_max.variable_values:
                    samples.append(res_max.variable_values)
                if res_min.variable_values:
                    samples.append(res_min.variable_values)
        return tight, samples

    def _perturbed_tight_point(
        self,
        ir: CanonicalIR,
        target: CanonicalConstraint,
        target_idx: int,
        var_specs: Dict[str, Tuple[str, float, float]],
        constraints_list: List[Tuple[Dict[str, float], str, float]],
        sign: float,
        perturbation_scale: float,
    ) -> Optional[Dict[str, float]]:
        """
        Finds a point tight for the constraint whose rhs was perturbed by
        `sign * perturbation_scale * s_c`. The perturbed constraint replaces
        the original one in the feasible set.
        """
        s_c = max(1.0, sum(abs(c) for c in target.coeffs.values()), abs(target.constant))
        b_prime = target.constant + sign * perturbation_scale * s_c

        perturbed_constraints = list(constraints_list)
        perturbed_constraints[target_idx] = (target.coeffs, target.sense, -b_prime)

        res = self._solve(var_specs, dict(target.coeffs), "maximize", perturbed_constraints)
        if res.status == "Optimal" and res.variable_values:
            return res.variable_values
        return None

    def _jitter(
        self,
        point: Dict[str, float],
        var_specs: Dict[str, Tuple[str, float, float]],
        rng: random.Random,
        perturbation_scale: float,
    ) -> Optional[Dict[str, float]]:
        """
        Jitters continuous coordinates of a point by relative noise. Integer /
        binary variables are kept fixed to respect integrality. Unbounded
        continuous variables are clamped to a safety box.
        """
        out: Dict[str, float] = {}
        changed = False
        for v, val in point.items():
            spec = var_specs.get(v)
            if spec is None:
                out[v] = val
                continue
            vtype, low, high = spec
            if vtype.lower() in ("binary", "integer"):
                out[v] = val
                continue
            scale = max(1.0, abs(val))
            new_val = val + rng.uniform(-perturbation_scale, perturbation_scale) * scale
            if low > float("-inf"):
                new_val = max(low, new_val)
            if high < float("inf"):
                new_val = min(high, new_val)
            else:
                new_val = max(-_SAFETY_BOX, min(_SAFETY_BOX, new_val))
            if new_val != val:
                changed = True
            out[v] = new_val
        return out if changed else None

    # ─────────────────────────────────────────────────────────────────────────
    # Solver plumbing (same transformation as matching.py / precheck.py)
    # ─────────────────────────────────────────────────────────────────────────

    def _solve(
        self,
        var_specs: Dict[str, Tuple[str, float, float]],
        objective_coeffs: Dict[str, float],
        objective_sense: str,
        constraints_list: List[Tuple[Dict[str, float], str, float]],
    ) -> SolverResult:
        self._num_calls += 1
        if not var_specs:
            return SolverResult(
                status="Error", objective_value=None, variable_values={}, solve_time_seconds=0.0,
            )
        if self.budget is not None:
            return self.budget.solve(
                "behavioral",
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


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _var_specs(ir: CanonicalIR) -> Dict[str, Tuple[str, float, float]]:
    return {
        vname: (var.var_type, var.lower_bound, var.upper_bound)
        for vname, var in ir.variables.items()
    }


def _constraints_list(ir: CanonicalIR) -> List[Tuple[Dict[str, float], str, float]]:
    # c.constant is negated because the canonical form is a·y + b ≤ 0, which the
    # solver wrapper receives as a·y ≤ -b (rhs).
    return [(c.coeffs, c.sense, -c.constant) for c in ir.constraints]


def _dedup(points: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Deduplicates identical points (rounded to ~1e-9 before comparing)."""
    seen = set()
    out: List[Dict[str, float]] = []
    for p in points:
        key = tuple((v, round(p.get(v, 0.0), 9)) for v in sorted(p))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out
