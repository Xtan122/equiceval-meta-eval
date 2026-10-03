"""Reference-form baseline (Refai & Ahmed reimplementation), main-compatible.

Ported out of ``experiments/prove_improvement.py`` because that module cannot be
imported on this branch (it pulls the retired ``detection_cost_frontier`` API).
This copy only depends on ``CanonicalIR`` and numpy/scipy, so both the legacy
experiment and the new meta-evaluation runner can share one implementation.

mode='refform'  : exact variable names + raw constraint signature (scale-sensitive)
mode='strong'   : underscore-insensitive variable names + L1-normalized signature
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.equiceval.constraint_rmse import compute_cons_rmse
from src.equiceval.canonical_ir import (
    CanonicalConstraint,
    CanonicalIR,
    _normalize_name,
)


def _pr(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return p, r


def _constraint_eval_func(c: CanonicalConstraint) -> Callable[[Dict[str, float]], float]:
    return lambda x, c=c: sum(
        coef * x.get(v, 0.0) for v, coef in c.coeffs.items()
    ) + c.constant


class ReferenceFormBaseline:
    """
    Baseline theo tinh thần paper gốc:
      - decision-variable P/R: so khớp tên biến;
      - constraint P/R: Hungarian matching theo (coeffs, constant, sense);
      - Cons-RMSE: chỉ trên constraint đã match.

    ``evaluate`` trả về một dict; ``score`` càng lớn càng dễ coi là sai
    (fault), dùng làm điểm bất thường khi so AUROC/RQ2.
    """

    def __init__(self, mode: str = "refform", num_samples: int = 100, seed: int = 42,
                 cons_rmse_tol: float = 1e-6):
        if mode not in ("refform", "strong"):
            raise ValueError("mode must be 'refform' or 'strong'")
        if cons_rmse_tol < 0:
            raise ValueError("cons_rmse_tol must be nonnegative")
        self.mode = mode
        self.num_samples = num_samples
        self.seed = seed
        # Paper §8.1 baseline #1 reports component P/R *and* Cons-RMSE over the
        # 100 sampled points. ``cons_rmse_tol`` is the pre-registered relative
        # tolerance that turns Cons-RMSE into a fault indicator; 0 disables the
        # Cons-RMSE term and keeps the legacy P/R-only decision rule.
        self.cons_rmse_tol = cons_rmse_tol

    # -- variable mapping --------------------------------------------------
    def _var_map(self, ref_vars: List[str], cand_vars: List[str]) -> Dict[str, str]:
        if self.mode == "refform":
            return {c: c for c in cand_vars if c in ref_vars}
        ref_norm = {_normalize_name(r): r for r in ref_vars}
        mapping: Dict[str, str] = {}
        for c in cand_vars:
            r = ref_norm.get(_normalize_name(c))
            if r is not None:
                mapping[c] = r
        return mapping

    # -- constraint signature ---------------------------------------------
    def _signature(
        self,
        c: CanonicalConstraint,
        var_map: Dict[str, str],
        ref_vars: set,
    ) -> Optional[Tuple[Dict[str, float], float, str]]:
        coeffs: Dict[str, float] = {}
        for v, coef in c.coeffs.items():
            rv = var_map.get(v, v if v in ref_vars else None)
            if rv is None:
                return None  # unmapped variable => cannot compare
            coeffs[rv] = coeffs.get(rv, 0.0) + coef
        const = c.constant
        if self.mode == "strong":
            l1 = sum(abs(x) for x in coeffs.values())
            if l1 > 1e-12:
                coeffs = {k: v / l1 for k, v in coeffs.items()}
                const = const / l1
        return coeffs, const, c.sense

    @staticmethod
    def _distance(
        a: Tuple[Dict[str, float], float, str],
        b: Tuple[Dict[str, float], float, str],
    ) -> float:
        ca, ka, sa = a
        cb, kb, sb = b
        if sa != sb:
            return 1e6
        keys = set(ca) | set(cb)
        return sum(abs(ca.get(k, 0.0) - cb.get(k, 0.0)) for k in keys) + abs(ka - kb)

    def _cons_func(
        self,
        c: CanonicalConstraint,
        var_map: Dict[str, str],
    ) -> Optional[Callable[[Dict[str, float]], float]]:
        if any(v not in var_map for v in c.coeffs):
            return None
        return lambda x, c=c, vm=var_map: sum(
            coef * x.get(vm[v], 0.0) for v, coef in c.coeffs.items()
        ) + c.constant

    def _sample_points(self, ref_ir: CanonicalIR) -> List[Dict[str, float]]:
        rng = np.random.RandomState(self.seed)
        samples: List[Dict[str, float]] = []
        for _ in range(self.num_samples):
            point: Dict[str, float] = {}
            for name, var in ref_ir.variables.items():
                lb, ub = var.lower_bound, var.upper_bound
                if var.var_type.lower() in ("binary", "bin"):
                    point[name] = float(rng.randint(0, 2))
                else:
                    lo = 0.0 if lb == float("-inf") else float(lb)
                    hi = 1.0 if ub == float("inf") else float(ub)
                    if hi <= lo:
                        hi = lo + 1.0
                    point[name] = float(rng.uniform(lo, hi))
            samples.append(point)
        return samples

    def evaluate(self, ref_ir: CanonicalIR, cand_ir: CanonicalIR) -> Dict[str, Any]:
        ref_vars = list(ref_ir.variables)
        cand_vars = list(cand_ir.variables)
        var_map = self._var_map(ref_vars, cand_vars)

        # decision-variable P/R
        tp_v = len(var_map)
        fp_v = len(cand_vars) - tp_v
        fn_v = len(ref_vars) - tp_v
        dv_p, dv_r = _pr(tp_v, fp_v, fn_v)

        # constraint matching (Hungarian on distance matrix)
        ref_sigs = [self._signature(c, {r: r for r in ref_vars}, set(ref_vars))
                    for c in ref_ir.constraints]
        cand_sigs = [self._signature(c, var_map, set(ref_vars)) for c in cand_ir.constraints]
        n, m = len(ref_sigs), len(cand_sigs)
        matched: List[Tuple[int, int]] = []
        if n and m:
            cost = np.full((n, m), 1e6, dtype=np.float64)
            for i, rs in enumerate(ref_sigs):
                if rs is None:
                    continue
                for j, cs in enumerate(cand_sigs):
                    if cs is not None:
                        cost[i, j] = self._distance(rs, cs)
            rows, cols = linear_sum_assignment(cost)
            matched = [(i, j) for i, j in zip(rows, cols) if cost[i, j] <= 1e-6]
        tp_c = len(matched)
        fp_c = m - tp_c
        fn_c = n - tp_c
        cons_p, cons_r = _pr(tp_c, fp_c, fn_c)

        # Cons-RMSE over matched pairs
        samples = self._sample_points(ref_ir)
        pairs = []
        row_scales = []
        for gi, cj in matched:
            gf = _constraint_eval_func(ref_ir.constraints[gi])
            cf = self._cons_func(cand_ir.constraints[cj], var_map)
            if cf is not None:
                pairs.append((gf, cf))
                row_scales.append(float(ref_ir.constraints[gi].normalization_scale))
        cons_rmse = compute_cons_rmse(pairs, samples) if pairs else 0.0
        cons_rmse_scale = max([1.0] + row_scales)
        cons_rmse_rel = float(cons_rmse) / cons_rmse_scale
        cons_rmse_exceeds = bool(
            self.cons_rmse_tol > 0 and pairs and cons_rmse_rel > self.cons_rmse_tol)

        # Pre-registered decision rule: a pair is a fault when the component
        # structure differs (P/R < 1) *or* the matched rows behave differently
        # (relative Cons-RMSE above the declared tolerance). The continuous
        # ``score`` stays in [0, 1] for AUROC/AUPRC.
        score = float(max(1.0 - cons_p, 1.0 - cons_r, 1.0 if cons_rmse_exceeds else 0.0))
        return {
            "cons_precision": float(cons_p),
            "cons_recall": float(cons_r),
            "dv_precision": float(dv_p),
            "dv_recall": float(dv_r),
            "cons_rmse": float(cons_rmse),
            "cons_rmse_rel": cons_rmse_rel,
            "cons_rmse_scale": cons_rmse_scale,
            "cons_rmse_exceeds": cons_rmse_exceeds,
            "cons_rmse_tol": self.cons_rmse_tol,
            "score": score,
            "matched": tp_c,
            "n_ref": n,
            "n_cand": m,
        }
