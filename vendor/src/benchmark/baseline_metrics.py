"""
Baseline Metrics — 9 metric từ bài báo gốc (arXiv:2510.16943).

Metric được tính KHÔNG có filter Module 0 (như bài báo gốc làm):
  - Con-Precision / Con-Recall: structural constraint matching
  - Var-Precision / Var-Recall: variable set matching
  - Optimality-Gap:   |LLM_obj - REF_obj| / |REF_obj|
  - Obj-RMSE:         RMSE giữa objective value
  - Cons-RMSE:        RMSE functional evaluation của matched constraints
  - Latency:          từ LLMResponse
  - Output-Tokens:    từ LLMResponse

Đây là "before" baseline để so sánh với Module 0+1 "after".
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from src.equiceval.canonical_ir import CanonicalIR, CanonicalConstraint
from src.utils.solver_wrapper import UnifiedSolver

# Fixed seed như bài báo gốc (n=100 samples)
_SAMPLE_SEED = 42
_N_SAMPLES    = 100


@dataclass
class BaselineMetrics:
    """9 metric từ bài báo gốc."""
    # Structural
    con_precision: Optional[float] = None   # % cand constraints matched in ref
    con_recall:    Optional[float] = None   # % ref  constraints covered by cand
    var_precision: Optional[float] = None   # % cand variables in ref
    var_recall:    Optional[float] = None   # % ref  variables in cand
    # Solution quality
    optimality_gap: Optional[float] = None  # |LLM_obj - REF_obj| / |REF_obj|
    obj_rmse:       Optional[float] = None  # RMSE of objective values
    cons_rmse:      Optional[float] = None  # RMSE of matched constraint evaluations
    # Efficiency (từ LLMResponse)
    latency_s:     Optional[float] = None
    output_tokens: Optional[int]   = None
    input_tokens:  Optional[int]   = None

    note: str = ""   # Lý do nếu một số metric là None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "con_precision":  self.con_precision,
            "con_recall":     self.con_recall,
            "var_precision":  self.var_precision,
            "var_recall":     self.var_recall,
            "optimality_gap": self.optimality_gap,
            "obj_rmse":       self.obj_rmse,
            "cons_rmse":      self.cons_rmse,
            "latency_s":      self.latency_s,
            "output_tokens":  self.output_tokens,
            "input_tokens":   self.input_tokens,
            "note":           self.note,
        }


class BaselineEvaluator:
    """
    Tính 9 metric bài báo gốc cho một cặp (ref_ir, cand_ir).

    Dùng heuristic name-matching (đúng như bài gốc):
      - Constraints matched theo tên
      - Variables matched theo tên

    Không dùng Module 0+1 projection certificate ở đây.
    """

    def __init__(self, tolerance: float = 1e-6, solver_timeout: float = 10.0):
        self.tolerance     = tolerance
        self.solver_timeout = solver_timeout

    # ── Public entry ─────────────────────────────────────────────────────────

    def evaluate(
        self,
        ref_ir: CanonicalIR,
        cand_ir: Optional[CanonicalIR],
        latency_s: float = 0.0,
        output_tokens: int = 0,
        input_tokens: int = 0,
    ) -> BaselineMetrics:
        """
        Tính đầy đủ 9 metric. Nếu cand_ir là None (parse fail), trả về
        tất cả None với note='parse-error'.
        """
        m = BaselineMetrics(
            latency_s=latency_s,
            output_tokens=output_tokens,
            input_tokens=input_tokens,
        )

        if cand_ir is None:
            m.note = "parse-error: no candidate IR"
            return m

        # ── Structural metrics ────────────────────────────────────────────
        m.var_precision, m.var_recall = self._var_precision_recall(ref_ir, cand_ir)
        m.con_precision, m.con_recall = self._con_precision_recall(ref_ir, cand_ir)

        # ── Solution quality ──────────────────────────────────────────────
        ref_obj = self._solve_optimal(ref_ir)
        cand_obj = self._solve_optimal(cand_ir)

        if ref_obj is not None and cand_obj is not None:
            denom = abs(ref_obj) if abs(ref_obj) > self.tolerance else 1.0
            m.optimality_gap = abs(cand_obj - ref_obj) / denom
            m.obj_rmse = math.sqrt((cand_obj - ref_obj) ** 2)
        else:
            m.note = (
                "optimality metrics N/A: "
                f"ref_obj={ref_obj}, cand_obj={cand_obj}"
            )

        # ── Cons-RMSE ─────────────────────────────────────────────────────
        m.cons_rmse = self._cons_rmse(ref_ir, cand_ir)

        return m

    # ── Variable precision / recall ───────────────────────────────────────

    def _var_precision_recall(
        self, ref_ir: CanonicalIR, cand_ir: CanonicalIR
    ) -> Tuple[float, float]:
        ref_vars  = set(ref_ir.variables.keys())
        cand_vars = set(cand_ir.variables.keys())
        if not cand_vars:
            return 0.0, 0.0
        precision = len(ref_vars & cand_vars) / len(cand_vars)
        recall    = len(ref_vars & cand_vars) / len(ref_vars) if ref_vars else 1.0
        return round(precision, 4), round(recall, 4)

    # ── Constraint precision / recall (name-based heuristic) ─────────────

    def _con_precision_recall(
        self, ref_ir: CanonicalIR, cand_ir: CanonicalIR
    ) -> Tuple[float, float]:
        """
        Matching theo normalized constraint fingerprint (coefficient ratios + sense).
        Bài báo gốc dùng structural matching — ta dùng fingerprint thay cho tên
        để tránh phụ thuộc vào tên đặt bởi LLM.
        """
        ref_fps  = self._constraint_fingerprints(ref_ir)
        cand_fps = self._constraint_fingerprints(cand_ir)

        matched = sum(1 for fp in cand_fps if fp in ref_fps)
        precision = matched / len(cand_fps) if cand_fps else 0.0
        recall    = matched / len(ref_fps)  if ref_fps  else 1.0
        return round(precision, 4), round(recall, 4)

    @staticmethod
    def _constraint_fingerprints(ir: CanonicalIR) -> List[tuple]:
        """
        Fingerprint mỗi constraint bằng cách chuẩn hóa:
          - Sort coefficients by variable name
          - Chia cho L1 norm
          - Làm tròn 4 chữ số
        """
        fps = []
        for c in ir.constraints:
            l1 = sum(abs(v) for v in c.coeffs.values())
            if l1 < 1e-9:
                continue
            sorted_items = tuple(
                sorted((k, round(v / l1, 4)) for k, v in c.coeffs.items())
            )
            fps.append((sorted_items, c.sense))
        return fps

    # ── Optimal objective value ───────────────────────────────────────────

    def _solve_optimal(self, ir: CanonicalIR) -> Optional[float]:
        var_specs = {
            n: (v.var_type, v.lower_bound, v.upper_bound)
            for n, v in ir.variables.items()
        }
        constraints = [
            (c.coeffs, c.sense, -c.constant)
            for c in ir.constraints
        ]
        result = UnifiedSolver.solve_pulp_model(
            var_specs=var_specs,
            objective_coeffs=ir.objective_coeffs,
            objective_sense=ir.objective_sense,
            constraints_list=constraints,
            time_limit=self.solver_timeout,
        )
        if result.status == "Optimal" and result.objective_value is not None:
            return result.objective_value
        return None

    # ── Cons-RMSE ─────────────────────────────────────────────────────────

    def _cons_rmse(self, ref_ir: CanonicalIR, cand_ir: CanonicalIR) -> Optional[float]:
        """
        Tính RMSE của constraint evaluations trên 100 random samples.
        Chỉ tính trên matched constraints (theo fingerprint).
        """
        ref_fps  = {fp: c for fp, c in zip(
            self._constraint_fingerprints(ref_ir), ref_ir.constraints
        )}
        cand_fps = {fp: c for fp, c in zip(
            self._constraint_fingerprints(cand_ir), cand_ir.constraints
        )}

        matched_pairs: List[Tuple[CanonicalConstraint, CanonicalConstraint]] = []
        for fp, cand_c in cand_fps.items():
            if fp in ref_fps:
                matched_pairs.append((ref_fps[fp], cand_c))

        if not matched_pairs:
            return None

        # Generate 100 random samples (fixed seed)
        all_vars = set(ref_ir.variables) | set(cand_ir.variables)
        rng = random.Random(_SAMPLE_SEED)

        def _sample() -> Dict[str, float]:
            return {
                v: rng.uniform(0.0, 1.0)
                for v in all_vars
            }

        samples = [_sample() for _ in range(_N_SAMPLES)]

        rmse_per_constraint = []
        for ref_c, cand_c in matched_pairs:
            sq_errors = []
            for x in samples:
                ref_val  = sum(ref_c.coeffs.get(v, 0.0)  * x.get(v, 0.0) for v in ref_c.coeffs)  + ref_c.constant
                cand_val = sum(cand_c.coeffs.get(v, 0.0) * x.get(v, 0.0) for v in cand_c.coeffs) + cand_c.constant
                sq_errors.append((ref_val - cand_val) ** 2)
            rmse_per_constraint.append(math.sqrt(sum(sq_errors) / len(sq_errors)))

        return round(sum(rmse_per_constraint) / len(rmse_per_constraint), 6)
