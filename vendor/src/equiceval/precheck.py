"""
Module 0 — Pre-check & Evaluator State Machine (EquiCEval upgrade paper §4.1).

Mỗi record trả về một status tuple 5 chiều TRƯỚC khi tính bất kỳ metric nào:

    S = (S_parse, S_map, S_feas, S_bound, S_solve)

Các trạng thái:
    S_parse  ∈ {parsed, parse-error}
    S_map    ∈ {verified, unresolved, unsupported}
    S_feas   ∈ {feasible, infeasible, unknown}
    S_bound  ∈ {bounded, domain-unbounded, unknown}
    S_solve  ∈ {precheck-passed, timeout, numerical-failure, inconclusive}

Quy tắc xử lý (Table 3 trong upgrade paper):
  - Parse error         → báo, giữ raw output, không gán metric
  - Mapping unresolved  → báo metric có điều kiện, KHÔNG chạy projected discrepancy
  - Infeasible          → ghi kết luận số; chưa có chứng cứ IIS
  - Domain-unbounded    → ghi đúng trạng thái, không đổi thành lỗi cú pháp
  - Timeout/numerical   → giữ witness dương nếu đã tìm; còn lại inconclusive
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import logging

from src.equiceval.canonical_ir import (
    CanonicalIR,
    CanonicalConstraint,
    ProjectionCertificate,
    build_projection_certificate,
)
from src.utils.solver_wrapper import UnifiedSolver, SolverResult, QueryBudget
from src.equiceval.evidence import point_is_feasible

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations (mirroring Table 2 in upgrade paper)
# ─────────────────────────────────────────────────────────────────────────────

class ParseStatus(str, Enum):
    PARSED      = "parsed"
    PARSE_ERROR = "parse-error"


class MapStatus(str, Enum):
    VERIFIED   = "verified"       # Affine projection / shared vars confirmed
    UNRESOLVED = "unresolved"     # Heuristic mapping attempted, not certified
    UNSUPPORTED = "unsupported"   # Auxiliary-variable projection not available


class FeasStatus(str, Enum):
    FEASIBLE   = "feasible"
    INFEASIBLE = "infeasible"
    UNKNOWN    = "unknown"        # Solver did not conclude (timeout)


class BoundStatus(str, Enum):
    BOUNDED          = "bounded"
    DOMAIN_UNBOUNDED = "domain-unbounded"
    UNKNOWN          = "unknown"


class SolveStatus(str, Enum):
    PRECHECK_PASSED    = "precheck-passed"
    CERTIFIED         = "certified"  # Historical value only.
    TIMEOUT           = "timeout"
    NUMERICAL_FAILURE = "numerical-failure"
    INCONCLUSIVE      = "inconclusive"   # No witness, solver did not certify


# ─────────────────────────────────────────────────────────────────────────────
# Status tuple
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluatorStatus:
    """
    The 5-dimensional status tuple returned by Module 0.
    All downstream modules MUST check this before computing metrics.
    """
    s_parse : ParseStatus  = ParseStatus.PARSED
    s_map   : MapStatus    = MapStatus.UNRESOLVED
    s_feas  : FeasStatus   = FeasStatus.UNKNOWN
    s_bound : BoundStatus  = BoundStatus.UNKNOWN
    s_solve : SolveStatus  = SolveStatus.INCONCLUSIVE

    # Human-readable diagnostics per status
    parse_error_msg   : Optional[str]              = None
    map_reason        : Optional[str]              = None
    feas_iis_hints    : List[str]                  = field(default_factory=list)
    bound_violated_var: Optional[str]              = None
    solve_wall_time   : float                      = 0.0
    solve_node_count  : Optional[int]              = None  # CBC wrapper does not expose it.
    witness_found     : bool                       = False
    scope_issues      : List[str]                  = field(default_factory=list)

    # Module 1 projection certificate — set by check_mapping()
    projection_certificate: Optional[ProjectionCertificate] = None

    # Quyết định xuôi từ state machine về pipeline (paper §4.1)
    #   can_run_projected_discrepancy : map=verified
    #   can_run_delta_right  (Δ→)    : verified map + source feasibility
    #   can_run_delta_left   (Δ←)    : map=verified (infeasible candidate: Δ← chỉ chạy nếu đã ở Y)
    #   can_run_objective_comparison : permission to try, not proof of finite extrema
    can_run_projected_discrepancy : bool = False
    can_run_delta_right           : bool = False   # Δ→ = NA khi candidate infeasible
    can_run_delta_left            : bool = False   # Δ← chạy nếu map=verified, kể cả infeasible
    can_run_objective_comparison  : bool = False
    should_report_as_inconclusive : bool = True

    @property
    def can_run_directed_queries(self) -> bool:
        """Backward-compatible alias: True khi cả Δ→ lẫn Δ← đều chạy được."""
        return self.can_run_delta_right and self.can_run_delta_left

    def is_fully_certified(self) -> bool:
        """A precheck never certifies semantic correctness (legacy API)."""
        return False

    def short_repr(self) -> str:
        """Compact 5-char string for logging."""
        return (
            f"({self.s_parse.value}|{self.s_map.value}|"
            f"{self.s_feas.value}|{self.s_bound.value}|{self.s_solve.value})"
        )

    def to_dict(self) -> Dict[str, Any]:
        cert_dict = self.projection_certificate.to_dict() if self.projection_certificate else None
        return {
            "s_parse":  self.s_parse.value,
            "s_map":    self.s_map.value,
            "s_feas":   self.s_feas.value,
            "s_bound":  self.s_bound.value,
            "s_solve":  self.s_solve.value,
            "parse_error_msg":    self.parse_error_msg,
            "map_reason":         self.map_reason,
            "feas_iis_hints":     self.feas_iis_hints,
            "bound_violated_var": self.bound_violated_var,
            "solve_wall_time":    self.solve_wall_time,
            "witness_found":      self.witness_found,
            "scope_issues":       self.scope_issues,
            "projection_certificate":        cert_dict,
            "can_run_projected_discrepancy": self.can_run_projected_discrepancy,
            "can_run_delta_right":           self.can_run_delta_right,
            "can_run_delta_left":            self.can_run_delta_left,
            "can_run_directed_queries":      self.can_run_directed_queries,
            "can_run_objective_comparison":  self.can_run_objective_comparison,
            "should_report_as_inconclusive": self.should_report_as_inconclusive,
            "fully_certified": self.is_fully_certified(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main Pre-checker class
# ─────────────────────────────────────────────────────────────────────────────

class ModelPreChecker:
    """
    Chạy Module 0: Pre-check & Evaluator State Machine.

    Sequence:
        1. Kiểm tra parse status (truyền vào từ parser, không tự parse lại)
        2. Kiểm tra variable-space mapping (shared vars hoặc affine projection)
        3. Kiểm tra feasibility bằng solver
        4. Kiểm tra boundedness theo từng tọa độ và cả hai chiều
        5. Xác định solve status dựa trên kết quả solver
        6. Thiết lập can_run_* flags cho các module xuôi

    Parameters
    ----------
    solver_time_limit : float
        Giới hạn thời gian solver cho mỗi query (giây).
    tolerance : float
        Tolerance ε cho numerical checks.
    safety_box_bound : float
        Bound B khi kiểm tra domain-unbounded models trong safety box.
    """

    def __init__(
        self,
        solver_time_limit: float = 30.0,
        tolerance: float = 1e-5,
        safety_box_bound: float = 1e6,
        budget: Optional[QueryBudget] = None,
    ):
        self.solver_time_limit = solver_time_limit
        self.tolerance = tolerance
        self.safety_box_bound = safety_box_bound
        self.budget = budget
        self.last_solver_status = None

    def _solve(self, **kwargs):
        result = (self.budget.solve("precheck", **kwargs) if self.budget is not None
                  else UnifiedSolver.solve_pulp_model(**kwargs))
        self.last_solver_status = result.status
        return result

    # ── Step 1: Parse ────────────────────────────────────────────────────────

    def check_parse(
        self,
        ir: Optional[CanonicalIR],
        parse_error_msg: Optional[str] = None
    ) -> Tuple[ParseStatus, Optional[str]]:
        """
        Nhận kết quả parse từ bên ngoài.
        Nếu ir is None → parse-error.
        """
        if ir is None:
            msg = parse_error_msg or "IR is None: parser returned nothing"
            logger.warning("[Module 0 / parse] PARSE-ERROR: %s", msg)
            return ParseStatus.PARSE_ERROR, msg
        return ParseStatus.PARSED, None

    # ── Step 2: Variable-space mapping ──────────────────────────────────────

    def check_mapping(
        self,
        ref_ir: CanonicalIR,
        cand_ir: CanonicalIR,
        provided_substitutions: Optional[Dict] = None,
    ) -> Tuple[MapStatus, str, ProjectionCertificate]:
        """
        Kiểm tra variable-space mapping bằng cách gọi Module 1's
        build_projection_certificate().

        Thứ tự: ánh xạ affine khai báo, mã yêu cầu duy nhất, bí danh khai báo,
        rồi quy ước cùng tên biến (giả thiết được ghi lại). Không chứng nhận
        ánh xạ từ tên gần giống hoặc mẫu hệ số.

        MapStatus mapping:
          VERIFIED    — certificate.is_verified = True
          UNRESOLVED  — ánh xạ chưa đủ, dù tỷ lệ đối chiếu là bao nhiêu
          UNSUPPORTED — unsupported features (nonlinear, aux-var không có elimination)

        Upgrade paper: chỉ chạy directed discrepancy khi VERIFIED.
        Returns (MapStatus, reason, ProjectionCertificate)
        """
        cert = build_projection_certificate(
            ref_ir=ref_ir,
            cand_ir=cand_ir,
            provided_substitutions=provided_substitutions,
            tolerance=self.tolerance,
        )

        if cert.is_verified:
            logger.debug("[Module 0 / map] VERIFIED via %s: %s", cert.detection_method, cert.reason)
            return MapStatus.VERIFIED, cert.reason, cert

        if cert.unsupported_features:
            logger.warning(
                "[Module 0 / map] UNSUPPORTED (scope): features=%s | %s",
                cert.unsupported_features, cert.reason,
            )
            return MapStatus.UNSUPPORTED, cert.reason, cert

        # Certificate không verified: tính coverage từ partial variable_map
        n_cand = len(cand_ir.variables)
        n_mapped = len(cert.variable_map)
        reason = f"Partial checked mapping ({n_mapped}/{n_cand} candidate vars): {cert.reason}"
        return MapStatus.UNRESOLVED, reason, cert

    # ── Step 3: Feasibility check ────────────────────────────────────────────

    def check_feasibility(
        self,
        ir: CanonicalIR,
    ) -> Tuple[FeasStatus, List[str], float]:
        """
        Giải feasibility check: minimize 0 s.t. constraints.
        Trả về (status, iis_hints, wall_time).

        Điểm khả thi phải được kiểm tra lại. Hints hiện để trống; không tự
        nhận một tập hàng nghi ngờ là IIS đã chứng minh.
        """
        var_specs = {
            vname: (var.var_type, var.lower_bound, var.upper_bound)
            for vname, var in ir.variables.items()
        }
        constraints_list = [
            (c.coeffs, c.sense, -c.constant)
            for c in ir.constraints
        ]
        dummy_obj = {vname: 0.0 for vname in var_specs}

        t0 = time.perf_counter()
        result = self._solve(
            var_specs=var_specs,
            objective_coeffs=dummy_obj,
            objective_sense="minimize",
            constraints_list=constraints_list,
            time_limit=self.solver_time_limit,
        )
        wall = time.perf_counter() - t0

        if result.status in ("Optimal", "Timeout") and point_is_feasible(ir, result.variable_values):
            return FeasStatus.FEASIBLE, [], wall

        if result.status == "Infeasible":
            return FeasStatus.INFEASIBLE, [], wall

        if result.status == "Timeout":
            # Solver hit time limit without certifying feasibility or infeasibility.
            # Paper Table 3: report inconclusive, do NOT flip to feasible.
            logger.warning(
                "[Module 0 / feas] Solver TIMEOUT after %.2fs for '%s'",
                wall, ir.problem_name,
            )
            return FeasStatus.UNKNOWN, [], wall

        # Error or unexpected status
        logger.warning("[Module 0 / feas] Solver status: %s", result.status)
        return FeasStatus.UNKNOWN, [], wall

    def _find_iis_hints(
        self,
        var_specs: Dict,
        constraints_list: List,
        ir: CanonicalIR,
    ) -> List[str]:
        """
        Legacy deletion heuristic, not invoked by the pipeline.
        Suspect names are NOT a certified IIS or a unique cause of infeasibility.
        """
        iis_hints = []
        dummy_obj = {vname: 0.0 for vname in var_specs}

        for skip_idx, c in enumerate(ir.constraints):
            reduced = [con for i, con in enumerate(constraints_list) if i != skip_idx]
            res = self._solve(
                var_specs=var_specs,
                objective_coeffs=dummy_obj,
                objective_sense="minimize",
                constraints_list=reduced,
                time_limit=self.solver_time_limit,
            )
            if res.status == "Optimal":
                iis_hints.append(c.name)

        if not iis_hints:
            # Không tìm được → báo tất cả là suspect
            iis_hints = [c.name for c in ir.constraints]

        logger.info("[Module 0 / IIS] Hints: %s", iis_hints)
        return iis_hints

    # ── Step 4: Boundedness check ────────────────────────────────────────────

    def check_boundedness(
        self,
        ir: CanonicalIR,
        use_safety_box: bool = False,
    ) -> Tuple[BoundStatus, Optional[str], float]:
        """
        Kiểm tra từng tọa độ theo hai chiều; cận hữu hạn khai báo không cần giải lại.
        Nếu unbounded → báo domain-unbounded.
        Trả về (status, violated_var_hint, wall_time).
        """
        var_specs = {
            vname: (
                var.var_type,
                self._safety_lower(var.lower_bound) if use_safety_box else var.lower_bound,
                self._safety_upper(var.upper_bound) if use_safety_box else var.upper_bound,
            )
            for vname, var in ir.variables.items()
        }
        constraints_list = [
            (c.coeffs, c.sense, -c.constant)
            for c in ir.constraints
        ]
        start = time.perf_counter()
        for name, (_, low, high) in var_specs.items():
            for sign, bound in ((1.0, high), (-1.0, low)):
                if math.isfinite(bound):
                    continue
                result = self._solve(var_specs=var_specs, objective_coeffs={name: sign},
                                     objective_sense="maximize", constraints_list=constraints_list,
                                     time_limit=self.solver_time_limit)
                if result.status == "Unbounded":
                    return BoundStatus.DOMAIN_UNBOUNDED, name, time.perf_counter() - start
                if result.status != "Optimal":
                    return BoundStatus.UNKNOWN, None, time.perf_counter() - start
        return BoundStatus.BOUNDED, None, time.perf_counter() - start

    def _safety_lower(self, lower_bound: float) -> float:
        if lower_bound == float("-inf"):
            return -self.safety_box_bound
        return max(lower_bound, -self.safety_box_bound)

    def _safety_upper(self, upper_bound: float) -> float:
        if upper_bound == float("inf"):
            return self.safety_box_bound
        return min(upper_bound, self.safety_box_bound)

    # ── Main entry: run full pre-check pipeline ──────────────────────────────

    def run(
        self,
        ref_ir: Optional[CanonicalIR],
        cand_ir: Optional[CanonicalIR],
        ref_parse_error: Optional[str] = None,
        cand_parse_error: Optional[str] = None,
        check_reference: bool = True,
        provided_substitutions: Optional[Dict] = None,
    ) -> Tuple[EvaluatorStatus, EvaluatorStatus]:
        """
        Chạy đầy đủ Module 0 cho cả reference và candidate IR.

        Parameters
        ----------
        provided_substitutions : dict, optional
            Explicit affine substitutions {cand_var: ref_var_or_AffineExpression}
            truyền vào Module 1's build_projection_certificate().

        Returns
        -------
        (ref_status, cand_status) : Tuple[EvaluatorStatus, EvaluatorStatus]
            ref_status  — nếu reference lỗi → record bị loại (benchmark error)
            cand_status — trạng thái candidate, quyết định pipeline xuôi

        Upgrade paper rules:
            Reference parse-error / infeasible / out-of-scope → loại record.
            Candidate lỗi → báo, tiếp tục với partial metrics.
            Infeasible candidate → Δ→=NA; Δ← chạy nếu map=verified (paper Table 3).
        """
        # ── Reference ─────────────────────────────────────────────────────
        ref_status = EvaluatorStatus()

        s_parse_ref, parse_msg_ref = self.check_parse(ref_ir, ref_parse_error)
        ref_status.s_parse = s_parse_ref
        ref_status.parse_error_msg = parse_msg_ref

        if s_parse_ref == ParseStatus.PARSE_ERROR:
            ref_status.should_report_as_inconclusive = True
            logger.error("[Module 0 / ref] Reference parse-error: %s", parse_msg_ref)
            cand_status = EvaluatorStatus()
            cand_status.s_parse = ParseStatus.PARSE_ERROR
            cand_status.parse_error_msg = "Reference parse failed, record excluded"
            return ref_status, cand_status

        if check_reference and ref_ir is not None:
            # ── Scope check (Module 1): nonlinear / stochastic / unsupported ──
            scope_issues = ref_ir.unsupported_scope_features()
            if scope_issues:
                logger.error(
                    "[Module 0 / ref] Reference out-of-scope: %s → record excluded",
                    scope_issues,
                )
                ref_status.scope_issues = scope_issues
                ref_status.should_report_as_inconclusive = True
                cand_status = EvaluatorStatus()
                cand_status.s_parse = ParseStatus.PARSE_ERROR
                cand_status.parse_error_msg = f"Reference out-of-scope: {scope_issues}"
                return ref_status, cand_status

            # ── Feasibility ───────────────────────────────────────────────
            s_feas_ref, iis_ref, wt_feas = self.check_feasibility(ref_ir)
            ref_status.s_feas = s_feas_ref
            ref_status.feas_iis_hints = iis_ref
            ref_status.solve_wall_time += wt_feas

            if s_feas_ref == FeasStatus.INFEASIBLE:
                logger.error("[Module 0 / ref] Reference is INFEASIBLE → record excluded")
                ref_status.should_report_as_inconclusive = True
                cand_status = EvaluatorStatus()
                cand_status.s_parse = ParseStatus.PARSE_ERROR
                cand_status.parse_error_msg = "Reference infeasible, record excluded"
                return ref_status, cand_status

            # ── Boundedness ───────────────────────────────────────────────
            s_bound_ref, bound_var_ref, wt_bound = self.check_boundedness(ref_ir)
            ref_status.s_bound = s_bound_ref
            ref_status.bound_violated_var = bound_var_ref
            ref_status.solve_wall_time += wt_bound

            if s_bound_ref == BoundStatus.DOMAIN_UNBOUNDED:
                logger.info("[Module 0 / ref] Domain unbounded; preserve status, not a parse error")

            if s_feas_ref != FeasStatus.FEASIBLE or s_bound_ref == BoundStatus.UNKNOWN:
                ref_status.should_report_as_inconclusive = True
                return ref_status, EvaluatorStatus()

            ref_status.s_map = MapStatus.VERIFIED   # ref maps to itself
            ref_status.s_solve = SolveStatus.PRECHECK_PASSED
            ref_status.should_report_as_inconclusive = False
            ref_status.can_run_projected_discrepancy = True
            ref_status.can_run_delta_right = True
            ref_status.can_run_delta_left = True
            ref_status.can_run_objective_comparison = True

        # ── Candidate ─────────────────────────────────────────────────────
        cand_status = EvaluatorStatus()

        s_parse_cand, parse_msg_cand = self.check_parse(cand_ir, cand_parse_error)
        cand_status.s_parse = s_parse_cand
        cand_status.parse_error_msg = parse_msg_cand

        if s_parse_cand == ParseStatus.PARSE_ERROR:
            logger.warning("[Module 0 / cand] Candidate PARSE-ERROR: %s", parse_msg_cand)
            return ref_status, cand_status

        # ── Scope check (Module 1) for candidate ──────────────────────────
        if cand_ir is not None:
            cand_scope_issues = cand_ir.unsupported_scope_features()
            if cand_scope_issues:
                logger.warning(
                    "[Module 0 / cand] Candidate out-of-scope: %s", cand_scope_issues
                )
                cand_status.scope_issues = cand_scope_issues
                cand_status.s_map = MapStatus.UNSUPPORTED
                cand_status.map_reason = f"Out-of-scope features: {cand_scope_issues}"
                return ref_status, cand_status

        # ── Mapping check — calls Module 1's build_projection_certificate() ─
        if ref_ir is not None and cand_ir is not None:
            s_map, map_reason, cert = self.check_mapping(
                ref_ir, cand_ir, provided_substitutions
            )
            cand_status.s_map = s_map
            cand_status.map_reason = map_reason
            cand_status.projection_certificate = cert
            cand_status.can_run_projected_discrepancy = (s_map == MapStatus.VERIFIED)
        else:
            cand_status.s_map = MapStatus.UNSUPPORTED
            cand_status.map_reason = "ref_ir or cand_ir is None"

        # ── Feasibility ───────────────────────────────────────────────────
        s_feas_cand, iis_cand, wt_feas_c = self.check_feasibility(cand_ir)
        cand_status.s_feas = s_feas_cand
        cand_status.feas_iis_hints = iis_cand
        cand_status.solve_wall_time += wt_feas_c

        # ── Boundedness (chỉ check nếu feasible) ─────────────────────────
        if s_feas_cand == FeasStatus.FEASIBLE:
            s_bound_cand, bound_var_cand, wt_bound_c = self.check_boundedness(cand_ir)
            cand_status.s_bound = s_bound_cand
            cand_status.bound_violated_var = bound_var_cand
            cand_status.solve_wall_time += wt_bound_c
        else:
            cand_status.s_bound = BoundStatus.UNKNOWN
            logger.info(
                "[Module 0 / cand] Infeasible candidate → Δ→=NA, "
                "Δ← runs only if map=verified. IIS: %s", iis_cand
            )

        # ── Flags: Δ→ / Δ← theo paper Table 3 ───────────────────────────

        map_verified = cand_status.s_map == MapStatus.VERIFIED
        ref_ready_for_left = (
            ref_status.s_feas == FeasStatus.FEASIBLE
        )
        is_cand_feasible = s_feas_cand == FeasStatus.FEASIBLE

        # Permission to TRY queries is not a boundedness or correctness proof.
        # Unbounded sources can still yield valid witnesses or exact implications.
        cand_status.can_run_delta_right = map_verified and is_cand_feasible

        # Δ← (over-constraining): paper Table 3 — chỉ cần map=verified
        # Kể cả khi candidate infeasible, constraints đã ở Y nên Δ← vẫn valid
        cand_status.can_run_delta_left = map_verified and ref_ready_for_left

        # can_run_directed_queries là @property = (can_run_delta_right AND can_run_delta_left)
        cand_status.can_run_objective_comparison = map_verified and is_cand_feasible

        # ── Solve status: set TIMEOUT when solver timed out, else CERTIFIED / INCONCLUSIVE ──
        feas_timed_out = (s_feas_cand == FeasStatus.UNKNOWN)
        bound_timed_out = (cand_status.s_bound == BoundStatus.UNKNOWN and s_feas_cand == FeasStatus.FEASIBLE)

        if feas_timed_out or bound_timed_out:
            # Solver did not conclude within the budget.
            # Per upgrade paper: keep any positive witness already found; else inconclusive.
            cand_status.s_solve = (SolveStatus.NUMERICAL_FAILURE
                                   if self.last_solver_status == "Error" else SolveStatus.TIMEOUT)
            cand_status.should_report_as_inconclusive = True
            logger.warning(
                "[Module 0] cand='%s' → s_solve=TIMEOUT (feas_timeout=%s, bound_timeout=%s)",
                cand_ir.problem_name if cand_ir else "?", feas_timed_out, bound_timed_out,
            )
        elif cand_status.can_run_delta_right or cand_status.can_run_delta_left:
            cand_status.s_solve = SolveStatus.PRECHECK_PASSED
            cand_status.should_report_as_inconclusive = False
        elif s_feas_cand == FeasStatus.INFEASIBLE:
            cand_status.s_solve = SolveStatus.PRECHECK_PASSED  # numerical infeasibility, not an exact certificate
            cand_status.should_report_as_inconclusive = False
        else:
            cand_status.s_solve = SolveStatus.INCONCLUSIVE
            cand_status.should_report_as_inconclusive = True

        logger.info(
            "[Module 0] ref=%s | cand=%s",
            ref_status.short_repr(), cand_status.short_repr()
        )
        return ref_status, cand_status


# ─────────────────────────────────────────────────────────────────────────────
# Helper: kiểm tra nhanh một IR đơn (dùng khi không có ref so sánh)
# ─────────────────────────────────────────────────────────────────────────────

def quick_precheck(
    ir: Optional[CanonicalIR],
    parse_error: Optional[str] = None,
    solver_time_limit: float = 10.0,
) -> EvaluatorStatus:
    """
    Quick single-model pre-check (không có reference model).
    Dùng để validate candidate IR trước khi đưa vào full pipeline.
    """
    checker = ModelPreChecker(solver_time_limit=solver_time_limit)
    _, cand_status = checker.run(
        ref_ir=ir,
        cand_ir=ir,
        ref_parse_error=parse_error,
        cand_parse_error=parse_error,
        check_reference=False,
    )
    return cand_status
