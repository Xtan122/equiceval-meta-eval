"""Validated multi-backend solver interface.

Backends (auto-selected):
  - HiGHS (`highspy`)   → pure LP
  - SCIP  (`pyscipopt`) → MILP
  - PuLP/CBC            → fallback (always available)

Numerical bounds are telemetry, not exact proof certificates: ``bound_kind``
records whether a global bound was published (upgrade paper §9, Bảng 9).
``SolverResult`` keeps the extended telemetry (``node_count``, ``mip_gap``,
``best_bound``, ``solver``) alongside the compact ``objective_bound``/
``bound_kind`` used by the verified evaluator.
"""
from dataclasses import dataclass, field
from math import isfinite, isinf
from time import perf_counter
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import json

import numpy as np
import pulp

CBC_SETTINGS = {"threads": 1, "randomSeed": 42, "randomCbcSeed": 42,
                "primalTolerance": 1e-7, "integerTolerance": 1e-7, "gapRel": 0.0}

_VALID_TYPES = ("continuous", "cont", "real", "integer", "int", "binary", "bin")


@dataclass
class SolverResult:
    status: str
    objective_value: Optional[float]
    variable_values: Dict[str, float]
    solve_time_seconds: float
    objective_bound: Optional[float] = None
    bound_kind: str = "unavailable"
    message: str = ""
    # Extended telemetry (restored from the branch-`thanh` solver wrapper).
    node_count: int = 0
    mip_gap: Optional[float] = None
    best_bound: Optional[float] = None
    solver: str = ""
    warm_start_used: bool = False
    incumbent_feasible: Optional[bool] = None


@dataclass
class QueryBudget:
    """One cooperative wall-clock budget, shared by all solves for a pair.

    Construction/shutdown can overrun the deadline; no new solve starts after it.
    Actual elapsed time and every solver call are reported.
    """
    seconds: float = 30.0
    max_calls: Optional[int] = None
    started: float = field(default_factory=perf_counter)
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not isfinite(self.seconds) or self.seconds < 0:
            raise ValueError("Budget must be finite and nonnegative")
        if self.max_calls is not None and self.max_calls < 0:
            raise ValueError("max_calls must be nonnegative")

    @property
    def remaining(self):
        if self.max_calls is not None and len(self.calls) >= self.max_calls:
            return 0.0
        return max(0.0, self.seconds - (perf_counter() - self.started))

    def solve(self, phase: str, **kwargs):
        remaining = self.remaining
        if remaining <= 0:
            return SolverResult("Timeout", None, {}, 0.0, message="Pair budget exhausted")
        kwargs["time_limit"] = min(kwargs.get("time_limit", remaining), remaining)
        start = perf_counter()
        result = UnifiedSolver.solve_pulp_model(**kwargs)
        self.calls.append({"phase": phase, "status": result.status,
                           "elapsed_seconds": perf_counter() - start,
                           "time_limit": kwargs["time_limit"]})
        return result

    def to_dict(self):
        counts = {}
        for call in self.calls:
            counts[call["phase"]] = counts.get(call["phase"], 0) + 1
        return {"seconds": self.seconds, "max_calls": self.max_calls,
                "elapsed_seconds": perf_counter() - self.started,
                "solver_calls": len(self.calls), "calls_by_phase": counts,
                "calls": list(self.calls)}


class UnifiedSolver:
    """Auto-selecting LP/MILP solver wrapper with warm-start caching."""

    _warm_start_cache: Dict[str, Dict[str, float]] = {}
    _AVAILABLE: Optional[Dict[str, bool]] = None

    # ── Backend probing ────────────────────────────────────────────────
    @classmethod
    def _available_backends(cls) -> Dict[str, bool]:
        if cls._AVAILABLE is not None:
            return cls._AVAILABLE
        backends = {"cbc": True}
        try:
            import highspy  # noqa: F401
            backends["highs"] = True
        except ImportError:
            backends["highs"] = False
        try:
            import pyscipopt  # noqa: F401
            backends["scip"] = True
        except ImportError:
            backends["scip"] = False
        cls._AVAILABLE = backends
        return backends

    @staticmethod
    def _is_milp(var_specs: Dict[str, Tuple[str, float, float]]) -> bool:
        return any(str(vtype).lower() in ("binary", "bin", "integer", "int")
                   for vtype, _low, _high in var_specs.values())

    @staticmethod
    def _fingerprint(var_specs, objective_coeffs, objective_sense, constraints_list) -> str:
        payload = {"vars": var_specs, "obj": objective_coeffs,
                   "sense": objective_sense, "cons": constraints_list}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    @staticmethod
    def _check_feasibility(var_specs, constraints_list, values, tol: float = 1e-6):
        """Return (is_feasible, max_violation) for an incumbent solution."""
        max_violation = 0.0
        for name, (vtype, low, high) in var_specs.items():
            v = values.get(name, 0.0)
            if low not in (None, float("-inf")) and v < low - tol:
                max_violation = max(max_violation, low - v)
            if high not in (None, float("inf")) and v > high + tol:
                max_violation = max(max_violation, v - high)
            if str(vtype).lower() in ("binary", "bin", "integer", "int") and abs(v - round(v)) > tol:
                max_violation = max(max_violation, abs(v - round(v)))
        for coeffs, c_sense, rhs in constraints_list:
            lhs = sum(coeffs.get(name, 0.0) * values.get(name, 0.0) for name in coeffs)
            violation = (lhs - rhs if c_sense == "<=" else rhs - lhs if c_sense == ">="
                         else abs(lhs - rhs))
            if violation > tol:
                max_violation = max(max_violation, violation)
        return max_violation <= tol, float(max_violation)

    @staticmethod
    def _validate(var_specs, objective_coeffs, objective_sense, constraints_list):
        if str(objective_sense).lower() not in ("minimize", "maximize"):
            raise ValueError("Unknown objective sense")
        names = set(var_specs)
        for coeffs, sense, rhs in list(constraints_list) + [(objective_coeffs, "<=", 0.0)]:
            if sense not in ("<=", ">=", "=="):
                raise ValueError("Unknown constraint sense")
            if not set(coeffs) <= names:
                raise ValueError("Unknown variable in expression")
            if not isfinite(rhs) or any(not isfinite(c) for c in coeffs.values()):
                raise ValueError("Non-finite coefficient")
        for _name, (kind, low, high) in var_specs.items():
            if str(kind).lower() not in _VALID_TYPES:
                raise ValueError("Unknown variable type")
            if low != low or high != high or low == float("inf") or high == -float("inf"):
                raise ValueError("Invalid variable bound")

    @staticmethod
    def _finalize(res: SolverResult) -> SolverResult:
        if res.best_bound is None and res.objective_bound is not None:
            res.best_bound = res.objective_bound
        if res.objective_bound is None and res.best_bound is not None:
            res.objective_bound = res.best_bound
        res.bound_kind = ("global_bound" if res.objective_bound is not None
                          else "unavailable")
        return res

    # ── Public entry points ────────────────────────────────────────────
    @staticmethod
    def solve_pulp_model(var_specs, objective_coeffs, objective_sense,
                         constraints_list, time_limit=30.0) -> SolverResult:
        """Backward-compatible entry point — delegates to auto-selecting solve()."""
        return UnifiedSolver.solve(
            var_specs=var_specs, objective_coeffs=objective_coeffs,
            objective_sense=objective_sense, constraints_list=constraints_list,
            time_limit=time_limit)

    @staticmethod
    def solve(var_specs, objective_coeffs, objective_sense, constraints_list,
              time_limit=30.0, warm_start: Optional[Dict[str, float]] = None,
              force_backend: Optional[str] = None) -> SolverResult:
        """Auto-selects the backend: LP → HiGHS, MILP → SCIP, else CBC."""
        try:
            UnifiedSolver._validate(var_specs, objective_coeffs, objective_sense, constraints_list)
        except Exception as exc:
            return SolverResult("Error", None, {}, 0.0, message=str(exc))
        if time_limit <= 0:
            return SolverResult("Timeout", None, {}, 0.0)

        fingerprint = UnifiedSolver._fingerprint(
            var_specs, objective_coeffs, objective_sense, constraints_list)
        if warm_start is None:
            warm_start = UnifiedSolver._warm_start_cache.get(fingerprint)

        available = UnifiedSolver._available_backends()
        is_milp = UnifiedSolver._is_milp(var_specs)
        if force_backend is not None:
            backend = force_backend
        elif is_milp and available.get("scip"):
            backend = "scip"
        elif not is_milp and available.get("highs"):
            backend = "highs"
        else:
            backend = "cbc"

        try:
            if backend == "highs":
                res = UnifiedSolver._solve_highs(
                    var_specs, objective_coeffs, objective_sense, constraints_list,
                    time_limit, warm_start)
            elif backend == "scip":
                res = UnifiedSolver._solve_scip(
                    var_specs, objective_coeffs, objective_sense, constraints_list,
                    time_limit, warm_start)
            else:
                res = UnifiedSolver._solve_cbc(
                    var_specs, objective_coeffs, objective_sense, constraints_list,
                    time_limit)
        except Exception as exc:
            return SolverResult("Error", None, {}, 0.0, message=str(exc))

        if res.status == "Optimal" and res.variable_values:
            UnifiedSolver._warm_start_cache[fingerprint] = dict(res.variable_values)
        return UnifiedSolver._finalize(res)

    # ── HiGHS backend (LP) ─────────────────────────────────────────────
    @staticmethod
    def _solve_highs(var_specs, objective_coeffs, objective_sense,
                     constraints_list, time_limit, warm_start) -> SolverResult:
        import highspy

        m = highspy.Highs()
        m.setOptionValue("time_limit", max(float(time_limit), 1e-9))
        m.setOptionValue("output_flag", False)
        inf = highspy.kHighsInf

        name_to_idx: Dict[str, int] = {}
        for name, (vtype, low, high) in var_specs.items():
            idx = m.getNumCol()
            if str(vtype).lower() in ("binary", "bin"):
                low, high = 0.0, 1.0
            lb = -inf if low in (None, float("-inf")) else float(low)
            ub = inf if high in (None, float("inf")) else float(high)
            m.addVar(lb, ub)
            name_to_idx[name] = idx

        ncols = m.getNumCol()
        if ncols:
            m.changeColsCost(ncols, list(range(ncols)),
                             [float(objective_coeffs.get(name, 0.0)) for name in var_specs])
        m.changeObjectiveSense(highspy.ObjSense.kMaximize
                               if str(objective_sense).lower() == "maximize"
                               else highspy.ObjSense.kMinimize)

        for coeffs, c_sense, rhs in constraints_list:
            idxs, vals = [], []
            for name, coeff in coeffs.items():
                if name in name_to_idx and abs(coeff) > 1e-12:
                    idxs.append(name_to_idx[name])
                    vals.append(float(coeff))
            if not idxs:
                continue
            if c_sense == "<=":
                lb, ub = -inf, float(rhs)
            elif c_sense == ">=":
                lb, ub = float(rhs), inf
            else:
                lb, ub = float(rhs), float(rhs)
            m.addRow(lb, ub, len(idxs), idxs, vals)

        warm_start_used = False
        if warm_start is not None:
            try:
                sol = highspy.HighsSolution()
                sol.col_value = np.array([float(warm_start.get(name, 0.0)) for name in var_specs],
                                         dtype=np.float64)
                sol.value_valid = True
                m.setSolution(sol)
                warm_start_used = True
            except Exception:
                warm_start_used = False

        t0 = perf_counter()
        m.run()
        elapsed = perf_counter() - t0
        status = m.getModelStatus()
        info = m.getInfo()
        node_count = max(0, int(getattr(info, "mip_node_count", 0) or 0))
        gap = getattr(info, "mip_gap", None)
        best_bound = getattr(info, "mip_dual_bound", None)

        solution = m.getSolution()
        var_vals = {name: float(solution.col_value[idx]) for name, idx in name_to_idx.items()}

        if status == highspy.HighsModelStatus.kOptimal:
            obj = float(info.objective_function_value)
            return SolverResult("Optimal", obj, var_vals, elapsed,
                                objective_bound=obj, node_count=node_count, mip_gap=None,
                                best_bound=obj, solver="HiGHS", warm_start_used=warm_start_used,
                                incumbent_feasible=True)
        if status == highspy.HighsModelStatus.kInfeasible:
            return SolverResult("Infeasible", None, {}, elapsed, node_count=node_count,
                                mip_gap=gap, best_bound=best_bound, solver="HiGHS",
                                warm_start_used=warm_start_used)
        if status == highspy.HighsModelStatus.kUnbounded:
            return SolverResult("Unbounded", None, {}, elapsed, node_count=node_count,
                                mip_gap=gap, best_bound=best_bound, solver="HiGHS",
                                warm_start_used=warm_start_used)
        if status == highspy.HighsModelStatus.kTimeLimit:
            feasible, _ = UnifiedSolver._check_feasibility(var_specs, constraints_list, var_vals)
            value = info.objective_function_value
            return SolverResult("Timeout", float(value) if value not in (None, float("inf")) else None,
                                var_vals, elapsed, node_count=node_count, mip_gap=gap,
                                best_bound=best_bound, solver="HiGHS",
                                warm_start_used=warm_start_used, incumbent_feasible=feasible)
        return SolverResult("Error", None, {}, elapsed, node_count=node_count, mip_gap=gap,
                            best_bound=best_bound, solver="HiGHS",
                            warm_start_used=warm_start_used)

    # ── SCIP backend (MILP) ────────────────────────────────────────────
    @staticmethod
    def _solve_scip(var_specs, objective_coeffs, objective_sense,
                    constraints_list, time_limit, warm_start) -> SolverResult:
        import pyscipopt as sp

        m = sp.Model("equiceval")
        m.hideOutput()
        m.setRealParam("limits/time", max(float(time_limit), 0.0))

        scip_vars: Dict[str, Any] = {}
        for name, (vtype, low, high) in var_specs.items():
            kind = str(vtype).lower()
            if kind in ("binary", "bin"):
                low, high = max(0.0, low), min(1.0, high)
                if low > high:
                    return SolverResult("Infeasible", None, {}, 0.0,
                                        message="Contradictory variable bounds")
                scip_vars[name] = m.addVar(name, vtype="B", lb=float(low), ub=float(high))
            else:
                low = None if low in (None, float("-inf")) else float(low)
                high = None if high in (None, float("inf")) else float(high)
                if low is not None and high is not None and low > high:
                    return SolverResult("Infeasible", None, {}, 0.0,
                                        message="Contradictory variable bounds")
                scip_vars[name] = m.addVar(name, vtype="I" if kind in ("integer", "int") else "C",
                                           lb=low, ub=high)

        obj_expr = sp.quicksum(objective_coeffs[name] * scip_vars[name]
                               for name in scip_vars
                               if abs(objective_coeffs.get(name, 0.0)) > 1e-12)
        m.setObjective(obj_expr, sense="maximize" if str(objective_sense).lower() == "maximize"
                       else "minimize")

        for i, (coeffs, c_sense, rhs) in enumerate(constraints_list):
            expr = sp.quicksum(coeffs[name] * scip_vars[name] for name in coeffs
                               if name in scip_vars and abs(coeffs.get(name, 0.0)) > 1e-12)
            if c_sense == "<=":
                m.addCons(expr <= rhs, name=f"c_{i}")
            elif c_sense == ">=":
                m.addCons(expr >= rhs, name=f"c_{i}")
            else:
                m.addCons(expr == rhs, name=f"c_{i}")

        warm_start_used = False
        if warm_start is not None:
            try:
                sol = m.createSol()
                for name, sv in scip_vars.items():
                    m.setSolVal(sol, sv, float(warm_start.get(name, 0.0)))
                m.addSol(sol, free=True)
                warm_start_used = True
            except Exception:
                warm_start_used = False

        t0 = perf_counter()
        m.optimize()
        elapsed = perf_counter() - t0
        status = m.getStatus()
        node_count = int(m.getNNodes())
        try:
            gap = float(m.getGap())
        except Exception:
            gap = None
        try:
            best_bound = float(m.getDualbound())
        except Exception:
            best_bound = None

        def _vals_for(sol) -> Dict[str, float]:
            return {name: float(m.getSolVal(sol, sv)) for name, sv in scip_vars.items()}

        if status == "optimal":
            obj = float(m.getObjVal())
            return SolverResult("Optimal", obj, _vals_for(m.getBestSol()), elapsed,
                                objective_bound=obj, node_count=node_count, mip_gap=gap,
                                best_bound=best_bound, solver="SCIP",
                                warm_start_used=warm_start_used, incumbent_feasible=True)
        if status == "infeasible":
            return SolverResult("Infeasible", None, {}, elapsed, node_count=node_count,
                                mip_gap=gap, best_bound=best_bound, solver="SCIP",
                                warm_start_used=warm_start_used)
        if status == "unbounded":
            return SolverResult("Unbounded", None, {}, elapsed, node_count=node_count,
                                mip_gap=gap, best_bound=best_bound, solver="SCIP",
                                warm_start_used=warm_start_used)
        if status == "timelimit":
            best_sol = m.getBestSol()
            if best_sol is not None:
                vals = _vals_for(best_sol)
                feasible, _ = UnifiedSolver._check_feasibility(var_specs, constraints_list, vals)
                return SolverResult("Timeout", float(m.getSolObjVal(best_sol)), vals, elapsed,
                                    node_count=node_count, mip_gap=gap, best_bound=best_bound,
                                    solver="SCIP", warm_start_used=warm_start_used,
                                    incumbent_feasible=feasible)
            return SolverResult("Timeout", None, {}, elapsed, node_count=node_count,
                                mip_gap=gap, best_bound=best_bound, solver="SCIP",
                                warm_start_used=warm_start_used)
        return SolverResult("Error", None, {}, elapsed, node_count=node_count, mip_gap=gap,
                            best_bound=best_bound, solver="SCIP",
                            warm_start_used=warm_start_used)

    # ── PuLP/CBC backend (fallback) ────────────────────────────────────
    @staticmethod
    def _solve_cbc(var_specs, objective_coeffs, objective_sense,
                   constraints_list, time_limit) -> SolverResult:
        start = perf_counter()
        try:
            prob = pulp.LpProblem("Model", pulp.LpMaximize
                                  if str(objective_sense).lower() == "maximize"
                                  else pulp.LpMinimize)
            variables = {}
            for name, (kind, low, high) in var_specs.items():
                kind = str(kind).lower()
                if kind in ("binary", "bin"):
                    low, high = max(0.0, low), min(1.0, high)
                if low > high:
                    return SolverResult("Infeasible", None, {}, perf_counter() - start,
                                        message="Contradictory variable bounds")
                cat = pulp.LpContinuous if kind in ("continuous", "cont", "real") else pulp.LpInteger
                # Internal IDs prevent PuLP sanitization from merging distinct names.
                variables[name] = pulp.LpVariable(f"v{len(variables)}", None if isinf(low) else low,
                                                  None if isinf(high) else high, cat)
            prob += pulp.lpSum(c * variables[v] for v, c in objective_coeffs.items())
            for i, (coeffs, sense, rhs) in enumerate(constraints_list):
                expr = pulp.lpSum(c * variables[v] for v, c in coeffs.items())
                prob += (expr <= rhs if sense == "<=" else expr >= rhs if sense == ">="
                         else expr == rhs), f"c{i}"
            prob.solve(pulp.PULP_CBC_CMD(
                msg=False, timeLimit=time_limit, threads=CBC_SETTINGS["threads"],
                gapRel=CBC_SETTINGS["gapRel"], timeMode="elapsed",
                options=[f"{key} {CBC_SETTINGS[key]}" for key in
                         ("randomSeed", "randomCbcSeed", "primalTolerance", "integerTolerance")]))
            elapsed = perf_counter() - start
            status = pulp.LpStatus[prob.status]
            # PuLP may mark a time-limited integer incumbent LpStatusOptimal.
            if status == "Optimal" and prob.sol_status != pulp.LpSolutionOptimal:
                status = "Timeout"
            if status in ("Not Solved", "Undefined"):
                status = "Timeout"
            node_count = int(getattr(prob, "numNodes", 0) or 0)
            try:
                gap = float(getattr(prob, "gap", 0.0) or 0.0)
            except (TypeError, ValueError):
                gap = None
            values = {}
            if status in ("Optimal", "Timeout"):
                for name, var in variables.items():
                    value = var.varValue
                    if value is None:
                        _, low, high = var_specs[name]
                        value = max(low, min(high, 0.0))
                    values[name] = float(value)
                obj = sum(c * values[v] for v, c in objective_coeffs.items())
            else:
                obj = None
            feasible = None
            if status == "Timeout" and values:
                feasible, _ = UnifiedSolver._check_feasibility(var_specs, constraints_list, values)
            best_bound = obj if status == "Optimal" else None
            return SolverResult(status, obj, values if status in ("Optimal", "Timeout") else {},
                                elapsed, objective_bound=best_bound, node_count=node_count,
                                mip_gap=gap, best_bound=best_bound, solver="CBC",
                                incumbent_feasible=feasible)
        except Exception as exc:
            return SolverResult("Error", None, {}, perf_counter() - start, message=str(exc))
