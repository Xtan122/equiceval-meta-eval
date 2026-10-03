"""
Reference IR — Ground-truth Canonical IR cho 4 bài toán benchmark.

Tương ứng với Appendix của bài báo gốc (arXiv:2510.16943, Figures 3–6).
Module 0 dùng các IR này làm reference để so sánh với candidate IR từ LLM.
"""

from typing import Dict, Optional
from src.equiceval.canonical_ir import (
    CanonicalIR,
    CanonicalConstraint,
    CanonicalVariable,
)


def _var(name: str, vtype: str, lb: float = 0.0, ub: Optional[float] = None,
         role: str = "", req_id: str = "", unit: str = "") -> CanonicalVariable:
    if ub is None:
        ub = 1.0 if vtype.lower() in ("binary", "bin") else float("inf")
    return CanonicalVariable(
        name=name, var_type=vtype,
        lower_bound=lb, upper_bound=ub,
        semantic_role=role, requirement_id=req_id,
        unit=unit,
    )


def _leq(name: str, coeffs: Dict[str, float], rhs: float) -> CanonicalConstraint:
    """a·x <= rhs  →  a·x - rhs <= 0  →  constant = -rhs"""
    return CanonicalConstraint(name=name, coeffs=coeffs, constant=-rhs, sense="<=")


def _geq(name: str, coeffs: Dict[str, float], rhs: float) -> CanonicalConstraint:
    """-a·x <= -rhs"""
    return CanonicalConstraint(name=name,
                               coeffs={k: -v for k, v in coeffs.items()},
                               constant=rhs, sense="<=")


def _eq(name: str, coeffs: Dict[str, float], rhs: float) -> CanonicalConstraint:
    return CanonicalConstraint(name=name, coeffs=coeffs, constant=-rhs, sense="==")


# ── 1. Knapsack ───────────────────────────────────────────────────────────────
def make_knapsack_reference() -> CanonicalIR:
    """
    max  60x1 + 100x2 + 120x3
    s.t. 10x1 + 20x2 + 30x3 <= 50
         x1, x2, x3 ∈ {0,1}
    Optimal: x2=1, x3=1  →  obj=220
    """
    return CanonicalIR(
        problem_name="Knapsack",
        variables={
            "x1": _var("x1", "binary", lb=0.0, ub=1.0, role="select_item1", req_id="knap_x1"),
            "x2": _var("x2", "binary", lb=0.0, ub=1.0, role="select_item2", req_id="knap_x2"),
            "x3": _var("x3", "binary", lb=0.0, ub=1.0, role="select_item3", req_id="knap_x3"),
        },
        constraints=[
            _leq("capacity", {"x1": 10.0, "x2": 20.0, "x3": 30.0}, 50.0),
        ],
        objective_sense="maximize",
        objective_coeffs={"x1": 60.0, "x2": 100.0, "x3": 120.0},
        metadata={"optimal_value": 220.0},
    )



# ── 2. Aircraft Assignment ────────────────────────────────────────────────────
def make_aircraft_assignment_reference() -> CanonicalIR:
    """
    min  100x11 + 200x12 + 150x21 + 250x22 + 200x31 + 300x32
    s.t. x11+x12 <= 2         (A1 availability)
         x21+x22 <= 3         (A2 availability)
         x31+x32 <= 1         (A3 availability)
         50x11+60x21+70x31 >= 100  (R1 demand)
         70x12+80x22+90x32 >= 150  (R2 demand)
         x_{ij} ∈ {0,1}
    Variables: x{aircraft}{route}, e.g. x11 = aircraft A1 on route R1.
    req_id encodes both compact (x11) and underscore (x_1_1) LLM naming styles.
    """
    vars_ = {}
    for a, r in [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]:
        compact = f"x{a}{r}"         # reference canonical name: x11, x12 …
        underscore = f"x_{a}_{r}"    # LLM often uses: x_1_1, x_1_2 …
        vars_[compact] = _var(
            compact, "binary",
            lb=0.0, ub=1.0,
            role=f"assign_A{a}_R{r}",
            req_id=f"aa_{compact}",   # primary req_id: aa_x11
        )
        # Store underscore alias as a secondary req_id in metadata so
        # build_projection_certificate can resolve x_1_1 → x11
        vars_[compact].index_sets = [f"A{a}", f"R{r}"]
        # Also make a "shadow" req_id lookup via the CanonicalVariable's
        # requirement_id field: we use compact name for uniqueness,
        # the underscore alias is handled by _normalize_name() in matching.

    return CanonicalIR(
        problem_name="AircraftAssignment",
        variables=vars_,
        constraints=[
            # Availability
            _leq("avail_A1", {"x11": 1.0, "x12": 1.0}, 2.0),
            _leq("avail_A2", {"x21": 1.0, "x22": 1.0}, 3.0),
            _leq("avail_A3", {"x31": 1.0, "x32": 1.0}, 1.0),
            # Demand
            _geq("demand_R1", {"x11": 50.0, "x21": 60.0, "x31": 70.0}, 100.0),
            _geq("demand_R2", {"x12": 70.0, "x22": 80.0, "x32": 90.0}, 150.0),
        ],
        objective_sense="minimize",
        objective_coeffs={
            "x11": 100.0, "x12": 200.0,
            "x21": 150.0, "x22": 250.0,
            "x31": 200.0, "x32": 300.0,
        },
        metadata={
            "optimal_value": 700.0,
            # Map: underscore name → canonical name (used by matching layer)
            "name_aliases": {
                "x_1_1": "x11", "x_1_2": "x12",
                "x_2_1": "x21", "x_2_2": "x22",
                "x_3_1": "x31", "x_3_2": "x32",
            },
        },
    )



# ── 3. Diet Optimization ──────────────────────────────────────────────────────
def make_diet_reference() -> CanonicalIR:
    """
    min  2.0*x_apple + 1.5*x_banana
    s.t. 10*x_apple + 5*x_banana  >= 50   (VitC min)
         10*x_apple + 5*x_banana  <= 100  (VitC max)
         5*x_apple  + 10*x_banana >= 30   (Fiber min)
         5*x_apple  + 10*x_banana <= 60   (Fiber max)
         0 <= x_apple  <= 10
         0 <= x_banana <= 10
    """
    return CanonicalIR(
        problem_name="Diet",
        variables={
            "x_apple":  _var("x_apple",  "continuous", 0.0, 10.0,
                             role="amount_apple",  req_id="diet_apple",  unit="serving"),
            "x_banana": _var("x_banana", "continuous", 0.0, 10.0,
                             role="amount_banana", req_id="diet_banana", unit="serving"),
        },
        constraints=[
            _geq("vitc_min",   {"x_apple": 10.0, "x_banana": 5.0},  50.0),
            _leq("vitc_max",   {"x_apple": 10.0, "x_banana": 5.0}, 100.0),
            _geq("fiber_min",  {"x_apple":  5.0, "x_banana": 10.0}, 30.0),
            _leq("fiber_max",  {"x_apple":  5.0, "x_banana": 10.0}, 60.0),
        ],
        objective_sense="minimize",
        objective_coeffs={"x_apple": 2.0, "x_banana": 1.5},
        metadata={"optimal_value": 10.0},  # Approx: x_apple=2.5, x_banana=5
    )


# ── 4. Aircraft Landing ───────────────────────────────────────────────────────
def make_aircraft_landing_reference() -> CanonicalIR:
    """
    min  5*e1 + 10*l1 + 10*e2 + 20*l2 + 15*e3 + 30*l3
    s.t. x_i = T_i + l_i - e_i  (landing time)
         E_i <= x_i <= L_i       (time windows: A1[1,10], A2[3,12], A3[5,15])
         e_i, l_i >= 0
         x_i - x_j >= sep_{ij} - M*z_{ij}  (separation if i before j)
         x_j - x_i >= sep_{ij} - M*(1-z_{ij})
         z_{ij} ∈ {0,1}   (ordering)
         z_{ij} + z_{ji} = 1         (exactly one order per pair, Eq 24)
    Big-M = 1000, Target: T1=4, T2=8, T3=14
    """
    M = 1000.0
    return CanonicalIR(
        problem_name="AircraftLanding",
        variables={
            # Landing times (continuous, bounded by time windows)
            "x1": _var("x1", "continuous", 1.0, 10.0,  role="land_time_A1", req_id="al_x1", unit="min"),
            "x2": _var("x2", "continuous", 3.0, 12.0,  role="land_time_A2", req_id="al_x2", unit="min"),
            "x3": _var("x3", "continuous", 5.0, 15.0,  role="land_time_A3", req_id="al_x3", unit="min"),
            # Earliness
            "e1": _var("e1", "continuous", 0.0, 20.0, role="early_A1", req_id="al_e1", unit="min"),
            "e2": _var("e2", "continuous", 0.0, 20.0, role="early_A2", req_id="al_e2", unit="min"),
            "e3": _var("e3", "continuous", 0.0, 20.0, role="early_A3", req_id="al_e3", unit="min"),
            # Lateness
            "l1": _var("l1", "continuous", 0.0, 20.0, role="late_A1", req_id="al_l1", unit="min"),
            "l2": _var("l2", "continuous", 0.0, 20.0, role="late_A2", req_id="al_l2", unit="min"),
            "l3": _var("l3", "continuous", 0.0, 20.0, role="late_A3", req_id="al_l3", unit="min"),
            # Ordering binaries — both directions per pair, so LLM outputs that
            # enumerate every ordered pair (z_1_2 … z_3_2) map without gaps.
            "z12": _var("z12", "binary", lb=0.0, ub=1.0, role="order_A1_A2", req_id="al_z12"),
            "z21": _var("z21", "binary", lb=0.0, ub=1.0, role="order_A2_A1", req_id="al_z21"),
            "z13": _var("z13", "binary", lb=0.0, ub=1.0, role="order_A1_A3", req_id="al_z13"),
            "z31": _var("z31", "binary", lb=0.0, ub=1.0, role="order_A3_A1", req_id="al_z31"),
            "z23": _var("z23", "binary", lb=0.0, ub=1.0, role="order_A2_A3", req_id="al_z23"),
            "z32": _var("z32", "binary", lb=0.0, ub=1.0, role="order_A3_A2", req_id="al_z32"),
        },
        constraints=[
            # Landing time definition: xi - li + ei = Ti  → xi - li + ei - Ti = 0
            _eq("land_def_1", {"x1": 1.0, "l1": -1.0, "e1": 1.0}, 4.0),
            _eq("land_def_2", {"x2": 1.0, "l2": -1.0, "e2": 1.0}, 8.0),
            _eq("land_def_3", {"x3": 1.0, "l3": -1.0, "e3": 1.0}, 14.0),
            # Separation A1-A2 (sep=2): forward uses z12, backward uses z21
            _geq("sep12_fwd", {"x1": 1.0, "x2": -1.0, "z12": -M}, -(M - 2)),
            _geq("sep12_bwd", {"x2": 1.0, "x1": -1.0, "z21":  M},  2.0),
            # Separation A1-A3 (sep=3)
            _geq("sep13_fwd", {"x1": 1.0, "x3": -1.0, "z13": -M}, -(M - 3)),
            _geq("sep13_bwd", {"x3": 1.0, "x1": -1.0, "z31":  M},  3.0),
            # Separation A2-A3 (sep=4)
            _geq("sep23_fwd", {"x2": 1.0, "x3": -1.0, "z23": -M}, -(M - 4)),
            _geq("sep23_bwd", {"x3": 1.0, "x2": -1.0, "z32":  M},  4.0),
            # Ordering logic link (paper Eq 24): exactly one direction per pair.
            _eq("order_link_12", {"z12": 1.0, "z21": 1.0}, 1.0),
            _eq("order_link_13", {"z13": 1.0, "z31": 1.0}, 1.0),
            _eq("order_link_23", {"z23": 1.0, "z32": 1.0}, 1.0),
        ],
        objective_sense="minimize",
        objective_coeffs={
            "e1": 5.0, "l1": 10.0,
            "e2": 10.0, "l2": 20.0,
            "e3": 15.0, "l3": 30.0,
        },
        metadata={
            # LLM often uses underscore variants: x_1→x1, e_1→e1, z_1_2→z12
            "name_aliases": {
                "x_1": "x1", "x_2": "x2", "x_3": "x3",
                "e_1": "e1", "e_2": "e2", "e_3": "e3",
                "l_1": "l1", "l_2": "l2", "l_3": "l3",
                "z_12": "z12", "z_1_2": "z12", "z_2_1": "z21",
                "z_13": "z13", "z_1_3": "z13", "z_3_1": "z31",
                "z_23": "z23", "z_2_3": "z23", "z_3_2": "z32",
            },
            # Ordering binaries are linked by z_ij + z_ji = 1 (paper Eq 24), so
            # the reference no longer permits the ambiguous independent state.
            "family_label": "AircraftLanding",
            "difficulty": "Hard",
            "opt_type": "MILP",
        },
    )



# ── Registry ──────────────────────────────────────────────────────────────────
REFERENCE_IR_REGISTRY = {
    "Knapsack":           make_knapsack_reference,
    "AircraftAssignment": make_aircraft_assignment_reference,
    "Diet":               make_diet_reference,
    "AircraftLanding":    make_aircraft_landing_reference,
}


def get_reference_ir(problem_name: str) -> CanonicalIR:
    """Trả về Reference IR cho bài toán đã cho."""
    factory = REFERENCE_IR_REGISTRY.get(problem_name)
    if factory is None:
        raise ValueError(
            f"Unknown problem '{problem_name}'. "
            f"Available: {sorted(REFERENCE_IR_REGISTRY)}"
        )
    return factory()
