"""
Ground truth benchmark problem specifications from Refai & Ahmed (arXiv:2510.16943).
Covers the 4 benchmark problems from the ComplexOR dataset:
1. Knapsack Problem (Easy - MILP)
2. Aircraft Assignment Problem (Medium - MILP)
3. Diet Optimization Problem (Medium - LP)
4. Aircraft Landing Problem (Hard - MILP)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
import numpy as np


@dataclass
class VariableSpec:
    name: str
    var_type: str  # 'binary', 'integer', 'continuous'
    lower_bound: float = 0.0
    upper_bound: float = float('inf')
    description: str = ""


@dataclass
class ConstraintSpec:
    name: str
    expression_func: Any  # Function x_dict -> float representing LHS - RHS (<= 0)
    canonical_repr: str
    category: str = "general"  # e.g., 'capacity', 'demand', 'bounds', 'separation'
    is_essential: bool = True  # True if missing this constraint breaks feasibility/optimality


@dataclass
class GroundTruthProblem:
    problem_name: str
    difficulty: str  # 'Easy', 'Medium', 'Hard'
    opt_type: str    # 'ILP', 'MILP', 'LP'
    variables: Dict[str, VariableSpec]
    constraints: List[ConstraintSpec]
    objective_sense: str  # 'maximize' or 'minimize'
    objective_func: Any   # Function x_dict -> float
    optimal_value: float
    sample_domain: Dict[str, Tuple[float, float]]  # Domain for sampling 100 evaluation points


# -----------------------------------------------------------------------------
# 1. Knapsack Problem (Easy - MILP)
# -----------------------------------------------------------------------------
def _knapsack_obj(x: Dict[str, float]) -> float:
    # Maximize sum(x_i * ItemValue_i)
    values = [60.0, 100.0, 120.0]
    return sum(x[f"x_{i+1}"] * values[i] for i in range(3))


def _knapsack_capacity(x: Dict[str, float]) -> float:
    # 10 x_1 + 20 x_2 + 30 x_3 <= 50  =>  LHS - 50 <= 0
    weights = [10.0, 20.0, 30.0]
    return sum(x[f"x_{i+1}"] * weights[i] for i in range(3)) - 50.0


KNAPSACK_PROBLEM = GroundTruthProblem(
    problem_name="Knapsack",
    difficulty="Easy",
    opt_type="ILP",
    variables={
        "x_1": VariableSpec("x_1", "binary", 0.0, 1.0, "Item 1 selected"),
        "x_2": VariableSpec("x_2", "binary", 0.0, 1.0, "Item 2 selected"),
        "x_3": VariableSpec("x_3", "binary", 0.0, 1.0, "Item 3 selected"),
    },
    constraints=[
        ConstraintSpec(
            name="capacity_constraint",
            expression_func=_knapsack_capacity,
            canonical_repr="10*x_1 + 20*x_2 + 30*x_3 <= 50",
            category="capacity",
            is_essential=True,
        )
    ],
    objective_sense="maximize",
    objective_func=_knapsack_obj,
    optimal_value=220.0,
    sample_domain={"x_1": (0.0, 1.0), "x_2": (0.0, 1.0), "x_3": (0.0, 1.0)},
)


# -----------------------------------------------------------------------------
# 2. Aircraft Assignment Problem (Medium - MILP)
# -----------------------------------------------------------------------------
def _aircraft_assignment_obj(x: Dict[str, float]) -> float:
    # Minimize Costs: A1=[100,200], A2=[150,250], A3=[200,300]
    costs = {
        ("1", "1"): 100.0, ("1", "2"): 200.0,
        ("2", "1"): 150.0, ("2", "2"): 250.0,
        ("3", "1"): 200.0, ("3", "2"): 300.0,
    }
    return sum(x[f"x_{a}_{r}"] * c for (a, r), c in costs.items())


def _aircraft_avail(a: str, max_avail: float):
    def func(x: Dict[str, float]) -> float:
        return (x[f"x_{a}_1"] + x[f"x_{a}_2"]) - max_avail
    return func


def _aircraft_demand(r: str, min_demand: float, cap_dict: Dict[str, float]):
    def func(x: Dict[str, float]) -> float:
        # sum_a x_{a,r} * cap_{a,r} >= demand => demand - sum(...) <= 0
        supplied = sum(x[f"x_{a}_{r}"] * cap_dict[a] for a in ["1", "2", "3"])
        return min_demand - supplied
    return func


AIRCRAFT_ASSIGNMENT_PROBLEM = GroundTruthProblem(
    problem_name="AircraftAssignment",
    difficulty="Medium",
    opt_type="MILP",
    variables={
        f"x_{a}_{r}": VariableSpec(f"x_{a}_{r}", "binary", 0.0, 1.0, f"Assign aircraft {a} to route {r}")
        for a in ["1", "2", "3"] for r in ["1", "2"]
    },
    constraints=[
        # Availability constraints
        ConstraintSpec("avail_A1", _aircraft_avail("1", 2.0), "x_1_1 + x_1_2 <= 2", "availability", True),
        ConstraintSpec("avail_A2", _aircraft_avail("2", 3.0), "x_2_1 + x_2_2 <= 3", "availability", True),
        ConstraintSpec("avail_A3", _aircraft_avail("3", 1.0), "x_3_1 + x_3_2 <= 1", "availability", True),
        # Demand constraints
        ConstraintSpec("demand_R1", _aircraft_demand("1", 100.0, {"1": 50.0, "2": 60.0, "3": 70.0}),
                       "50*x_1_1 + 60*x_2_1 + 70*x_3_1 >= 100", "demand", True),
        ConstraintSpec("demand_R2", _aircraft_demand("2", 150.0, {"1": 70.0, "2": 80.0, "3": 90.0}),
                       "70*x_1_2 + 80*x_2_2 + 90*x_3_2 >= 150", "demand", True),
    ],
    objective_sense="minimize",
    objective_func=_aircraft_assignment_obj,
    optimal_value=700.0,
    sample_domain={f"x_{a}_{r}": (0.0, 1.0) for a in ["1", "2", "3"] for r in ["1", "2"]},
)


# -----------------------------------------------------------------------------
# 3. Diet Optimization Problem (Medium - LP)
# -----------------------------------------------------------------------------
def _diet_obj(x: Dict[str, float]) -> float:
    # Cost: Apple=2.0, Banana=1.5
    return 2.0 * x["x_apple"] + 1.5 * x["x_banana"]


DIET_PROBLEM = GroundTruthProblem(
    problem_name="Diet",
    difficulty="Medium",
    opt_type="LP",
    variables={
        "x_apple": VariableSpec("x_apple", "continuous", 0.0, 10.0, "Quantity of Apple"),
        "x_banana": VariableSpec("x_banana", "continuous", 0.0, 10.0, "Quantity of Banana"),
    },
    constraints=[
        # Nutrient constraints: Min required
        ConstraintSpec("min_vit_c", lambda x: 50.0 - (10.0 * x["x_apple"] + 5.0 * x["x_banana"]),
                       "10*x_apple + 5*x_banana >= 50", "nutrient_min", True),
        ConstraintSpec("min_fiber", lambda x: 30.0 - (5.0 * x["x_apple"] + 10.0 * x["x_banana"]),
                       "5*x_apple + 10*x_banana >= 30", "nutrient_min", True),
        # Nutrient constraints: Max allowed
        ConstraintSpec("max_vit_c", lambda x: (10.0 * x["x_apple"] + 5.0 * x["x_banana"]) - 100.0,
                       "10*x_apple + 5*x_banana <= 100", "nutrient_max", True),
        ConstraintSpec("max_fiber", lambda x: (5.0 * x["x_apple"] + 10.0 * x["x_banana"]) - 60.0,
                       "5*x_apple + 10*x_banana <= 60", "nutrient_max", True),
        # Food bounds
        ConstraintSpec("max_apple", lambda x: x["x_apple"] - 10.0, "x_apple <= 10", "food_bound", False),
        ConstraintSpec("max_banana", lambda x: x["x_banana"] - 10.0, "x_banana <= 10", "food_bound", False),
        ConstraintSpec("min_apple", lambda x: 0.0 - x["x_apple"], "x_apple >= 0", "food_bound", False),
        ConstraintSpec("min_banana", lambda x: 0.0 - x["x_banana"], "x_banana >= 0", "food_bound", False),
    ],
    objective_sense="minimize",
    objective_func=_diet_obj,
    optimal_value=10.333333333333334,
    sample_domain={"x_apple": (0.0, 10.0), "x_banana": (0.0, 10.0)},
)


# -----------------------------------------------------------------------------
# 4. Aircraft Landing Problem (Hard - MILP)
# -----------------------------------------------------------------------------
def _alp_obj(x: Dict[str, float]) -> float:
    # P_early = [5, 10, 15], P_late = [10, 20, 30]
    return (5.0 * x["e_1"] + 10.0 * x["l_1"] +
            10.0 * x["e_2"] + 20.0 * x["l_2"] +
            15.0 * x["e_3"] + 30.0 * x["l_3"])


ALP_PROBLEM = GroundTruthProblem(
    problem_name="AircraftLanding",
    difficulty="Hard",
    opt_type="MILP",
    variables={
        "x_1": VariableSpec("x_1", "continuous", 1.0, 10.0, "Landing time A1"),
        "x_2": VariableSpec("x_2", "continuous", 3.0, 12.0, "Landing time A2"),
        "x_3": VariableSpec("x_3", "continuous", 5.0, 15.0, "Landing time A3"),
        "e_1": VariableSpec("e_1", "continuous", 0.0, 20.0, "Earliness A1"),
        "e_2": VariableSpec("e_2", "continuous", 0.0, 20.0, "Earliness A2"),
        "e_3": VariableSpec("e_3", "continuous", 0.0, 20.0, "Earliness A3"),
        "l_1": VariableSpec("l_1", "continuous", 0.0, 20.0, "Lateness A1"),
        "l_2": VariableSpec("l_2", "continuous", 0.0, 20.0, "Lateness A2"),
        "l_3": VariableSpec("l_3", "continuous", 0.0, 20.0, "Lateness A3"),
        "z_1_2": VariableSpec("z_1_2", "binary", 0.0, 1.0, "A1 lands before A2"),
        "z_2_1": VariableSpec("z_2_1", "binary", 0.0, 1.0, "A2 lands before A1"),
        "z_1_3": VariableSpec("z_1_3", "binary", 0.0, 1.0, "A1 lands before A3"),
        "z_3_1": VariableSpec("z_3_1", "binary", 0.0, 1.0, "A3 lands before A1"),
        "z_2_3": VariableSpec("z_2_3", "binary", 0.0, 1.0, "A2 lands before A3"),
        "z_3_2": VariableSpec("z_3_2", "binary", 0.0, 1.0, "A3 lands before A2"),
    },
    constraints=[
        # Time Windows
        ConstraintSpec("tw_min_A1", lambda x: 1.0 - x["x_1"], "x_1 >= 1", "time_window", True),
        ConstraintSpec("tw_max_A1", lambda x: x["x_1"] - 10.0, "x_1 <= 10", "time_window", True),
        ConstraintSpec("tw_min_A2", lambda x: 3.0 - x["x_2"], "x_2 >= 3", "time_window", True),
        ConstraintSpec("tw_max_A2", lambda x: x["x_2"] - 12.0, "x_2 <= 12", "time_window", True),
        ConstraintSpec("tw_min_A3", lambda x: 5.0 - x["x_3"], "x_3 >= 5", "time_window", True),
        ConstraintSpec("tw_max_A3", lambda x: x["x_3"] - 15.0, "x_3 <= 15", "time_window", True),

        # Target definitions: e_i >= T_i - x_i, l_i >= x_i - T_i (T = [4, 8, 14])
        ConstraintSpec("earliness_A1", lambda x: (4.0 - x["x_1"]) - x["e_1"], "e_1 >= 4 - x_1", "earliness", True),
        ConstraintSpec("lateness_A1", lambda x: (x["x_1"] - 4.0) - x["l_1"], "l_1 >= x_1 - 4", "lateness", True),
        ConstraintSpec("earliness_A2", lambda x: (8.0 - x["x_2"]) - x["e_2"], "e_2 >= 8 - x_2", "earliness", True),
        ConstraintSpec("lateness_A2", lambda x: (x["x_2"] - 8.0) - x["l_2"], "l_2 >= x_2 - 8", "lateness", True),
        ConstraintSpec("earliness_A3", lambda x: (14.0 - x["x_3"]) - x["e_3"], "e_3 >= 14 - x_3", "earliness", True),
        ConstraintSpec("lateness_A3", lambda x: (x["x_3"] - 14.0) - x["l_3"], "l_3 >= x_3 - 14", "lateness", True),

        # Order Completeness: z_ij + z_ji = 1
        ConstraintSpec("order_1_2", lambda x: abs((x["z_1_2"] + x["z_2_1"]) - 1.0), "z_1_2 + z_2_1 == 1", "order", True),
        ConstraintSpec("order_1_3", lambda x: abs((x["z_1_3"] + x["z_3_1"]) - 1.0), "z_1_3 + z_3_1 == 1", "order", True),
        ConstraintSpec("order_2_3", lambda x: abs((x["z_2_3"] + x["z_3_2"]) - 1.0), "z_2_3 + z_3_2 == 1", "order", True),

        # Separation Constraints: x_j >= x_i + S_ij - M(1 - z_ij)  (M = 1000)
        ConstraintSpec("sep_1_2", lambda x: (x["x_1"] + 2.0 - 1000.0 * (1.0 - x["z_1_2"])) - x["x_2"],
                       "x_2 >= x_1 + 2 - M(1 - z_1_2)", "separation", True),
        ConstraintSpec("sep_2_1", lambda x: (x["x_2"] + 2.0 - 1000.0 * (1.0 - x["z_2_1"])) - x["x_1"],
                       "x_1 >= x_2 + 2 - M(1 - z_2_1)", "separation", True),
        ConstraintSpec("sep_1_3", lambda x: (x["x_1"] + 3.0 - 1000.0 * (1.0 - x["z_1_3"])) - x["x_3"],
                       "x_3 >= x_1 + 3 - M(1 - z_1_3)", "separation", True),
        ConstraintSpec("sep_3_1", lambda x: (x["x_3"] + 3.0 - 1000.0 * (1.0 - x["z_3_1"])) - x["x_1"],
                       "x_1 >= x_3 + 3 - M(1 - z_3_1)", "separation", True),
        ConstraintSpec("sep_2_3", lambda x: (x["x_2"] + 4.0 - 1000.0 * (1.0 - x["z_2_3"])) - x["x_3"],
                       "x_3 >= x_2 + 4 - M(1 - z_2_3)", "separation", True),
        ConstraintSpec("sep_3_2", lambda x: (x["x_3"] + 4.0 - 1000.0 * (1.0 - x["z_3_2"])) - x["x_2"],
                       "x_2 >= x_3 + 4 - M(1 - z_3_2)", "separation", True),
    ],
    objective_sense="minimize",
    objective_func=_alp_obj,
    optimal_value=0.0,
    sample_domain={
        "x_1": (1.0, 10.0), "x_2": (3.0, 12.0), "x_3": (5.0, 15.0),
        "e_1": (0.0, 5.0), "e_2": (0.0, 5.0), "e_3": (0.0, 5.0),
        "l_1": (0.0, 5.0), "l_2": (0.0, 5.0), "l_3": (0.0, 5.0),
        "z_1_2": (0.0, 1.0), "z_2_1": (0.0, 1.0),
        "z_1_3": (0.0, 1.0), "z_3_1": (0.0, 1.0),
        "z_2_3": (0.0, 1.0), "z_3_2": (0.0, 1.0),
    },
)

PROBLEMS_REGISTRY: Dict[str, GroundTruthProblem] = {
    "Knapsack": KNAPSACK_PROBLEM,
    "AircraftAssignment": AIRCRAFT_ASSIGNMENT_PROBLEM,
    "Diet": DIET_PROBLEM,
    "AircraftLanding": ALP_PROBLEM,
}
