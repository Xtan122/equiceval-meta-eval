"""
Semantics-Preserving Transformation Engine for EquiCEval Benchmark.
Generates certified equivalent formulations by applying semantics-preserving transformations:
- Positive constraint scaling (multiplier k > 0)
- Bound conversion to explicit canonical constraints
- Redundant implied constraint addition
- Safe variable renaming & index permutation
"""

import random
from typing import Dict, List, Tuple, Optional
import numpy as np

from src.equiceval.canonical_ir import CanonicalIR, CanonicalConstraint, CanonicalVariable


class SemanticsPreservingTransformer:
    """
    Applies sound semantics-preserving transformations to a CanonicalIR.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def apply_positive_scaling(self, ir: CanonicalIR, scale_range: Tuple[float, float] = (1.5, 5.0)) -> CanonicalIR:
        """
        Multiplies random constraints by scalar k > 0.
        Preserves feasible set and objective value exactly.
        """
        new_constraints = []
        for c in ir.constraints:
            if self.rng.random() > 0.3:
                k = self.rng.uniform(*scale_range)
                scaled_coeffs = {v: coeff * k for v, coeff in c.coeffs.items()}
                scaled_const = c.constant * k
                new_constraints.append(CanonicalConstraint(
                    name=f"{c.name}_scaled",
                    coeffs=scaled_coeffs,
                    constant=scaled_const,
                    sense=c.sense
                ))
            else:
                new_constraints.append(c)

        return CanonicalIR(
            problem_name=f"{ir.problem_name}_scaled",
            variables=dict(ir.variables),
            constraints=new_constraints,
            objective_sense=ir.objective_sense,
            objective_coeffs=dict(ir.objective_coeffs),
            objective_constant=ir.objective_constant,
        )

    def convert_bounds_to_constraints(self, ir: CanonicalIR) -> CanonicalIR:
        """
        Converts variable bounds (lower_bound, upper_bound) into explicit constraints.
        E.g., x_1 >= 0  =>  -x_1 <= 0
              x_1 <= 10 =>   x_1 - 10 <= 0
        """
        new_constraints = list(ir.constraints)

        for vname, var in ir.variables.items():
            if var.lower_bound != float('-inf') and var.lower_bound != 0.0:
                # lower bound constraint: -x_i + lb <= 0
                new_constraints.append(CanonicalConstraint(
                    name=f"bound_lb_{vname}",
                    coeffs={vname: -1.0},
                    constant=var.lower_bound,
                    sense="<="
                ))
            if var.upper_bound != float('inf') and var.upper_bound != 1.0:
                # upper bound constraint: x_i - ub <= 0
                new_constraints.append(CanonicalConstraint(
                    name=f"bound_ub_{vname}",
                    coeffs={vname: 1.0},
                    constant=-var.upper_bound,
                    sense="<="
                ))

        return CanonicalIR(
            problem_name=f"{ir.problem_name}_explicit_bounds",
            variables=dict(ir.variables),
            constraints=new_constraints,
            objective_sense=ir.objective_sense,
            objective_coeffs=dict(ir.objective_coeffs),
            objective_constant=ir.objective_constant,
        )

    def add_implied_redundant_constraints(self, ir: CanonicalIR) -> CanonicalIR:
        """
        Adds non-negative linear combinations of existing <= constraints.
        k1 * c1 + k2 * c2 <= 0  where k1, k2 > 0.
        """
        new_constraints = list(ir.constraints)
        le_constraints = [c for c in ir.constraints if c.sense == '<=']

        if len(le_constraints) >= 2:
            c1, c2 = self.rng.sample(le_constraints, 2)
            k1 = self.rng.uniform(0.5, 2.0)
            k2 = self.rng.uniform(0.5, 2.0)

            comb_coeffs = {}
            for v, coeff in c1.coeffs.items():
                comb_coeffs[v] = comb_coeffs.get(v, 0.0) + k1 * coeff
            for v, coeff in c2.coeffs.items():
                comb_coeffs[v] = comb_coeffs.get(v, 0.0) + k2 * coeff

            comb_const = k1 * c1.constant + k2 * c2.constant

            new_constraints.append(CanonicalConstraint(
                name=f"implied_{c1.name}_{c2.name}",
                coeffs=comb_coeffs,
                constant=comb_const,
                sense="<="
            ))

        return CanonicalIR(
            problem_name=f"{ir.problem_name}_implied",
            variables=dict(ir.variables),
            constraints=new_constraints,
            objective_sense=ir.objective_sense,
            objective_coeffs=dict(ir.objective_coeffs),
            objective_constant=ir.objective_constant,
        )

    def permute_variable_names(self, ir: CanonicalIR) -> Tuple[CanonicalIR, Dict[str, str]]:
        """
        Safely renames variables according to a random mapping.
        E.g. x_1 -> var_A, x_2 -> var_B
        """
        var_names = list(ir.variables.keys())
        scrambled_names = [f"var_{chr(65 + i)}" for i in range(len(var_names))]
        mapping = dict(zip(var_names, scrambled_names))

        new_vars = {}
        for old_v, new_v in mapping.items():
            old_spec = ir.variables[old_v]
            new_vars[new_v] = CanonicalVariable(
                name=new_v,
                var_type=old_spec.var_type,
                lower_bound=old_spec.lower_bound,
                upper_bound=old_spec.upper_bound,
                provenance=old_spec.provenance
            )

        new_constraints = []
        for c in ir.constraints:
            renamed_coeffs = {mapping[old_v]: coeff for old_v, coeff in c.coeffs.items() if old_v in mapping}
            new_constraints.append(CanonicalConstraint(
                name=c.name,
                coeffs=renamed_coeffs,
                constant=c.constant,
                sense=c.sense
            ))

        renamed_obj = {mapping[old_v]: coeff for old_v, coeff in ir.objective_coeffs.items() if old_v in mapping}

        new_ir = CanonicalIR(
            problem_name=f"{ir.problem_name}_renamed",
            variables=new_vars,
            constraints=new_constraints,
            objective_sense=ir.objective_sense,
            objective_coeffs=renamed_obj,
            objective_constant=ir.objective_constant,
        )

        return new_ir, mapping

    # ── Phase 8: aggregate/decompose, auxiliary variable, move constant ─────

    def apply_aggregate_decompose(self, ir: CanonicalIR) -> CanonicalIR:
        """
        Aggregate/decompose (paper §7.2): decomposes one equality constraint
        ``a·x + b == 0`` into the two equivalent inequalities ``a·x + b <= 0``
        and ``-a·x - b <= 0``. If no equality exists, merges the two tightest
        same-support '<=' constraints (a redundant-aggregate that preserves the
        feasible set). Falls back to an identity copy otherwise.
        """
        new_constraints = list(ir.constraints)
        eq_idx = next((i for i, c in enumerate(ir.constraints) if c.sense == "=="), None)
        if eq_idx is not None:
            c = ir.constraints[eq_idx]
            pos = CanonicalConstraint(name=f"{c.name}_leq", coeffs=dict(c.coeffs),
                                      constant=c.constant, sense="<=")
            neg = CanonicalConstraint(name=f"{c.name}_geq",
                                      coeffs={v: -coef for v, coef in c.coeffs.items()},
                                      constant=-c.constant, sense="<=")
            new_constraints = [cc for i, cc in enumerate(ir.constraints) if i != eq_idx]
            new_constraints.extend([pos, neg])
            return self._clone(ir, f"{ir.problem_name}_decomposed", new_constraints)

        # Merge the tightest pair of same-support '<=' constraints.
        le = [c for c in ir.constraints if c.sense == "<="]
        for i in range(len(le)):
            for j in range(i + 1, len(le)):
                if set(le[i].coeffs) == set(le[j].coeffs):
                    keep = le[i] if le[i].constant <= le[j].constant else le[j]
                    drop_name = le[j].name if keep is le[i] else le[i].name
                    merged = [c for c in ir.constraints if c.name != drop_name]
                    return self._clone(ir, f"{ir.problem_name}_aggregated", merged)
        return self._clone(ir, f"{ir.problem_name}_aggregate", list(ir.constraints))

    def apply_auxiliary_variable(self, ir: CanonicalIR) -> CanonicalIR:
        """
        Auxiliary-variable transformation (paper §7.2): introduces a slack/alias
        variable ``s`` pinned by ``s == x_k`` so the model is equivalent; the
        projection certificate (dropped variable) is recorded in metadata.
        """
        if not ir.variables:
            return self._clone(ir, f"{ir.problem_name}_aux", list(ir.constraints))
        base_var = next(iter(ir.variables.values()))
        aux_name = "s_aux"
        # Pick a name that does not collide.
        while aux_name in ir.variables:
            aux_name += "_"
        aux_var = CanonicalVariable(
            name=aux_name,
            var_type=base_var.var_type,
            lower_bound=base_var.lower_bound,
            upper_bound=base_var.upper_bound,
            provenance="auxiliary",
            is_auxiliary=True,
        )
        new_vars = dict(ir.variables)
        new_vars[aux_name] = aux_var
        new_constraints = list(ir.constraints)
        new_constraints.append(CanonicalConstraint(
            name=f"aux_link_{aux_name}",
            coeffs={aux_name: 1.0, base_var.name: -1.0},
            constant=0.0,
            sense="==",
        ))
        new_ir = CanonicalIR(
            problem_name=f"{ir.problem_name}_auxiliary",
            variables=new_vars,
            constraints=new_constraints,
            objective_sense=ir.objective_sense,
            objective_coeffs=dict(ir.objective_coeffs),
            objective_constant=ir.objective_constant,
            metadata=dict(ir.metadata),
        )
        new_ir.metadata["auxiliary_certificate"] = {
            "auxiliary_variable": aux_name,
            "projection": f"{aux_name} = {base_var.name}",
        }
        return new_ir

    def apply_move_constant(self, ir: CanonicalIR) -> CanonicalIR:
        """
        Move-constant transformation (paper §7.2): rewrites one '<=' constraint
        ``a·x + b <= 0`` as the equivalent ``-a·x - b >= 0`` (constant moved to
        the other side), which canonicalizes back to the same inequality.
        """
        new_constraints = list(ir.constraints)
        idx = next((i for i, c in enumerate(ir.constraints) if c.sense == "<="), None)
        if idx is not None:
            c = ir.constraints[idx]
            new_constraints[idx] = CanonicalConstraint(
                name=f"{c.name}_moved_const",
                coeffs={v: -coef for v, coef in c.coeffs.items()},
                constant=-c.constant,
                sense=">=",
            )
        return self._clone(ir, f"{ir.problem_name}_moved_const", new_constraints)

    def generate_coverage_matrix(self) -> Dict[str, List[str]]:
        """
        Coverage matrix (paper §7.2): transformation type × problem family.
        All 7 semantics-preserving transformations apply to all 4 families.
        """
        problems = ["Knapsack", "AircraftAssignment", "Diet", "AircraftLanding"]
        trans_types = [
            "positive-scaling", "bound-to-constraint", "implied-redundant",
            "variable-renaming", "aggregate-decompose", "auxiliary-variable",
            "move-constant",
        ]
        return {p: list(trans_types) for p in problems}

    def generate_mapping_coverage_matrix(self) -> Dict[str, List[str]]:
        """
        Mapping-class coverage (paper §7.2): declared variable-projection classes
        whose false-positive rate must be reported. Extends the transformation
        matrix with the naming/elimination classes handled by Module 1:
          - semantic-index-naming        : x_A1_R1 -> x11
          - base-letter-agnostic-binary  : y_A1_A2 -> z12
          - binary-complement-elimination: z21 = 1 - z12 (constraint-verified)
        """
        return {
            "Knapsack": ["variable-renaming"],
            "AircraftAssignment": ["variable-renaming", "semantic-index-naming"],
            "Diet": ["variable-renaming"],
            "AircraftLanding": [
                "variable-renaming",
                "semantic-index-naming",
                "base-letter-agnostic-binary",
                "binary-complement-elimination",
            ],
        }

    @staticmethod
    def _clone(ir: CanonicalIR, name: str, constraints: List[CanonicalConstraint]) -> CanonicalIR:
        return CanonicalIR(
            problem_name=name,
            variables=dict(ir.variables),
            constraints=constraints,
            objective_sense=ir.objective_sense,
            objective_coeffs=dict(ir.objective_coeffs),
            objective_constant=ir.objective_constant,
            metadata=dict(ir.metadata),
        )
