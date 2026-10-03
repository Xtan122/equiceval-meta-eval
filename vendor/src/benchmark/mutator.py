"""
Controlled Mutation Engine for EquiCEval Benchmark.
Generates solver-certified defective formulations based on OptArgus taxonomy:
- Omission errors (OMIT_CONSTRAINT)
- Sense reversal errors (REVERSE_SENSE)
- Coefficient / scale errors (COEFFICIENT_ERROR)
- Over-constraining extra constraints (OVER_CONSTRAINING_EXTRA)
- Multi-error compositions
"""

from enum import Enum
from dataclasses import dataclass, field
import random
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from src.equiceval.canonical_ir import CanonicalIR, CanonicalConstraint, CanonicalVariable
from src.utils.solver_wrapper import UnifiedSolver


class MutationType(Enum):
    OMIT_CONSTRAINT = "omit_constraint"
    REVERSE_SENSE = "reverse_sense"
    COEFFICIENT_ERROR = "coefficient_error"
    OVER_CONSTRAINING_EXTRA = "over_constraining_extra"
    MULTI_ERROR = "multi_error"
    CONSTANT_ERROR = "constant_error"
    WRONG_DOMAIN = "wrong_domain"
    MISSING_INDEX_RANGE = "missing_index_range"
    BROKEN_BIG_M = "broken_big_m"


@dataclass
class MutationSurvivalReport:
    """Survival statistics across a batch of generated mutants (paper §7.3)."""
    n_generated: int = 0
    n_survived: int = 0
    n_excluded: int = 0
    exclusion_reasons: Dict[str, int] = field(default_factory=dict)

    @property
    def survival_rate(self) -> float:
        return self.n_survived / self.n_generated if self.n_generated else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_generated": self.n_generated,
            "n_survived": self.n_survived,
            "n_excluded": self.n_excluded,
            "survival_rate": self.survival_rate,
            "exclusion_reasons": dict(self.exclusion_reasons),
        }


def generate_survival_report(results: List[Tuple[Any, bool, Optional[str]]]) -> MutationSurvivalReport:
    """
    Builds a MutationSurvivalReport from a list of (mutant, survived, reason).
    reason is None for survivors; a code (e.g. 'infeasible', 'equivalent') for exclusions.
    """
    report = MutationSurvivalReport(n_generated=len(results))
    for _mutant, survived, reason in results:
        if survived:
            report.n_survived += 1
        else:
            report.n_excluded += 1
            code = reason or "excluded"
            report.exclusion_reasons[code] = report.exclusion_reasons.get(code, 0) + 1
    return report


class ControlledMutator:
    """
    Applies controlled mutation operators to generate defective candidate formulations.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def mutate_omit_constraint(self, ir: CanonicalIR) -> Tuple[CanonicalIR, str]:
        """
        Omits one essential constraint.
        """
        if not ir.constraints:
            return ir, "none"

        omit_idx = self.rng.randrange(len(ir.constraints))
        omitted_name = ir.constraints[omit_idx].name
        new_constraints = [c for i, c in enumerate(ir.constraints) if i != omit_idx]

        mutated_ir = CanonicalIR(
            problem_name=f"{ir.problem_name}_omit_{omitted_name}",
            variables=dict(ir.variables),
            constraints=new_constraints,
            objective_sense=ir.objective_sense,
            objective_coeffs=dict(ir.objective_coeffs),
            objective_constant=ir.objective_constant,
        )
        return mutated_ir, f"omitted_{omitted_name}"

    def mutate_reverse_sense(self, ir: CanonicalIR) -> Tuple[CanonicalIR, str]:
        """
        Flips constraint sense (<= to >= or vice versa).
        """
        if not ir.constraints:
            return ir, "none"

        flip_idx = self.rng.randrange(len(ir.constraints))
        target_c = ir.constraints[flip_idx]

        # Flip coeffs and constant to reverse <=
        flipped_coeffs = {v: -coeff for v, coeff in target_c.coeffs.items()}
        flipped_const = -target_c.constant

        new_constraints = list(ir.constraints)
        new_constraints[flip_idx] = CanonicalConstraint(
            name=f"{target_c.name}_sense_reversed",
            coeffs=flipped_coeffs,
            constant=flipped_const,
            sense=target_c.sense
        )

        mutated_ir = CanonicalIR(
            problem_name=f"{ir.problem_name}_reversed_{target_c.name}",
            variables=dict(ir.variables),
            constraints=new_constraints,
            objective_sense=ir.objective_sense,
            objective_coeffs=dict(ir.objective_coeffs),
            objective_constant=ir.objective_constant,
        )
        return mutated_ir, f"reversed_{target_c.name}"

    def mutate_coefficient_error(self, ir: CanonicalIR) -> Tuple[CanonicalIR, str]:
        """
        Alters a single coefficient by bad factor (e.g. 0.5x, 2.0x, -1x).
        """
        if not ir.constraints:
            return ir, "none"

        target_idx = self.rng.randrange(len(ir.constraints))
        target_c = ir.constraints[target_idx]

        if not target_c.coeffs:
            return ir, "none"

        v_target = self.rng.choice(list(target_c.coeffs.keys()))
        bad_factor = self.rng.choice([0.2, 0.5, 2.5, -1.0])

        new_coeffs = dict(target_c.coeffs)
        new_coeffs[v_target] = new_coeffs[v_target] * bad_factor

        new_constraints = list(ir.constraints)
        new_constraints[target_idx] = CanonicalConstraint(
            name=f"{target_c.name}_bad_coeff",
            coeffs=new_coeffs,
            constant=target_c.constant,
            sense=target_c.sense
        )

        mutated_ir = CanonicalIR(
            problem_name=f"{ir.problem_name}_bad_coeff_{target_c.name}",
            variables=dict(ir.variables),
            constraints=new_constraints,
            objective_sense=ir.objective_sense,
            objective_coeffs=dict(ir.objective_coeffs),
            objective_constant=ir.objective_constant,
        )
        return mutated_ir, f"bad_coeff_{v_target}_{target_c.name}"

    def mutate_over_constraining_extra(self, ir: CanonicalIR) -> Tuple[CanonicalIR, str]:
        """
        Adds an overly restrictive extra constraint that excludes valid solutions.
        E.g. sum(x_i) <= 0.5
        """
        new_constraints = list(ir.constraints)
        var_names = list(ir.variables.keys())

        if not var_names:
            return ir, "none"

        v_target = self.rng.choice(var_names)
        extra_c = CanonicalConstraint(
            name=f"over_restrictive_{v_target}",
            coeffs={v_target: 1.0},
            constant=-0.0,
            sense="<="
        )
        new_constraints.append(extra_c)

        mutated_ir = CanonicalIR(
            problem_name=f"{ir.problem_name}_over_constrained",
            variables=dict(ir.variables),
            constraints=new_constraints,
            objective_sense=ir.objective_sense,
            objective_coeffs=dict(ir.objective_coeffs),
            objective_constant=ir.objective_constant,
        )
        return mutated_ir, f"over_constrained_{v_target}"

    def mutate_constant_error(self, ir: CanonicalIR) -> Tuple[CanonicalIR, str]:
        """
        Perturbs a constraint's RHS constant by a bad factor (paper §7.3).
        """
        if not ir.constraints:
            return ir, "none"
        target_idx = self.rng.randrange(len(ir.constraints))
        target_c = ir.constraints[target_idx]
        factor = self.rng.choice([0.3, 0.5, 1.7, 2.5])
        new_constant = target_c.constant * factor
        new_constraints = list(ir.constraints)
        new_constraints[target_idx] = CanonicalConstraint(
            name=f"{target_c.name}_const_err",
            coeffs=dict(target_c.coeffs),
            constant=new_constant,
            sense=target_c.sense,
        )
        mutated_ir = self._rebuild(ir, f"{ir.problem_name}_const_err_{target_c.name}", new_constraints)
        return mutated_ir, f"constant_error_{target_c.name}"

    def mutate_wrong_domain(self, ir: CanonicalIR) -> Tuple[CanonicalIR, str]:
        """
        Changes the domain of the first continuous variable (continuous → integer).
        """
        target_name = next(
            (n for n, v in ir.variables.items() if v.var_type.lower() == "continuous"), None
        )
        if target_name is None:
            return ir, "none"
        target = ir.variables[target_name]
        new_vars = dict(ir.variables)
        new_vars[target_name] = CanonicalVariable(
            name=target.name,
            var_type="integer",
            lower_bound=target.lower_bound,
            upper_bound=target.upper_bound,
            provenance=target.provenance,
            semantic_role=target.semantic_role,
            requirement_id=target.requirement_id,
            unit=target.unit,
            index_sets=list(target.index_sets),
        )
        mutated_ir = CanonicalIR(
            problem_name=f"{ir.problem_name}_wrong_domain_{target_name}",
            variables=new_vars,
            constraints=list(ir.constraints),
            objective_sense=ir.objective_sense,
            objective_coeffs=dict(ir.objective_coeffs),
            objective_constant=ir.objective_constant,
            metadata=dict(ir.metadata),
        )
        return mutated_ir, f"wrong_domain_{target_name}"

    def mutate_missing_index_range(self, ir: CanonicalIR) -> Tuple[CanonicalIR, str]:
        """
        Truncates an index range: drops variables of the extreme last index-set
        value and any constraints that lose all their variables (paper §7.3).
        """
        indexed = [(n, v) for n, v in ir.variables.items() if v.index_sets]
        if not indexed:
            return ir, "none"
        dim = len(indexed[0][1].index_sets)
        group = [(n, v) for n, v in indexed if len(v.index_sets) == dim]
        last_vals = sorted({v.index_sets[-1] for _, v in group})
        if len(last_vals) < 2:
            return ir, "none"
        drop_val = last_vals[-1]
        dropped = [n for n, v in group if v.index_sets[-1] == drop_val]
        keep = {n for n, v in ir.variables.items() if n not in dropped}

        new_vars = {n: v for n, v in ir.variables.items() if n in keep}
        new_constraints = []
        for c in ir.constraints:
            coeffs = {v: coeff for v, coeff in c.coeffs.items() if v in keep}
            if coeffs:
                new_constraints.append(CanonicalConstraint(
                    name=c.name, coeffs=coeffs, constant=c.constant, sense=c.sense,
                    scale_factor=c.scale_factor,
                ))
        mutated_ir = CanonicalIR(
            problem_name=f"{ir.problem_name}_missing_index_{drop_val}",
            variables=new_vars,
            constraints=new_constraints,
            objective_sense=ir.objective_sense,
            objective_coeffs={v: c for v, c in ir.objective_coeffs.items() if v in keep},
            objective_constant=ir.objective_constant,
            metadata=dict(ir.metadata),
        )
        return mutated_ir, f"missing_index_range_{drop_val}"

    def mutate_broken_big_m(self, ir: CanonicalIR, big_m_threshold: float = 100.0) -> Tuple[CanonicalIR, str]:
        """
        Breaks a Big-M logic link: zeroes out a coefficient whose magnitude
        exceeds big_m_threshold (paper §7.3).
        """
        for idx, c in enumerate(ir.constraints):
            big = [v for v, coeff in c.coeffs.items() if abs(coeff) >= big_m_threshold]
            if big:
                v_target = big[0]
                new_coeffs = dict(c.coeffs)
                new_coeffs[v_target] = 0.0
                new_constraints = list(ir.constraints)
                new_constraints[idx] = CanonicalConstraint(
                    name=f"{c.name}_broken_big_m",
                    coeffs=new_coeffs,
                    constant=c.constant,
                    sense=c.sense,
                )
                mutated_ir = self._rebuild(ir, f"{ir.problem_name}_broken_big_m_{c.name}", new_constraints)
                return mutated_ir, f"broken_big_m_{c.name}"
        return ir, "none"

    @staticmethod
    def _rebuild(ir: CanonicalIR, name: str, constraints: List[CanonicalConstraint]) -> CanonicalIR:
        return CanonicalIR(
            problem_name=name,
            variables=dict(ir.variables),
            constraints=constraints,
            objective_sense=ir.objective_sense,
            objective_coeffs=dict(ir.objective_coeffs),
            objective_constant=ir.objective_constant,
            metadata=dict(ir.metadata),
        )

    def verify_mutant_semantics_changed(self, reference_ir: CanonicalIR, mutant_ir: CanonicalIR) -> bool:
        """
        Uses solver to verify that mutant_ir is semantics-changing relative to reference_ir.
        Checks if optimal values or feasible solutions differ.
        """
        def solve_ir(ir: CanonicalIR) -> Tuple[Optional[float], str]:
            var_specs = {
                vname: (var.var_type, var.lower_bound, var.upper_bound)
                for vname, var in ir.variables.items()
            }
            constraints_list = [
                (c.coeffs, c.sense, -c.constant) for c in ir.constraints
            ]
            res = UnifiedSolver.solve_pulp_model(
                var_specs=var_specs,
                objective_coeffs=ir.objective_coeffs,
                objective_sense=ir.objective_sense,
                constraints_list=constraints_list,
            )
            return res.objective_value, res.status

        ref_val, ref_status = solve_ir(reference_ir)
        mut_val, mut_status = solve_ir(mutant_ir)

        # If feasibility status changed (e.g. Optimal vs Infeasible), semantics changed!
        if ref_status != mut_status:
            return True

        # If objective value changed beyond 1e-4, semantics changed!
        if ref_val is not None and mut_val is not None:
            if abs(ref_val - mut_val) > 1e-4:
                return True

        # Paper §7.3: solver-witness — a point feasible in one model but not the
        # other proves the feasible set changed even when the optimal value is
        # identical (e.g. a loosened Big-M link on an already-satisfied schedule).
        changed, _, _ = self.find_semantic_witness(reference_ir, mutant_ir)
        return changed

    def find_semantic_witness(
        self,
        reference_ir: CanonicalIR,
        mutant_ir: CanonicalIR,
        cap: int = 4,
    ) -> Tuple[bool, Optional[Dict[str, float]], Optional[str]]:
        """
        Returns (changed, witness_point, violated_constraint_name).
        A witness is a solver-verified point that is feasible in one model and
        violates a constraint of the other.
        """
        witness = self._find_witness(mutant_ir, reference_ir.constraints, cap)
        if witness is not None:
            return True, witness[0], witness[1]
        witness = self._find_witness(reference_ir, mutant_ir.constraints, cap)
        if witness is not None:
            return True, witness[0], witness[1]
        return False, None, None

    def _find_witness(
        self,
        domain_ir: CanonicalIR,
        target_constraints: List[CanonicalConstraint],
        cap: int,
    ) -> Optional[Tuple[Dict[str, float], str]]:
        var_specs = {
            vname: (var.var_type, var.lower_bound, var.upper_bound)
            for vname, var in domain_ir.variables.items()
        }
        constraints_list = [
            (c.coeffs, c.sense, -c.constant) for c in domain_ir.constraints
        ]
        dummy = {v: 0.0 for v in var_specs}
        feas = UnifiedSolver.solve_pulp_model(var_specs, dummy, "minimize", constraints_list)
        if feas.status != "Optimal":
            return None

        for c in target_constraints[:cap]:
            coeffs = {v: coeff for v, coeff in c.coeffs.items() if v in var_specs and abs(coeff) > 1e-12}
            if not coeffs:
                continue
            # To violate a '<='/'==' constraint we maximize its residual; for '>='
            # we minimize it (violation means residual < 0).
            sense = "minimize" if c.sense == ">=" else "maximize"
            res = UnifiedSolver.solve_pulp_model(var_specs, dict(coeffs), sense, constraints_list)
            if res.status != "Optimal" or not res.variable_values:
                continue
            residual = res.objective_value + c.constant
            if c.sense == ">=":
                residual = -residual
            if residual > 1e-5:
                return res.variable_values, c.name
        return None
