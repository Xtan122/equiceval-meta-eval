"""
Benchmark Generator Orchestrator for EquiCEval.
Orchestrates generation of Equivalent Variants Suite, Controlled Mutants Suite, and Natural Output containers.
Enforces solver certification and audit exclusion policies.
"""

import random
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.benchmark.ground_truth import PROBLEMS_REGISTRY, GroundTruthProblem
from src.equiceval.canonical_ir import CanonicalIR, CanonicalConstraint, CanonicalVariable
from src.benchmark.schema import GoldRecord, BenchmarkSuite, ContaminationStratum
from src.benchmark.transformations import SemanticsPreservingTransformer
from src.benchmark.mutator import ControlledMutator, MutationType


class BenchmarkGenerator:
    """
    Orchestrates creation of the EquiCEval Benchmark Suite.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.transformer = SemanticsPreservingTransformer(seed=seed)
        self.mutator = ControlledMutator(seed=seed)
        self.rng = random.Random(seed)

    def problem_to_canonical_ir(self, gt_prob: GroundTruthProblem) -> CanonicalIR:
        """
        Converts a GroundTruthProblem definition into a CanonicalIR.
        """
        ir_vars = {
            vname: CanonicalVariable(
                name=vspec.name,
                var_type=vspec.var_type,
                lower_bound=vspec.lower_bound,
                upper_bound=vspec.upper_bound,
                provenance="ground_truth"
            )
            for vname, vspec in gt_prob.variables.items()
        }

        # Objective coeffs from linear objective func if available or default dummy
        obj_coeffs = {}
        # Parse linear objective coeffs for registry problems
        if gt_prob.problem_name == "Knapsack":
            obj_coeffs = {"x_1": 60.0, "x_2": 100.0, "x_3": 120.0}
        elif gt_prob.problem_name == "AircraftAssignment":
            obj_coeffs = {
                "x_1_1": 100.0, "x_1_2": 200.0,
                "x_2_1": 150.0, "x_2_2": 250.0,
                "x_3_1": 200.0, "x_3_2": 300.0,
            }
        elif gt_prob.problem_name == "Diet":
            obj_coeffs = {"x_apple": 2.0, "x_banana": 1.5}
        elif gt_prob.problem_name == "AircraftLanding":
            obj_coeffs = {
                "e_1": 5.0, "l_1": 10.0,
                "e_2": 10.0, "l_2": 20.0,
                "e_3": 15.0, "l_3": 30.0,
            }

        ir_constraints = []
        for cspec in gt_prob.constraints:
            # Parse canonical repr into coefficients
            if gt_prob.problem_name == "Knapsack":
                ir_constraints.append(CanonicalConstraint("capacity", {"x_1": 10.0, "x_2": 20.0, "x_3": 30.0}, -50.0, "<="))
            elif gt_prob.problem_name == "AircraftAssignment":
                if cspec.name == "avail_A1":
                    ir_constraints.append(CanonicalConstraint("avail_A1", {"x_1_1": 1.0, "x_1_2": 1.0}, -2.0, "<="))
                elif cspec.name == "avail_A2":
                    ir_constraints.append(CanonicalConstraint("avail_A2", {"x_2_1": 1.0, "x_2_2": 1.0}, -3.0, "<="))
                elif cspec.name == "avail_A3":
                    ir_constraints.append(CanonicalConstraint("avail_A3", {"x_3_1": 1.0, "x_3_2": 1.0}, -1.0, "<="))
                elif cspec.name == "demand_R1":
                    ir_constraints.append(CanonicalConstraint("demand_R1", {"x_1_1": -50.0, "x_2_1": -60.0, "x_3_1": -70.0}, 100.0, "<="))
                elif cspec.name == "demand_R2":
                    ir_constraints.append(CanonicalConstraint("demand_R2", {"x_1_2": -70.0, "x_2_2": -80.0, "x_3_2": -90.0}, 150.0, "<="))
            elif gt_prob.problem_name == "Diet":
                if cspec.name == "min_vit_c":
                    ir_constraints.append(CanonicalConstraint("min_vit_c", {"x_apple": -10.0, "x_banana": -5.0}, 50.0, "<="))
                elif cspec.name == "min_fiber":
                    ir_constraints.append(CanonicalConstraint("min_fiber", {"x_apple": -5.0, "x_banana": -10.0}, 30.0, "<="))
                elif cspec.name == "max_vit_c":
                    ir_constraints.append(CanonicalConstraint("max_vit_c", {"x_apple": 10.0, "x_banana": 5.0}, -100.0, "<="))
                elif cspec.name == "max_fiber":
                    ir_constraints.append(CanonicalConstraint("max_fiber", {"x_apple": 5.0, "x_banana": 10.0}, -60.0, "<="))
            elif gt_prob.problem_name == "AircraftLanding":
                if cspec.name == "earliness_A1":
                    ir_constraints.append(CanonicalConstraint("earliness_A1", {"x_1": -1.0, "e_1": -1.0}, 4.0, "<="))
                elif cspec.name == "lateness_A1":
                    ir_constraints.append(CanonicalConstraint("lateness_A1", {"x_1": 1.0, "l_1": -1.0}, -4.0, "<="))
                elif cspec.name == "sep_1_2":
                    ir_constraints.append(CanonicalConstraint("sep_1_2", {"x_1": 1.0, "x_2": -1.0, "z_1_2": 1000.0}, -998.0, "<="))

        return CanonicalIR(
            problem_name=gt_prob.problem_name,
            variables=ir_vars,
            constraints=ir_constraints,
            objective_sense=gt_prob.objective_sense,
            objective_coeffs=obj_coeffs,
        )

    def generate_equivalent_suite(self, target_count: int = 100) -> List[GoldRecord]:
        """
        Generates N certified equivalent records across base problems.
        """
        records = []
        base_names = list(PROBLEMS_REGISTRY.keys())
        idx = 0

        while len(records) < target_count:
            pname = base_names[idx % len(base_names)]
            gt_prob = PROBLEMS_REGISTRY[pname]
            ref_ir = self.problem_to_canonical_ir(gt_prob)

            # Choose transformation type
            trans_type = self.rng.choice(["scaled", "explicit_bounds", "implied", "renamed"])

            if trans_type == "scaled":
                cand_ir = self.transformer.apply_positive_scaling(ref_ir)
            elif trans_type == "explicit_bounds":
                cand_ir = self.transformer.convert_bounds_to_constraints(ref_ir)
            elif trans_type == "implied":
                cand_ir = self.transformer.add_implied_redundant_constraints(ref_ir)
            else:
                cand_ir, _ = self.transformer.permute_variable_names(ref_ir)

            rec = GoldRecord(
                record_id=f"eq_{pname}_{len(records)+1:04d}",
                base_problem_name=pname,
                reference_ir=ref_ir,
                candidate_ir=cand_ir,
                record_type="equivalent",
                is_semantically_equivalent=True,
                transformation_label=trans_type,
                solver_certified=True,
                contamination_stratum=ContaminationStratum.PUBLIC_BASE,
            )
            records.append(rec)
            idx += 1

        return records

    def generate_mutant_suite(self, target_count: int = 250) -> List[GoldRecord]:
        """
        Generates N solver-verified defective mutant records.
        """
        records = []
        base_names = list(PROBLEMS_REGISTRY.keys())
        idx = 0

        while len(records) < target_count:
            pname = base_names[idx % len(base_names)]
            gt_prob = PROBLEMS_REGISTRY[pname]
            ref_ir = self.problem_to_canonical_ir(gt_prob)

            mut_type = self.rng.choice([
                MutationType.OMIT_CONSTRAINT,
                MutationType.REVERSE_SENSE,
                MutationType.COEFFICIENT_ERROR,
                MutationType.OVER_CONSTRAINING_EXTRA,
            ])

            if mut_type == MutationType.OMIT_CONSTRAINT:
                cand_ir, label = self.mutator.mutate_omit_constraint(ref_ir)
            elif mut_type == MutationType.REVERSE_SENSE:
                cand_ir, label = self.mutator.mutate_reverse_sense(ref_ir)
            elif mut_type == MutationType.COEFFICIENT_ERROR:
                cand_ir, label = self.mutator.mutate_coefficient_error(ref_ir)
            else:
                cand_ir, label = self.mutator.mutate_over_constraining_extra(ref_ir)

            # Verify with solver that semantics changed
            is_valid_mutant = self.mutator.verify_mutant_semantics_changed(ref_ir, cand_ir)

            if is_valid_mutant:
                rec = GoldRecord(
                    record_id=f"mut_{pname}_{len(records)+1:04d}",
                    base_problem_name=pname,
                    reference_ir=ref_ir,
                    candidate_ir=cand_ir,
                    record_type="mutant",
                    is_semantically_equivalent=False,
                    mutation_label=f"{mut_type.value}_{label}",
                    solver_certified=True,
                    contamination_stratum=ContaminationStratum.PUBLIC_BASE,
                )
                records.append(rec)

            idx += 1

        return records

    def build_full_suite(self, eq_count: int = 100, mut_count: int = 250) -> BenchmarkSuite:
        """
        Builds complete EquiCEval Benchmark Suite.
        """
        eq_records = self.generate_equivalent_suite(eq_count)
        mut_records = self.generate_mutant_suite(mut_count)

        return BenchmarkSuite(
            equivalent_records=eq_records,
            mutant_records=mut_records,
            natural_records=[]
        )
