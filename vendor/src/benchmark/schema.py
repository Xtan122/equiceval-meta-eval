"""
Benchmark Schema definitions for EquiCEval Benchmark Suite.
Defines GoldRecord structures, contamination strata, and dataset containers.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from src.equiceval.canonical_ir import CanonicalIR


class ContaminationStratum(Enum):
    PUBLIC_BASE = "public_base"             # Known base problems (e.g. ComplexOR)
    HELD_OUT_FAMILY = "held_out_family"     # Held-out problem families
    POST_CUTOFF = "post_cutoff"             # Private/Post-release randomized instances


@dataclass
class GoldRecord:
    record_id: str
    base_problem_name: str
    reference_ir: CanonicalIR
    candidate_ir: CanonicalIR
    record_type: str                         # 'equivalent', 'mutant', 'natural'
    is_semantically_equivalent: bool
    mutation_label: Optional[str] = None     # e.g., 'omit_constraint', 'coefficient_error'
    transformation_label: Optional[str] = None # e.g., 'scaled_positive', 'bound_to_constraint'
    solver_certified: bool = True
    contamination_stratum: ContaminationStratum = ContaminationStratum.PUBLIC_BASE
    provenance: str = "generated"
    annotator_adjudication_needed: bool = False
    # Paper §7.4 audit fields
    projection_status: str = ""                      # e.g. 'verified' | 'unresolved' | 'unsupported'
    matching_evidence: str = ""                      # e.g. 'exact' | 'normalized-equivalent' | 'context-equivalent'
    solver_witness: Optional[Dict[str, Any]] = None  # witness point confirming feasible-set change
    exclusion_reason: str = ""                       # e.g. 'solver_certification_failed'
    family_label: str = ""                           # problem family for family-level splits

    def to_dict(self) -> Dict[str, Any]:
        def canonical_to_dict(ir: CanonicalIR) -> Dict[str, Any]:
            return {
                "problem_name": ir.problem_name,
                "objective_sense": ir.objective_sense,
                "objective_coeffs": ir.objective_coeffs,
                "objective_constant": ir.objective_constant,
                "variables": {
                    vname: {
                        "name": var.name,
                        "var_type": var.var_type,
                        "lower_bound": var.lower_bound,
                        "upper_bound": var.upper_bound,
                        "provenance": var.provenance
                    }
                    for vname, var in ir.variables.items()
                },
                "constraints": [
                    {
                        "name": c.name,
                        "coeffs": c.coeffs,
                        "constant": c.constant,
                        "sense": c.sense,
                        "scale_factor": c.scale_factor
                    }
                    for c in ir.constraints
                ]
            }

        return {
            "record_id": self.record_id,
            "base_problem_name": self.base_problem_name,
            "reference_ir": canonical_to_dict(self.reference_ir),
            "candidate_ir": canonical_to_dict(self.candidate_ir),
            "record_type": self.record_type,
            "is_semantically_equivalent": self.is_semantically_equivalent,
            "mutation_label": self.mutation_label,
            "transformation_label": self.transformation_label,
            "solver_certified": self.solver_certified,
            "contamination_stratum": self.contamination_stratum.value,
            "provenance": self.provenance,
            "annotator_adjudication_needed": self.annotator_adjudication_needed,
            "projection_status": self.projection_status,
            "matching_evidence": self.matching_evidence,
            "solver_witness": self.solver_witness,
            "exclusion_reason": self.exclusion_reason,
            "family_label": self.family_label,
        }


@dataclass
class BenchmarkSuite:
    equivalent_records: List[GoldRecord] = field(default_factory=list)
    mutant_records: List[GoldRecord] = field(default_factory=list)
    natural_records: List[GoldRecord] = field(default_factory=list)

    @property
    def total_records(self) -> int:
        return len(self.equivalent_records) + len(self.mutant_records) + len(self.natural_records)

    def summary(self) -> Dict[str, int]:
        return {
            "equivalent_count": len(self.equivalent_records),
            "mutant_count": len(self.mutant_records),
            "natural_count": len(self.natural_records),
            "total_count": self.total_records,
        }
