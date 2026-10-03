"""
EquiCEval Core Framework Package.
Equivalence-Aware and Counterexample-Guided Evaluation for Optimization Formulations.
"""

from src.equiceval.canonical_ir import (
    AffineExpression,
    CanonicalIR,
    CanonicalConstraint,
    CanonicalVariable,
    ProjectionCertificate,
    build_projection_certificate,
    check_affine_substitution,
)
from src.equiceval.matching import (
    EquivalenceMatcher,
    MatchingLabel,
    ConstraintMatchResult,
)
from src.equiceval.directed_query import (
    DirectedDiscrepancyEvaluator,
    CounterexampleWitness,
)
from src.equiceval.behavioral_discrepancy import (
    BehavioralDiscrepancyEvaluator,
    BehavioralDiscrepancyResult,
    PairDiscrepancy,
)
from src.equiceval.objective_discrepancy import (
    ObjectiveDiscrepancyEvaluator,
    ObjectiveDiscrepancyResult,
    ObjectiveAlignment,
)
from src.equiceval.evaluator import EquiCEvalEvaluator, EquiCEvalResult
from src.equiceval.precheck import (
    ModelPreChecker,
    EvaluatorStatus,
    ParseStatus,
    MapStatus,
    FeasStatus,
    BoundStatus,
    SolveStatus,
    quick_precheck,
)

__all__ = [
    # Canonical IR
    "CanonicalIR",
    "CanonicalConstraint",
    "CanonicalVariable",
    "AffineExpression",
    "ProjectionCertificate",
    "build_projection_certificate",
    "check_affine_substitution",
    # Matching
    "EquivalenceMatcher",
    "MatchingLabel",
    "ConstraintMatchResult",
    # Directed queries
    "DirectedDiscrepancyEvaluator",
    "CounterexampleWitness",
    # Module 3 — Behavioral discrepancy
    "BehavioralDiscrepancyEvaluator",
    "BehavioralDiscrepancyResult",
    "PairDiscrepancy",
    # Module 5 — Objective discrepancy
    "ObjectiveDiscrepancyEvaluator",
    "ObjectiveDiscrepancyResult",
    "ObjectiveAlignment",
    # Evaluator
    "EquiCEvalEvaluator",
    "EquiCEvalResult",
    # Module 0 — Pre-check & State Machine
    "ModelPreChecker",
    "EvaluatorStatus",
    "ParseStatus",
    "MapStatus",
    "FeasStatus",
    "BoundStatus",
    "SolveStatus",
    "quick_precheck",
]
