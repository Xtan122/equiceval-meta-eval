"""
Benchmark Suite Generator Package for EquiCEval.
Provides certified equivalent variant transformations and solver-verified mutation operators.
"""

from src.benchmark.schema import GoldRecord, BenchmarkSuite, ContaminationStratum
from src.benchmark.transformations import SemanticsPreservingTransformer
from src.benchmark.mutator import ControlledMutator, MutationType
from src.benchmark.generator import BenchmarkGenerator

__all__ = [
    "GoldRecord",
    "BenchmarkSuite",
    "ContaminationStratum",
    "SemanticsPreservingTransformer",
    "ControlledMutator",
    "MutationType",
    "BenchmarkGenerator",
]
