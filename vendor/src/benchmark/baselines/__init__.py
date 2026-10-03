"""Baseline registry.

Only the original-paper reference-form methods are registered today. External
baselines (EquivaMap, ReLoop, Falsification) will be added here once a verified
published dataset is chosen; the tier-2 runner stays unchanged.
"""
from __future__ import annotations

from typing import Dict, Iterable

from src.benchmark.baselines.base import Baseline, BaselineResult
from src.benchmark.baselines.reference_form import ReferenceFormAdapter

BASELINES: Dict[str, Baseline] = {
    "refform": ReferenceFormAdapter("refform"),
    "strong": ReferenceFormAdapter("strong"),
}


def build_baselines(names: Iterable[str]) -> Dict[str, Baseline]:
    return {name: BASELINES[name] for name in names}
