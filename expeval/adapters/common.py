"""Common comparison interface for meta-evaluating EquiCEval against EquivaMap."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class Pair:
    pair_id: str
    family: Optional[str]
    reference_ir: Dict[str, Any]
    candidate_ir: Dict[str, Any]
    label: Optional[bool]
    contract: str
    provided_substitutions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Decision:
    pair_id: str
    method: str
    outcome: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    cost: Dict[str, Any] = field(default_factory=dict)


def load_pairs(path, default_contract: str) -> List[Pair]:
    import json
    from pathlib import Path

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    records = raw["pairs"] if isinstance(raw, dict) else raw
    return [
        Pair(
            pair_id=record["pair_id"],
            family=record.get("family") or record.get("base_problem_name"),
            reference_ir=record["reference_ir"],
            candidate_ir=record["candidate_ir"],
            label=record.get("label", record.get("is_semantically_equivalent")),
            contract=record.get("contract", default_contract),
            provided_substitutions=record.get("provided_substitutions") or {},
            metadata=record.get("metadata", {}),
        )
        for record in records
    ]


def _rate(numerator: int, denominator: int) -> Dict[str, Any]:
    return {"numerator": numerator, "denominator": denominator,
            "value": (numerator / denominator) if denominator else None}


def rates(labels: Sequence[Optional[bool]], outcomes: Sequence[str],
          accepted=("equivalent",)) -> Dict[str, Any]:
    if len(labels) != len(outcomes):
        raise ValueError("labels and outcomes must have equal length")
    eq = [i for i, label in enumerate(labels) if label is True]
    bad = [i for i, label in enumerate(labels) if label is False]
    accepted = set(accepted)
    return {
        "n_total": len(labels),
        "n_equivalent": len(eq),
        "n_faulty": len(bad),
        "n_unverified": sum(label is None for label in labels),
        "false_alarm_rate": _rate(sum(outcomes[i] == "faulty" for i in eq), len(eq)),
        "error_recall": _rate(sum(outcomes[i] == "faulty" for i in bad), len(bad)),
        "equivalent_confirmation_rate": _rate(sum(outcomes[i] in accepted for i in eq), len(eq)),
        "unresolved_rate": _rate(sum(o == "unresolved" for o in outcomes), len(outcomes)),
        "unsupported_rate": _rate(sum(o == "unsupported" for o in outcomes), len(outcomes)),
        "outcome_counts": dict(Counter(outcomes)),
    }


def binary_metrics(labels: Sequence[Optional[bool]], decisions: Sequence[Optional[bool]]) -> Dict[str, Any]:
    """FPR/Recall against labels when a method only yields a boolean decision."""
    eq = [i for i, label in enumerate(labels) if label is True]
    bad = [i for i, label in enumerate(labels) if label is False]
    return {
        "false_alarm_rate": _rate(sum(decisions[i] is False for i in eq), len(eq)),
        "error_recall": _rate(sum(decisions[i] is True for i in bad), len(bad)),
        "equivalent_confirmation_rate": _rate(sum(decisions[i] is True for i in eq), len(eq)),
    }
