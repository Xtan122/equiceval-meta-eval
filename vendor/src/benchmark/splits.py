"""Family-level train/test splits (paper §7.1).

All variants of a problem family stay on the same side of the split so a renamed
or rescaled variant cannot leak from the threshold-selection set into the final
test set. Records may additionally carry a ``contamination_stratum`` field.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List


def _family_of(record: Dict[str, Any], family_key: str) -> str:
    return (record.get(family_key) or record.get("family")
            or record.get("base_problem_name") or "unknown")


def family_split(records: Iterable[Dict[str, Any]], test_families: Iterable[str],
                 family_key: str = "base_problem_name") -> Dict[str, str]:
    """Map ``record_id -> 'selection' | 'test'`` by problem family."""
    test = set(test_families)
    return {
        record["record_id"]: ("test" if _family_of(record, family_key) in test else "selection")
        for record in records
    }


def split_summary(records: Iterable[Dict[str, Any]], split: Dict[str, str],
                  family_key: str = "base_problem_name") -> Dict[str, Dict[str, int]]:
    """Count records per family within each split side."""
    summary: Dict[str, Dict[str, int]] = {"selection": {}, "test": {}}
    for record in records:
        family = _family_of(record, family_key)
        side = split.get(record["record_id"], "selection")
        summary.setdefault(side, {})[family] = summary.setdefault(side, {}).get(family, 0) + 1
    return summary
