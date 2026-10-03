"""Independently relabel EquivaFormulation pairs under the settled contract.

Uses the vendored ``src.benchmark.independent_relabel.check_equivalence`` (MILP
containment + affine objective check), which does not call the EquiCEval engine.
Unverifiable pairs get ``label = null``.

Usage:
    PYTHONPATH=. python -m expeval.relabel \
        --pairs data/equiva_pairs.json \
        --contract feasible_set_and_objective_affine \
        --output data/equiva_pairs_labeled.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.benchmark.independent_relabel import check_equivalence
from src.benchmark.verified_evaluation import ir_from_dict
from src.equiceval.contracts import PRIMARY_EQUIVAFORMULATION_CONTRACT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--contract", default=PRIMARY_EQUIVAFORMULATION_CONTRACT)
    parser.add_argument("--safety-box", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    raw = json.loads(args.pairs.read_text(encoding="utf-8"))
    records = raw["pairs"] if isinstance(raw, dict) else raw

    verified = 0
    for record in records:
        reference = ir_from_dict(record["reference_ir"])
        candidate = ir_from_dict(record["candidate_ir"])
        label, evidence = check_equivalence(
            reference, candidate, args.contract, safety_box_bound=args.safety_box)
        record["label"] = label
        record["label_evidence"] = evidence
        verified += 1 if label is not None else 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = records if isinstance(raw, list) else {**raw, "pairs": records}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Verified {verified}/{len(records)} labels -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
