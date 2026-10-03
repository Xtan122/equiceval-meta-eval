"""Convert an EquivaFormulation benchmark file into the pair schema.

Usage:
    PYTHONPATH=. python -m expeval.build_pairs \
        --benchmark data/equivaformulation_affine_v2c_benchmark.json \
        --output data/equiva_pairs.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.equiceval.contracts import PRIMARY_EQUIVAFORMULATION_CONTRACT


def build(benchmark_path: Path):
    raw = json.loads(benchmark_path.read_text(encoding="utf-8"))
    contract = raw.get("contract", PRIMARY_EQUIVAFORMULATION_CONTRACT)
    records = (raw.get("equivalent_records", []) + raw.get("mutant_records", [])
               + raw.get("natural_records", []))
    pairs = [
        {
            "pair_id": record["record_id"],
            "family": record.get("family") or record.get("base_problem_name"),
            "reference_ir": record["reference_ir"],
            "candidate_ir": record["candidate_ir"],
            "label": record.get("is_semantically_equivalent"),
            "contract": contract,
            "provided_substitutions": record.get("provided_substitutions") or {},
            "metadata": {
                "published_label": record.get("published_label"),
                "variation_suffix": record.get("variation_suffix"),
            },
        }
        for record in records
    ]
    return contract, pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    contract, pairs = build(args.benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"contract": contract, "pairs": pairs},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(pairs)} pairs (contract={contract}) -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
