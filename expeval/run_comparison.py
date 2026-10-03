"""Meta-evaluate EquiCEval against EquivaMap on EquivaFormulation.

Reports FPR / Recall for both methods under their own criteria, plus the
definitional disagreement between EquiCEval's contract (feasible set + affine
objective) and EquivaMap's quasi-Karp label.

Usage:
    PYTHONPATH=. python -m expeval.run_comparison \
        --pairs data/equiva_pairs.json \
        --contract feasible_set_and_objective_affine --limit 24 \
        --output output/meta_eval_report.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from src.equiceval.contracts import PRIMARY_EQUIVAFORMULATION_CONTRACT

from expeval.adapters import EquiCEvalAdapter, EquivaMapAdapter
from expeval.adapters.common import Pair, load_pairs, rates


def stratified_limit(pairs: List[Pair], limit: int) -> List[Pair]:
    """Keep the equivalent/non-equivalent ratio while subsampling."""
    if not limit or limit >= len(pairs):
        return pairs
    eq = [p for p in pairs if p.label is True]
    bad = [p for p in pairs if p.label is False]
    n_eq = min(len(eq), max(1, limit // 2))
    n_bad = min(len(bad), limit - n_eq)
    return eq[:n_eq] + bad[:n_bad]


def _fmt(metric):
    value = metric.get("value") if isinstance(metric, dict) else metric
    return "n/a" if value is None else f"{value:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--contract", default=PRIMARY_EQUIVAFORMULATION_CONTRACT)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--mode", default="certificates",
                        choices=["direct_only", "certificates", "early_stop"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        parser.error("Output already exists; choose a new path")

    pairs = stratified_limit(load_pairs(args.pairs, args.contract), args.limit)
    labels = [p.label for p in pairs]
    families = [p.family or "unknown" for p in pairs]

    equiceval = EquiCEvalAdapter(seconds=args.seconds, mode=args.mode)
    equivamap = EquivaMapAdapter()
    evaluations = [equiceval.score(p) for p in pairs]
    equivamap_eval = [equivamap.score(p) for p in pairs]

    report = {
        "config": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pairs": str(args.pairs),
            "n_pairs": len(pairs),
            "contract": args.contract,
            "mode": args.mode,
            "seconds": args.seconds,
            "limit": args.limit,
        },
        "equiceval": {
            "metrics": rates(labels, [d.outcome for d in evaluations]),
            "decisions": [d.__dict__ for d in evaluations],
        },
        "equivamap": {
            "metrics": rates(labels, [d.outcome for d in equivamap_eval]),
            "decisions": [d.__dict__ for d in equivamap_eval],
        },
    }

    # Definitional disagreements between the two criteria.
    counts, examples = Counter(), []
    for pair, decision in zip(pairs, evaluations):
        published = pair.metadata.get("published_label")
        if published is None:
            continue
        if decision.outcome == "faulty" and published:
            counts["equiceval_faulty_quasikarp_equivalent"] += 1
            examples.append(pair.pair_id)
        elif decision.outcome in {"equivalent", "tolerance"} and not published:
            counts["equiceval_equivalent_quasikarp_faulty"] += 1
            examples.append(pair.pair_id)
    report["definitional_disagreements"] = {
        "counts": dict(counts), "examples": examples[:20],
        "note": "Quasi-Karp (EquivaMap) is weaker than feasible-set + affine objective; "
                "disagreements are definitional, not method errors.",
    }

    # Fair head-to-head on the intersection where both criteria agree.
    intersection = [
        (pair, decision) for pair, decision in zip(pairs, evaluations)
        if pair.label is not None and pair.label == pair.metadata.get("published_label")
    ]
    report["agreement_intersection"] = {
        "n": len(intersection),
        "equiceval_metrics": rates(
            [p.label for p, _ in intersection], [d.outcome for _, d in intersection])
        if intersection else None,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved -> {args.output}  (n={len(pairs)})")
    for name in ("equiceval", "equivamap"):
        m = report[name]["metrics"]
        print(f"  {name:10s} FPR={_fmt(m['false_alarm_rate'])} "
              f"recall={_fmt(m['error_recall'])} "
              f"unresolved={_fmt(m['unresolved_rate'])} "
              f"unverified={m['n_unverified']}")
    print(f"  definitional disagreements: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
