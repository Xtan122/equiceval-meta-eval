"""Meta-evaluate EquiCEval on a labelled dataset.

Supports both labelled datasets this repo is meant to test on:

* **self-generated benchmark** (``data/equiceval_full_benchmark.json``):
  compare EquiCEval with the reference-form baseline ``--baseline refform``.
* **EquivaFormulation** (``data/equivaformulation_affine_v2c_benchmark.json``):
  compare EquiCEval with EquivaMap ``--baseline equivamap`` (quasi-Karp), and
  report the definitional disagreements between the two criteria.

EquiCEval is always scored. ``--baseline none`` gives EquiCEval-only metrics.

Usage:
    PYTHONPATH=. python -m expeval.run_comparison \
        --pairs data/equiva_pairs.json --baseline equivamap --limit 240 \
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

from expeval.adapters import EquiCEvalAdapter, EquivaMapAdapter, ReferenceFormAdapter
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
    parser.add_argument("--baseline", choices=["equivamap", "refform", "none"],
                        default="equivamap",
                        help="Second method: EquivaMap (EquivaFormulation) or the "
                             "reference-form baseline (self-generated benchmark)")
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

    equiceval = EquiCEvalAdapter(seconds=args.seconds, mode=args.mode)
    evaluations = [equiceval.score(p) for p in pairs]
    report = {
        "config": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pairs": str(args.pairs),
            "n_pairs": len(pairs),
            "contract": args.contract,
            "mode": args.mode,
            "seconds": args.seconds,
            "limit": args.limit,
            "baseline": args.baseline,
        },
        "equiceval": {
            "metrics": rates(labels, [d.outcome for d in evaluations]),
            "decisions": [d.__dict__ for d in evaluations],
        },
    }

    if args.baseline == "equivamap":
        baseline = EquivaMapAdapter()
    elif args.baseline == "refform":
        baseline = ReferenceFormAdapter()
    else:
        baseline = None

    if baseline is not None:
        baseline_eval = [baseline.score(p) for p in pairs]
        report[baseline.name] = {
            "metrics": rates(labels, [d.outcome for d in baseline_eval]),
            "decisions": [d.__dict__ for d in baseline_eval],
        }

    if args.baseline == "equivamap":
        # Definitional disagreements between EquiCEval (FS + affine) and quasi-Karp.
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
            "note": "Quasi-Karp is weaker than feasible-set + affine objective; "
                    "disagreements are definitional, not method errors.",
        }
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

    print(f"Saved -> {args.output}  (n={len(pairs)}, baseline={args.baseline})")
    for name in [k for k in report if k in ("equiceval", "equivamap", "refform", "strong")]:
        m = report[name]["metrics"]
        print(f"  {name:10s} FPR={_fmt(m['false_alarm_rate'])} "
              f"recall={_fmt(m['error_recall'])} "
              f"unresolved={_fmt(m['unresolved_rate'])} "
              f"unsupported={_fmt(m['unsupported_rate'])} "
              f"unverified={m['n_unverified']}")
    if "definitional_disagreements" in report:
        print(f"  definitional disagreements: {report['definitional_disagreements']['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
