# equiceval-meta-eval

Meta-evaluation repo: **does EquiCEval grade correctly?** It compares the
EquiCEval engine with **EquivaMap** (ICML 2025), the published equivalence
checker, on the **EquivaFormulation** benchmark.

Two different criteria are involved and must not be conflated:

- **EquiCEval** uses the settled contract `feasible_set_and_objective_affine`
  (same feasible set + `f* = a·f̂ + b`, `a > 0`).
- **EquivaMap** uses **quasi-Karp**: preserve an optimal solution, directionally.
  Quasi-Karp is weaker, so disagreements where EquiCEval says "faulty" but
  EquivaMap says "equivalent" are **definitional**, not method errors. The repo
  reports them separately and also scores both on the agreement intersection.

The EquiCEval engine is vendored from `../EquiCEval`; the EquivaFormulation
adapter and the independent relabel oracle are vendored from the monorepo;
EquivaMap is vendored from `https://github.com/HumainLab/EquivaMap` at commit
`69d9c90edacb862824141b903b832fd2a1353d9e`. `VENDOR.lock` pins the file hashes.

## Install

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Test

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

EquivaMap itself needs an OpenAI key and a Gurobi license; the adapter here scores
the dataset's published quasi-Karp label (EquivaMap's criterion), so the tests
run offline.

## Run

```bash
# 1. EquivaFormulation benchmark -> pair schema
PYTHONPATH=. .venv/bin/python -m expeval.build_pairs \
  --benchmark data/equivaformulation_affine_v2c_benchmark.json \
  --output data/equiva_pairs.json

# 2. Independent relabel under the settled contract (MILP containment)
PYTHONPATH=. .venv/bin/python -m expeval.relabel \
  --pairs data/equiva_pairs.json \
  --contract feasible_set_and_objective_affine \
  --output data/equiva_pairs_labeled.json

# 3. Compare EquiCEval vs EquivaMap
PYTHONPATH=. .venv/bin/python -m expeval.run_comparison \
  --pairs data/equiva_pairs_labeled.json \
  --contract feasible_set_and_objective_affine --limit 240 \
  --output output/meta_eval_report.json
```

`data/sample_pairs.json` is a 12-pair smoke set; the full benchmark is
`data/equivaformulation_affine_v2c_benchmark.json` (2,307 pairs).

## Layout

- `vendor/src/` — vendored EquiCEval engine + EquivaFormulation adapter + relabel.
- `vendor/EquivaMap/` — vendored third-party clone.
- `expeval/adapters/` — `Pair`/`Decision`, EquiCEval and EquivaMap adapters.
- `expeval/build_pairs.py`, `expeval/relabel.py`, `expeval/run_comparison.py`.
- `VENDOR.lock` — vendored file hashes (`EquivaMap` commit pinned).
