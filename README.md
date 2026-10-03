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

## Datasets (tested in order)

1. **Self-generated benchmark** — `data/equiceval_full_benchmark.json`
   (400 equivalent + 800 mutants, solver-certified labels, 4 base problems).
   No EquivaMap label exists here, so the second method is the **reference-form
   baseline** (`--baseline refform`); this validates the engine mechanics.
2. **EquivaFormulation** — `data/equivaformulation_affine_v2c_benchmark.json`
   (2,307 pairs, published quasi-Karp labels). Compared with **EquivaMap**
   (`--baseline equivamap`); disagreements are definitional.

`data/sample_pairs.json` (EquivaFormulation) and
`data/sample_pairs_selfgen.json` (self-generated) are 12-pair smoke sets.

## Run

```bash
# 1. Any labelled benchmark -> pair schema
PYTHONPATH=. .venv/bin/python -m expeval.build_pairs \
  --benchmark data/equiceval_full_benchmark.json \
  --output data/selfgen_pairs.json

# 2. EquiCEval only  (self-generated mechanics)
PYTHONPATH=. .venv/bin/python -m expeval.run_comparison \
  --pairs data/selfgen_pairs.json --baseline refform --limit 240 \
  --output output/selfgen_report.json

# 3. EquivaFormulation -> independent relabel under the settled contract
PYTHONPATH=. .venv/bin/python -m expeval.build_pairs \
  --benchmark data/equivaformulation_affine_v2c_benchmark.json \
  --output data/equiva_pairs.json
PYTHONPATH=. .venv/bin/python -m expeval.relabel \
  --pairs data/equiva_pairs.json \
  --contract feasible_set_and_objective_affine \
  --output data/equiva_pairs_labeled.json

# 4. Compare EquiCEval vs EquivaMap (+ definitional disagreements)
PYTHONPATH=. .venv/bin/python -m expeval.run_comparison \
  --pairs data/equiva_pairs_labeled.json --baseline equivamap --limit 240 \
  --output output/meta_eval_report.json
```

`--baseline` accepts `equivamap` (default), `refform`, or `none` (EquiCEval-only).

## Layout

- `vendor/src/` — vendored EquiCEval engine + EquivaFormulation adapter + relabel.
- `vendor/EquivaMap/` — vendored third-party clone.
- `expeval/adapters/` — `Pair`/`Decision`, EquiCEval and EquivaMap adapters.
- `expeval/build_pairs.py`, `expeval/relabel.py`, `expeval/run_comparison.py`.
- `VENDOR.lock` — vendored file hashes (`EquivaMap` commit pinned).
