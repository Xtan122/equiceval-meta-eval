"""Adapter for EquivaMap (quasi-Karp equivalence).

EquivaMap requires an LLM (mapping discovery) and Gurobi; the vendored clone under
``vendor/EquivaMap`` is run separately. Its decisive criterion is **quasi-Karp**:
preserve an optimal solution, directionally. The EquivaFormulation dataset pins
that criterion in each record's ``published_label``, so this adapter scores the
published quasi-Karp label directly and marks it ``criterion = "quasi-Karp"``.

Disagreements with EquiCEval's stricter (feasible-set + affine objective)
contract are therefore *definitional*, not method errors.
"""
from __future__ import annotations

from expeval.adapters.common import Decision, Pair


class EquivaMapAdapter:
    name = "equivamap"

    def score(self, pair: Pair) -> Decision:
        published = pair.metadata.get("published_label")
        if published is None:
            return Decision(pair.pair_id, self.name, "unsupported",
                            {"reason": "record has no published_label"})
        return Decision(
            pair_id=pair.pair_id,
            method=self.name,
            outcome="equivalent" if published else "faulty",
            diagnostics={"published_label": bool(published), "criterion": "quasi-Karp"},
        )
