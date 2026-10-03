"""Component explanations, separate from model-level verification in M4.

Context is built only from independently algebraically matched rows. Neither
complete model nor either tested row is allowed to serve as the common context.
Callers must supply explicit bound rows if those bounds should be compared.
"""
from enum import Enum
from dataclasses import dataclass, field, replace
from fractions import Fraction
from itertools import combinations
from math import nan, isfinite
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.equiceval.canonical_ir import CanonicalIR
from src.equiceval.directed_query import DirectedDiscrepancyEvaluator
from src.utils.solver_wrapper import QueryBudget


class MatchingLabel(Enum):
    EXACT = "exact"
    NORMALIZED_EQUIVALENT = "normalized"
    CONTEXT_EQUIVALENT = "context-equivalent"
    GROUP_EQUIVALENT = "group-equivalent"
    DISPROVED_WITH_WITNESS = "disproved-with-witness"
    UNRESOLVED = "unresolved"
    MUTUALLY_IMPLIED = "mutually-implied"  # Kept for reading historical results.
    PARTIALLY_ALIGNED = "partially-aligned"
    CONTRADICTED = "contradicted"


# Compatibility with branch `thanh` consumers (experiment_runner, tests):
# labels that count as semantically equivalent evidence.
EQUIVALENT_LABELS = {
    MatchingLabel.EXACT,
    MatchingLabel.NORMALIZED_EQUIVALENT,
    MatchingLabel.CONTEXT_EQUIVALENT,
    MatchingLabel.GROUP_EQUIVALENT,
}


@dataclass
class ConstraintMatchResult:
    gt_index: int
    candidate_index: int
    gt_name: str
    candidate_name: str
    label: MatchingLabel
    implication_violation_gt_to_cand: float
    implication_violation_cand_to_gt: float
    status: str = "certified"
    evidence_level: str = "exact_algebra"
    gt_indices: tuple = ()
    candidate_indices: tuple = ()
    evidence: dict = field(default_factory=dict)

    def to_dict(self):
        def safe(value):
            return value if isfinite(value) else None
        return {"reference_indices": list(self.gt_indices or (self.gt_index,)),
                "candidate_indices": list(self.candidate_indices or (self.candidate_index,)),
                "reference_name": self.gt_name, "candidate_name": self.candidate_name,
                "label": self.label.value, "status": self.status,
                "evidence_level": self.evidence_level,
                "eta_reference_to_candidate": safe(self.implication_violation_gt_to_cand),
                "eta_candidate_to_reference": safe(self.implication_violation_cand_to_gt),
                "evidence": self.evidence}


def normalized_key(row):
    """Exact decimal-rational key; equality signs include constant-only rows."""
    c = row.flip_to_leq()
    coeffs = {v: Fraction(str(a)) for v, a in c.coeffs.items() if a != 0}
    constant = Fraction(str(c.constant))
    q = max(sum(abs(a) for a in coeffs.values()), abs(constant))
    first = coeffs[sorted(coeffs)[0]] if coeffs else constant
    sign = -1 if c.sense == "==" and first < 0 else 1
    return (c.sense, tuple(sorted((v, sign*a/q) for v, a in coeffs.items())) if q else (),
            sign*constant/q if q else 0)


def _select_pairs(edges, nr, nc):
    """Lexicographic: certified count, total count, algebraic count, exact count.

Zero-weight dummy columns permit leaving a row unmatched. Equal optima use the
deterministic SciPy assignment algorithm on sorted indices (version is logged).
"""
    if not edges or not nr or not nc:
        return []
    k = min(nr, nc)
    algebraic = k + 1
    cardinality = (k + 1) * (algebraic + 1)
    certified = (k + 1) * cardinality
    weights = np.zeros((nr, nc + nr), dtype=np.int64)
    by_pair = {}
    for edge in edges:
        score = cardinality + (certified if edge.status == "certified" else 0)
        if edge.label in (MatchingLabel.EXACT, MatchingLabel.NORMALIZED_EQUIVALENT):
            score += algebraic
        if edge.label == MatchingLabel.EXACT:
            score += 1
        weights[edge.gt_index, edge.candidate_index] = score
        by_pair[edge.gt_index, edge.candidate_index] = edge
    rows, cols = linear_sum_assignment(weights, maximize=True)
    return [by_pair[i, j] for i, j in zip(rows, cols) if (i, j) in by_pair]


class EquivalenceMatcher:
    def __init__(self, solver_tolerance=1e-5):
        self.tolerance = solver_tolerance

    def check_implication(self, base_ir, target_constraint):
        """Compatibility API: unresolved/infeasible is NaN, never zero."""
        target = CanonicalIR("single_target", base_ir.variables, [target_constraint],
                             "minimize", {})
        result = DirectedDiscrepancyEvaluator(self.tolerance).evaluate_delta_right(target, base_ir)
        if result.status in ("certified", "within_tolerance", "disproved"):
            return result.discrepancy_value
        return nan

    def match_constraints(self, gt_ir, cand_ir):
        return _select_pairs(self._algebraic_edges(gt_ir, cand_ir),
                             len(gt_ir.constraints), len(cand_ir.constraints))

    @staticmethod
    def _algebraic_edges(gt_ir, cand_ir):
        matches, candidates = [], {}
        for j, row in enumerate(cand_ir.constraints):
            candidates.setdefault(normalized_key(row), []).append(j)
        for i, a in enumerate(gt_ir.constraints):
            for j in candidates.get(normalized_key(a), []):
                b = cand_ir.constraints[j]
                label = MatchingLabel.EXACT if (a.coeffs, a.constant, a.sense) == (b.coeffs, b.constant, b.sense) else MatchingLabel.NORMALIZED_EQUIVALENT
                matches.append(ConstraintMatchResult(i, j, a.name, b.name, label, 0.0, 0.0))
        return matches

    def evaluate_components(self, gt_ir, cand_ir, *, budget=None, directed=None,
                            safety_box_bound=None, context_matching=True,
                            group_matching=False, max_context_pairs=32,
                            max_group_trials=16):
        """Bounded contextual search; one-versus-two groups are opt-in.

        Search filtering is not proof. Untested pairs remain unexplored, and
        neither contextual witnesses nor low matching coverage change M4.
        Group selection is deterministic greedy over residual rows, not a
        globally optimal hypergraph matching or a minimal explanation claim.
        """
        for limit in (max_context_pairs, max_group_trials):
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
                raise ValueError("Component search limits must be nonnegative integers")
        budget = budget if budget is not None else QueryBudget()
        directed = directed if directed is not None else DirectedDiscrepancyEvaluator(self.tolerance)
        nr, nc = len(gt_ir.constraints), len(cand_ir.constraints)
        algebraic_edges = self._algebraic_edges(gt_ir, cand_ir)
        algebraic_pairs = _select_pairs(algebraic_edges, nr, nc)
        edges, trials = list(algebraic_edges), []
        used_ref = {m.gt_index for m in algebraic_pairs}
        used_cand = {m.candidate_index for m in algebraic_pairs}
        # Common variable TYPES, not either model's bounds. Bound conditions
        # enter H only when present as independently verified common rows.
        variables = {n: replace(v, lower_bound=-float("inf"), upper_bound=float("inf"))
                     for n, v in gt_ir.variables.items()}

        def support(rows):
            return {v for row in rows for v, a in row.coeffs.items() if a != 0}

        def plausible(left, right):
            a, b = support(left), support(right)
            return bool(a & b) or not (a or b)

        def check(ri, ci, group=False):
            left_rows = [gt_ir.constraints[i] for i in ri]
            right_rows = [cand_ir.constraints[j] for j in ci]
            excluded = {normalized_key(c) for c in left_rows + right_rows}
            h_pairs = [m for m in algebraic_pairs if m.gt_index not in ri and
                       m.candidate_index not in ci and
                       normalized_key(gt_ir.constraints[m.gt_index]) not in excluded]
            context = [gt_ir.constraints[m.gt_index] for m in h_pairs]
            left = CanonicalIR("context_reference", variables, context + left_rows, "minimize", {})
            right = CanonicalIR("context_candidate", variables, context + right_rows, "minimize", {})
            left_target = replace(left, constraints=left_rows)
            right_target = replace(right, constraints=right_rows)
            phase = "group" if group else "context"
            lr = directed._solve_max_violation(left, right_target, f"{phase}_reference_to_candidate",
                                               safety_box_bound, budget, phase=phase)
            rl = directed._solve_max_violation(right, left_target, f"{phase}_candidate_to_reference",
                                               safety_box_bound, budget, phase=phase)
            accepted = ("certified", "within_tolerance")
            if lr.is_feasible_candidate and rl.is_feasible_candidate and lr.status in accepted and rl.status in accepted:
                label = MatchingLabel.GROUP_EQUIVALENT if group else MatchingLabel.CONTEXT_EQUIVALENT
                status = "certified" if lr.status == rl.status == "certified" else "within_tolerance"
            elif "disproved" in (lr.status, rl.status):
                label, status = MatchingLabel.DISPROVED_WITH_WITNESS, "disproved"
            else:
                # Empty domains, even two empty domains, are not equivalence.
                label, status = MatchingLabel.UNRESOLVED, "unresolved"
            def row_dict(c):
                return {"name": c.name, "coeffs": dict(c.coeffs), "constant": c.constant, "sense": c.sense}
            evidence = {
                "scope": "component_context_only", "box_bound": safety_box_bound,
                "variables": {n: {"type": v.var_type, "lower_bound": None, "upper_bound": None}
                              for n, v in variables.items()},
                "common_pairs": [[m.gt_index, m.candidate_index] for m in h_pairs],
                "common_rows": [row_dict(c) for c in context],
                "reference_rows": [row_dict(c) for c in left_rows],
                "candidate_rows": [row_dict(c) for c in right_rows],
                "reference_to_candidate": lr.to_dict(), "candidate_to_reference": rl.to_dict(),
                "empty_domain_mismatch": bool(
                    (lr.source_status == "Infeasible" and rl.is_feasible_candidate) or
                    (rl.source_status == "Infeasible" and lr.is_feasible_candidate)),
                "is_model_level_witness": False,
            }
            return ConstraintMatchResult(
                ri[0], ci[0], "; ".join(c.name for c in left_rows),
                "; ".join(c.name for c in right_rows), label,
                lr.discrepancy_value, rl.discrepancy_value, status,
                "rational_row_certificates" if status == "certified" else
                "numerical_solver_bounds" if status == "within_tolerance" else
                "exact_decimal_witness" if status == "disproved" else "none",
                tuple(ri), tuple(ci), evidence)

        algebraic_keys = {(m.gt_index, m.candidate_index) for m in algebraic_edges}
        pairs = [(i, j) for i in range(nr) for j in range(nc)
                 if (i, j) not in algebraic_keys and
                 plausible([gt_ir.constraints[i]], [cand_ir.constraints[j]])]
        pairs.sort(key=lambda ij: (ij[0] in used_ref or ij[1] in used_cand, ij))
        # No additional calls are useful when an algebraic perfect matching exists.
        if context_matching and len(algebraic_pairs) < min(nr, nc):
            for i, j in pairs[:max_context_pairs]:
                if budget.remaining <= 0:
                    break
                trial = check((i,), (j,))
                trials.append(trial)
                if trial.status in ("certified", "within_tolerance"):
                    edges.append(trial)
        selected = _select_pairs(edges, nr, nc)
        used_ref, used_cand = {m.gt_index for m in selected}, {m.candidate_index for m in selected}
        groups = []
        if group_matching:
            rr, cc = [i for i in range(nr) if i not in used_ref], [j for j in range(nc) if j not in used_cand]
            candidates = [((i,), js) for i in rr for js in combinations(cc, 2)] + [
                (ii, (j,)) for ii in combinations(rr, 2) for j in cc]
            candidates = [(ii, jj) for ii, jj in candidates if plausible(
                [gt_ir.constraints[i] for i in ii], [cand_ir.constraints[j] for j in jj])]
            for ii, jj in candidates[:max_group_trials]:
                if budget.remaining <= 0:
                    break
                trial = check(ii, jj, group=True)
                trials.append(trial)
                if trial.status in ("certified", "within_tolerance"):
                    groups.append(trial)
            groups.sort(key=lambda m: (m.status != "certified", -len(m.gt_indices), m.gt_indices, m.candidate_indices))
            for group in groups:
                if not (used_ref.intersection(group.gt_indices) or used_cand.intersection(group.candidate_indices)):
                    selected.append(group)
                    used_ref.update(group.gt_indices)
                    used_cand.update(group.candidate_indices)
        certified_ref = {i for m in selected if m.status == "certified" for i in (m.gt_indices or (m.gt_index,))}
        return {
            "matches": selected, "trials": trials,
            "certified_reference_rows": len(certified_ref), "matched_reference_rows": len(used_ref),
            "unmatched_reference_indices": [i for i in range(nr) if i not in used_ref],
            "unmatched_candidate_indices": [j for j in range(nc) if j not in used_cand],
            "reference_row_count": nr, "candidate_row_count": nc,
            "controls": {"context_matching": context_matching, "group_matching": group_matching,
                         "max_context_pairs": max_context_pairs, "max_group_trials": max_group_trials,
                         "plausible_pair_count": len(pairs),
                         "tested_context_pairs": sum(len(t.gt_indices) == len(t.candidate_indices) == 1 for t in trials),
                         "tested_group_trials": sum(len(t.gt_indices) != len(t.candidate_indices) for t in trials),
                         "pair_selection": "certified_count,total_count,algebraic_count,exact_count; scipy deterministic ties",
                         "group_selection": "bounded one-versus-two, greedy residual rows",
                         "context_source": "independent algebraic pairs; tested keys and duplicate copies excluded"},
        }
