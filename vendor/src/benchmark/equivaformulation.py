"""Adapter: EquivaFormulation (Zhai et al., ICML 2025) -> EquiCEval records.

EquivaFormulation is the open equivalence-checking dataset released with
*EquivaMap: Leveraging LLMs for Automatic Equivalence Checking of Optimization
Formulations*. Each base problem is published together with ten transformed
formulations whose equivalence status is fixed by the published transformation
table.

Provenance (pinned; see ``data/third_party`` manifest for the license):

    dataset : https://huggingface.co/datasets/humainlab/EquivaFormulation
    revision: e0566ed8e60942dc83e1420557e0afc593983672
    source  : https://github.com/HumainLab/EquivaMap
    license : CC-BY-NC-SA-4.0 (NON-COMMERCIAL, share-alike)

Released archive layout (after extracting ``EquivaFormulation.zip``)::

    sample-data-easy/<instance_id>/
        <instance_id>_c/model.lp
        <instance_id>_d/model.lp
        ... <instance_id>_l/model.lp

Variation label convention (from the dataset card)::

    c,d,e,f,g,h,i -> equivalent to the original
    j,k,l         -> NOT equivalent

For EquiCEval we use ``<instance_id>_c`` as the reference formulation.  The
``_c`` transformation only renames parameters and variables, so it is the
original problem up to a bijective renaming and preserves the optimal
solutions; every other suffix is a candidate.  ``_c`` itself is therefore
skipped as a candidate.

This module never writes into the source archive: it only reads ``model.lp``
files and converts them, through HiGHS, into ``CanonicalIR`` objects.
"""
from __future__ import annotations

import json
import hashlib
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.equiceval.canonical_ir import CanonicalConstraint, CanonicalIR, CanonicalVariable
from src.benchmark.verified_evaluation import ir_to_dict

# ---------------------------------------------------------------------------
# Pinned provenance
# ---------------------------------------------------------------------------
SOURCE = {
    "dataset": "EquivaFormulation",
    "huggingface_repo": "humainlab/EquivaFormulation",
    "huggingface_revision": "e0566ed8e60942dc83e1420557e0afc593983672",
    "source_repo": "https://github.com/HumainLab/EquivaMap",
    "paper": "https://arxiv.org/abs/2502.14760",
    "license": "CC-BY-NC-SA-4.0",
    "license_note": "non-commercial; share-alike",
    "transactions_sha256": {
        "test.jsonl": "aabdca24836dc9689b147f86c5e0ad02fc30badf9a915ce87dd4e46a4a4bb101",
        "EquivaFormulation.zip": "3e6b841e4a1b5c09d9142f93a6fc0fc112157d46ce28645e1f45024b439e39cb",
    },
}

REFERENCE_SUFFIX = "c"
CANDIDATE_SUFFIXES = ("d", "e", "f", "g", "h", "i", "j", "k", "l")
# With a bare original reference (two-roots layout) every published variant is a
# candidate, including the rename-only ``_c``.
ALL_CANDIDATE_SUFFIXES = ("c",) + CANDIDATE_SUFFIXES
EQUIVALENT_SUFFIXES = frozenset(("c", "d", "e", "f", "g", "h", "i"))
NON_EQUIVALENT_SUFFIXES = frozenset(("j", "k", "l"))

# Transformation type per suffix (dataset card, variation correspondence table).
TRANSFORMATION_NAME = {
    "c": "rename_parameters_and_variables",
    "d": "binary_substitution",
    "e": "add_valid_inequalities",
    "f": "substitute_objective_function_with_constraint",
    "g": "add_slack_variables",
    "h": "linear_substitution",
    "i": "rescale_objective",
    "j": "random_order_unrelated_instance",
    "k": "feasibility_problem",
    "l": "loose_active_constraints_at_optimum",
}

_COEFF_TOL = 1e-9


def file_sha256(path: Path) -> Optional[str]:
    """Hash a source artifact; absent optional metadata is recorded as null."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class EquivaPair:
    """One (reference, candidate) pair with its published label."""

    instance_id: str
    suffix: str
    reference_path: Path
    candidate_path: Path
    equivalent: bool

    @property
    def record_id(self) -> str:
        return f"equivaformulation_{self.instance_id}_{self.suffix}"

    @property
    def family(self) -> str:
        return f"equivaformulation_{self.instance_id}"

    @property
    def transformation_label(self) -> str:
        return TRANSFORMATION_NAME[self.suffix]


# ---------------------------------------------------------------------------
# LP -> CanonicalIR
# ---------------------------------------------------------------------------
def _var_type(integrality: Any, lower: float, upper: float) -> str:
    is_integer = getattr(integrality, "name", str(integrality)) == "kInteger"
    if not is_integer:
        return "continuous"
    if lower == 0.0 and upper == 1.0:
        return "binary"
    return "integer"


_INDEXED_NAME = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\[([0-9,]+)\]")


def _sanitize_lp_names(text: str) -> str:
    """Rewrite ``h[0]`` -> ``h_0`` (HiGHS' LP reader rejects square brackets)."""
    return _INDEXED_NAME.sub(
        lambda m: m.group(1) + "_" + "_".join(m.group(2).split(",")), text)


def _sanitize_lp_name(name: str) -> str:
    """Sanitize a single variable name the same way ``lp_to_canonical_ir`` does."""
    return _sanitize_lp_names(str(name))


VARIABLE_MAPPINGS_FILENAME = "variable_mappings.json"

_INDEX_RE = re.compile(r"^(?P<base>.*?)(?P<idx>(?:\[[^\]]*\]|_\d+)+)$")


def _split_base_index(name: str) -> Tuple[str, str]:
    """Split ``PackageCount[0]`` into ``("PackageCount", "[0]")``."""
    match = _INDEX_RE.match(name)
    if match:
        return match.group("base"), match.group("idx")
    return name, ""


def _indexed_targets(base: str, variables: Iterable[str]) -> Dict[str, str]:
    """``index_suffix -> full_name`` for every variable sharing ``base``.

    The published ``variable_mappings.json`` is written at the base level
    (``PackageCount -> b``) while the LP files carry indexed variables
    (``PackageCount[0]``, ``b[0]``); this expands the former onto the latter.
    """
    out: Dict[str, str] = {}
    for name in variables:
        candidate_base, index = _split_base_index(name)
        if candidate_base == base:
            out[index] = name
    return out


def _read_variable_terms(path: Path) -> Dict[str, List[Tuple[float, str]]]:
    """Read ``variable_mappings.json`` into ``{original: [(coeff, variant_var)]}``.

    Each entry encodes ``original = sum(coeff * variant_var)`` for the variant in
    that directory.  Malformed / null entries are dropped rather than trusted.
    """
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    terms: Dict[str, List[Tuple[float, str]]] = {}
    for original, entries in raw.items():
        if not isinstance(entries, list):
            continue
        parsed: List[Tuple[float, str]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            variable = entry.get("variable")
            constant = entry.get("constant")
            if variable is None or constant is None:
                continue
            try:
                parsed.append((float(constant), str(variable)))
            except (TypeError, ValueError):
                continue
        if parsed:
            terms[str(original)] = parsed
    return terms


def _close(a: float, b: float, relative_tolerance: float) -> bool:
    return abs(a - b) <= relative_tolerance * max(1.0, abs(a), abs(b))


def _snap_scale(scale: float, relative_tolerance: float = 1e-9) -> float:
    """Snap a ratio to a short rational ``p/q`` when it is within tolerance.

    Ratios of decimal LP coefficients carry float noise (``99.99999999999999``),
    which makes exact affine maps fragile.  Snapping to the nearest fraction
    with bounded denominator removes that noise while leaving genuinely
    irregular scales untouched.
    """
    approximation = Fraction(str(scale)).limit_denominator(10_000)
    exact = float(approximation)
    if _close(exact, scale, relative_tolerance):
        return exact
    return scale


def infer_diagonal_substitutions(
    reference_ir: CanonicalIR,
    candidate_ir: CanonicalIR,
    relative_tolerance: float = 1e-6,
) -> Dict[str, Dict[str, Any]]:
    """Infer ``candidate_var = k * reference_var`` from matching constraint rows.

    EquivaFormulation reuses variable names across the ``_i`` rescaling variant
    while changing scales, which defeats the shared-name convention.  Rows keep
    their identifiers (``R0``, ``R1``, ...), so for every candidate variable the
    ratio ``reference_coeff / candidate_coeff`` is a direct estimate of ``k``.

    A substitution is emitted only when the evidence is *complete and
    consistent*: the two models must expose the same row identifiers, every
    shared row must agree coefficient-by-coefficient and in its right-hand side,
    and each ratio must be observed more than once unless it is the only row.
    Random formulations (``_j``) share variable names but not rows, so they are
    rejected here and left to structural reasoning instead of being silently
    mapped.  Nothing assumes ``variable_mappings.json`` -- the scale is read off
    the numeric models, the fair detector for a comparison with EquivaMap.
    """
    reference_rows = {c.name: c for c in reference_ir.constraints}
    candidate_rows = {c.name: c for c in candidate_ir.constraints}
    if not reference_rows or not candidate_rows:
        return {}
    if set(reference_rows) != set(candidate_rows):
        return {}

    votes: Dict[str, List[float]] = {}
    for name, constraint in candidate_rows.items():
        reference_row = reference_rows[name]
        if constraint.sense != reference_row.sense:
            return {}
        if not _close(constraint.constant, reference_row.constant, relative_tolerance):
            return {}
        if set(constraint.coeffs) != set(reference_row.coeffs):
            return {}
        for cand_name, cand_coeff in constraint.coeffs.items():
            ref_coeff = reference_row.coeffs[cand_name]
            if cand_coeff == 0:
                return {}
            votes.setdefault(cand_name, []).append(ref_coeff / cand_coeff)

    substitutions: Dict[str, Dict[str, Any]] = {}
    multi_row = len(reference_rows) > 1
    for cand_name, ratios in votes.items():
        if multi_row and len(ratios) < 2:
            continue  # a unique ratio from a single row is too weak to trust
        scale = ratios[0]
        if not math.isfinite(scale) or scale == 0:
            continue
        if not all(_close(scale, other, relative_tolerance) for other in ratios[1:]):
            continue
        substitutions[cand_name] = {"coeffs": {cand_name: _snap_scale(scale)}, "constant": 0.0}
    return substitutions


def affine_substitutions_for_pair(
    reference_dir: Path,
    candidate_dir: Path,
    reference_ir: CanonicalIR,
    candidate_ir: CanonicalIR,
) -> Dict[str, Dict[str, Any]]:
    """Explicit candidate->reference affine maps for a variant pair.

    Two sources are composed, with the numeric inference taking precedence:

    1. ``infer_diagonal_substitutions`` reads the scale off matching rows of the
       two ``model.lp`` files (deterministic, dataset-agnostic).  Some published
       ``variable_mappings.json`` files carry swapped constants, so this is the
       primary detector.
    2. ``variable_mappings.json`` is a fallback for variables that never appear
       in a shared constraint row.

    Only invertible single-term maps are emitted; decomposition/auxiliary
    variants (``_d``, ``_h``, ``_f``, ``_g``) need real elimination and are left
    to structural handling.
    """
    inferred = infer_diagonal_substitutions(reference_ir, candidate_ir)

    reference_terms = _read_variable_terms(reference_dir / VARIABLE_MAPPINGS_FILENAME)
    candidate_terms = _read_variable_terms(candidate_dir / VARIABLE_MAPPINGS_FILENAME)
    # Bare original reference (two-roots layout): the published mapping keys are
    # the original names, so the reference side is the identity.
    bare_reference = not reference_terms
    declared: Dict[str, Dict[str, Any]] = {}
    if candidate_terms:
        for original, cand_terms in candidate_terms.items():
            ref_terms = [(1.0, original)] if bare_reference else reference_terms.get(original)
            if not ref_terms or len(cand_terms) != 1 or len(ref_terms) != 1:
                continue
            cand_coeff, cand_var = cand_terms[0]
            ref_coeff, ref_var = ref_terms[0]
            if cand_coeff == 0 or not math.isfinite(cand_coeff):
                continue
            cand_name = _sanitize_lp_name(cand_var)
            ref_name = _sanitize_lp_name(ref_var)
            if cand_name not in candidate_ir.variables or ref_name not in reference_ir.variables:
                continue
            scale = ref_coeff / cand_coeff
            if not math.isfinite(scale) or scale == 0:
                continue
            proposed = {"coeffs": {ref_name: _snap_scale(scale)}, "constant": 0.0}
            existing = declared.get(cand_name)
            if existing is not None and existing != proposed:
                declared.pop(cand_name, None)
                continue
            declared[cand_name] = proposed

    # Indexed expansion: base-level map ("PackageCount -> b") onto the indexed
    # variables the LP files actually contain ("PackageCount[0] -> b[0]").
    if candidate_terms:
        for original, cand_terms in candidate_terms.items():
            if len(cand_terms) != 1:
                continue
            ref_terms = [(1.0, original)] if bare_reference else reference_terms.get(original)
            if not ref_terms or len(ref_terms) != 1:
                continue
            cand_coeff, cand_base = cand_terms[0]
            ref_coeff, ref_base = ref_terms[0]
            if cand_coeff == 0 or not math.isfinite(cand_coeff):
                continue
            cand_indexed = _indexed_targets(_sanitize_lp_name(cand_base),
                                            candidate_ir.variables)
            ref_indexed = _indexed_targets(_sanitize_lp_name(ref_base),
                                           reference_ir.variables)
            if not cand_indexed or not ref_indexed:
                continue
            for index in set(cand_indexed) & set(ref_indexed):
                cand_name = cand_indexed[index]
                ref_name = ref_indexed[index]
                scale = ref_coeff / cand_coeff
                if not math.isfinite(scale) or scale == 0 or cand_name in declared:
                    continue
                declared[cand_name] = {
                    "coeffs": {ref_name: _snap_scale(scale)}, "constant": 0.0}

    # The published ``variable_mappings.json`` is the *declared* correspondence
    # for the mapping-provided experiment (§9.1 D1), so it takes precedence over
    # the row-inferred scale. Inferring from rows can be right for a pure rescale
    # but wrong for ``_i`` (objective and constraints scaled differently), which
    # would let the engine certify a pair the declared correspondence rejects.
    # Inferred values only fill variables the published map does not cover.
    combined = dict(inferred)
    combined.update(declared)
    return combined


# ---------------------------------------------------------------------------
# Auxiliary-variable elimination (presolve-style, certificate-backed)
# ---------------------------------------------------------------------------
def _expression_from_terms(terms: List[Tuple[float, str]]) -> Dict[str, float]:
    coeffs: Dict[str, float] = {}
    for coeff, name in terms:
        coeffs[name] = coeffs.get(name, 0.0) + coeff
    return {name: c for name, c in coeffs.items() if c != 0}


def eliminate_defined_auxiliaries(
    candidate_ir: CanonicalIR,
    reference_ir: Optional[CanonicalIR] = None,
    relative_tolerance: float = 1e-9,
) -> Tuple[CanonicalIR, Dict[str, Dict[str, Any]]]:
    """Substitute variables that are *defined* by an equality row out of the model.

    EquivaFormulation's ``_f`` adds ``zed = objective`` and ``_g`` adds slack
    definitions like ``slack_1 = 10a + 7i - 30``.  Both are equalities whose
    auxiliary appears in no other row, so the auxiliary (and its defining row)
    can be removed exactly.  The eliminated variable is recorded as a
    candidate->reference substitution when it maps onto reference variables, so
    ``zed`` (which only carried the objective) is dropped and the recognised
    substitution lets EquiCEval compare the true objective.

    A variable is only eliminated when it is *auxiliary*: it is absent from the
    reference IR (when one is supplied) or appears in the objective but in no
    reference variable set.  This avoids consuming genuine model variables that
    merely happen to appear in a single row.
    """
    reference_names = set(reference_ir.variables) if reference_ir is not None else set()
    variables = dict(candidate_ir.variables)
    constraints = list(candidate_ir.constraints)
    objective_coeffs = dict(candidate_ir.objective_coeffs)
    objective_constant = candidate_ir.objective_constant
    substitutions: Dict[str, Dict[str, Any]] = {}

    changed = True
    while changed:
        changed = False
        usage: Dict[str, int] = {}
        for c in constraints:
            for v in c.coeffs:
                usage[v] = usage.get(v, 0) + 1
        for c in constraints:
            if c.sense != "==" or not c.coeffs:
                continue
            defined = [
                v for v in c.coeffs
                if usage.get(v, 0) == 1
                and (not reference_names or v not in reference_names)
                # A [0, inf) variable that is *not* in the objective is a slack,
                # already handled by eliminate_slack_variables; deleting its row
                # here would drop a genuine inequality.  A defined objective
                # alias such as `zed` carries an objective coefficient and is
                # safe to substitute out.
                and (v in objective_coeffs
                     or not (variables.get(v) is not None
                             and variables[v].lower_bound == 0
                             and float(variables[v].upper_bound) == float("inf")))
            ]
            if len(defined) != 1:
                continue
            target = defined[0]
            # Removing an integer/binary auxiliary through a linear equality
            # may erase a lattice restriction. Keep it unless a dedicated
            # integer-preservation certificate is available.
            target_var = variables.get(target)
            if (target_var is not None
                    and target_var.var_type.lower() in {"integer", "int", "binary", "bin"}):
                continue
            coeff = c.coeffs[target]
            if abs(coeff) <= relative_tolerance:
                continue
            # target = -(sum(other coeffs) + constant) / coeff
            expr_coeffs = {v: -a / coeff for v, a in c.coeffs.items() if v != target}
            expr_constant = -c.constant / coeff
            # Substitute target into every other row and the objective.
            def substitute(coeffs: Dict[str, float], constant: float):
                if target not in coeffs:
                    return coeffs, constant
                factor = coeffs[target]
                new_coeffs = {v: a for v, a in coeffs.items() if v != target}
                for v, a in expr_coeffs.items():
                    new_coeffs[v] = new_coeffs.get(v, 0.0) + factor * a
                return ({v: a for v, a in new_coeffs.items()
                         if abs(a) > relative_tolerance},
                        constant + factor * expr_constant)

            new_constraints = []
            for other in constraints:
                if other is c:
                    continue
                nc, nconst = substitute(dict(other.coeffs), other.constant)
                new_constraints.append(CanonicalConstraint(
                    other.name, nc, nconst, other.sense))
            constraints = new_constraints
            objective_coeffs, objective_constant = substitute(
                objective_coeffs, objective_constant)
            variables.pop(target, None)
            substitutions[target] = {
                "coeffs": expr_coeffs, "constant": expr_constant}
            changed = True
            break

    reduced = CanonicalIR(
        candidate_ir.problem_name, variables, constraints,
        candidate_ir.objective_sense, objective_coeffs, objective_constant,
        metadata=dict(candidate_ir.metadata),
    )
    return reduced, substitutions


def eliminate_slack_variables(
    candidate_ir: CanonicalIR,
    reference_ir: CanonicalIR,
    relative_tolerance: float = 1e-9,
) -> Tuple[CanonicalIR, Dict[str, Dict[str, Any]]]:
    """Recover a reference inequality hidden as an equality with a slack.

    ``_g`` rewrites ``g(x) >= b`` as ``g(x) - s = b`` and ``h(x) <= d`` as
    ``h(x) + s = d``, in both cases with ``s >= 0``.  The slack is nonnegative,
    appears in exactly one equality and carries a *unit* coefficient (either
    sign); the row is then the original inequality and the slack can be dropped.
    A free auxiliary with a positive coefficient (e.g. ``zed``) is a defined
    objective alias, not a slack, and is left to
    ``eliminate_defined_auxiliaries``.
    """
    constraints: List[CanonicalConstraint] = []
    variables = dict(candidate_ir.variables)
    dropped: Dict[str, Dict[str, Any]] = {}
    usage: Dict[str, int] = {}
    for c in candidate_ir.constraints:
        for v in c.coeffs:
            usage[v] = usage.get(v, 0) + 1

    for c in candidate_ir.constraints:
        slack = None
        if c.sense == "==":
            candidates = [
                v for v in c.coeffs
                if usage.get(v, 0) == 1
                and abs(abs(c.coeffs[v]) - 1.0) <= relative_tolerance
                # A variable that carries an objective coefficient is a defined
                # objective alias (e.g. `zed`), not a slack.
                and v not in candidate_ir.objective_coeffs
            ]
            if len(candidates) == 1:
                name = candidates[0]
                var = candidate_ir.variables.get(name)
                if (var is not None and var.lower_bound == 0
                        and float(var.upper_bound) == float("inf")
                        and var.var_type.lower() not in {"integer", "int", "binary", "bin"}):
                    slack = name
        if slack is None:
            constraints.append(c)
            continue
        coeff = c.coeffs[slack]
        body = {v: a for v, a in c.coeffs.items() if v != slack}
        # body - s = b  =>  body >= b ;  body + s = b  =>  body <= b
        # In canonical convention sum(a*x) + constant sense 0:
        constant = c.constant
        sense = ">=" if coeff < 0 else "<="
        constraints.append(CanonicalConstraint(c.name, body, constant, sense))
        variables.pop(slack, None)
        dropped[slack] = {"coeffs": {v: -a / coeff for v, a in body.items()},
                          "constant": -constant / coeff}

    reduced = CanonicalIR(
        candidate_ir.problem_name, variables, constraints,
        candidate_ir.objective_sense, dict(candidate_ir.objective_coeffs),
        candidate_ir.objective_constant, metadata=dict(candidate_ir.metadata),
    )
    return reduced, dropped


def _reference_term_map(
    reference_dir: Path,
    candidate_dir: Path,
    reference_ir: CanonicalIR,
    candidate_ir: CanonicalIR,
) -> Dict[str, Dict[str, float]]:
    """Reference variable -> linear combination of candidate variables.

    Built from a symmetric reading of the published ``variable_mappings.json``.
    Each file encodes ``original = sum coeff * variant_var``.  Composing the
    reference and candidate files gives
    ``reference_var = sum (cand_coeff / ref_coeff) * candidate_var``
    (e.g. ``g = g0 + 10 g1``).  Used to recognise grouped reconstruction.
    """
    reference_terms = _read_variable_terms(reference_dir / VARIABLE_MAPPINGS_FILENAME)
    candidate_terms = _read_variable_terms(candidate_dir / VARIABLE_MAPPINGS_FILENAME)
    # A bare original reference (two-roots layout) carries no mapping file: it
    # already *is* the original variable space, so treat it as the identity and
    # expand the published base-level map onto the indexed variables the LP
    # files actually contain (PackageCount[0] = b1[0] + b2[0]).
    bare_reference = not reference_terms
    result: Dict[str, Dict[str, float]] = {}
    if bare_reference:
        for ref_name in reference_ir.variables:
            base, index = _split_base_index(ref_name)
            cand_terms = candidate_terms.get(base)
            if not cand_terms:
                return {}
            coeffs: Dict[str, float] = {}
            for cand_coeff, cand_base in cand_terms:
                cand_sanitized = _sanitize_lp_name(cand_base)
                targets = _indexed_targets(cand_sanitized, candidate_ir.variables)
                cand_full = targets.get(index)
                if cand_full is None and not index and cand_sanitized in candidate_ir.variables:
                    cand_full = cand_sanitized
                if cand_full is None:
                    return {}
                coeffs[cand_full] = coeffs.get(cand_full, 0.0) + cand_coeff
            coeffs = {v: c for v, c in coeffs.items() if c != 0}
            if not coeffs:
                return {}
            result[ref_name] = {v: _snap_scale(c) for v, c in coeffs.items()}
        # Projection is used for every published map: it is exact algebra and
        # reproduces the candidate rows, so it is safer than inferred
        # substitutions. Coefficients are snapped to short rationals to remove
        # float noise (e.g. 0.1*3 -> 0.30000000000000004) that would otherwise
        # make the engine reject an exact comparison.
        return result

    for original, cand_terms in candidate_terms.items():
        ref_terms = reference_terms.get(original)
        if not ref_terms or len(ref_terms) != 1:
            continue
        ref_coeff, ref_var = ref_terms[0]
        if ref_coeff == 0:
            continue
        ref_name = _sanitize_lp_name(ref_var)
        if ref_name not in reference_ir.variables:
            continue
        coeffs = {}
        valid = True
        for cand_coeff, cand_var in cand_terms:
            cand_name = _sanitize_lp_name(cand_var)
            if cand_name not in candidate_ir.variables:
                valid = False
                break
            coeffs[cand_name] = coeffs.get(cand_name, 0.0) + cand_coeff / ref_coeff
        if valid and coeffs:
            result[ref_name] = {v: _snap_scale(c) for v, c in coeffs.items()}
    return result


def _row_implied_by_candidate_bounds(
    expr: Dict[str, float],
    rhs: float,
    candidate_ir: CanonicalIR,
    tolerance: float = 1e-9,
) -> bool:
    """True when ``expr <= rhs`` holds for every point in the candidate's box.

    Used to avoid emitting a reference-bound row (e.g. ``i1 + i2 >= 0``) that is
    already implied by the candidate's own bounds, which would otherwise make
    ``_rows_reproduce`` reject an otherwise exact grouped projection.
    """
    maximum = 0.0
    for name, coeff in expr.items():
        var = candidate_ir.variables.get(name)
        if var is None:
            return False
        lb, ub = var.lower_bound, var.upper_bound
        maximum += coeff * (ub if coeff >= 0 else lb)
    return math.isfinite(maximum) and maximum <= rhs + tolerance


def _project_grouped_reference_to_candidate(
    reference_ir: CanonicalIR,
    candidate_ir: CanonicalIR,
    term_map: Dict[str, Dict[str, float]],
    relative_tolerance: float,
) -> Optional[CanonicalIR]:
    """Rewrite a reference whose variables are linear combinations of candidates.

    Handles ``_d`` (base-10 digit substitution) and ``_h`` (linear substitution)
    where ``reference_var = sum a_i * candidate_var`` with more than one term.
    The candidate's own variable domain is used for the image (the comparison is
    in candidate space and the caller only accepts the projection when it
    reproduces the candidate rows), while finite reference bounds are carried as
    linear rows. Integer roots require integral coefficients and integer
    candidate variables so the lattice is preserved.
    """
    image = {v for expr in term_map.values() for v in expr}
    if not image or any(v not in candidate_ir.variables for v in image):
        return None
    for ref_name, expr in term_map.items():
        root = reference_ir.variables[ref_name]
        if root.var_type.lower() not in {"integer", "int", "binary", "bin"}:
            continue
        for cand_name, coeff in expr.items():
            cand_var = candidate_ir.variables[cand_name]
            if cand_var.var_type.lower() not in {"integer", "int", "binary", "bin"}:
                return None
            if not _close(coeff, round(coeff), relative_tolerance):
                return None

    def rewrite(coeffs: Dict[str, float]):
        out: Dict[str, float] = {}
        for ref_name, coeff in coeffs.items():
            for cand_name, factor in term_map[ref_name].items():
                out[cand_name] = out.get(cand_name, 0.0) + coeff * factor
        return {v: _snap_scale(c) for v, c in out.items()
                if abs(c) > relative_tolerance}

    variables = {name: candidate_ir.variables[name] for name in sorted(image)}
    constraints: List[CanonicalConstraint] = []
    for c in reference_ir.constraints:
        constraints.append(CanonicalConstraint(c.name, rewrite(c.coeffs),
                                               c.constant, c.sense))
    for ref_name, var in reference_ir.variables.items():
        if math.isfinite(var.lower_bound):
            # Row stored as ``expr + lower_bound <= 0`` i.e. ``expr <= -lower_bound``.
            expr = rewrite({ref_name: -1.0})
            if not _row_implied_by_candidate_bounds(
                    expr, -var.lower_bound, candidate_ir):
                constraints.append(CanonicalConstraint(
                    f"bound_lo_{ref_name}", expr, var.lower_bound, "<="))
        if math.isfinite(var.upper_bound):
            # Row stored as ``expr - upper_bound <= 0`` i.e. ``expr <= upper_bound``.
            expr = rewrite({ref_name: 1.0})
            if not _row_implied_by_candidate_bounds(
                    expr, var.upper_bound, candidate_ir):
                constraints.append(CanonicalConstraint(
                    f"bound_hi_{ref_name}", expr, -var.upper_bound, "<="))
    return CanonicalIR(
        reference_ir.problem_name,
        variables=variables,
        constraints=constraints,
        objective_sense=reference_ir.objective_sense,
        objective_coeffs=rewrite(reference_ir.objective_coeffs),
        objective_constant=reference_ir.objective_constant,
        metadata=dict(reference_ir.metadata),
    )


def project_reference_to_candidate(
    reference_ir: CanonicalIR,
    candidate_ir: CanonicalIR,
    term_map: Dict[str, Dict[str, float]],
    relative_tolerance: float = 1e-9,
) -> Optional[CanonicalIR]:
    """Rewrite the reference IR into the candidate's variable space.

    The current adapter accepts only a complete one-to-one coordinate change.
    A grouped expression such as ``x = x_0 + 10 x_1`` is useful metadata, but
    is not by itself an invertible map: copying the candidate digits' bounds or
    integrality into the reference can manufacture false equivalence. Such a
    pair therefore remains unsupported here until it has a separate,
    certificate-backed projection.

    Returns ``None`` when the map is incomplete or the rewritten reference does
    not reproduce the candidate's rows, in which case the caller keeps the
    original pair for honest ``unresolved`` handling.
    """
    if set(term_map) != set(reference_ir.variables):
        return None
    if any(not expr for expr in term_map.values()):
        return None
    if any(len(expr) != 1 for expr in term_map.values()):
        return _project_grouped_reference_to_candidate(
            reference_ir, candidate_ir, term_map, relative_tolerance)

    targets = [next(iter(expr)) for expr in term_map.values()]
    if len(set(targets)) != len(targets):
        return None

    transformed_variables: Dict[str, CanonicalVariable] = {}
    for ref_name, expr in term_map.items():
        cand_name, scale = next(iter(expr.items()))
        root = reference_ir.variables[ref_name]
        if scale == 0:
            return None
        if (root.var_type.lower() in {"integer", "int", "binary", "bin"}
                and not _close(abs(scale), 1.0, relative_tolerance)):
            return None
        lower, upper = root.lower_bound / scale, root.upper_bound / scale
        if scale < 0:
            lower, upper = upper, lower
        transformed_variables[cand_name] = CanonicalVariable(
            name=cand_name,
            var_type=root.var_type,
            lower_bound=lower,
            upper_bound=upper,
            provenance=root.provenance,
            semantic_role=root.semantic_role,
            requirement_id=root.requirement_id,
            unit=root.unit,
            is_auxiliary=root.is_auxiliary,
            index_sets=list(root.index_sets),
        )

    def rewrite(coeffs: Dict[str, float], constant: float):
        out: Dict[str, float] = {}
        for ref_name, coeff in coeffs.items():
            for cand_name, factor in term_map[ref_name].items():
                out[cand_name] = out.get(cand_name, 0.0) + coeff * factor
        return ({v: _snap_scale(c) for v, c in out.items()
                 if abs(c) > relative_tolerance}, constant)

    constraints = []
    for c in reference_ir.constraints:
        coeffs, constant = rewrite(c.coeffs, c.constant)
        constraints.append(CanonicalConstraint(c.name, coeffs, constant, c.sense))
    objective_coeffs, objective_constant = rewrite(
        reference_ir.objective_coeffs, reference_ir.objective_constant)

    return CanonicalIR(
        reference_ir.problem_name,
        variables=transformed_variables,
        constraints=constraints,
        objective_sense=reference_ir.objective_sense,
        objective_coeffs=objective_coeffs,
        objective_constant=objective_constant,
        metadata=dict(reference_ir.metadata),
    )


def _row_key(coeffs: Dict[str, float], constant: float, sense: str,
             tolerance: float = 1e-9) -> Tuple[Any, ...]:
    """Sign/scale-invariant key for a row, used to match rewritten rows.

    Uses the paper's normalization ``q_c = max(||a||_1, |b|)`` with **no floor
    of 1** (§453-454, §473-474): a floor would make ``x <= 1`` and
    ``0.1 x <= 0.1`` normalize differently.
    """
    if sense == ">=":
        coeffs = {v: -c for v, c in coeffs.items()}
        constant = -constant
        sense = "<="
    scale = max(sum(abs(c) for c in coeffs.values()), abs(constant))
    if scale == 0:
        return ("trivial",)
    return (
        sense,
        tuple(sorted((v, round(c / scale, 9)) for v, c in coeffs.items())),
        round(constant / scale, 9),
    )


def _rows_reproduce(reference_ir: CanonicalIR, candidate_ir: CanonicalIR) -> bool:
    """True when the two IRs expose the same set of rows (normalized).

    Used to accept a grouped reference projection only when it exactly
    reproduces the candidate's constraints, so a wrong map can never silently
    certify a pair.
    """
    def merged(ir):
        return {_row_key(c.coeffs, c.constant, c.sense)
                for c in ir.merge_bounds_into_constraints().constraints}
    return merged(reference_ir) == merged(candidate_ir)


def constant_objective_certificate(
    reference_ir: CanonicalIR,
    candidate_ir: CanonicalIR,
) -> Optional[Dict[str, Any]]:
    """Sound non-equivalence certificate for a constant-objective candidate.

    Under ``feasible_set_and_argmin`` semantics, if the candidate's objective is
    constant over every *free* variable (all nonzero objective coefficients sit
    on variables fixed to a constant), then every feasible candidate point is an
    optimizer: its argmin set equals its whole feasible set.  A reference whose
    objective is *not* constant and is bounded has an argmin set that is a proper
    face.  No feasibility-preserving, optimality-preserving correspondence can
    map a proper face onto a full-dimensional feasible set, so the two
    formulations are not equivalent under the argmin contract.

    Returns a certificate dict (verdict ``disproved``) or ``None`` when the
    pattern does not hold.  This fires only on the ``_k`` feasibility-problem
    family and never on the equivalent variants (verified over the archive).
    """
    fixed = {
        name for name, var in candidate_ir.variables.items()
        if var.lower_bound == var.upper_bound
    }
    free_objective = {
        name: coeff for name, coeff in candidate_ir.objective_coeffs.items()
        if abs(coeff) > _COEFF_TOL and name not in fixed
    }
    if free_objective:
        return None
    reference_objective = {
        name: coeff for name, coeff in reference_ir.objective_coeffs.items()
        if abs(coeff) > _COEFF_TOL
    }
    if not reference_objective:
        return None
    if not _reference_objective_varies(reference_ir):
        return None
    return {
        "verdict": "disproved",
        "kind": "constant_objective_feasibility_problem",
        "contract": ["feasible_set_and_objective_affine",
                     "feasible_set_and_argmin"],
        "reason": ("Candidate objective is constant over all free variables, so it "
                   "cannot be related to a non-constant reference objective by any "
                   "positive affine map (min/max) nor by an argmin correspondence."),
        "candidate_free_objective_vars": sorted(free_objective),
        "reference_objective_vars": sorted(reference_objective),
    }


def _reference_objective_varies(reference_ir: CanonicalIR) -> bool:
    """True when the reference objective takes at least two values on its feasible set.

    Guards the constant-objective certificate: on a singleton feasible set every
    objective is trivially constant, so a constant candidate could still be
    affine-equivalent. Uses the same MILP oracle as the relabel path.
    """
    from src.benchmark.independent_relabel import _max_linear, _names_union

    names = _names_union(reference_ir, reference_ir)
    sign = 1.0 if reference_ir.objective_sense != "maximize" else -1.0
    coeffs = {n: sign * a for n, a in reference_ir.objective_coeffs.items()}
    status_hi, max_pos, _ = _max_linear(reference_ir, coeffs, names)
    status_lo, max_neg, _ = _max_linear(
        reference_ir, {n: -c for n, c in coeffs.items()}, names)
    if status_hi == "unbounded" or status_lo == "unbounded":
        return True
    if status_hi == "ok" and status_lo == "ok":
        return (max_pos + max_neg) > 1e-9
    return False


def _wl_signature(ir: CanonicalIR, iterations: int = 3) -> Tuple[Any, ...]:
    """Weisfeiler-Lehman colour histogram of the variable-constraint graph.

    Purely diagnostic.  Equivalent variants also restructure the graph, so a
    differing signature is *evidence* of a structural mismatch, never a proof of
    non-equivalence; the caller must not turn it into a `disproved` verdict.
    """
    labels: Dict[str, str] = {}
    adjacency: Dict[str, List[Tuple[str, float]]] = {}
    for name, var in ir.variables.items():
        labels[name] = (f"V:{var.var_type.lower()}:{round(var.lower_bound, 6)}:"
                        f"{round(var.upper_bound, 6)}:"
                        f"{round(ir.objective_coeffs.get(name, 0.0), 6)}")
        adjacency.setdefault(name, [])
    for c in ir.constraints:
        labels[c.name] = f"C:{c.sense}:{round(c.constant, 6)}"
        adjacency.setdefault(c.name, [])
        for name, coeff in c.coeffs.items():
            adjacency[c.name].append((name, round(coeff, 9)))
            adjacency[name].append((c.name, round(coeff, 9)))

    colours = dict(labels)
    for _ in range(iterations):
        signatures = {
            node: (colours[node], tuple(sorted(
                (colours[nbr], edge) for nbr, edge in adjacency[node])))
            for node in colours
        }
        palette = {sig: index for index, sig in enumerate(sorted(set(signatures.values())))}
        colours = {node: str(palette[sig]) for node, sig in signatures.items()}
    histogram: Dict[str, int] = {}
    for colour in colours.values():
        histogram[colour] = histogram.get(colour, 0) + 1
    return tuple(sorted(histogram.items()))


def lp_to_canonical_ir(lp_path: Path, problem_name: Optional[str] = None) -> CanonicalIR:
    """Parse a Gurobi ``model.lp`` file into a ``CanonicalIR`` via HiGHS.

    Constraint convention matches ``CanonicalConstraint``: the stored
    ``constant`` is ``-rhs`` so that ``sum(coeffs*x) + constant sense 0``.
    """
    import highspy
    import tempfile

    lp_path = Path(lp_path)
    sanitized = _sanitize_lp_names(lp_path.read_text())
    handle = tempfile.NamedTemporaryFile("w", suffix=".lp", delete=False)
    try:
        handle.write(sanitized)
        handle.close()
        highs = highspy.Highs()
        highs.setOptionValue("output_flag", False)
        status = highs.readModel(handle.name)
    finally:
        Path(handle.name).unlink(missing_ok=True)
    if status != highspy.HighsStatus.kOk:
        raise ValueError(f"HiGHS could not read {lp_path}: {status}")

    lp = highs.getLp()
    n_col, n_row = lp.num_col_, lp.num_row_
    names = [str(n) for n in lp.col_names_]

    variables: Dict[str, CanonicalVariable] = {}
    integrality = lp.integrality_
    for j, name in enumerate(names):
        lower = float(lp.col_lower_[j])
        upper = float(lp.col_upper_[j])
        kind = integrality[j] if j < len(integrality) else highspy.HighsVarType.kContinuous
        variables[name] = CanonicalVariable(
            name=name,
            var_type=_var_type(kind, lower, upper),
            lower_bound=lower,
            upper_bound=upper,
            provenance="equivaformulation",
        )

    # hiGHS stores the matrix column-wise; invert it into row coefficient maps.
    start = list(lp.a_matrix_.start_)
    index = list(lp.a_matrix_.index_)
    value = list(lp.a_matrix_.value_)
    rows: List[Dict[str, float]] = [dict() for _ in range(n_row)]
    for j in range(n_col):
        for k in range(start[j], start[j + 1]):
            rows[index[k]][names[j]] = float(value[k])

    row_names = [str(n) for n in lp.row_names_]
    constraints: List[CanonicalConstraint] = []
    for i in range(n_row):
        coeffs = {v: c for v, c in rows[i].items() if abs(c) > _COEFF_TOL}
        lower = float(lp.row_lower_[i])
        upper = float(lp.row_upper_[i])
        base = row_names[i] if i < len(row_names) else f"R{i}"
        if math.isinf(lower) and math.isinf(upper):
            continue  # free row carries no information
        if lower == upper:
            constraints.append(CanonicalConstraint(base, coeffs, -lower, "=="))
        elif math.isinf(upper):
            constraints.append(CanonicalConstraint(base, coeffs, -lower, ">="))
        elif math.isinf(lower):
            constraints.append(CanonicalConstraint(base, coeffs, -upper, "<="))
        else:  # ranged row: split into two one-sided rows
            constraints.append(CanonicalConstraint(f"{base}_lo", coeffs, -lower, ">="))
            constraints.append(CanonicalConstraint(f"{base}_hi", dict(coeffs), -upper, "<="))

    objective_coeffs = {
        names[j]: float(lp.col_cost_[j])
        for j in range(n_col)
        if abs(float(lp.col_cost_[j])) > _COEFF_TOL
    }
    objective_sense = "minimize" if lp.sense_.name == "kMinimize" else "maximize"

    return CanonicalIR(
        problem_name or lp_path.parent.name,
        variables,
        constraints,
        objective_sense,
        objective_coeffs,
        objective_constant=float(lp.offset_),
    )


# ---------------------------------------------------------------------------
# Archive -> pairs -> records
# ---------------------------------------------------------------------------
def discover_pairs(
    data_root: Path,
    instance_ids: Optional[Iterable[str]] = None,
    suffixes: Optional[Iterable[str]] = None,
    reference_root: Optional[Path] = None,
) -> List[EquivaPair]:
    """Enumerate (reference, candidate) pairs from the extracted archive.

    ``reference_root`` opts into the "two roots" layout: the reference is the
    original problem O materialised as ``<reference_root>/<id>/model.lp`` (see
    ``experiments/gen_equivaformulation_root_lp.py``), while candidates stay in
    ``data_root/<id>/<id>_<suffix>/model.lp``.  In that layout every published
    variant including the rename-only ``_c`` is a candidate.  When
    ``reference_root`` is ``None`` the legacy in-archive ``_c`` proxy is used and
    ``_c`` is excluded (it would be compared against itself).
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"EquivaFormulation data root not found: {data_root}")
    reference_root = Path(reference_root) if reference_root is not None else None
    wanted = set(instance_ids) if instance_ids is not None else None
    if suffixes is None:
        suffixes = (ALL_CANDIDATE_SUFFIXES if reference_root is not None
                    else CANDIDATE_SUFFIXES)
    suffixes = tuple(suffixes)

    pairs: List[EquivaPair] = []
    for instance_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        instance_id = instance_dir.name
        if wanted is not None and instance_id not in wanted:
            continue
        if reference_root is not None:
            reference = reference_root / instance_id / "model.lp"
        else:
            reference = instance_dir / f"{instance_id}_{REFERENCE_SUFFIX}" / "model.lp"
        if not reference.exists():
            continue
        for suffix in suffixes:
            candidate = instance_dir / f"{instance_id}_{suffix}" / "model.lp"
            if candidate.exists():
                pairs.append(EquivaPair(instance_id, suffix, reference, candidate,
                                        suffix in EQUIVALENT_SUFFIXES))
    return pairs


def build_records(
    data_root: Path,
    instance_ids: Optional[Iterable[str]] = None,
    suffixes: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    reference_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Convert the archive into EquiCEval records (``reference_ir``/``candidate_ir``)."""
    pairs = discover_pairs(data_root, instance_ids=instance_ids, suffixes=suffixes,
                           reference_root=reference_root)
    if limit is not None:
        pairs = pairs[:limit]

    records: List[Dict[str, Any]] = []
    for pair in pairs:
        reference_ir = lp_to_canonical_ir(pair.reference_path, pair.family)
        candidate_ir = lp_to_canonical_ir(pair.candidate_path, pair.family)
        # Raw IRs as an external baseline (e.g. the original-paper reference-form
        # method) would see them, before EquiCEval's own reduction/projection.
        raw_reference_ir = reference_ir
        raw_candidate_ir = candidate_ir
        provided_substitutions: Dict[str, Any] = {}
        reference_projection: Optional[str] = None
        reference_projection_map: Dict[str, Dict[str, float]] = {}
        auxiliary_eliminations: Dict[str, Dict[str, Any]] = {}
        # In the legacy ``_c``-proxy layout a suffix-``c`` candidate is the
        # reference itself; in the two-roots layout ``_c`` is a real candidate.
        is_self_reference = reference_root is None and pair.suffix == REFERENCE_SUFFIX
        if not is_self_reference:
            # A verified one-to-one coordinate change may rewrite the reference
            # into the candidate's variable space. Grouped maps deliberately
            # remain unsupported: their extra domain restrictions need an
            # explicit projection certificate, not copied candidate bounds.
            term_map = _reference_term_map(
                pair.reference_path.parent, pair.candidate_path.parent,
                reference_ir, candidate_ir)
            projected_reference = project_reference_to_candidate(
                reference_ir, candidate_ir, term_map)
            if (projected_reference is not None
                    and _rows_reproduce(projected_reference, candidate_ir)):
                reference_ir = projected_reference
                reference_projection = "grouped-substitution"
                reference_projection_map = term_map
            else:
                # Slack definitions first (inequalities in disguise), then the
                # remaining singleton equality (e.g. the ``zed`` objective alias).
                candidate_ir, slack_eliminations = eliminate_slack_variables(
                    candidate_ir, reference_ir)
                candidate_ir, defined_eliminations = eliminate_defined_auxiliaries(
                    candidate_ir, reference_ir)
                auxiliary_eliminations.update(slack_eliminations)
                auxiliary_eliminations.update(defined_eliminations)
                provided_substitutions = affine_substitutions_for_pair(
                    pair.reference_path.parent, pair.candidate_path.parent,
                    reference_ir, candidate_ir,
                )
        # Diagnostic only: the WL signature is recorded as evidence of a
        # structural mismatch (mostly _j/_k).  It is deliberately not used to
        # return `disproved`, since equivalent variants also restructure.
        structural_match = _wl_signature(reference_ir) == _wl_signature(candidate_ir)
        # Sound certificate: a constant-objective candidate under the argmin
        # contract cannot be equivalent to a non-constant-objective reference.
        certificate = constant_objective_certificate(reference_ir, candidate_ir)
        records.append({
            "record_id": pair.record_id,
            "family": pair.family,
            "base_problem_name": pair.family,
            "provenance": "equivaformulation",
            "source_instance_id": pair.instance_id,
            "variation_suffix": pair.suffix,
            "transformation_label": pair.transformation_label,
            "mutation_label": None,
            "contamination_stratum": "public_base",
            "is_semantically_equivalent": pair.equivalent,
            "provided_substitutions": provided_substitutions,
            "reference_projection": reference_projection,
            "reference_projection_map": reference_projection_map,
            "auxiliary_eliminations": auxiliary_eliminations,
            "source_artifacts": {
                "reference_lp": {
                    "path": str(pair.reference_path),
                    "sha256": file_sha256(pair.reference_path),
                },
                "candidate_lp": {
                    "path": str(pair.candidate_path),
                    "sha256": file_sha256(pair.candidate_path),
                },
                "candidate_variable_mapping": {
                    "path": str(pair.candidate_path.parent / VARIABLE_MAPPINGS_FILENAME),
                    "sha256": file_sha256(
                        pair.candidate_path.parent / VARIABLE_MAPPINGS_FILENAME),
                },
            },
            "structural_signature_match": structural_match,
            "certificate": certificate,
            "raw_reference_ir": ir_to_dict(raw_reference_ir),
            "raw_candidate_ir": ir_to_dict(raw_candidate_ir),
            "reference_ir": ir_to_dict(reference_ir),
            "candidate_ir": ir_to_dict(candidate_ir),
        })
    return records


def summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    equivalent = sum(1 for r in records if r["is_semantically_equivalent"] is True)
    non_equivalent = sum(1 for r in records if r["is_semantically_equivalent"] is False)
    return {
        "source": SOURCE,
        "total_pairs": len(records),
        "equivalent": equivalent,
        "non_equivalent": non_equivalent,
        "unknown_label": len(records) - equivalent - non_equivalent,
        "instances": len({r["source_instance_id"] for r in records}),
        "by_suffix": {
            suffix: sum(1 for r in records if r["variation_suffix"] == suffix)
            for suffix in sorted({r["variation_suffix"] for r in records})
        },
    }
