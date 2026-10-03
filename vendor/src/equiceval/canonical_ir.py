"""
Typed Canonical Intermediate Representation (Canonical IR) for LP/MILP.
Performs semantic-preserving transformations:
- Normalizes affine expressions: sum(a_i * x_i) + b <= 0 or == 0
- Uses q=max(||a||_1, |b|), without a floor that would break scaling invariance
- Merges bounds and explicitly standardizes senses
"""

from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np

_NUMERIC_TOL = 1e-9  # global tolerance for bound-merging and sense checks


@dataclass
class CanonicalVariable:
    name: str
    var_type: str      # 'binary', 'integer', 'continuous'
    lower_bound: float = 0.0
    upper_bound: float = float('inf')
    provenance: str = ""
    semantic_role: str = ""
    requirement_id: str = ""
    unit: str = ""              # restored from branch `thanh` (Module 5 unit conversion)
    # Module 1 additions (upgrade paper §4.2)
    is_auxiliary: bool = False   # True for slack / Big-M / reformulation auxiliaries
    index_sets: List[str] = field(default_factory=list)  # e.g. ['i', 'j'] for x[i,j]


@dataclass
class AffineExpression:
    coeffs: Dict[str, float]
    constant: float = 0.0

    def is_affine_over(self, variable_names: Set[str], tolerance: float = 1e-9) -> bool:
        if not np.isfinite(self.constant):
            return False
        for vname, coeff in self.coeffs.items():
            if vname not in variable_names or not np.isfinite(coeff):
                return False
            if abs(coeff) <= tolerance:
                continue
        return True


@dataclass
class ProjectionCertificate:
    """
    Module 1 certificate mapping candidate variables into reference space.

    affine_substitutions are interpreted as:
        candidate_var = sum(coeffs[ref_var] * ref_var) + constant
    """
    is_verified: bool
    variable_map: Dict[str, str] = field(default_factory=dict)  # candidate var -> reference var
    affine_substitutions: Dict[str, AffineExpression] = field(default_factory=dict)
    ref_eliminations: Dict[str, AffineExpression] = field(default_factory=dict)  # ref var -> f(candidate)
    reason: str = ""
    detection_method: str = "none"
    unsupported_features: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_identity(self) -> bool:
        if not self.is_verified:
            return False
        for cand, ref in self.variable_map.items():
            if cand != ref:
                return False
            expr = self.affine_substitutions.get(cand)
            # A declared unit conversion (same name, scale != 1) is not the identity.
            if expr is not None and (expr.coeffs != {ref: 1.0} or expr.constant != 0.0):
                return False
        return True

    @property
    def is_renaming(self) -> bool:
        return self.is_verified and bool(self.variable_map) and not self.is_identity

    def has_affine_projection(self) -> bool:
        return self.is_verified and not self.unsupported_features

    def project_constraint(self, constraint: "CanonicalConstraint") -> "CanonicalConstraint":
        coeffs: Dict[str, float] = {}
        constant = Fraction(str(constraint.constant))
        for cand_var, coeff in constraint.coeffs.items():
            expr = self.affine_substitutions.get(cand_var)
            if expr is None:
                raise ValueError(f"No affine substitution for candidate variable '{cand_var}'")
            for ref_var, ref_coeff in expr.coeffs.items():
                coeffs[ref_var] = coeffs.get(ref_var, Fraction(0)) + Fraction(str(coeff)) * Fraction(str(ref_coeff))
            constant += Fraction(str(coeff)) * Fraction(str(expr.constant))

        return CanonicalConstraint(
            name=constraint.name,
            coeffs={v: _exact_decimal_float(c) for v, c in coeffs.items() if c != 0},
            constant=_exact_decimal_float(constant),
            sense=constraint.sense,
            scale_factor=constraint.scale_factor,
        )

    def project_ir(self, cand_ir: "CanonicalIR", ref_ir: "CanonicalIR") -> "CanonicalIR":
        if not self.has_affine_projection():
            raise ValueError(f"Projection is not verified: {self.reason}")

        objective_coeffs: Dict[str, float] = {}
        objective_constant = Fraction(str(cand_ir.objective_constant))
        for cand_var, coeff in cand_ir.objective_coeffs.items():
            expr = self.affine_substitutions.get(cand_var)
            if expr is None:
                raise ValueError(f"No affine substitution for objective variable '{cand_var}'")
            for ref_var, ref_coeff in expr.coeffs.items():
                objective_coeffs[ref_var] = objective_coeffs.get(ref_var, Fraction(0)) + Fraction(str(coeff)) * Fraction(str(ref_coeff))
            objective_constant += Fraction(str(coeff)) * Fraction(str(expr.constant))

        constraints = [
            self.project_constraint(c)
            for c in cand_ir.merge_bounds_into_constraints().constraints
        ]
        # Eliminated reference variables (e.g. z_ji = 1 - z_ij) must be pinned in
        # the projected candidate space, otherwise the directed query would treat
        # them as free and over-approximate the candidate feasible set.
        for ref_var, expr in self.ref_eliminations.items():
            coeffs: Dict[str, Fraction] = {ref_var: Fraction(1)}
            constant = -Fraction(str(expr.constant))
            for cand_var, coeff in expr.coeffs.items():
                sub = self.affine_substitutions.get(cand_var)
                if sub is None:
                    raise ValueError(
                        f"No affine substitution for eliminated variable '{cand_var}'"
                    )
                for rv, rc in sub.coeffs.items():
                    coeffs[rv] = coeffs.get(rv, Fraction(0)) - Fraction(str(coeff)) * Fraction(str(rc))
                constant -= Fraction(str(coeff)) * Fraction(str(sub.constant))
            constraints.append(CanonicalConstraint(
                name=f"_elim_{ref_var}",
                coeffs={v: _exact_decimal_float(c) for v, c in coeffs.items() if c != 0},
                constant=_exact_decimal_float(constant),
                sense="==",
            ))

        return CanonicalIR(
            problem_name=f"{cand_ir.problem_name}_projected",
            variables={n: replace(v, lower_bound=-float("inf"), upper_bound=float("inf"))
                       for n, v in ref_ir.variables.items()},
            constraints=constraints,
            objective_sense=cand_ir.objective_sense,
            objective_coeffs={v: _exact_decimal_float(c) for v, c in objective_coeffs.items() if c != 0},
            objective_constant=_exact_decimal_float(objective_constant),
            metadata=dict(cand_ir.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_verified": self.is_verified,
            "variable_map": dict(self.variable_map),
            "affine_substitutions": {
                name: {"coeffs": dict(expr.coeffs), "constant": expr.constant}
                for name, expr in self.affine_substitutions.items()
            },
            "ref_eliminations": {
                name: {"coeffs": dict(expr.coeffs), "constant": expr.constant}
                for name, expr in self.ref_eliminations.items()
            },
            "reason": self.reason,
            "detection_method": self.detection_method,
            "unsupported_features": list(self.unsupported_features),
            "warnings": list(self.warnings),
            "is_identity": self.is_identity,
            "is_renaming": self.is_renaming,
        }


@dataclass
class CanonicalConstraint:
    name: str
    coeffs: Dict[str, float]  # var_name -> coefficient
    constant: float           # c in sum(a_i * x_i) + c <= 0 or == 0
    sense: str                # '<=' or '=='
    scale_factor: float = 1.0

    @property
    def normalization_scale(self):
        return max(sum(abs(c) for c in self.coeffs.values()), abs(self.constant))

    def evaluate_residual(self, x_dict: Dict[str, float]) -> float:
        c = self.flip_to_leq()
        if c.sense not in ("<=", "=="):
            raise ValueError("Unsupported constraint sense")
        val = sum(coef * x_dict[v] for v, coef in c.coeffs.items()) + c.constant
        q = c.normalization_scale
        return (max(0.0, val) if c.sense == "<=" else abs(val)) / q if q else 0.0

    def flip_to_leq(self) -> 'CanonicalConstraint':
        """
        Converts a '>=' constraint to '<=' by negating coefficients and constant.
        g(x) >= 0  ≡  -g(x) <= 0
        """
        if self.sense == '>=':
            return CanonicalConstraint(
                name=self.name,
                coeffs={v: -c for v, c in self.coeffs.items()},
                constant=-self.constant,
                sense='<=',
                scale_factor=self.scale_factor,
            )
        return self

    def normalize(self) -> 'CanonicalConstraint':
        """
        Returns a normalized copy of the constraint.
        First flips '>=' to '<=', then divides by q=max(||a||_1, |b|).
        """
        # Step 0: normalize sense — only '<=', '==' are canonical
        c = self.flip_to_leq()

        l1_norm = c.normalization_scale
        if l1_norm == 0:
            return c

        norm_coeffs = {v: coeff / l1_norm for v, coeff in c.coeffs.items() if coeff != 0}
        norm_const = c.constant / l1_norm

        # Enforce canonical sign: make first non-zero coefficient positive if equality
        if c.sense == '==' and norm_coeffs:
            first_var = sorted(norm_coeffs.keys())[0]
            if norm_coeffs[first_var] < 0:
                norm_coeffs = {v: -coef for v, coef in norm_coeffs.items()}
                norm_const = -norm_const

        return CanonicalConstraint(
            name=c.name,
            coeffs=norm_coeffs,
            constant=norm_const,
            sense=c.sense,
            scale_factor=l1_norm,
        )


@dataclass
class CanonicalIR:
    problem_name: str
    variables: Dict[str, CanonicalVariable]
    constraints: List[CanonicalConstraint]
    objective_sense: str                # 'minimize' or 'maximize'
    objective_coeffs: Dict[str, float]
    objective_constant: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def merge_bounds_into_constraints(self) -> 'CanonicalIR':
        """
        Module 1, canonicalization step 5:
        Convert every finite variable bound into explicit constraints so
        that a candidate expressing ``x <= 10`` as an inequality and a reference
        storing ``upper_bound=10`` both produce the same constraint fingerprint.

        Zero lower bounds and implicit binary bounds are included.

        The resulting CanonicalIR keeps the original variable bounds AND adds
        synthetic constraints named ``_lb_{var}`` / ``_ub_{var}``.
        Existing constraints are preserved unchanged.
        """
        new_constraints: List[CanonicalConstraint] = list(self.constraints)
        existing = {(tuple(sorted(c.coeffs.items())), c.constant, c.sense)
                    for c in new_constraints}
        def add_bound(row):
            key = (tuple(sorted(row.coeffs.items())), row.constant, row.sense)
            if key not in existing:
                new_constraints.append(row)
                existing.add(key)
        for vname, var in self.variables.items():
            # Lower-bound constraint:  -x + lb <= 0  →  x >= lb
            lb = max(0.0, var.lower_bound) if var.var_type.lower() in ('binary', 'bin') else var.lower_bound
            if np.isfinite(lb):
                add_bound(
                    CanonicalConstraint(
                        name=f"_lb_{vname}",
                        coeffs={vname: -1.0},
                        constant=lb,      # -x + lb <= 0  →  x >= lb
                        sense='<=',
                    )
                )
            # Upper-bound constraint:  x - ub <= 0  →  x <= ub
            ub = min(1.0, var.upper_bound) if var.var_type.lower() in ('binary', 'bin') else var.upper_bound
            if np.isfinite(ub):
                add_bound(
                    CanonicalConstraint(
                        name=f"_ub_{vname}",
                        coeffs={vname: 1.0},
                        constant=-ub,     # x - ub <= 0  →  x <= ub
                        sense='<=',
                    )
                )
        return CanonicalIR(
            problem_name=self.problem_name,
            variables=dict(self.variables),
            constraints=new_constraints,
            objective_sense=self.objective_sense,
            objective_coeffs=dict(self.objective_coeffs),
            objective_constant=self.objective_constant,
            metadata=dict(self.metadata),
        )

    def canonicalize(self, merge_bounds: bool = True) -> 'CanonicalIR':
        """
        Applies canonicalization to all constraints and objective.

        Steps (matching upgrade paper §4.2):
          1. Optionally merge finite variable bounds as explicit constraints.
          2. Normalize each constraint (flip '>=' → '<=', q-scale, sign-fix).
          3. Preserve objective coefficients, constants, and units.
        """
        ir = self.merge_bounds_into_constraints() if merge_bounds else self
        norm_constraints = [c.normalize() for c in ir.constraints]

        # Objective values must keep their units and scale.
        norm_obj_coeffs = dict(ir.objective_coeffs)
        norm_obj_const = ir.objective_constant

        return CanonicalIR(
            problem_name=ir.problem_name,
            variables=dict(ir.variables),
            constraints=norm_constraints,
            objective_sense=ir.objective_sense,
            objective_coeffs=norm_obj_coeffs,
            objective_constant=norm_obj_const,
            metadata=dict(ir.metadata),
        )

    def unsupported_scope_features(self) -> List[str]:
        features: List[str] = []
        metadata_features = self.metadata.get("unsupported_features", [])
        if isinstance(metadata_features, str):
            metadata_features = [metadata_features]
        features.extend(str(f) for f in metadata_features)

        if self.metadata.get("nonlinear") or self.metadata.get("has_nonlinear_terms"):
            features.append("nonlinear")

        for c in self.constraints:
            if getattr(c, "nonlinear_terms", None):
                features.append(f"nonlinear:{c.name}")
            # '>='' is handled by normalize()/flip_to_leq(), not unsupported.
            # Only truly unknown senses (e.g. '!=') are unsupported.
            if c.sense not in ('<=', '==', '>='):
                features.append(f"unsupported_sense:{c.name}:{c.sense}")

        return sorted(set(features))


def check_affine_substitution(
    substitution: Any,
    ref_ir: CanonicalIR,
    tolerance: float = 1e-9,
) -> Tuple[bool, Optional[AffineExpression], str]:
    """
    Validates and normalizes one affine substitution over reference variables.
    Accepted forms:
      - AffineExpression(coeffs={...}, constant=...)
      - {"coeffs": {...}, "constant": ...}
      - ("x_ref") as a direct alias
    """
    if isinstance(substitution, str):
        expr = AffineExpression({substitution: 1.0}, 0.0)
    elif isinstance(substitution, AffineExpression):
        expr = substitution
    elif isinstance(substitution, dict):
        raw_coeffs = substitution.get("coeffs", substitution)
        raw_constant = substitution.get("constant", 0.0)
        try:
            expr = AffineExpression(
                coeffs={str(v): float(c) for v, c in raw_coeffs.items()},
                constant=float(raw_constant),
            )
        except (TypeError, ValueError, AttributeError):
            return False, None, "substitution is not a numeric affine dictionary"
    else:
        return False, None, f"unsupported substitution type: {type(substitution).__name__}"

    if not expr.is_affine_over(set(ref_ir.variables), tolerance=tolerance):
        return False, None, "substitution references unknown variables or non-finite coefficients"

    expr.coeffs = {v: c for v, c in expr.coeffs.items() if c != 0}
    return True, expr, "affine"


def build_projection_certificate(
    ref_ir: CanonicalIR, cand_ir: CanonicalIR,
    provided_substitutions: Optional[Dict[str, Any]] = None,
    tolerance: float = 1e-9,
) -> ProjectionCertificate:
    """Verify an invertible diagonal affine map on a declared common variable space.

    Explicit maps and unique requirement IDs take precedence. Shared names are
    an explicit identity convention, reported as an assumption, never inferred
    from coefficient similarity. General projections/auxiliaries are unsupported.
    """
    unsupported = ref_ir.unsupported_scope_features() + cand_ir.unsupported_scope_features()
    if unsupported:
        return ProjectionCertificate(False, reason="Unsupported scope",
                                     unsupported_features=unsupported)
    provided = provided_substitutions or {}
    if not set(provided) <= set(cand_ir.variables):
        return ProjectionCertificate(False, reason="Map contains unknown candidate variables")
    req_index = _unique_index(ref_ir.variables, "requirement_id")
    aliases = ref_ir.metadata.get("name_aliases", {})
    ref_sig_index = _unique_signature_index(ref_ir)
    ref_bin_digit_index = _unique_binary_digit_index(ref_ir)
    ref_norm_index = _unique_normalized_name_index(ref_ir)
    cand_norm_index = _unique_normalized_name_index(cand_ir)
    ref_role_index = _unique_index(ref_ir.variables, "semantic_role")
    ref_fp_index = _unique_fingerprint_index(ref_ir)
    cand_fp_index = _unique_fingerprint_index(cand_ir)
    subs, mapping, used, methods, warnings = {}, {}, set(), [], []

    def failed(reason, unsupported_features=None):
        # Retain only locally checked pairs. Partial coverage is an explanation,
        # never permission to project or certify the complete model.
        return ProjectionCertificate(
            is_verified=False,
            variable_map=dict(mapping),
            affine_substitutions=dict(subs),
            reason=reason,
            detection_method="+".join(sorted(set(methods))) or "none",
            unsupported_features=unsupported_features or [],
            warnings=list(warnings),
        )

    for name, var in cand_ir.variables.items():
        unit_applied = False
        if name in provided:
            ok, expr, reason = check_affine_substitution(provided[name], ref_ir, tolerance)
            method = "explicit-affine"
            if not ok:
                return failed(reason)
        else:
            target = req_index.get(var.requirement_id) if var.requirement_id else None
            method = "requirement_id"
            if target is None and name in aliases:
                target, method = aliases[name], "declared-alias"
            if target is None and name in ref_ir.variables:
                target, method = name, "assumed-shared-name"
                warnings.append(f"Identity of variable '{name}' is assumed from the shared-name convention.")
            if target is None:
                # Underscore/space/case-insensitive name match: x_1_1 -> x11.
                # A naming convention, not proof of meaning; only unambiguous
                # normalized names on both sides are accepted, and a bare
                # separator such as 'x_' must not collapse onto 'x' (that fuzzy
                # match is deliberately rejected by the evidence tests).
                normalized = _normalize_name(name)
                ref_name = ref_norm_index.get(normalized)
                if (ref_name is not None and cand_norm_index.get(normalized) == name
                        and _normalization_is_informative(name, ref_name)):
                    target, method = ref_name, "normalized-name"
                    warnings.append(
                        f"Identity of variable '{name}' is proposed from an underscore/space-"
                        "normalized name match; recorded as an assumption, not a verified "
                        "semantic equivalence."
                    )
            if target is None and getattr(var, "semantic_role", ""):
                # Declared semantic role (e.g. assign_A1_R1). Stronger than a bare
                # name convention but still a declared label, so recorded as an
                # assumption. The role must be unambiguous on the reference side.
                role = var.semantic_role
                if role in ref_role_index:
                    target, method = ref_role_index[role], "semantic-role"
                    warnings.append(
                        f"Identity of variable '{name}' is proposed from declared semantic_role "
                        f"'{role}'; recorded as an assumption, not a verified semantic equivalence."
                    )
            if target is None:
                # Structural index naming: x_A1_R1 -> x11 (ordered index digits).
                # A naming convention, not proof of meaning; recorded as its own
                # detection method and flagged as an assumption. Fuzzy names
                # without index digits do not match.
                signature = _index_signature(name)
                if signature is not None and signature in ref_sig_index:
                    target, method = ref_sig_index[signature], "index-signature"
                    warnings.append(
                        f"Identity of variable '{name}' is proposed from the structural "
                        "index-name convention; recorded as an assumption, not a verified "
                        "semantic equivalence."
                    )
            if target is None and var.var_type.lower() in ("binary", "bin"):
                # Base-letter-agnostic ordering binary: y_A1_A2 -> z12.
                digits = _digit_signature(name)
                if digits is not None and digits in ref_bin_digit_index:
                    target, method = ref_bin_digit_index[digits], "binary-digit-signature"
                    warnings.append(
                        f"Identity of variable '{name}' is proposed from the ordering-digit "
                        "convention; recorded as an assumption, not a verified semantic "
                        "equivalence."
                    )
            if target is None:
                # Structural coefficient fingerprint: variable type, bounds and
                # normalized row/objective participation. Name-agnostic, so it can
                # recover an arbitrary rename where names carry no shared signal.
                # Accepted only when the fingerprint is unique on BOTH sides;
                # recorded as an assumption, never proof of meaning.
                fingerprint = _variable_fingerprint(cand_ir, name)
                ref_name = ref_fp_index.get(fingerprint)
                # Do not let the name-agnostic fingerprint re-admit a fuzzy name
                # pair ('x_' vs 'x') that the normalized-name guard deliberately
                # rejects; that would bypass the evidence test's invariant.
                blocked = (ref_name is not None
                           and _normalize_name(name) == _normalize_name(ref_name)
                           and not _normalization_is_informative(name, ref_name))
                if (not blocked and cand_fp_index.get(fingerprint) == name
                        and ref_name is not None):
                    target, method = ref_name, "structural-fingerprint"
                    warnings.append(
                        f"Identity of variable '{name}' is proposed from a structural coefficient "
                        "fingerprint (type, bounds, row/objective participation); recorded as an "
                        "assumption, not a verified semantic equivalence."
                    )
            if target not in ref_ir.variables:
                return failed(f"No verified mapping for {name}")
            # Name/pattern matches are only proposals: declared index sets must
            # not contradict them before we keep the pair.
            if method in ("assumed-shared-name", "normalized-name", "index-signature",
                          "binary-digit-signature"):
                if not _index_sets_compatible(var, ref_ir.variables[target]):
                    return failed(
                        f"Index-set cardinality mismatch for {name} -> {target}",
                        ["index-cardinality-mismatch"],
                    )
            expr = AffineExpression({target: 1.0}, 0.0)
        if len(expr.coeffs) != 1:
            return failed("Only invertible diagonal affine maps supported", ["general-projection"])
        target, scale = next(iter(expr.coeffs.items()))
        if target in used or scale == 0:
            return failed("Map is not one-to-one")
        ref_var = ref_ir.variables[target]
        def kind(v):
            return {"bin": "binary", "int": "integer", "cont": "continuous", "real": "continuous"}.get(v.var_type.lower(), v.var_type.lower())
        if kind(var) != kind(ref_var):
            return failed("Variable type mismatch", ["variable-type-mismatch"])
        if name not in provided:
            # Explicitly declared units are a priority-1 source (§5.4): apply the
            # conversion and record it, or refuse a mapping whose declared units
            # conflict. Integer/binary domains cannot absorb a non-unit scale.
            unit_factor, unit_status = _unit_relation(var, ref_var)
            if unit_status == "unit-mismatch":
                return failed(
                    f"Declared units are incompatible for {name} -> {target}",
                    ["unit-mismatch"],
                )
            if unit_status == "unit-conversion":
                if kind(var) != "continuous":
                    return failed(
                        f"Unit conversion would change the {kind(var)} lattice for {name}",
                        ["unit-domain-change"],
                    )
                expr = AffineExpression({target: unit_factor}, 0.0)
                scale = unit_factor
                unit_applied = True
                warnings.append(
                    f"Unit conversion applied for '{name}': '{var.unit}' -> "
                    f"'{ref_var.unit}' (value_candidate = {unit_factor:g} * value_reference)."
                )
        if var.requirement_id and ref_var.requirement_id and var.requirement_id != ref_var.requirement_id:
            return failed("Conflicting requirement IDs")
        if kind(var) == "integer" and (abs(scale) != 1 or not float(expr.constant).is_integer()):
            return failed("Map does not preserve the integer lattice", ["integer-map"])
        if kind(var) == "binary" and (scale, expr.constant) not in ((1, 0), (-1, 1)):
            return failed("Unsupported binary map", ["binary-map"])
        mapping[name], subs[name] = target, expr
        used.add(target)
        methods.append(method)
        if unit_applied:
            methods.append("unit-conversion")

    # Reference variables absent from the candidate may be recoverable as a
    # verified complement/identity of a candidate binary (e.g. z_ji = 1 - z_ij).
    # Accepted only when the eliminated reference rows are reproduced verbatim.
    ref_only = set(ref_ir.variables) - used
    ref_eliminations: Dict[str, AffineExpression] = {}
    if ref_only:
        ref_eliminations = _resolve_ref_eliminations(
            ref_ir, cand_ir, ref_only, mapping, tolerance
        )
        if not ref_eliminations:
            return failed("Map does not cover reference variables")
        used |= set(ref_eliminations)
        methods.append("ref-elimination")
    if used != set(ref_ir.variables):
        return failed("Map does not cover reference variables")
    return ProjectionCertificate(
        is_verified=True,
        variable_map=mapping,
        affine_substitutions=subs,
        ref_eliminations=ref_eliminations,
        reason="Verified under recorded mapping assumptions",
        detection_method="+".join(sorted(set(methods))),
        warnings=warnings,
    )


def _exact_decimal_float(value):
    """Reject transformations that cannot be stored faithfully in this float IR."""
    converted = float(value)
    if not np.isfinite(converted) or Fraction(str(converted)) != value:
        raise ValueError("Affine projection requires higher-precision coefficient storage")
    return converted


def _normalize_name(name: str) -> str:
    """Normalize variable name for fuzzy matching: lowercase, strip underscores/spaces/hyphens.
    e.g. 'x_1_1' -> 'x11', 'z_12' -> 'z12', 'x 1' -> 'x1'.
    """
    import re as _re
    return _re.sub(r'[\s_\-]+', '', name.lower())


def _normalization_is_informative(cand_name: str, ref_name: str) -> bool:
    """Guard against separator-only matches such as 'x_' collapsing onto 'x'.

    Both names must split into non-empty tokens on whitespace/underscore/hyphen,
    so a trailing/dangling separator (``x_``, ``_x``) never produces a match.
    """
    import re as _re
    pattern = r'[\s_\-]+'
    cand_tokens = _re.split(pattern, cand_name)
    ref_tokens = _re.split(pattern, ref_name)
    return all(cand_tokens) and all(ref_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Structural index-name matching + verified reference elimination
# (ported from branch `thanh`; additive detection methods, no fuzzy names).
# ─────────────────────────────────────────────────────────────────────────────

def _index_signature(name: str) -> Optional[str]:
    """Leading letter run + ordered index digits, ignoring semantic tokens.
        x_A1_R1 -> 'x11' ; x_A1 -> 'x1' ; z_A1_A2 -> 'z12' ; x_1_1 -> 'x11'
    Returns None when there are no digits (plain identifiers keep name rules).
    """
    import re as _re
    match = _re.match(r'^\s*([A-Za-z]+)(.*)$', name)
    if not match:
        return None
    base = match.group(1).lower()
    digits = _re.findall(r'\d+', match.group(2))
    if not digits:
        return None
    return base + "".join(digits)


def _digit_signature(name: str) -> Optional[str]:
    """Ordered index digits only, ignoring the base letter (y_A1_A2 -> '12')."""
    import re as _re
    digits = "".join(_re.findall(r'\d+', name))
    return digits or None


def _reversed_index_signature(name: str) -> Optional[str]:
    """Signature with index digits reversed: z21 -> 'z12', x_A2_R1 -> 'x12'."""
    import re as _re
    match = _re.match(r'^\s*([A-Za-z]+)(.*)$', name)
    if not match:
        return None
    base = match.group(1).lower()
    digit_str = "".join(_re.findall(r'\d+', match.group(2)))
    if not digit_str:
        return None
    return base + digit_str[::-1]


def _index_set_pattern(index_label: str) -> str:
    """Index-set name without its values: 'A1' -> 'A', 'R2' -> 'R'."""
    import re as _re
    return _re.sub(r'\d+', '', str(index_label)).strip().lower()


def _index_sets_compatible(
    cand_var: CanonicalVariable,
    ref_var: CanonicalVariable,
) -> bool:
    """Declared index sets must not contradict a name-proposed pairing.

    Missing index sets carry no evidence and never block. When both sides
    declare them, the number of index dimensions must match and the per-dimension
    index names must line up (so x[i,j] cannot silently pair with x[j,i]).
    """
    cand_sets, ref_sets = cand_var.index_sets, ref_var.index_sets
    if not cand_sets or not ref_sets:
        return True
    if len(cand_sets) != len(ref_sets):
        return False
    return [_index_set_pattern(s) for s in cand_sets] == [
        _index_set_pattern(s) for s in ref_sets
    ]


def _unit_relation(
    cand_var: CanonicalVariable,
    ref_var: CanonicalVariable,
) -> Tuple[Optional[float], str]:
    """Declared-unit relation between candidate and reference variables.

    Returns ``(factor, status)`` with ``value_candidate = factor * value_reference``:
      * ``(1.0, "")``            no declared conversion (missing or equal units);
      * ``(factor, "unit-conversion")`` declared, comparable units differ;
      * ``(None, "unit-mismatch")`` declared units are not convertible.
    """
    if cand_var.unit and ref_var.unit:
        factor = _unit_conversion_factor(cand_var.unit, ref_var.unit)
        if factor is None:
            return None, "unit-mismatch"
        if factor != 1.0:
            return factor, "unit-conversion"
    return 1.0, ""


def _unique_signature_index(ir: CanonicalIR) -> Dict[str, str]:
    """signature -> variable name, keeping only unambiguous signatures."""
    buckets: Dict[str, List[str]] = {}
    for name in ir.variables:
        signature = _index_signature(name)
        if signature is not None:
            buckets.setdefault(signature, []).append(name)
    return {sig: names[0] for sig, names in buckets.items() if len(names) == 1}


def _unique_normalized_name_index(ir: CanonicalIR) -> Dict[str, str]:
    """normalized name -> variable, keeping only unambiguous normalized names.

    Two distinct names that collapse to the same normalized form (``x_1`` and
    ``x1``) make the entry ambiguous and it is dropped, so normalization can
    never silently merge two variables.
    """
    buckets: Dict[str, List[str]] = {}
    for name in ir.variables:
        buckets.setdefault(_normalize_name(name), []).append(name)
    return {key: names[0] for key, names in buckets.items() if len(names) == 1}


def _unique_binary_digit_index(ir: CanonicalIR) -> Dict[str, str]:
    """digit-signature -> binary variable name, unambiguous matches only."""
    buckets: Dict[str, List[str]] = {}
    for name, var in ir.variables.items():
        if var.var_type.lower() not in ("binary", "bin"):
            continue
        signature = _digit_signature(name)
        if signature is not None:
            buckets.setdefault(signature, []).append(name)
    return {sig: names[0] for sig, names in buckets.items() if len(names) == 1}


def _constraints_equivalent(
    a: CanonicalConstraint,
    b: CanonicalConstraint,
    tolerance: float = 1e-6,
) -> bool:
    """Compare two constraints after normalization (sign/scale invariant)."""
    na, nb = a.normalize(), b.normalize()
    if na.sense != nb.sense:
        return False
    keys = set(na.coeffs) | set(nb.coeffs)
    if any(abs(na.coeffs.get(k, 0.0) - nb.coeffs.get(k, 0.0)) > tolerance for k in keys):
        return False
    return abs(na.constant - nb.constant) <= tolerance


def _substitute_ref_constraint(
    constraint: CanonicalConstraint,
    eliminations: Dict[str, AffineExpression],
    ref_to_cand: Dict[str, str],
    tolerance: float,
) -> Optional[CanonicalConstraint]:
    """Rewrite a reference constraint into candidate space, eliminating ref vars."""
    coeffs: Dict[str, float] = {}
    constant = constraint.constant
    for ref_var, coeff in constraint.coeffs.items():
        if ref_var in eliminations:
            expr = eliminations[ref_var]
        elif ref_var in ref_to_cand:
            expr = AffineExpression({ref_to_cand[ref_var]: 1.0}, 0.0)
        else:
            return None
        for cand_var, c2 in expr.coeffs.items():
            coeffs[cand_var] = coeffs.get(cand_var, 0.0) + coeff * c2
        constant += coeff * expr.constant
    return CanonicalConstraint(
        name=constraint.name,
        coeffs={v: c for v, c in coeffs.items() if abs(c) > tolerance},
        constant=constant,
        sense=constraint.sense,
    )


def _verify_ref_eliminations(
    ref_ir: CanonicalIR,
    cand_ir: CanonicalIR,
    ref_to_cand: Dict[str, str],
    eliminations: Dict[str, AffineExpression],
    tolerance: float,
) -> bool:
    """Every reference constraint using an eliminated var must match a candidate one."""
    cand_constraints = list(cand_ir.constraints)
    for ref_c in ref_ir.constraints:
        if not any(v in eliminations for v in ref_c.coeffs):
            continue
        projected = _substitute_ref_constraint(ref_c, eliminations, ref_to_cand, tolerance)
        if projected is None:
            return False
        if not any(_constraints_equivalent(projected, c, tolerance) for c in cand_constraints):
            return False
    return True


def _resolve_ref_eliminations(
    ref_ir: CanonicalIR,
    cand_ir: CanonicalIR,
    ref_only: Set[str],
    variable_map: Dict[str, str],
    tolerance: float,
) -> Dict[str, AffineExpression]:
    """Resolve reference variables absent from the candidate via a candidate
    binary driving both ordering directions (complement `1 - cand` or identity
    `cand`). The relation is chosen by constraint-level verification, never
    assumed; {} is returned when any check fails.
    """
    if not ref_only:
        return {}
    ref_to_cand: Dict[str, str] = {}
    for cand_name, ref_name in variable_map.items():
        ref_to_cand.setdefault(ref_name, cand_name)

    cand_by_sig = _unique_signature_index(cand_ir)
    cand_bin_digit = _unique_binary_digit_index(cand_ir)
    proposals_per_ref: List[Tuple[str, List[AffineExpression]]] = []
    for ref_name in sorted(ref_only):
        ref_var = ref_ir.variables[ref_name]
        if ref_var.var_type.lower() not in ("binary", "bin"):
            return {}
        partner = cand_by_sig.get(_reversed_index_signature(ref_name) or "")
        if partner is None:
            digits = _digit_signature(ref_name)
            if digits is not None:
                partner = cand_bin_digit.get(digits[::-1])
        if partner is None or partner not in cand_ir.variables:
            return {}
        if cand_ir.variables[partner].var_type.lower() not in ("binary", "bin"):
            return {}
        proposals_per_ref.append((
            ref_name,
            [
                AffineExpression({partner: -1.0}, 1.0),  # complement: ref = 1 - cand
                AffineExpression({partner: 1.0}, 0.0),   # identity:   ref = cand
            ],
        ))

    if not proposals_per_ref or len(proposals_per_ref) > 6:
        return {}

    import itertools
    for combo in itertools.product(*[opts for _, opts in proposals_per_ref]):
        proposals = {name: expr for (name, _), expr in zip(proposals_per_ref, combo)}
        if _verify_ref_eliminations(ref_ir, cand_ir, ref_to_cand, proposals, tolerance):
            return proposals
    return {}


def _unique_index(variables: Dict[str, CanonicalVariable], attr_name: str) -> Dict[str, str]:
    buckets: Dict[str, List[str]] = {}
    for name, var in variables.items():
        value = getattr(var, attr_name, "")
        if value:
            buckets.setdefault(str(value), []).append(name)
    return {value: names[0] for value, names in buckets.items() if len(names) == 1}


def _unique_fingerprint_index(ir: CanonicalIR) -> Dict[Tuple[Any, ...], str]:
    buckets: Dict[Tuple[Any, ...], List[str]] = {}
    for name in ir.variables:
        buckets.setdefault(_variable_fingerprint(ir, name), []).append(name)
    return {fp: names[0] for fp, names in buckets.items() if len(names) == 1}


def _variable_fingerprint(ir: CanonicalIR, var_name: str) -> Tuple[Any, ...]:
    var = ir.variables[var_name]
    constraint_coeffs = sorted(
        round(c.coeffs.get(var_name, 0.0) / max(1.0, sum(abs(v) for v in c.coeffs.values()), abs(c.constant)), 10)
        for c in ir.constraints
        if abs(c.coeffs.get(var_name, 0.0)) > 1e-12
    )
    obj_scale = max(1.0, sum(abs(v) for v in ir.objective_coeffs.values()), abs(ir.objective_constant))
    return (
        var.var_type.lower(),
        _round_bound(var.lower_bound),
        _round_bound(var.upper_bound),
        round(ir.objective_coeffs.get(var_name, 0.0) / obj_scale, 10),
        tuple(constraint_coeffs),
    )


def _round_bound(value: float) -> Any:
    if value == float("inf") or value == float("-inf"):
        return value
    return round(float(value), 10)


# ─────────────────────────────────────────────────────────────────────────────
# Unit conversion helper — restored from branch `thanh` (Module 5 objective
# unit conversion). Main removed objective unit conversion; Module 5 needs it.
# ─────────────────────────────────────────────────────────────────────────────

# Conversion factor F such that:  value_in_from_unit = F * value_in_to_unit.
# e.g. ('kg','tonne') → 1000.0  because 1 tonne = 1000 kg.
_UNIT_FACTORS: Dict[Tuple[str, str], float] = {
    # mass
    ("kg", "tonne"): 1000.0,
    ("kg", "ton"): 1000.0,
    ("kg", "metricton"): 1000.0,
    ("tonne", "kg"): 0.001,
    ("ton", "kg"): 0.001,
    ("g", "kg"): 1000.0,
    ("kg", "g"): 0.001,
    ("g", "tonne"): 1e6,
    ("tonne", "g"): 1e-6,
    ("mg", "g"): 1000.0,
    ("g", "mg"): 0.001,
    # length
    ("m", "km"): 1000.0,
    ("km", "m"): 0.001,
    ("cm", "m"): 100.0,
    ("m", "cm"): 0.01,
    ("mm", "m"): 1000.0,
    ("m", "mm"): 0.001,
    # time
    ("s", "min"): 60.0,
    ("min", "s"): 1.0 / 60.0,
    ("s", "h"): 3600.0,
    ("h", "s"): 1.0 / 3600.0,
    ("min", "h"): 60.0,
    ("h", "min"): 1.0 / 60.0,
    # currency (placeholder scale — deterministic, unit-preserving only when same currency)
    ("vnd", "vnd"): 1.0,
}


def _canonical_unit(unit: str) -> str:
    """Lowercase, strip whitespace, and drop plural markers for unit lookup."""
    u = unit.lower().strip().replace(" ", "")
    if u.endswith("s") and u not in ("m", "s", "km", "g", "kg", "h", "min"):
        u = u[:-1]
    return u


def _unit_conversion_factor(from_unit: str, to_unit: str) -> Optional[float]:
    """
    Returns F such that value_in_from_unit = F * value_in_to_unit,
    or None when the units are unknown / not comparable.
    """
    if from_unit == to_unit:
        return 1.0
    if not from_unit or not to_unit:
        return None
    u_from = _canonical_unit(from_unit)
    u_to = _canonical_unit(to_unit)
    if u_from == u_to:
        return 1.0
    return _UNIT_FACTORS.get((u_from, u_to))
