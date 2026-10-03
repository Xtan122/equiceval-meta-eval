"""Small independent checkers for primal points and rational LP implications.

Fractions interpret each input float's decimal spelling; they do not recover an
unknown intended real number. Certificates are checked against the original rows.
"""
from fractions import Fraction
from math import isfinite
from src.equiceval.canonical_ir import CanonicalConstraint


def rational(value):
    return Fraction(str(value))


# Relative slack for the exact checker. A semantics-preserving rescaling stores
# each coefficient's decimal spelling independently, so a point exactly on the
# boundary can show a few-ulp residual (~1e-13) against the constant. The slack
# is relative to the row scale and far below any semantic violation.
DEFAULT_RELATIVE_TOLERANCE = Fraction(1, 10 ** 9)


def snapped_point(point, max_denominator=10 ** 9):
    """Rational reconstruction of a float point with a bounded denominator."""
    return {v: rational(x).limit_denominator(max_denominator) for v, x in point.items()}


def rows(ir):
    """All original rows and explicit bounds; equalities have two directions."""
    result = []
    for i, c in enumerate(ir.merge_bounds_into_constraints().constraints):
        c = c.flip_to_leq()
        result.append((f"{i}:{c.name}:+", c))
        if c.sense == "==":
            result.append((f"{i}:{c.name}:-", CanonicalConstraint(
                c.name, {v: -a for v, a in c.coeffs.items()}, -c.constant, "<=")))
    return result


def _row_scale(c, p):
    return abs(rational(c.constant)) + sum(
        abs(rational(a)) * abs(p[v]) for v, a in c.coeffs.items())


def _point_ok(ir, p, box, tolerance=Fraction(0)):
    for v, spec in ir.variables.items():
        if spec.var_type.lower() in ("integer", "int", "binary", "bin") and p[v].denominator != 1:
            return False
        if box is not None and abs(p[v]) > rational(box):
            return False
    for _, c in rows(ir):
        value = sum(rational(a) * p[v] for v, a in c.coeffs.items()) + rational(c.constant)
        if value <= 0:
            continue
        if tolerance > 0 and value <= tolerance * max(_row_scale(c, p), Fraction(1)):
            continue
        return False
    return True


def point_is_feasible(ir, point, box=None, snap=True, tolerance=None):
    """Exact check of the submitted decimal point, including integrality.

    Floating solver output can miss a feasible vertex by one ulp (HiGHS may
    report ``0.6666666666666657`` for ``2/3``), and the exact decimal spelling
    would then be rejected as infeasible. When ``snap`` is set, a nearby
    rational reconstruction (bounded denominator) is checked exactly as well;
    the accepted point is still an exactly verified rational point.

    ``tolerance`` is a *relative* slack (fraction of the row scale); ``None``
    uses ``DEFAULT_RELATIVE_TOLERANCE`` for decimal-representation noise, and
    ``0`` restores the fully strict check.
    """
    tol = DEFAULT_RELATIVE_TOLERANCE if tolerance is None else Fraction(tolerance)
    try:
        if set(point) != set(ir.variables) or any(not isfinite(x) for x in point.values()):
            return False
        if _point_ok(ir, {v: rational(x) for v, x in point.items()}, box, tol):
            return True
        return snap and _point_ok(ir, snapped_point(point), box, tol)
    except (ValueError, KeyError, OverflowError, TypeError):
        return False


def point_violates(ir, point, tolerance=DEFAULT_RELATIVE_TOLERANCE):
    """Independent check that ``point`` violates a row of ``ir`` beyond slack.

    Rows include the explicit variable bounds. Returns False when the point is
    not a complete finite assignment, so a partial/empty witness is never counted.
    """
    tol = DEFAULT_RELATIVE_TOLERANCE if tolerance is None else Fraction(tolerance)
    try:
        if set(point) != set(ir.variables) or any(not isfinite(x) for x in point.values()):
            return False
        p = {v: rational(x) for v, x in point.items()}
        for _, c in rows(ir):
            value = sum(rational(a) * p[v] for v, a in c.coeffs.items()) + rational(c.constant)
            if value > tol * max(_row_scale(c, p), Fraction(1)):
                return True
        return False
    except (ValueError, KeyError, OverflowError, TypeError):
        return False


def violating_row_names(ir, point, tolerance=DEFAULT_RELATIVE_TOLERANCE):
    """Names of the rows (bounds included) that ``point`` violates beyond slack."""
    tol = DEFAULT_RELATIVE_TOLERANCE if tolerance is None else Fraction(tolerance)
    names = []
    try:
        if set(point) != set(ir.variables) or any(not isfinite(x) for x in point.values()):
            return []
        p = {v: rational(x) for v, x in point.items()}
        for _, c in rows(ir):
            value = sum(rational(a) * p[v] for v, a in c.coeffs.items()) + rational(c.constant)
            if value > tol * max(_row_scale(c, p), Fraction(1)):
                names.append(c.name)
    except (ValueError, KeyError, OverflowError, TypeError):
        return []
    return list(dict.fromkeys(names))


def check_implication_certificate(source, target, certificate):
    """Verify nonnegative weights imply target <= 0; no solver is called."""
    try:
        source_rows = dict(rows(source))
        weights = {key: Fraction(value) for key, value in certificate["weights"].items()}
        if not set(weights) <= set(source_rows) or any(x < 0 for x in weights.values()):
            return False
        for v in source.variables:
            if sum(w * rational(source_rows[k].coeffs.get(v, 0)) for k, w in weights.items()) != rational(target.coeffs.get(v, 0)):
                return False
        if not set(target.coeffs) <= set(source.variables):
            return False
        # sum lambda_i (a_i x + b_i) <= 0 implies c x + d <= 0
        return sum(w * rational(source_rows[k].constant) for k, w in weights.items()) >= rational(target.constant)
    except (ValueError, KeyError, ZeroDivisionError, TypeError):
        return False


def find_implication_certificate(source, target, budget, phase="implication_certificate"):
    """Search an LP dual, then validate rationalized weights independently.

    This is also sufficient for integer sources, but not complete for them.
    """
    source_rows = rows(source)
    trivial = {"kind": "rational_nonnegative_combination", "weights": {},
               "number_semantics": "exact decimal spelling of input coefficients",
               "support_rows": []}
    if check_implication_certificate(source, target, trivial):
        return trivial
    # Cheap exact single-row scaling before a dual LP.
    for key, row in source_rows:
        variable = next((v for v, a in row.coeffs.items() if a != 0), None)
        if variable is None:
            continue
        weight = rational(target.coeffs.get(variable, 0)) / rational(row.coeffs[variable])
        cert = dict(trivial, weights={key: str(weight)}, support_rows=[key])
        if weight > 0 and check_implication_certificate(source, target, cert):
            return cert
    specs = {f"w{i}": ("continuous", 0.0, float("inf")) for i in range(len(source_rows))}
    constraints = [
        ({f"w{i}": c.coeffs.get(v, 0.0) for i, (_, c) in enumerate(source_rows)},
         "==", target.coeffs.get(v, 0.0)) for v in source.variables
    ]
    result = budget.solve(
        phase, var_specs=specs,
        objective_coeffs={f"w{i}": -c.constant for i, (_, c) in enumerate(source_rows)},
        objective_sense="minimize", constraints_list=constraints)
    if result.status not in ("Optimal", "Timeout") or not result.variable_values:
        return None
    for max_denominator in (1000, 1000000, 1000000000):
        weights = {}
        for i, (key, _) in enumerate(source_rows):
            value = result.variable_values.get(f"w{i}", 0)
            if not isfinite(value):
                return None
            weight = rational(value).limit_denominator(max_denominator)
            if weight:
                weights[key] = str(weight)
        certificate = {"kind": "rational_nonnegative_combination", "weights": weights,
                       "number_semantics": "exact decimal spelling of input coefficients",
                       "support_rows": list(weights)}
        if check_implication_certificate(source, target, certificate):
            return certificate
    return None
