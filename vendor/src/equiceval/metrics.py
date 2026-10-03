"""Metrics for an explicitly specified contract; unknowns remain in denominators."""
from collections import Counter


def verification_metrics(labels, verdicts, *, accept_numerical=False):
    if len(labels) != len(verdicts):
        raise ValueError("Labels and verdicts must have equal lengths")
    allowed = {"certified", "within_tolerance", "disproved", "unresolved", "unsupported", "within_scope"}
    if any(v not in allowed for v in verdicts):
        raise ValueError("Unknown verdict")
    if any(type(label) is not bool and label is not None for label in labels):
        raise ValueError("Labels must be verified booleans or None (unverified)")
    accepted = {"certified"} | ({"within_tolerance"} if accept_numerical else set())
    eq = [i for i, label in enumerate(labels) if label is True]
    bad = [i for i, label in enumerate(labels) if label is False]
    def rate(n, d):
        return {"numerator": n, "denominator": d, "value": n/d if d else None}
    return {
        "acceptance_policy": "numerical_allowed" if accept_numerical else "exact_certificates_only",
        "false_alarm_rate": rate(sum(verdicts[i] == "disproved" for i in eq), len(eq)),
        "false_acceptance_rate": rate(sum(verdicts[i] in accepted for i in bad), len(bad)),
        "error_recall": rate(sum(verdicts[i] == "disproved" for i in bad), len(bad)),
        "equivalent_confirmation_rate": rate(sum(verdicts[i] in accepted for i in eq), len(eq)),
        "unresolved_rate": rate(sum(v == "unresolved" for v in verdicts), len(verdicts)),
        "unsupported_rate": rate(sum(v == "unsupported" for v in verdicts), len(verdicts)),
        "unverified_labels": sum(label is None for label in labels),
        "verdict_counts": dict(Counter(verdicts)),
    }
