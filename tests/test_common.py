from expeval.adapters.common import binary_metrics, rates


def test_rates_class_conditional():
    labels = [True, True, False, False]
    outcomes = ["faulty", "equivalent", "faulty", "equivalent"]
    metrics = rates(labels, outcomes)
    assert metrics["false_alarm_rate"]["value"] == 0.5
    assert metrics["error_recall"]["value"] == 0.5


def test_binary_metrics_for_boolean_decision():
    labels = [True, True, False, False]
    decisions = [False, True, True, False]  # True = faulty
    metrics = binary_metrics(labels, decisions)
    assert metrics["false_alarm_rate"]["value"] == 0.5
    assert metrics["error_recall"]["value"] == 0.5
