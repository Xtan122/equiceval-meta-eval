from expeval.adapters import EquiCEvalAdapter, EquivaMapAdapter
from expeval.adapters.common import load_pairs
from src.equiceval.contracts import PRIMARY_EQUIVAFORMULATION_CONTRACT


def _pairs():
    return load_pairs("data/sample_pairs.json", PRIMARY_EQUIVAFORMULATION_CONTRACT)


def test_equiceval_certifies_labelled_equivalent():
    pair = next(p for p in _pairs() if p.label is True)
    decision = EquiCEvalAdapter(seconds=1.0).score(pair)
    assert decision.outcome in {"equivalent", "tolerance"}


def test_equivamap_uses_published_quasi_karp_label():
    for pair in _pairs():
        published = pair.metadata.get("published_label")
        decision = EquivaMapAdapter().score(pair)
        if published is None:
            assert decision.outcome == "unsupported"
        else:
            assert decision.outcome == ("equivalent" if published else "faulty")
            assert decision.diagnostics["criterion"] == "quasi-Karp"
