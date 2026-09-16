"""Deterministic offline contradiction evaluator (bench/eval_contradictions.py).

Covers the labeled fixture set and the precision/recall gate without needing
the full query suite:
  * the fixture file loads and validates,
  * a deliberately wrong label trips the threshold gate (the gate must bite),
  * every shipped fixture case classifies to its declared label,
  * malformed fixtures fail loudly at load time.
"""

from __future__ import annotations

from bench import eval_contradictions
from bench.golden import contradictions_v1


# ---------------------------------------------------------------------------
# Fixture loading / validation
# ---------------------------------------------------------------------------

def test_fixtures_load_and_validate():
    cases = contradictions_v1.load_cases()
    assert cases
    assert len({c["id"] for c in cases}) == len(cases)
    labels = {c["expected_label"] for c in cases}
    assert labels <= contradictions_v1.LABELS
    # The set must cover all three labels.
    assert labels == {"contradiction", "not_contradiction", "resolved_explained"}


def test_fixture_categories_cover_the_brief():
    cases = contradictions_v1.load_cases()
    categories = {c["category"] for c in cases}
    assert {
        "same_measure_conflict",
        "different_period",
        "different_scope",
        "primary_secondary_conflict",
        "shared_number_unrelated",
        "different_metric",
        "polarity_conflict",
    } <= categories


def test_malformed_fixture_is_rejected():
    errors = contradictions_v1.validate(
        {"cases": [{"id": "x", "expected_label": "nope", "claims": []}]}
    )
    assert any("unknown expected_label" in e for e in errors)
    assert any("needs at least two claims" in e for e in errors)


def test_missing_claim_source_is_rejected():
    errors = contradictions_v1.validate({
        "cases": [{
            "id": "y",
            "expected_label": "contradiction",
            "claims": [
                {"claim": "a", "source": ""},
                {"claim": "b", "source": "https://b.example/y"},
            ],
        }]
    })
    assert any("empty source" in e for e in errors)


# ---------------------------------------------------------------------------
# Classification + gate
# ---------------------------------------------------------------------------

def test_every_fixture_case_classifies_correctly():
    cases = contradictions_v1.load_cases()
    rows = [eval_contradictions.classify_case(c) for c in cases]
    wrong = [(r["id"], r["expected_label"], r["observed_label"])
             for r in rows if not r["correct"]]
    assert wrong == [], f"misclassified fixtures: {wrong}"


def test_aggregate_metrics_perfect_on_shipped_set():
    report = eval_contradictions.evaluate()
    m = report["aggregate"]["metrics"]
    assert m["detection_precision"] == 1.0
    assert m["detection_recall"] == 1.0
    assert m["resolution_precision"] == 1.0
    assert m["classification_accuracy"] == 1.0
    assert report["passed"] is True


def test_wrong_label_trips_the_gate():
    """A deliberately wrong fixture label must produce a threshold failure —
    otherwise the gate is decorative."""
    cases = contradictions_v1.load_cases()
    bad = [dict(c) for c in cases]
    bad[0] = {**bad[0], "expected_label": "not_contradiction"}
    per_case = [eval_contradictions.classify_case(c) for c in bad]
    aggregate = eval_contradictions.aggregate(per_case)
    thresholds = {
        "version": "v1",
        "contradictions": {k: 1.0 for k in eval_contradictions.CONTRADICTION_METRICS},
    }
    failures = eval_contradictions.check_thresholds(aggregate, thresholds)
    assert failures, "gate failed to bite on a wrong label"
    assert any(f["metric"] == "detection_precision" for f in failures)


def test_resolved_requires_an_explanation():
    """Every resolved finding from the fixtures must carry a non-empty
    explanation (the resolution string)."""
    from app.core.contradiction_resolution import resolve_contradictions
    from app.core.contradictions import find_contradictions

    for case in contradictions_v1.load_cases():
        claims = [
            {"claim": c["claim"], "source": c["source"]}
            for c in case["claims"][:2]
        ]
        for resolved in resolve_contradictions(find_contradictions(claims)):
            if resolved.get("resolved"):
                assert str(resolved.get("resolution", "")).strip(), case["id"]


def test_thresholds_file_loads_and_is_additive():
    import json
    from pathlib import Path

    path = Path(eval_contradictions.DEFAULT_THRESHOLDS)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == "v1"
    assert data["contradictions"]
