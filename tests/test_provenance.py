import pytest

from risklab.governance.provenance import (
    AuditTrail,
    ComputationMethod,
    StrictScoreFactory,
)


def _score(value: float, name: str):
    return StrictScoreFactory.from_rule(
        value=value,
        metric_name=name,
        rule_id=f"{name}-rule",
        evidence=[f"{name} evidence"],
    )


def test_average_aggregation_preserves_parent_provenance():
    first = _score(0.2, "first")
    second = _score(0.6, "second")

    aggregate = StrictScoreFactory.aggregate(
        [first, second],
        metric_name="combined",
    )

    assert aggregate.value == pytest.approx(0.4)
    assert aggregate.provenance.parent_provenances == [
        first.provenance.provenance_id,
        second.provenance.provenance_id,
    ]


def test_weighted_aggregation_uses_effective_weights():
    first = _score(0.2, "first")
    second = _score(0.8, "second")

    aggregate = StrictScoreFactory.aggregate(
        [first, second],
        metric_name="combined",
        method=ComputationMethod.WEIGHTED,
        weights={"first": 1.0, "second": 3.0},
    )

    assert aggregate.value == pytest.approx(0.65)
    assert aggregate.provenance.weights_applied == {
        "first": 1.0,
        "second": 3.0,
    }


def test_weighted_aggregation_rejects_non_positive_total_weight():
    first = _score(0.2, "first")
    second = _score(0.8, "second")

    with pytest.raises(ValueError, match="positive total weight"):
        StrictScoreFactory.aggregate(
            [first, second],
            metric_name="combined",
            method=ComputationMethod.WEIGHTED,
            weights={"first": 0.0, "second": 0.0},
        )


def test_compute_requires_upstream_scores():
    with pytest.raises(ValueError, match="at least one input"):
        StrictScoreFactory.compute(
            value=0.5,
            metric_name="derived",
            formula="x",
            input_scores={},
        )


def test_audit_trail_detects_missing_parent_reference():
    first = _score(0.2, "first")
    second = _score(0.8, "second")
    aggregate = StrictScoreFactory.aggregate(
        [first, second],
        metric_name="combined",
    )

    trail = AuditTrail(scores={"first": first, "combined": aggregate})

    issues = trail.verify_provenance_chain()

    assert any(second.provenance.provenance_id in issue for issue in issues)
