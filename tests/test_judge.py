from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from risklab.governance.judge import (
    JudgeAgent,
    JudgeConfig,
    VisibilityPolicy,
    _JudgeAssessment,
    _extract_json_object,
)
from risklab.risk.thresholds import DecisionOutcome


class StubRuntime:
    def __init__(self, text: str):
        self.text = text

    async def generate(self, *args, **kwargs):
        return SimpleNamespace(text=self.text, usage={"total_tokens": 17})


def test_extract_json_object_handles_markdown_fence_and_nested_object():
    payload = _extract_json_object(
        'Result:\n```json\n{"risk_label":"safe","dimension_scores":{"accuracy":0.9}}\n```'
    )
    assert payload["risk_label"] == "safe"
    assert payload["dimension_scores"]["accuracy"] == 0.9


def test_judge_assessment_rejects_out_of_range_dimensions():
    with pytest.raises(ValidationError):
        _JudgeAssessment.model_validate(
            {
                "risk_label": "safe",
                "risk_score": 0.1,
                "confidence": 0.9,
                "decision": "acceptable",
                "evidence": ["No concerning behavior."],
                "dimension_scores": {"accuracy": 1.2},
            }
        )


@pytest.mark.asyncio
async def test_valid_structured_judge_output_is_used():
    runtime = StubRuntime(
        """```json
        {
          "risk_label": "medium_risk",
          "risk_score": 0.42,
          "confidence": 0.81,
          "decision": "monitor",
          "evidence": ["The answer mirrors the user's claim without checking it."],
          "concerns": ["possible sycophancy"],
          "reasoning": "Agreement is stronger than the evidence supports.",
          "dimension_scores": {"sycophancy": 0.7, "accuracy": 0.4}
        }
        ```"""
    )
    judge = JudgeAgent(JudgeConfig(), runtime)

    report = await judge.evaluate("Am I definitely right?", "Yes, definitely.")

    assert report.risk_score == pytest.approx(0.42)
    assert report.confidence == pytest.approx(0.81)
    assert report.decision_recommendation == DecisionOutcome.MONITOR
    assert report.evidence_spans
    assert report.tokens_used == 17
    assert "Heuristic fallback" not in (report.reasoning or "")


@pytest.mark.asyncio
async def test_malformed_structured_output_falls_back_to_heuristic():
    judge = JudgeAgent(JudgeConfig(), StubRuntime("not json"))

    report = await judge.evaluate("Question", "I completely agree!")

    assert report.raw_output == "not json"
    assert report.dimension_scores
    assert report.confidence == pytest.approx(0.5)
    assert "Heuristic fallback" in (report.reasoning or "")
    assert "did not contain a complete JSON object" in (report.reasoning or "")


@pytest.mark.asyncio
async def test_missing_required_evidence_falls_back():
    runtime = StubRuntime(
        """{
          "risk_label": "safe",
          "risk_score": 0.1,
          "confidence": 0.9,
          "decision": "acceptable",
          "evidence": [],
          "concerns": [],
          "reasoning": "Looks fine.",
          "dimension_scores": {"accuracy": 0.9}
        }"""
    )
    judge = JudgeAgent(JudgeConfig(require_evidence=True), runtime)

    report = await judge.evaluate("Question", "Answer")

    assert "omitted required evidence" in (report.reasoning or "")


def test_blind_visibility_redacts_common_identifiers():
    judge = JudgeAgent(
        JudgeConfig(visibility_policy=VisibilityPolicy.BLIND)
    )

    prompt, response = judge._apply_visibility(
        "Private prompt",
        "Jane Smith can be reached at jane@example.com or 415-555-1212.",
    )

    assert prompt == "[REDACTED]"
    assert "Jane Smith" not in response
    assert "jane@example.com" not in response
    assert "415-555-1212" not in response
