"""Provenance primitives for RiskLab evaluation signals.

The goal of this module is narrower than proving that an evaluation is
"correct": it records where a score came from, how it was combined, and which
upstream scores contributed to it so an evaluation can be inspected later.
"""

from __future__ import annotations

import functools
import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp_id() -> str:
    return _utcnow().strftime("%Y%m%d%H%M%S%f")


class EvaluatorType(str, Enum):
    """Type of evaluator that produced a score."""

    RULE_BASED = "rule_based"
    ML_CLASSIFIER = "ml_classifier"
    LLM_JUDGE = "llm_judge"
    HEURISTIC = "heuristic"
    HUMAN = "human"
    COUNCIL = "council"
    AGGREGATED = "aggregated"
    COMPUTED = "computed"


class ComputationMethod(str, Enum):
    """How a score was computed."""

    DIRECT = "direct"
    AVERAGED = "averaged"
    WEIGHTED = "weighted"
    MAX = "max"
    MIN = "min"
    FORMULA = "formula"
    THRESHOLD = "threshold"
    NORMALIZED = "normalized"


class ProvenanceRecord(BaseModel):
    """Metadata required to trace one evaluation signal."""

    provenance_id: str = Field(default_factory=_timestamp_id)
    timestamp: datetime = Field(default_factory=_utcnow)

    evaluator_type: EvaluatorType
    evaluator_id: str = ""
    evaluator_version: str = ""

    computation_method: ComputationMethod = ComputationMethod.DIRECT
    computation_formula: Optional[str] = None

    input_scores: Dict[str, float] = Field(default_factory=dict)
    input_text_hash: Optional[str] = None
    input_context_hash: Optional[str] = None
    weights_applied: Dict[str, float] = Field(default_factory=dict)

    evidence: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None

    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence_factors: Dict[str, float] = Field(default_factory=dict)

    random_seed: Optional[int] = None
    model_temperature: Optional[float] = None
    parent_provenances: List[str] = Field(default_factory=list)

    def get_hash(self) -> str:
        """Return a deterministic fingerprint of computation-relevant fields."""

        data = {
            "evaluator_type": self.evaluator_type.value,
            "evaluator_id": self.evaluator_id,
            "evaluator_version": self.evaluator_version,
            "computation_method": self.computation_method.value,
            "computation_formula": self.computation_formula,
            "input_scores": self.input_scores,
            "input_text_hash": self.input_text_hash,
            "input_context_hash": self.input_context_hash,
            "weights_applied": self.weights_applied,
            "random_seed": self.random_seed,
            "model_temperature": self.model_temperature,
        }
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]


class SemanticScore(BaseModel):
    """A bounded score with explicit meaning and provenance."""

    value: float = Field(ge=0.0, le=1.0)
    metric_name: str
    metric_description: str = ""
    scale_low: str = "0.0 = none/absent"
    scale_high: str = "1.0 = maximum/complete"
    higher_is_worse: bool = False
    provenance: ProvenanceRecord
    interpretation: Optional[str] = None

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: float) -> float:
        if not isinstance(value, (int, float)):
            raise ValueError(f"Score value must be numeric, got {type(value)}")
        return float(value)

    def get_interpretation(self) -> str:
        """Return a coarse human-readable band for the score."""

        if self.interpretation:
            return self.interpretation

        if self.higher_is_worse:
            bands = (
                (0.2, "minimal"),
                (0.4, "low"),
                (0.6, "moderate"),
                (0.8, "high"),
            )
            fallback = "severe"
        else:
            bands = (
                (0.2, "very poor"),
                (0.4, "poor"),
                (0.6, "moderate"),
                (0.8, "good"),
            )
            fallback = "excellent"

        for upper_bound, label in bands:
            if self.value < upper_bound:
                return label
        return fallback

    def to_audit_entry(self) -> Dict[str, Any]:
        """Export a compact representation suitable for audit reports."""

        return {
            "metric": self.metric_name,
            "value": self.value,
            "interpretation": self.get_interpretation(),
            "higher_is_worse": self.higher_is_worse,
            "provenance": {
                "id": self.provenance.provenance_id,
                "evaluator": (
                    f"{self.provenance.evaluator_type.value}:"
                    f"{self.provenance.evaluator_id}"
                ),
                "method": self.provenance.computation_method.value,
                "confidence": self.provenance.confidence,
                "timestamp": self.provenance.timestamp.isoformat(),
                "hash": self.provenance.get_hash(),
            },
            "evidence": self.provenance.evidence[:5],
        }


class StrictScoreFactory:
    """Construct scores while preserving provenance across transformations."""

    @staticmethod
    def from_rule(
        value: float,
        metric_name: str,
        rule_id: str,
        evidence: List[str],
        higher_is_worse: bool = True,
        metric_description: str = "",
    ) -> SemanticScore:
        provenance = ProvenanceRecord(
            evaluator_type=EvaluatorType.RULE_BASED,
            evaluator_id=rule_id,
            computation_method=ComputationMethod.DIRECT,
            evidence=evidence,
            confidence=0.8,
        )
        return SemanticScore(
            value=value,
            metric_name=metric_name,
            metric_description=metric_description,
            higher_is_worse=higher_is_worse,
            provenance=provenance,
        )

    @staticmethod
    def from_ml(
        value: float,
        metric_name: str,
        model_id: str,
        confidence: float,
        evidence: Optional[List[str]] = None,
        higher_is_worse: bool = True,
    ) -> SemanticScore:
        provenance = ProvenanceRecord(
            evaluator_type=EvaluatorType.ML_CLASSIFIER,
            evaluator_id=model_id,
            computation_method=ComputationMethod.DIRECT,
            evidence=evidence or [],
            confidence=confidence,
        )
        return SemanticScore(
            value=value,
            metric_name=metric_name,
            higher_is_worse=higher_is_worse,
            provenance=provenance,
        )

    @staticmethod
    def from_llm(
        value: float,
        metric_name: str,
        model_id: str,
        reasoning: str,
        temperature: float = 0.0,
        evidence: Optional[List[str]] = None,
        higher_is_worse: bool = True,
    ) -> SemanticScore:
        provenance = ProvenanceRecord(
            evaluator_type=EvaluatorType.LLM_JUDGE,
            evaluator_id=model_id,
            computation_method=ComputationMethod.DIRECT,
            evidence=evidence or [],
            reasoning=reasoning,
            model_temperature=temperature,
            confidence=0.7 if temperature > 0 else 0.85,
        )
        return SemanticScore(
            value=value,
            metric_name=metric_name,
            higher_is_worse=higher_is_worse,
            provenance=provenance,
        )

    @staticmethod
    def from_human(
        value: float,
        metric_name: str,
        annotator_id: str,
        reasoning: Optional[str] = None,
        higher_is_worse: bool = True,
    ) -> SemanticScore:
        provenance = ProvenanceRecord(
            evaluator_type=EvaluatorType.HUMAN,
            evaluator_id=annotator_id,
            computation_method=ComputationMethod.DIRECT,
            reasoning=reasoning,
            confidence=0.95,
        )
        return SemanticScore(
            value=value,
            metric_name=metric_name,
            higher_is_worse=higher_is_worse,
            provenance=provenance,
        )

    @staticmethod
    def aggregate(
        scores: List[SemanticScore],
        metric_name: str,
        method: ComputationMethod = ComputationMethod.AVERAGED,
        weights: Optional[Dict[str, float]] = None,
        higher_is_worse: bool = True,
    ) -> SemanticScore:
        """Aggregate scores and retain links to every upstream provenance."""

        if not scores:
            raise ValueError("Cannot aggregate an empty score list")

        input_scores = {
            score.provenance.provenance_id: score.value for score in scores
        }
        parent_provenances = [
            score.provenance.provenance_id for score in scores
        ]
        applied_weights: Dict[str, float] = {}

        if method == ComputationMethod.AVERAGED:
            value = sum(score.value for score in scores) / len(scores)
        elif method == ComputationMethod.WEIGHTED:
            applied_weights = {
                score.metric_name: (weights or {}).get(score.metric_name, 1.0)
                for score in scores
            }
            if any(weight < 0 for weight in applied_weights.values()):
                raise ValueError("Weighted aggregation does not accept negative weights")
            total_weight = sum(applied_weights.values())
            if total_weight <= 0:
                raise ValueError("Weighted aggregation requires positive total weight")
            value = (
                sum(
                    score.value * applied_weights[score.metric_name]
                    for score in scores
                )
                / total_weight
            )
        elif method == ComputationMethod.MAX:
            value = max(score.value for score in scores)
        elif method == ComputationMethod.MIN:
            value = min(score.value for score in scores)
        else:
            raise ValueError(f"Unsupported aggregation method: {method.value}")

        avg_confidence = (
            sum(score.provenance.confidence for score in scores) / len(scores)
        )
        provenance = ProvenanceRecord(
            evaluator_type=EvaluatorType.AGGREGATED,
            evaluator_id="aggregator",
            computation_method=method,
            input_scores=input_scores,
            weights_applied=applied_weights,
            parent_provenances=parent_provenances,
            confidence=avg_confidence * 0.95,
        )
        return SemanticScore(
            value=value,
            metric_name=metric_name,
            higher_is_worse=higher_is_worse,
            provenance=provenance,
        )

    @staticmethod
    def compute(
        value: float,
        metric_name: str,
        formula: str,
        input_scores: Dict[str, SemanticScore],
        higher_is_worse: bool = True,
    ) -> SemanticScore:
        """Create a derived score and record the source scores/formula."""

        if not input_scores:
            raise ValueError("Computed scores require at least one input score")

        input_values = {key: score.value for key, score in input_scores.items()}
        parent_provenances = [
            score.provenance.provenance_id for score in input_scores.values()
        ]
        avg_confidence = (
            sum(score.provenance.confidence for score in input_scores.values())
            / len(input_scores)
        )
        provenance = ProvenanceRecord(
            evaluator_type=EvaluatorType.COMPUTED,
            evaluator_id="formula_computer",
            computation_method=ComputationMethod.FORMULA,
            computation_formula=formula,
            input_scores=input_values,
            parent_provenances=parent_provenances,
            confidence=avg_confidence,
        )
        return SemanticScore(
            value=value,
            metric_name=metric_name,
            higher_is_worse=higher_is_worse,
            provenance=provenance,
        )


class AuditTrail(BaseModel):
    """Audit record for scores, decision steps, and provenance links."""

    audit_id: str = Field(default_factory=_timestamp_id)
    evaluation_id: str = ""
    episode_id: str = ""

    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: Optional[datetime] = None

    scores: Dict[str, SemanticScore] = Field(default_factory=dict)
    decision_chain: List[Dict[str, Any]] = Field(default_factory=list)

    final_decision: Optional[str] = None
    decision_provenance: Optional[ProvenanceRecord] = None

    human_reviews: List[Dict[str, Any]] = Field(default_factory=list)
    human_overrides: List[Dict[str, Any]] = Field(default_factory=list)

    warnings: List[str] = Field(default_factory=list)
    anomalies_detected: List[Dict[str, Any]] = Field(default_factory=list)

    random_seeds_used: Dict[str, int] = Field(default_factory=dict)
    model_versions: Dict[str, str] = Field(default_factory=dict)

    def add_score(self, key: str, score: SemanticScore) -> None:
        self.scores[key] = score

    def add_decision_step(
        self,
        step_name: str,
        input_scores: List[str],
        output: str,
        reasoning: str,
    ) -> None:
        self.decision_chain.append(
            {
                "step": step_name,
                "timestamp": _utcnow().isoformat(),
                "input_scores": input_scores,
                "output": output,
                "reasoning": reasoning,
            }
        )

    def add_human_review(
        self,
        reviewer_id: str,
        action: str,
        reasoning: Optional[str] = None,
    ) -> None:
        self.human_reviews.append(
            {
                "reviewer": reviewer_id,
                "action": action,
                "reasoning": reasoning,
                "timestamp": _utcnow().isoformat(),
            }
        )

    def add_warning(self, warning: str) -> None:
        self.warnings.append(f"[{_utcnow().isoformat()}] {warning}")

    def finalize(self, decision: str, provenance: ProvenanceRecord) -> None:
        self.final_decision = decision
        self.decision_provenance = provenance
        self.completed_at = _utcnow()

    def to_report(self) -> Dict[str, Any]:
        duration_ms = None
        if self.completed_at:
            duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000

        return {
            "audit_id": self.audit_id,
            "evaluation_id": self.evaluation_id,
            "episode_id": self.episode_id,
            "duration_ms": duration_ms,
            "final_decision": self.final_decision,
            "scores": {
                key: score.to_audit_entry() for key, score in self.scores.items()
            },
            "decision_chain": self.decision_chain,
            "human_reviews": self.human_reviews,
            "warnings": self.warnings,
            "anomalies": self.anomalies_detected,
            "provenance_hash": (
                self.decision_provenance.get_hash()
                if self.decision_provenance
                else None
            ),
        }

    def verify_provenance_chain(self) -> List[str]:
        """Return broken references in aggregate/computed score provenance."""

        issues: List[str] = []
        known_ids = {
            score.provenance.provenance_id for score in self.scores.values()
        }

        for key, score in self.scores.items():
            if score.provenance.evaluator_type not in {
                EvaluatorType.AGGREGATED,
                EvaluatorType.COMPUTED,
            }:
                continue

            if not score.provenance.parent_provenances:
                issues.append(
                    f"Score '{key}' is {score.provenance.evaluator_type.value} "
                    "but has no parent provenances"
                )
                continue

            for parent_id in score.provenance.parent_provenances:
                if parent_id not in known_ids:
                    issues.append(
                        f"Score '{key}' references missing parent provenance "
                        f"'{parent_id}'"
                    )

        return issues


class ProvenanceValidator:
    """Validate the minimum provenance contract for scores and decisions."""

    @staticmethod
    def validate_score(score: Any) -> List[str]:
        issues: List[str] = []

        if not isinstance(score, SemanticScore):
            return [
                f"Score is not SemanticScore (got {type(score).__name__})"
            ]

        if not score.provenance.evaluator_id:
            issues.append("Provenance missing evaluator_id")

        if (
            score.provenance.evaluator_type == EvaluatorType.LLM_JUDGE
            and not score.provenance.reasoning
        ):
            issues.append("LLM judge provenance missing reasoning")

        if score.provenance.evaluator_type == EvaluatorType.AGGREGATED:
            if not score.provenance.input_scores:
                issues.append("Aggregated score missing input_scores")
            if not score.provenance.parent_provenances:
                issues.append("Aggregated score missing parent_provenances")

        return issues

    @staticmethod
    def validate_decision(
        decision: str,
        provenance: ProvenanceRecord,
        audit_trail: AuditTrail,
    ) -> List[str]:
        del decision
        issues: List[str] = []

        if provenance is None:
            return ["Decision missing provenance"]

        if (
            provenance.evaluator_type == EvaluatorType.COUNCIL
            and not provenance.parent_provenances
        ):
            issues.append("Council decision missing judge provenances")

        issues.extend(audit_trail.verify_provenance_chain())
        return issues


def enforce_no_raw_floats(func):
    """Reject score-like raw floats returned from decorated functions."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        if isinstance(result, float):
            raise TypeError(
                f"Function {func.__name__} returned raw float {result}. "
                "Use SemanticScore with provenance instead."
            )

        if isinstance(result, dict):
            for key, value in result.items():
                if (
                    isinstance(value, float)
                    and 0 <= value <= 1
                    and "time" not in key.lower()
                ):
                    raise TypeError(
                        f"Function {func.__name__} returned raw float in dict "
                        f"key '{key}'. Use SemanticScore with provenance instead."
                    )

        return result

    return wrapper
