"""
Provenance Tracking System for RiskLab.

Enforces the core design invariant:
"Every risk score must be explainable, decomposable, and reproducible"

Provides:
- Full computation chain tracking
- Semantic score wrappers (no float without semantics)
- Audit trail generation
- Reproducibility verification
"""

from typing import Optional, List, Dict, Any, Union, TypeVar, Generic
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator
import hashlib
import json
from dataclasses import dataclass


class EvaluatorType(str, Enum):
    """Type of evaluator that produced a score."""
    RULE_BASED = "rule_based"
    ML_CLASSIFIER = "ml_classifier"
    LLM_JUDGE = "llm_judge"
    HEURISTIC = "heuristic"
    HUMAN = "human"
    COUNCIL = "council"
    AGGREGATED = "aggregated"
    COMPUTED = "computed"  # Derived from other scores


class ComputationMethod(str, Enum):
    """How a score was computed."""
    DIRECT = "direct"  # Single evaluation
    AVERAGED = "averaged"  # Mean of multiple
    WEIGHTED = "weighted"  # Weighted combination
    MAX = "max"  # Maximum of inputs
    MIN = "min"  # Minimum of inputs
    FORMULA = "formula"  # Custom formula
    THRESHOLD = "threshold"  # Binary threshold
    NORMALIZED = "normalized"  # Normalized from raw


class ProvenanceRecord(BaseModel):
    """
    Complete provenance record for a score.
    
    This is the atomic unit of auditability - every score
    must have one of these attached.
    """
    
    # Unique identifier for this computation
    provenance_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"))
    
    # When was this computed
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # What type of evaluator produced this
    evaluator_type: EvaluatorType
    evaluator_id: str = ""  # Specific identifier (e.g., "gpt-4", "toxicity_bert", "rule_sycophancy_v2")
    evaluator_version: str = ""
    
    # How was it computed
    computation_method: ComputationMethod = ComputationMethod.DIRECT
    computation_formula: Optional[str] = None  # For FORMULA type
    
    # What were the inputs
    input_scores: Dict[str, float] = Field(default_factory=dict)  # provenance_id -> value
    input_text_hash: Optional[str] = None  # Hash of input text for reproducibility
    input_context_hash: Optional[str] = None  # Hash of context
    
    # Weights used (if any)
    weights_applied: Dict[str, float] = Field(default_factory=dict)
    
    # Evidence and reasoning
    evidence: List[str] = Field(default_factory=list)
    reasoning: Optional[str] = None
    
    # Confidence in this computation
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence_factors: Dict[str, float] = Field(default_factory=dict)
    
    # For reproducibility
    random_seed: Optional[int] = None
    model_temperature: Optional[float] = None
    
    # Parent provenance (for aggregated/computed scores)
    parent_provenances: List[str] = Field(default_factory=list)
    
    def get_hash(self) -> str:
        """Get deterministic hash of this provenance for verification."""
        data = {
            "evaluator_type": self.evaluator_type.value,
            "evaluator_id": self.evaluator_id,
            "computation_method": self.computation_method.value,
            "input_scores": self.input_scores,
            "input_text_hash": self.input_text_hash,
            "weights_applied": self.weights_applied,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


class SemanticScore(BaseModel):
    """
    A score with mandatory semantic context and provenance.
    
    This enforces the invariant: "No float without semantics"
    
    NEVER use raw floats for scores - always wrap in SemanticScore.
    """
    
    # The actual value
    value: float = Field(ge=0.0, le=1.0)
    
    # What does this score represent
    metric_name: str
    metric_description: str = ""
    
    # What does the scale mean
    scale_low: str = "0.0 = none/absent"
    scale_high: str = "1.0 = maximum/complete"
    
    # Is higher better or worse?
    higher_is_worse: bool = False  # True for risk scores, False for quality scores
    
    # Full provenance
    provenance: ProvenanceRecord
    
    # Optional: human-interpretable level
    interpretation: Optional[str] = None
    
    @field_validator('value')
    @classmethod
    def validate_value(cls, v):
        if not isinstance(v, (int, float)):
            raise ValueError(f"Score value must be numeric, got {type(v)}")
        return float(v)
    
    def get_interpretation(self) -> str:
        """Get human-readable interpretation."""
        if self.interpretation:
            return self.interpretation
        
        if self.higher_is_worse:
            if self.value < 0.2:
                return "minimal"
            elif self.value < 0.4:
                return "low"
            elif self.value < 0.6:
                return "moderate"
            elif self.value < 0.8:
                return "high"
            else:
                return "severe"
        else:
            if self.value < 0.2:
                return "very poor"
            elif self.value < 0.4:
                return "poor"
            elif self.value < 0.6:
                return "moderate"
            elif self.value < 0.8:
                return "good"
            else:
                return "excellent"
    
    def to_audit_entry(self) -> Dict[str, Any]:
        """Export as audit trail entry."""
        return {
            "metric": self.metric_name,
            "value": self.value,
            "interpretation": self.get_interpretation(),
            "higher_is_worse": self.higher_is_worse,
            "provenance": {
                "id": self.provenance.provenance_id,
                "evaluator": f"{self.provenance.evaluator_type.value}:{self.provenance.evaluator_id}",
                "method": self.provenance.computation_method.value,
                "confidence": self.provenance.confidence,
                "timestamp": self.provenance.timestamp.isoformat(),
                "hash": self.provenance.get_hash(),
            },
            "evidence": self.provenance.evidence[:5],
        }


class StrictScoreFactory:
    """
    Factory for creating scores with mandatory provenance.
    
    Use this instead of raw floats to ensure provenance is always tracked.
    """
    
    @staticmethod
    def from_rule(
        value: float,
        metric_name: str,
        rule_id: str,
        evidence: List[str],
        higher_is_worse: bool = True,
        metric_description: str = "",
    ) -> SemanticScore:
        """Create score from rule-based evaluation."""
        provenance = ProvenanceRecord(
            evaluator_type=EvaluatorType.RULE_BASED,
            evaluator_id=rule_id,
            computation_method=ComputationMethod.DIRECT,
            evidence=evidence,
            confidence=0.8,  # Rules are deterministic but may miss nuance
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
        """Create score from ML classifier."""
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
        """Create score from LLM judge."""
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
        """Create score from human evaluation."""
        provenance = ProvenanceRecord(
            evaluator_type=EvaluatorType.HUMAN,
            evaluator_id=annotator_id,
            computation_method=ComputationMethod.DIRECT,
            reasoning=reasoning,
            confidence=0.95,  # Humans are authoritative
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
        """Aggregate multiple scores with full provenance chain."""
        if not scores:
            raise ValueError("Cannot aggregate empty score list")
        
        input_scores = {s.provenance.provenance_id: s.value for s in scores}
        parent_provenances = [s.provenance.provenance_id for s in scores]
        
        # Compute aggregated value
        if method == ComputationMethod.AVERAGED:
            value = sum(s.value for s in scores) / len(scores)
        elif method == ComputationMethod.WEIGHTED and weights:
            total_weight = sum(weights.values())
            value = sum(s.value * weights.get(s.metric_name, 1.0) for s in scores) / total_weight
        elif method == ComputationMethod.MAX:
            value = max(s.value for s in scores)
        elif method == ComputationMethod.MIN:
            value = min(s.value for s in scores)
        else:
            value = sum(s.value for s in scores) / len(scores)
        
        # Aggregate confidence
        avg_confidence = sum(s.provenance.confidence for s in scores) / len(scores)
        
        provenance = ProvenanceRecord(
            evaluator_type=EvaluatorType.AGGREGATED,
            evaluator_id="aggregator",
            computation_method=method,
            input_scores=input_scores,
            weights_applied=weights or {},
            parent_provenances=parent_provenances,
            confidence=avg_confidence * 0.95,  # Slight penalty for aggregation
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
        """Create computed score with formula provenance."""
        input_values = {k: s.value for k, s in input_scores.items()}
        parent_provenances = [s.provenance.provenance_id for s in input_scores.values()]
        
        avg_confidence = sum(s.provenance.confidence for s in input_scores.values()) / len(input_scores)
        
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
    """
    Complete audit trail for an evaluation.
    
    Captures everything needed to explain and reproduce decisions.
    """
    
    # Identification
    audit_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"))
    evaluation_id: str = ""
    episode_id: str = ""
    
    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    # All scores with provenance
    scores: Dict[str, SemanticScore] = Field(default_factory=dict)
    
    # Decision chain
    decision_chain: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Final decision
    final_decision: Optional[str] = None
    decision_provenance: Optional[ProvenanceRecord] = None
    
    # Human interactions
    human_reviews: List[Dict[str, Any]] = Field(default_factory=list)
    human_overrides: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Warnings and anomalies
    warnings: List[str] = Field(default_factory=list)
    anomalies_detected: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Reproducibility info
    random_seeds_used: Dict[str, int] = Field(default_factory=dict)
    model_versions: Dict[str, str] = Field(default_factory=dict)
    
    def add_score(self, key: str, score: SemanticScore) -> None:
        """Add a score to the audit trail."""
        self.scores[key] = score
    
    def add_decision_step(
        self,
        step_name: str,
        input_scores: List[str],
        output: str,
        reasoning: str,
    ) -> None:
        """Record a decision step."""
        self.decision_chain.append({
            "step": step_name,
            "timestamp": datetime.utcnow().isoformat(),
            "input_scores": input_scores,
            "output": output,
            "reasoning": reasoning,
        })
    
    def add_human_review(
        self,
        reviewer_id: str,
        action: str,
        reasoning: Optional[str] = None,
    ) -> None:
        """Record human review."""
        self.human_reviews.append({
            "reviewer": reviewer_id,
            "action": action,
            "reasoning": reasoning,
            "timestamp": datetime.utcnow().isoformat(),
        })
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the audit trail."""
        self.warnings.append(f"[{datetime.utcnow().isoformat()}] {warning}")
    
    def finalize(self, decision: str, provenance: ProvenanceRecord) -> None:
        """Finalize the audit trail with final decision."""
        self.final_decision = decision
        self.decision_provenance = provenance
        self.completed_at = datetime.utcnow()
    
    def to_report(self) -> Dict[str, Any]:
        """Export as audit report."""
        return {
            "audit_id": self.audit_id,
            "evaluation_id": self.evaluation_id,
            "episode_id": self.episode_id,
            "duration_ms": (self.completed_at - self.started_at).total_seconds() * 1000 if self.completed_at else None,
            "final_decision": self.final_decision,
            "scores": {k: v.to_audit_entry() for k, v in self.scores.items()},
            "decision_chain": self.decision_chain,
            "human_reviews": self.human_reviews,
            "warnings": self.warnings,
            "anomalies": self.anomalies_detected,
            "provenance_hash": self.decision_provenance.get_hash() if self.decision_provenance else None,
        }
    
    def verify_provenance_chain(self) -> List[str]:
        """Verify all provenance chains are complete. Returns list of issues."""
        issues = []
        
        for key, score in self.scores.items():
            # Check aggregated/computed scores have parents
            if score.provenance.evaluator_type in [EvaluatorType.AGGREGATED, EvaluatorType.COMPUTED]:
                if not score.provenance.parent_provenances:
                    issues.append(f"Score '{key}' is {score.provenance.evaluator_type.value} but has no parent provenances")
                
                # Verify parents exist
                for parent_id in score.provenance.parent_provenances:
                    found = any(s.provenance.provenance_id == parent_id for s in self.scores.values())
                    if not found:
                        issues.append(f"Score '{key}' references missing parent provenance '{parent_id}'")
        
        return issues


class ProvenanceValidator:
    """
    Validates provenance requirements are met.
    
    Use this to enforce the invariant before accepting any evaluation.
    """
    
    @staticmethod
    def validate_score(score: Any) -> List[str]:
        """Validate a score has proper provenance. Returns list of issues."""
        issues = []
        
        # Must be SemanticScore
        if not isinstance(score, SemanticScore):
            issues.append(f"Score is not SemanticScore (got {type(score).__name__})")
            return issues
        
        # Must have provenance
        if not score.provenance:
            issues.append("Score missing provenance")
            return issues
        
        # Provenance must have evaluator
        if not score.provenance.evaluator_type:
            issues.append("Provenance missing evaluator_type")
        
        if not score.provenance.evaluator_id:
            issues.append("Provenance missing evaluator_id")
        
        # LLM evaluations must have reasoning
        if score.provenance.evaluator_type == EvaluatorType.LLM_JUDGE:
            if not score.provenance.reasoning:
                issues.append("LLM judge provenance missing reasoning")
        
        # Aggregated scores must have inputs
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
        """Validate a decision has proper provenance. Returns list of issues."""
        issues = []
        
        # Decision must have provenance
        if not provenance:
            issues.append("Decision missing provenance")
            return issues
        
        # Decision provenance should reference contributing scores
        if provenance.evaluator_type == EvaluatorType.COUNCIL:
            if not provenance.parent_provenances:
                issues.append("Council decision missing judge provenances")
        
        # Audit trail should be complete
        trail_issues = audit_trail.verify_provenance_chain()
        issues.extend(trail_issues)
        
        return issues


def enforce_no_raw_floats(func):
    """
    Decorator to enforce that functions don't return raw floats for scores.
    
    Use this on any function that returns scores to catch violations.
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        # Check return value
        if isinstance(result, float):
            raise TypeError(
                f"Function {func.__name__} returned raw float {result}. "
                "Use SemanticScore with provenance instead."
            )
        
        # Check dict values
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, float) and 0 <= value <= 1 and "time" not in key.lower():
                    raise TypeError(
                        f"Function {func.__name__} returned raw float in dict key '{key}'. "
                        "Use SemanticScore with provenance instead."
                    )
        
        return result
    
    return wrapper
