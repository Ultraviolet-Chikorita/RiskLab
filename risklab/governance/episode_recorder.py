"""
Comprehensive Episode Recorder.

Ensures episodes.json captures EVERYTHING as per the vision:
"Raw responses, Metrics, Signals, Conditioning, Decisions, 
 Council internals, Pipeline traces - Nothing is thrown away."

This is the authoritative record for auditability.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
import json
import hashlib

from risklab.governance.provenance import (
    SemanticScore, ProvenanceRecord, AuditTrail, EvaluatorType
)
from risklab.governance.escalation import EscalationRequest


class ResponseRecord(BaseModel):
    """Complete record of a model response."""
    
    # The response itself
    response_text: str
    response_hash: str = ""  # For verification
    
    # What generated it
    model_id: str
    model_version: str = ""
    
    # Input that produced it
    prompt: str
    prompt_hash: str = ""
    system_prompt: Optional[str] = None
    
    # Framing context
    framing: str = "neutral"
    
    # Generation parameters
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    
    # Timestamps
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    generation_time_ms: float = 0.0
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.response_hash:
            self.response_hash = hashlib.sha256(self.response_text.encode()).hexdigest()[:16]
        if not self.prompt_hash:
            self.prompt_hash = hashlib.sha256(self.prompt.encode()).hexdigest()[:16]


class MetricsRecord(BaseModel):
    """Complete metrics record with full provenance."""
    
    # Framing this was computed for
    framing: str
    
    # All metrics with provenance
    metrics: Dict[str, Any] = Field(default_factory=dict)  # metric_name -> SemanticScore dict
    
    # Computation metadata
    computed_at: datetime = Field(default_factory=datetime.utcnow)
    computation_time_ms: float = 0.0
    evaluators_used: List[str] = Field(default_factory=list)


class SignalsRecord(BaseModel):
    """Complete signals record."""
    
    # Derived signals
    signals: Dict[str, Any] = Field(default_factory=dict)
    
    # Cross-framing deltas
    framing_deltas: Dict[str, float] = Field(default_factory=dict)
    
    # Signal derivation metadata
    derived_from: List[str] = Field(default_factory=list)  # metric names used


class ConditioningRecord(BaseModel):
    """Risk conditioning record."""
    
    # Context used for conditioning
    domain: str
    stakes_level: str
    vulnerability_level: str
    
    # Weights applied
    domain_weight: float = 1.0
    stakes_weight: float = 1.0
    vulnerability_weight: float = 1.0
    behavior_weights: Dict[str, float] = Field(default_factory=dict)
    
    # Conditioned scores
    conditioned_metrics: Dict[str, Any] = Field(default_factory=dict)


class CouncilRecord(BaseModel):
    """Complete council evaluation record."""
    
    # Council configuration
    council_size: int = 0
    consensus_threshold: float = 0.0
    
    # Per-judge records (NEVER throw these away)
    judge_reports: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Consensus result
    consensus_reached: bool = False
    consensus_decision: Optional[str] = None
    consensus_score: Optional[float] = None
    
    # Disagreement analysis
    disagreement_score: float = 0.0
    dissenting_judges: List[str] = Field(default_factory=list)
    
    # Timing
    deliberation_time_ms: float = 0.0


class PipelineRecord(BaseModel):
    """Complete pipeline execution record."""
    
    # Pipeline identification
    pipeline_name: str
    pipeline_version: str = ""
    
    # Component-by-component trace
    component_traces: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Risk propagation
    risk_at_entry: Dict[str, float] = Field(default_factory=dict)
    risk_at_exit: Dict[str, float] = Field(default_factory=dict)
    risk_amplification_points: List[str] = Field(default_factory=list)
    
    # Gate decisions
    gates_encountered: List[Dict[str, Any]] = Field(default_factory=list)
    blocked_by: Optional[str] = None
    
    # Aggregate metrics
    total_cost: float = 0.0
    total_latency_ms: float = 0.0


class DecisionRecord(BaseModel):
    """Complete decision record with full provenance."""
    
    # The decision
    outcome: str  # accept, review, reject, escalate
    confidence: float
    
    # How it was made
    decision_method: str  # threshold, council, human, etc.
    threshold_used: Optional[float] = None
    
    # Full provenance
    provenance: Optional[Dict[str, Any]] = None
    
    # Contributing factors
    primary_factors: List[str] = Field(default_factory=list)
    risk_score: float = 0.0
    
    # Timestamps
    decided_at: datetime = Field(default_factory=datetime.utcnow)


class EpisodeRecord(BaseModel):
    """
    Complete, authoritative record of an episode evaluation.
    
    This contains EVERYTHING - nothing is thrown away.
    """
    
    # === Identification ===
    episode_id: str
    episode_name: str
    evaluation_id: str = ""
    
    # === Context ===
    context: Dict[str, Any] = Field(default_factory=dict)
    
    # === Raw Responses (per framing) ===
    responses: Dict[str, ResponseRecord] = Field(default_factory=dict)
    
    # === Metrics (per framing) ===
    metrics_by_framing: Dict[str, MetricsRecord] = Field(default_factory=dict)
    
    # === Derived Signals ===
    signals: Optional[SignalsRecord] = None
    
    # === Risk Conditioning ===
    conditioning: Optional[ConditioningRecord] = None
    
    # === Council Evaluation ===
    council: Optional[CouncilRecord] = None
    
    # === Pipeline Trace ===
    pipeline: Optional[PipelineRecord] = None
    
    # === Final Decision ===
    decision: Optional[DecisionRecord] = None
    
    # === Audit Trail ===
    audit_trail: Optional[Dict[str, Any]] = None
    
    # === Escalations ===
    escalations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # === Metadata ===
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    total_cost: float = 0.0
    total_time_ms: float = 0.0
    
    # === Invariant Violations ===
    invariant_violations: List[str] = Field(default_factory=list)
    
    # === Raw Data (catch-all for anything else) ===
    raw_data: Dict[str, Any] = Field(default_factory=dict)
    
    def finalize(self) -> None:
        """Mark episode as complete and run final checks."""
        self.completed_at = datetime.utcnow()
        
        # Check for missing data
        if not self.responses:
            self.invariant_violations.append("No responses recorded")
        if not self.metrics_by_framing:
            self.invariant_violations.append("No metrics recorded")
        if not self.decision:
            self.invariant_violations.append("No decision recorded")
    
    def to_json_safe(self) -> Dict[str, Any]:
        """Export as JSON-serializable dict."""
        def convert(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, BaseModel):
                return obj.model_dump()
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        
        return json.loads(json.dumps(self.model_dump(), default=convert))


class EpisodeRecorder:
    """
    Records all aspects of episode evaluation.
    
    Ensures nothing is lost and everything is auditable.
    """
    
    def __init__(self, evaluation_id: str = ""):
        self.evaluation_id = evaluation_id or datetime.utcnow().strftime("%Y%m%d%H%M%S")
        self.episodes: Dict[str, EpisodeRecord] = {}
        self.audit_trail = AuditTrail(evaluation_id=self.evaluation_id)
    
    def start_episode(
        self,
        episode_id: str,
        episode_name: str,
        context: Dict[str, Any],
    ) -> EpisodeRecord:
        """Start recording a new episode."""
        record = EpisodeRecord(
            episode_id=episode_id,
            episode_name=episode_name,
            evaluation_id=self.evaluation_id,
            context=context,
        )
        self.episodes[episode_id] = record
        return record
    
    def record_response(
        self,
        episode_id: str,
        framing: str,
        response_text: str,
        prompt: str,
        model_id: str,
        **kwargs,
    ) -> None:
        """Record a model response."""
        if episode_id not in self.episodes:
            raise ValueError(f"Episode {episode_id} not started")
        
        response = ResponseRecord(
            response_text=response_text,
            prompt=prompt,
            model_id=model_id,
            framing=framing,
            **kwargs,
        )
        self.episodes[episode_id].responses[framing] = response
    
    def record_metrics(
        self,
        episode_id: str,
        framing: str,
        metrics: Dict[str, SemanticScore],
        computation_time_ms: float = 0.0,
    ) -> None:
        """Record metrics for a framing."""
        if episode_id not in self.episodes:
            raise ValueError(f"Episode {episode_id} not started")
        
        # Convert SemanticScores to dicts for storage
        metrics_dict = {}
        evaluators = set()
        
        for name, score in metrics.items():
            if isinstance(score, SemanticScore):
                metrics_dict[name] = score.to_audit_entry()
                evaluators.add(f"{score.provenance.evaluator_type.value}:{score.provenance.evaluator_id}")
            else:
                metrics_dict[name] = score
        
        record = MetricsRecord(
            framing=framing,
            metrics=metrics_dict,
            computation_time_ms=computation_time_ms,
            evaluators_used=list(evaluators),
        )
        self.episodes[episode_id].metrics_by_framing[framing] = record
    
    def record_signals(
        self,
        episode_id: str,
        signals: Dict[str, Any],
        framing_deltas: Dict[str, float],
        derived_from: List[str],
    ) -> None:
        """Record derived signals."""
        if episode_id not in self.episodes:
            raise ValueError(f"Episode {episode_id} not started")
        
        self.episodes[episode_id].signals = SignalsRecord(
            signals=signals,
            framing_deltas=framing_deltas,
            derived_from=derived_from,
        )
    
    def record_conditioning(
        self,
        episode_id: str,
        domain: str,
        stakes_level: str,
        vulnerability_level: str,
        weights: Dict[str, float],
        conditioned_metrics: Dict[str, Any],
    ) -> None:
        """Record risk conditioning."""
        if episode_id not in self.episodes:
            raise ValueError(f"Episode {episode_id} not started")
        
        self.episodes[episode_id].conditioning = ConditioningRecord(
            domain=domain,
            stakes_level=stakes_level,
            vulnerability_level=vulnerability_level,
            domain_weight=weights.get("domain", 1.0),
            stakes_weight=weights.get("stakes", 1.0),
            vulnerability_weight=weights.get("vulnerability", 1.0),
            behavior_weights=weights.get("behavior", {}),
            conditioned_metrics=conditioned_metrics,
        )
    
    def record_council(
        self,
        episode_id: str,
        judge_reports: List[Dict[str, Any]],
        consensus_decision: Optional[str],
        consensus_score: Optional[float],
        disagreement_score: float,
        **kwargs,
    ) -> None:
        """Record council evaluation (NEVER omit judge reports)."""
        if episode_id not in self.episodes:
            raise ValueError(f"Episode {episode_id} not started")
        
        self.episodes[episode_id].council = CouncilRecord(
            council_size=len(judge_reports),
            judge_reports=judge_reports,  # MUST include all
            consensus_reached=consensus_decision is not None,
            consensus_decision=consensus_decision,
            consensus_score=consensus_score,
            disagreement_score=disagreement_score,
            **kwargs,
        )
    
    def record_pipeline(
        self,
        episode_id: str,
        pipeline_name: str,
        component_traces: List[Dict[str, Any]],
        risk_at_entry: Dict[str, float],
        risk_at_exit: Dict[str, float],
        **kwargs,
    ) -> None:
        """Record pipeline execution trace."""
        if episode_id not in self.episodes:
            raise ValueError(f"Episode {episode_id} not started")
        
        # Identify risk amplification points
        amplification_points = []
        for comp in component_traces:
            comp_name = comp.get("component", comp.get("name", "unknown"))
            entry_risk = comp.get("risk_at_entry", 0)
            exit_risk = comp.get("risk_at_exit", comp.get("risk", 0))
            if exit_risk > entry_risk + 0.1:
                amplification_points.append(comp_name)
        
        self.episodes[episode_id].pipeline = PipelineRecord(
            pipeline_name=pipeline_name,
            component_traces=component_traces,
            risk_at_entry=risk_at_entry,
            risk_at_exit=risk_at_exit,
            risk_amplification_points=amplification_points,
            **kwargs,
        )
    
    def record_decision(
        self,
        episode_id: str,
        outcome: str,
        confidence: float,
        decision_method: str,
        provenance: ProvenanceRecord,
        primary_factors: List[str],
        risk_score: float,
        **kwargs,
    ) -> None:
        """Record final decision with full provenance."""
        if episode_id not in self.episodes:
            raise ValueError(f"Episode {episode_id} not started")
        
        self.episodes[episode_id].decision = DecisionRecord(
            outcome=outcome,
            confidence=confidence,
            decision_method=decision_method,
            provenance=provenance.model_dump() if provenance else None,
            primary_factors=primary_factors,
            risk_score=risk_score,
            **kwargs,
        )
    
    def record_escalation(
        self,
        episode_id: str,
        escalation: EscalationRequest,
    ) -> None:
        """Record an escalation request."""
        if episode_id not in self.episodes:
            raise ValueError(f"Episode {episode_id} not started")
        
        self.episodes[episode_id].escalations.append(escalation.model_dump())
    
    def record_invariant_violation(
        self,
        episode_id: str,
        violation: str,
    ) -> None:
        """Record an invariant violation."""
        if episode_id not in self.episodes:
            raise ValueError(f"Episode {episode_id} not started")
        
        self.episodes[episode_id].invariant_violations.append(violation)
        self.audit_trail.add_warning(f"INVARIANT VIOLATION in {episode_id}: {violation}")
    
    def finalize_episode(self, episode_id: str) -> EpisodeRecord:
        """Finalize an episode and run completeness checks."""
        if episode_id not in self.episodes:
            raise ValueError(f"Episode {episode_id} not started")
        
        episode = self.episodes[episode_id]
        episode.finalize()
        
        # Check completeness
        self._check_completeness(episode)
        
        return episode
    
    def _check_completeness(self, episode: EpisodeRecord) -> None:
        """Check episode has all required data."""
        required_fields = [
            ("responses", "No responses recorded"),
            ("metrics_by_framing", "No metrics recorded"),
            ("decision", "No decision recorded"),
        ]
        
        for field, message in required_fields:
            if not getattr(episode, field):
                episode.invariant_violations.append(message)
        
        # Check provenance on decision
        if episode.decision and not episode.decision.provenance:
            episode.invariant_violations.append("Decision missing provenance")
        
        # Check council if high stakes
        stakes = episode.context.get("stakes_level", "medium")
        if stakes in ["high", "critical"] and not episode.council:
            episode.invariant_violations.append(
                f"High stakes ({stakes}) episode missing council evaluation"
            )
    
    def export_all(self, output_path: Path) -> Path:
        """Export all episodes to JSON file."""
        output_path = Path(output_path)
        
        episodes_data = [
            ep.to_json_safe() for ep in self.episodes.values()
        ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(episodes_data, f, indent=2, default=str)
        
        return output_path
    
    def get_completeness_report(self) -> Dict[str, Any]:
        """Get report on data completeness."""
        complete = 0
        incomplete = 0
        violations = []
        
        for ep_id, ep in self.episodes.items():
            if ep.invariant_violations:
                incomplete += 1
                violations.extend([
                    f"{ep_id}: {v}" for v in ep.invariant_violations
                ])
            else:
                complete += 1
        
        return {
            "total_episodes": len(self.episodes),
            "complete": complete,
            "incomplete": incomplete,
            "violations": violations,
            "completeness_rate": complete / len(self.episodes) if self.episodes else 1.0,
        }


def create_episode_recorder(evaluation_id: str = "") -> EpisodeRecorder:
    """Create a new episode recorder."""
    return EpisodeRecorder(evaluation_id)
