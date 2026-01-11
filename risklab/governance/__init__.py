"""
Governance and oversight simulation using LangGraph for multi-agent evaluation.

Core invariants enforced:
- No float without semantics (use SemanticScore)
- No risk score without provenance (use ProvenanceRecord)
- No LLM as sole authority (use enforce_multi_evaluator)
- Every decision explainable post-hoc (use AuditTrail)
"""

from risklab.governance.judge import JudgeAgent, JudgeReport, JudgeConfig
from risklab.governance.council import EvaluationCouncil, CouncilConfig, CouncilVerdict
from risklab.governance.resources import ResourceBudget, ResourceTracker
from risklab.governance.graph import create_evaluation_graph, EvaluationState
from risklab.governance.bias_detection import (
    EvaluatorBiasDetector,
    RewardHackingDetector,
    OversightGamingAnalyzer,
    OversightGamingReport,
    EvaluatorBiasSignal,
    RewardHackingSignal,
    EvaluatorBiasType,
)
from risklab.governance.human_calibration import (
    HumanCalibrationManager,
    HumanReviewRequest,
    HumanJudgment,
    ReviewFlaggingCriteria,
    ReviewPriority,
    ReviewReason,
    generate_review_interface_html,
)
from risklab.governance.cross_model_validation import (
    CrossModelValidator,
    CrossModelConsensus,
    CrossModelJudgment,
    ModelFamily,
    ModelEvaluatorConfig,
    create_diverse_validator,
)
from risklab.governance.provenance import (
    EvaluatorType,
    ComputationMethod,
    ProvenanceRecord,
    SemanticScore,
    StrictScoreFactory,
    AuditTrail,
    ProvenanceValidator,
    enforce_no_raw_floats,
)
from risklab.governance.escalation import (
    EscalationReason,
    EscalationPriority,
    EscalationRequest,
    DivergenceResult,
    DivergenceDetector,
    SingleEvaluatorDetector,
    HighStakesAnomalyDetector,
    EscalationManager,
    enforce_multi_evaluator,
)
from risklab.governance.episode_recorder import (
    ResponseRecord,
    MetricsRecord,
    SignalsRecord,
    ConditioningRecord,
    CouncilRecord,
    PipelineRecord,
    DecisionRecord,
    EpisodeRecord,
    EpisodeRecorder,
    create_episode_recorder,
)
from risklab.governance.invariant_checker import (
    InvariantType,
    InvariantViolation,
    InvariantCheckResult,
    InvariantChecker,
    check_invariants,
    enforce_invariants,
    validate_before_export,
    generate_invariant_report,
)

__all__ = [
    "JudgeAgent",
    "JudgeReport",
    "JudgeConfig",
    "EvaluationCouncil",
    "CouncilConfig",
    "CouncilVerdict",
    "ResourceBudget",
    "ResourceTracker",
    "create_evaluation_graph",
    "EvaluationState",
    # Bias and reward hacking detection
    "EvaluatorBiasDetector",
    "RewardHackingDetector",
    "OversightGamingAnalyzer",
    "OversightGamingReport",
    "EvaluatorBiasSignal",
    "RewardHackingSignal",
    "EvaluatorBiasType",
    # Human calibration
    "HumanCalibrationManager",
    "HumanReviewRequest",
    "HumanJudgment",
    "ReviewFlaggingCriteria",
    "ReviewPriority",
    "ReviewReason",
    "generate_review_interface_html",
    # Cross-model validation
    "CrossModelValidator",
    "CrossModelConsensus",
    "CrossModelJudgment",
    "ModelFamily",
    "ModelEvaluatorConfig",
    "create_diverse_validator",
    # Provenance tracking (core invariant)
    "EvaluatorType",
    "ComputationMethod",
    "ProvenanceRecord",
    "SemanticScore",
    "StrictScoreFactory",
    "AuditTrail",
    "ProvenanceValidator",
    "enforce_no_raw_floats",
    # Human escalation
    "EscalationReason",
    "EscalationPriority",
    "EscalationRequest",
    "DivergenceResult",
    "DivergenceDetector",
    "SingleEvaluatorDetector",
    "HighStakesAnomalyDetector",
    "EscalationManager",
    "enforce_multi_evaluator",
    # Episode recording (authoritative record)
    "ResponseRecord",
    "MetricsRecord",
    "SignalsRecord",
    "ConditioningRecord",
    "CouncilRecord",
    "PipelineRecord",
    "DecisionRecord",
    "EpisodeRecord",
    "EpisodeRecorder",
    "create_episode_recorder",
    # Invariant checking
    "InvariantType",
    "InvariantViolation",
    "InvariantCheckResult",
    "InvariantChecker",
    "check_invariants",
    "enforce_invariants",
    "validate_before_export",
    "generate_invariant_report",
]
