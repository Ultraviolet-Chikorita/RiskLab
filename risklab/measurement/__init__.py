"""
Behavioral measurement and manipulation signal detection.
"""

from risklab.measurement.metrics import BehavioralMetrics, MetricResult
from risklab.measurement.signals import ManipulationSignals, SignalResult
from risklab.measurement.analyzers import (
    ResponseAnalyzer,
    SycophancyAnalyzer,
    OmissionAnalyzer,
    HedgingAnalyzer,
    RefusalAnalyzer,
)
from risklab.measurement.comparators import FramingComparator, CrossFramingAnalysis
from risklab.measurement.norm_alignment import (
    NormAlignmentScore,
    NormAlignmentComputer,
    NormStabilityCurve,
    NormStabilityAnalyzer,
    JustificationConsistency,
    AlignmentDirection,
    InstitutionalBiasProfile,
    InstitutionalBiasAnalyzer,
)
from risklab.measurement.latent_bias_model import (
    BiasVector,
    SusceptibilityVector,
    ModelLatentProfile,
    LatentBiasModel,
    ModelEvolutionTracker,
)
from risklab.measurement.whitebox_analyzer import (
    WhiteBoxMetrics,
    WhiteBoxBehaviorAnalyzer,
    WhiteBoxAnalyzerFactory,
    DeceptionProbe,
    SycophancyProbe,
    UncertaintyProbe,
    compute_whitebox_manipulation_score,
)
from risklab.measurement.rule_based import (
    RuleBasedAnalyzer,
    RuleBasedResult,
    create_rule_based_analyzer,
    SYCOPHANCY_PATTERNS,
    HEDGING_PATTERNS,
    REFUSAL_PATTERNS,
)
from risklab.measurement.anomaly_detection import (
    StatisticalAnomalyDetector,
    AnomalyReport,
    AnomalySignal,
    AnomalyType,
    compute_divergence_summary,
)
from risklab.measurement.classifiers import (
    BaseClassifier,
    ClassifierOutput,
    ClassifierRegistry,
    SentimentClassifierML,
    IntentClassifierML,
    ToxicityClassifierML,
    QualityValidatorML,
)
from risklab.measurement.classifier_training import (
    TrainingDataset,
    TrainingExample,
    TrainingMetrics,
    ClassifierTrainer,
    create_sentiment_dataset,
    create_intent_dataset,
    create_quality_dataset,
    train_all_classifiers,
    load_pretrained_classifier,
    PRETRAINED_MODELS,
)
from risklab.measurement.enhanced_metrics import (
    EnhancedMetricType,
    ReasoningCoherenceAnalyzer,
    TemporalConsistencyTracker,
    BoundaryRespectAnalyzer,
    ManipulationResistanceAnalyzer,
    ValueStabilityAnalyzer,
    InstructionAdherenceAnalyzer,
    EnhancedMetricsComputer,
)

__all__ = [
    "BehavioralMetrics",
    "MetricResult",
    "ManipulationSignals",
    "SignalResult",
    "ResponseAnalyzer",
    "SycophancyAnalyzer",
    "OmissionAnalyzer",
    "HedgingAnalyzer",
    "RefusalAnalyzer",
    "FramingComparator",
    "CrossFramingAnalysis",
    # Norm alignment & institutional bias
    "NormAlignmentScore",
    "NormAlignmentComputer",
    "NormStabilityCurve",
    "NormStabilityAnalyzer",
    "JustificationConsistency",
    "AlignmentDirection",
    "InstitutionalBiasProfile",
    "InstitutionalBiasAnalyzer",
    # Latent bias model
    "BiasVector",
    "SusceptibilityVector",
    "ModelLatentProfile",
    "LatentBiasModel",
    "ModelEvolutionTracker",
    # White-box analysis
    "WhiteBoxMetrics",
    "WhiteBoxBehaviorAnalyzer",
    "WhiteBoxAnalyzerFactory",
    "DeceptionProbe",
    "SycophancyProbe",
    "UncertaintyProbe",
    "compute_whitebox_manipulation_score",
    # Rule-based (LLM-free) analysis
    "RuleBasedAnalyzer",
    "RuleBasedResult",
    "create_rule_based_analyzer",
    "SYCOPHANCY_PATTERNS",
    "HEDGING_PATTERNS",
    "REFUSAL_PATTERNS",
    # Statistical anomaly detection
    "StatisticalAnomalyDetector",
    "AnomalyReport",
    "AnomalySignal",
    "AnomalyType",
    "compute_divergence_summary",
    # ML Classifiers (fast, cheap, non-LLM)
    "BaseClassifier",
    "ClassifierOutput",
    "ClassifierRegistry",
    "SentimentClassifierML",
    "IntentClassifierML",
    "ToxicityClassifierML",
    "QualityValidatorML",
    # Classifier Training
    "TrainingDataset",
    "TrainingExample",
    "TrainingMetrics",
    "ClassifierTrainer",
    "create_sentiment_dataset",
    "create_intent_dataset",
    "create_quality_dataset",
    "train_all_classifiers",
    "load_pretrained_classifier",
    "PRETRAINED_MODELS",
    # Enhanced Metrics
    "EnhancedMetricType",
    "ReasoningCoherenceAnalyzer",
    "TemporalConsistencyTracker",
    "BoundaryRespectAnalyzer",
    "ManipulationResistanceAnalyzer",
    "ValueStabilityAnalyzer",
    "InstructionAdherenceAnalyzer",
    "EnhancedMetricsComputer",
]
