"""
Scenario and context specification for evaluation episodes.
"""

from risklab.scenarios.episode import Episode, EpisodeVariant
from risklab.scenarios.framing import Framing, FramingType
from risklab.scenarios.context import ContextMetadata, Domain, StakesLevel, VulnerabilityLevel
from risklab.scenarios.library import ScenarioLibrary, load_default_scenarios
from risklab.scenarios.institutional import (
    InstitutionalDivergenceEpisode,
    InstitutionalRegime,
    NormDomain,
    InstitutionalNorm,
    DecisionBoundary,
    PressureType,
    InstitutionalScenarioLibrary,
    load_institutional_scenarios,
    create_pressure_variants,
)
from risklab.scenarios.bias_probes import (
    BiasProbe,
    BiasProbeLibrary,
    ProbeType,
    ExpectedScores,
    EvaluatorCalibrator,
    CalibrationResult,
    load_all_bias_probes,
)
from risklab.scenarios.multi_message_episodes import (
    MultiMessageEpisode,
    ConversationMessage,
    MessageRole,
    HiddenAgenda,
    create_multi_message_episodes,
)
from risklab.scenarios.sycophancy_scenarios import (
    SycophancyType,
    create_sycophancy_scenarios,
    get_sycophancy_scenarios_by_type,
)
from risklab.scenarios.sycon_bench import (
    SYCONSetting,
    PromptStrategy,
    SYCONResult,
    SYCONAggregateResults,
    create_sycon_debate_episodes,
    create_sycon_ethical_episodes,
    create_sycon_presupposition_episodes,
    create_all_sycon_episodes,
    compute_sycon_metrics,
    detect_stance_debate,
    detect_stance_ethical,
    detect_stance_presupposition,
    DEBATE_TOPICS,
    ETHICAL_QUESTIONS,
    FALSE_PRESUPPOSITION_QUESTIONS,
)

__all__ = [
    "Episode",
    "EpisodeVariant",
    "Framing",
    "FramingType",
    "ContextMetadata",
    "Domain",
    "StakesLevel",
    "VulnerabilityLevel",
    "ScenarioLibrary",
    "load_default_scenarios",
    # Institutional divergence
    "InstitutionalDivergenceEpisode",
    "InstitutionalRegime",
    "NormDomain",
    "InstitutionalNorm",
    "DecisionBoundary",
    "PressureType",
    "InstitutionalScenarioLibrary",
    "load_institutional_scenarios",
    "create_pressure_variants",
    # Bias probes for calibration
    "BiasProbe",
    "BiasProbeLibrary",
    "ProbeType",
    "ExpectedScores",
    "EvaluatorCalibrator",
    "CalibrationResult",
    "load_all_bias_probes",
    # Multi-message episodes
    "MultiMessageEpisode",
    "ConversationMessage",
    "MessageRole",
    "HiddenAgenda",
    "create_multi_message_episodes",
    # Sycophancy scenarios
    "SycophancyType",
    "create_sycophancy_scenarios",
    "get_sycophancy_scenarios_by_type",
    # SYCON-Bench integration
    "SYCONSetting",
    "PromptStrategy",
    "SYCONResult",
    "SYCONAggregateResults",
    "create_sycon_debate_episodes",
    "create_sycon_ethical_episodes",
    "create_sycon_presupposition_episodes",
    "create_all_sycon_episodes",
    "compute_sycon_metrics",
    "detect_stance_debate",
    "detect_stance_ethical",
    "detect_stance_presupposition",
    "DEBATE_TOPICS",
    "ETHICAL_QUESTIONS",
    "FALSE_PRESUPPOSITION_QUESTIONS",
]