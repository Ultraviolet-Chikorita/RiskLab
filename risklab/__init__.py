"""
Risk-Conditioned AI Evaluation Lab

A System for Measuring, Analyzing, and Governing Manipulative Behavior in AI Systems
"""

__version__ = "0.1.0"
__author__ = "AI Safety Lab"

from risklab.config import LabConfig
from risklab.models import ModelProvider, ModelRef
from risklab.scenarios import Episode, Framing, ContextMetadata
from risklab.measurement import BehavioralMetrics, ManipulationSignals
from risklab.risk import RiskConditioner, RiskWeights
from risklab.governance import EvaluationCouncil, JudgeReport
from risklab.lab import RiskConditionedLab
from risklab.utils import (
    RiskLevel,
    compute_risk_level,
    extract_json_from_text,
    render_figure_to_output,
    compute_word_overlap_similarity,
)

__all__ = [
    "LabConfig",
    "ModelProvider",
    "ModelRef",
    "Episode",
    "Framing",
    "ContextMetadata",
    "BehavioralMetrics",
    "ManipulationSignals",
    "RiskConditioner",
    "RiskWeights",
    "EvaluationCouncil",
    "JudgeReport",
    "RiskConditionedLab",
    # Consolidated utilities
    "RiskLevel",
    "compute_risk_level",
    "extract_json_from_text",
    "render_figure_to_output",
    "compute_word_overlap_similarity",
]
