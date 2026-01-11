"""
Model management and provider abstractions.
"""

from risklab.models.provider import ModelProvider, ModelRef
from risklab.models.runtime import ModelRuntime, RuntimeConfig
from risklab.models.loader import load_model, get_available_providers
from risklab.models.whitebox import (
    WhiteBoxConfig,
    WhiteBoxInstrumenter,
    WhiteBoxAnalysisResult,
    ActivationCapture,
    AttentionPattern,
    LinearProbe,
    SparseAutoencoder,
    LayerType,
)

__all__ = [
    "ModelProvider",
    "ModelRef",
    "ModelRuntime",
    "RuntimeConfig",
    "load_model",
    "get_available_providers",
    # White-box instrumentation
    "WhiteBoxConfig",
    "WhiteBoxInstrumenter",
    "WhiteBoxAnalysisResult",
    "ActivationCapture",
    "AttentionPattern",
    "LinearProbe",
    "SparseAutoencoder",
    "LayerType",
]
