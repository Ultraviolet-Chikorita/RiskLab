"""
Model loading utilities and factory functions.
"""

from typing import Optional, List, Dict, Any

from risklab.config import (
    LabConfig,
    ModelConfig,
    APIConfig,
    ModelProviderType,
    InstrumentationMode,
)
from risklab.models.provider import ModelProvider, ModelRef
from risklab.models.runtime import ModelRuntime, RuntimeConfig


def get_available_providers(api_config: Optional[APIConfig] = None) -> List[ModelProviderType]:
    """Get list of available providers based on configured API keys."""
    if api_config is None:
        api_config = APIConfig.from_env()
    
    available = []
    
    if api_config.openai_api_key:
        available.append(ModelProviderType.OPENAI)
    
    if api_config.anthropic_api_key:
        available.append(ModelProviderType.ANTHROPIC)
    
    # HuggingFace is always available (can use public models)
    available.append(ModelProviderType.HUGGINGFACE)
    available.append(ModelProviderType.LOCAL)
    
    return available


def load_model(
    provider: Optional[ModelProviderType] = None,
    model_name: Optional[str] = None,
    config: Optional[LabConfig] = None,
    instrumentation_mode: InstrumentationMode = InstrumentationMode.BLACK_BOX,
    **kwargs,
) -> ModelRuntime:
    """
    Load a model and return a runtime interface.
    
    Args:
        provider: Model provider type. Defaults to config default.
        model_name: Model name/identifier. Defaults to provider default.
        config: Lab configuration. Loaded from env if not provided.
        instrumentation_mode: Level of instrumentation to enable.
        **kwargs: Additional model configuration options.
    
    Returns:
        ModelRuntime instance ready for inference.
    """
    if config is None:
        config = LabConfig.from_env()
    
    if provider is None:
        provider = config.default_provider
    
    if model_name is None:
        model_name = config.get_default_model_name(provider)
    
    model_config = ModelConfig(
        provider=provider,
        model_name=model_name,
        instrumentation_mode=instrumentation_mode,
        temperature=kwargs.get("temperature", 0.7),
        max_tokens=kwargs.get("max_tokens", 2048),
        revision=kwargs.get("revision"),
        device_map=kwargs.get("device_map", "auto"),
        torch_dtype=kwargs.get("torch_dtype", "auto"),
        load_in_8bit=kwargs.get("load_in_8bit", False),
        load_in_4bit=kwargs.get("load_in_4bit", False),
    )
    
    runtime_config = RuntimeConfig(
        max_retries=kwargs.get("max_retries", 3),
        retry_delay_seconds=kwargs.get("retry_delay_seconds", 1.0),
        timeout_seconds=kwargs.get("timeout_seconds", 120.0),
        enable_caching=kwargs.get("enable_caching", True),
    )
    
    return ModelRuntime(
        model_config=model_config,
        api_config=config.api,
        runtime_config=runtime_config,
    )


def create_model_ref(
    provider: ModelProviderType,
    model_name: str,
    revision: Optional[str] = None,
    **metadata,
) -> ModelRef:
    """Create a model reference without loading the model."""
    return ModelRef(
        provider=provider,
        model_name=model_name,
        revision=revision,
        metadata=metadata,
    )


class ModelRegistry:
    """Registry for managing multiple model runtimes."""
    
    def __init__(self, config: Optional[LabConfig] = None):
        self.config = config or LabConfig.from_env()
        self._runtimes: Dict[str, ModelRuntime] = {}
    
    def register(
        self,
        name: str,
        provider: ModelProviderType,
        model_name: str,
        **kwargs,
    ) -> ModelRuntime:
        """Register and load a model with a given name."""
        runtime = load_model(
            provider=provider,
            model_name=model_name,
            config=self.config,
            **kwargs,
        )
        self._runtimes[name] = runtime
        return runtime
    
    def get(self, name: str) -> Optional[ModelRuntime]:
        """Get a registered model runtime by name."""
        return self._runtimes.get(name)
    
    def list_registered(self) -> List[str]:
        """List all registered model names."""
        return list(self._runtimes.keys())
    
    def unregister(self, name: str) -> bool:
        """Unregister a model runtime."""
        if name in self._runtimes:
            del self._runtimes[name]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all registered runtimes."""
        self._runtimes.clear()
