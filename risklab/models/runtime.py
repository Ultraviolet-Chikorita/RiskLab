"""
Model runtime management and unified interface.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from risklab.config import ModelConfig, APIConfig, ModelProviderType, InstrumentationMode
from risklab.models.provider import (
    ModelProvider,
    ModelRef,
    GenerationResult,
    OpenAIProvider,
    AnthropicProvider,
    HuggingFaceProvider,
)


class RuntimeConfig(BaseModel):
    """Configuration for model runtime."""
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    timeout_seconds: float = 120.0
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600


class ModelRuntime:
    """Unified runtime interface for model inference."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        api_config: APIConfig,
        runtime_config: Optional[RuntimeConfig] = None,
    ):
        self.model_config = model_config
        self.api_config = api_config
        self.runtime_config = runtime_config or RuntimeConfig()
        self._provider: Optional[ModelProvider] = None
        self._cache: Dict[str, GenerationResult] = {}
    
    @property
    def provider(self) -> ModelProvider:
        """Get or create the model provider."""
        if self._provider is None:
            self._provider = self._create_provider()
        return self._provider
    
    def _create_provider(self) -> ModelProvider:
        """Create the appropriate provider based on configuration."""
        provider_map = {
            ModelProviderType.OPENAI: OpenAIProvider,
            ModelProviderType.ANTHROPIC: AnthropicProvider,
            ModelProviderType.HUGGINGFACE: HuggingFaceProvider,
            ModelProviderType.LOCAL: HuggingFaceProvider,
        }
        
        provider_class = provider_map.get(self.model_config.provider)
        if provider_class is None:
            raise ValueError(f"Unsupported provider: {self.model_config.provider}")
        
        return provider_class(self.model_config, self.api_config)
    
    @property
    def model_ref(self) -> ModelRef:
        """Get the model reference."""
        return self.provider.model_ref
    
    @property
    def instrumentation_mode(self) -> InstrumentationMode:
        """Get the instrumentation mode."""
        return self.model_config.instrumentation_mode
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> GenerationResult:
        """Generate text from a prompt with optional caching."""
        import hashlib
        import asyncio
        
        # Check cache
        if use_cache and self.runtime_config.enable_caching:
            cache_key = hashlib.sha256(
                f"{prompt}:{system_prompt}:{kwargs}".encode()
            ).hexdigest()
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Generate with retries
        last_error = None
        for attempt in range(self.runtime_config.max_retries):
            try:
                result = await asyncio.wait_for(
                    self.provider.generate(prompt, system_prompt, **kwargs),
                    timeout=self.runtime_config.timeout_seconds,
                )
                
                # Cache result
                if use_cache and self.runtime_config.enable_caching:
                    self._cache[cache_key] = result
                
                return result
                
            except Exception as e:
                last_error = e
                if attempt < self.runtime_config.max_retries - 1:
                    await asyncio.sleep(
                        self.runtime_config.retry_delay_seconds * (attempt + 1)
                    )
        
        raise RuntimeError(f"Generation failed after {self.runtime_config.max_retries} attempts: {last_error}")
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> GenerationResult:
        """Generate a chat response with optional caching."""
        import hashlib
        import asyncio
        
        # Check cache
        if use_cache and self.runtime_config.enable_caching:
            cache_key = hashlib.sha256(
                f"{messages}:{system_prompt}:{kwargs}".encode()
            ).hexdigest()
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Generate with retries
        last_error = None
        for attempt in range(self.runtime_config.max_retries):
            try:
                result = await asyncio.wait_for(
                    self.provider.chat(messages, system_prompt, **kwargs),
                    timeout=self.runtime_config.timeout_seconds,
                )
                
                # Cache result
                if use_cache and self.runtime_config.enable_caching:
                    self._cache[cache_key] = result
                
                return result
                
            except Exception as e:
                last_error = e
                if attempt < self.runtime_config.max_retries - 1:
                    await asyncio.sleep(
                        self.runtime_config.retry_delay_seconds * (attempt + 1)
                    )
        
        raise RuntimeError(f"Chat generation failed after {self.runtime_config.max_retries} attempts: {last_error}")
    
    def clear_cache(self) -> None:
        """Clear the generation cache."""
        self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "enabled": self.runtime_config.enable_caching,
        }
