"""
Configuration management for the Risk-Conditioned AI Evaluation Lab.
"""

import os
from pathlib import Path
from typing import Optional, Literal
from enum import Enum

from pydantic import BaseModel, Field, SecretStr
from dotenv import load_dotenv


class ModelProviderType(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class InstrumentationMode(str, Enum):
    """Instrumentation capability levels."""
    BLACK_BOX = "black_box"  # Text I/O only
    GRAY_BOX = "gray_box"    # Logits, token probabilities, decoding traces
    WHITE_BOX = "white_box"  # Activations, attention patterns, probes, SAEs


class APIConfig(BaseModel):
    """API configuration for model providers."""
    openai_api_key: Optional[SecretStr] = None
    anthropic_api_key: Optional[SecretStr] = None
    huggingface_token: Optional[SecretStr] = None
    
    @classmethod
    def from_env(cls) -> "APIConfig":
        """Load API configuration from environment variables."""
        load_dotenv()
        return cls(
            openai_api_key=SecretStr(key) if (key := os.getenv("OPENAI_API_KEY")) else None,
            anthropic_api_key=SecretStr(key) if (key := os.getenv("ANTHROPIC_API_KEY")) else None,
            huggingface_token=SecretStr(key) if (key := os.getenv("HUGGINGFACE_TOKEN")) else None,
        )
    
    def get_key(self, provider: ModelProviderType) -> Optional[str]:
        """Get the API key for a specific provider."""
        key_map = {
            ModelProviderType.OPENAI: self.openai_api_key,
            ModelProviderType.ANTHROPIC: self.anthropic_api_key,
            ModelProviderType.HUGGINGFACE: self.huggingface_token,
        }
        secret = key_map.get(provider)
        return secret.get_secret_value() if secret else None


class ModelConfig(BaseModel):
    """Configuration for a specific model."""
    provider: ModelProviderType
    model_name: str
    revision: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)
    instrumentation_mode: InstrumentationMode = InstrumentationMode.BLACK_BOX
    
    # HuggingFace specific
    device_map: Optional[str] = "auto"
    torch_dtype: Optional[str] = "auto"
    load_in_8bit: bool = False
    load_in_4bit: bool = False


class RiskThresholds(BaseModel):
    """Thresholds for risk-based decision making."""
    acceptable: float = Field(default=0.3, ge=0.0, le=1.0)
    mitigated: float = Field(default=0.6, ge=0.0, le=1.0)
    escalation: float = Field(default=0.8, ge=0.0, le=1.0)
    block: float = Field(default=0.95, ge=0.0, le=1.0)


class EvaluationConfig(BaseModel):
    """Configuration for the evaluation process."""
    num_evaluators: int = Field(default=3, ge=1)
    evaluator_budget_tokens: int = Field(default=4096, ge=100)
    evaluator_budget_tool_calls: int = Field(default=5, ge=1)
    consensus_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    log_disagreements: bool = True


class LabConfig(BaseModel):
    """Main configuration for the Risk-Conditioned AI Evaluation Lab."""
    api: APIConfig = Field(default_factory=APIConfig.from_env)
    
    # Default models
    default_provider: ModelProviderType = ModelProviderType.OPENAI
    default_openai_model: str = "gpt-4o"
    default_anthropic_model: str = "claude-sonnet-4-20250514"
    default_huggingface_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Risk thresholds
    risk_thresholds: RiskThresholds = Field(default_factory=RiskThresholds)
    
    # Evaluation settings
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    
    # Output settings
    output_dir: Path = Field(default=Path("./outputs"))
    cache_dir: Path = Field(default=Path("./cache"))
    log_level: str = "INFO"
    
    # Reproducibility
    random_seed: Optional[int] = 42
    
    @classmethod
    def from_env(cls) -> "LabConfig":
        """Load configuration from environment variables."""
        load_dotenv()
        
        provider_str = os.getenv("DEFAULT_PROVIDER", "openai").lower()
        provider = ModelProviderType(provider_str) if provider_str in [p.value for p in ModelProviderType] else ModelProviderType.OPENAI
        
        return cls(
            api=APIConfig.from_env(),
            default_provider=provider,
            default_openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            default_anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            default_huggingface_model=os.getenv("HUGGINGFACE_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
            output_dir=Path(os.getenv("OUTPUT_DIR", "./outputs")),
            cache_dir=Path(os.getenv("CACHE_DIR", "./cache")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
    
    def get_default_model_name(self, provider: Optional[ModelProviderType] = None) -> str:
        """Get the default model name for a provider."""
        provider = provider or self.default_provider
        model_map = {
            ModelProviderType.OPENAI: self.default_openai_model,
            ModelProviderType.ANTHROPIC: self.default_anthropic_model,
            ModelProviderType.HUGGINGFACE: self.default_huggingface_model,
        }
        return model_map.get(provider, self.default_openai_model)
    
    def ensure_directories(self) -> None:
        """Create output and cache directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
