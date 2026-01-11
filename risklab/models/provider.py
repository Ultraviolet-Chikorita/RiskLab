"""
Model provider abstractions and reference management.
"""

import hashlib
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from enum import Enum

from pydantic import BaseModel, Field

from risklab.config import ModelProviderType, InstrumentationMode, APIConfig, ModelConfig


class ModelCapability(str, Enum):
    """Capabilities that a model may support."""
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    LOGPROBS = "logprobs"
    EMBEDDINGS = "embeddings"
    STREAMING = "streaming"


class ModelRef(BaseModel):
    """Immutable reference to a specific model version."""
    provider: ModelProviderType
    model_name: str
    revision: Optional[str] = None
    model_hash: Optional[str] = None
    capabilities: List[ModelCapability] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @property
    def identifier(self) -> str:
        """Unique identifier for this model reference."""
        parts = [self.provider.value, self.model_name]
        if self.revision:
            parts.append(self.revision)
        return "/".join(parts)
    
    def compute_hash(self) -> str:
        """Compute a hash for this model reference."""
        content = f"{self.provider.value}:{self.model_name}:{self.revision or 'latest'}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class GenerationResult(BaseModel):
    """Result of a text generation request."""
    text: str
    finish_reason: Optional[str] = None
    usage: Dict[str, int] = Field(default_factory=dict)
    
    # Gray-box outputs (if available)
    logprobs: Optional[List[Dict[str, Any]]] = None
    token_ids: Optional[List[int]] = None
    
    # White-box outputs (if available)
    activations: Optional[Dict[str, Any]] = None
    attention_patterns: Optional[Dict[str, Any]] = None
    
    # Metadata
    model_ref: Optional[ModelRef] = None
    latency_ms: Optional[float] = None
    raw_response: Optional[Dict[str, Any]] = None


class ModelProvider(ABC):
    """Abstract base class for model providers."""
    
    def __init__(
        self,
        config: ModelConfig,
        api_config: APIConfig,
    ):
        self.config = config
        self.api_config = api_config
        self._model_ref: Optional[ModelRef] = None
    
    @property
    def model_ref(self) -> ModelRef:
        """Get the model reference for this provider."""
        if self._model_ref is None:
            self._model_ref = self._create_model_ref()
        return self._model_ref
    
    @abstractmethod
    def _create_model_ref(self) -> ModelRef:
        """Create a model reference for this provider."""
        pass
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate text from a prompt."""
        pass
    
    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        """Generate a response in a chat context."""
        pass
    
    def supports_capability(self, capability: ModelCapability) -> bool:
        """Check if this provider supports a specific capability."""
        return capability in self.model_ref.capabilities
    
    @property
    def instrumentation_mode(self) -> InstrumentationMode:
        """Get the instrumentation mode for this provider."""
        return self.config.instrumentation_mode


class OpenAIProvider(ModelProvider):
    """OpenAI API provider."""
    
    def __init__(self, config: ModelConfig, api_config: APIConfig):
        super().__init__(config, api_config)
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            api_key = self.api_config.get_key(ModelProviderType.OPENAI)
            if not api_key:
                raise ValueError("OpenAI API key not configured")
            self._client = AsyncOpenAI(api_key=api_key)
        return self._client
    
    def _create_model_ref(self) -> ModelRef:
        capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.FUNCTION_CALLING,
            ModelCapability.STREAMING,
        ]
        if "gpt-4" in self.config.model_name:
            capabilities.append(ModelCapability.VISION)
        if self.config.instrumentation_mode != InstrumentationMode.BLACK_BOX:
            capabilities.append(ModelCapability.LOGPROBS)
        
        return ModelRef(
            provider=ModelProviderType.OPENAI,
            model_name=self.config.model_name,
            revision=self.config.revision,
            capabilities=capabilities,
            metadata={"temperature": self.config.temperature},
        )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self.chat(messages, **kwargs)
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        import time
        start_time = time.time()
        
        if system_prompt and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        request_kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        
        # Enable logprobs for gray-box mode
        if self.config.instrumentation_mode != InstrumentationMode.BLACK_BOX:
            request_kwargs["logprobs"] = True
            request_kwargs["top_logprobs"] = 5
        
        response = await self.client.chat.completions.create(**request_kwargs)
        
        latency_ms = (time.time() - start_time) * 1000
        
        choice = response.choices[0]
        logprobs_data = None
        if choice.logprobs and choice.logprobs.content:
            logprobs_data = [
                {
                    "token": lp.token,
                    "logprob": lp.logprob,
                    "top_logprobs": [
                        {"token": t.token, "logprob": t.logprob}
                        for t in (lp.top_logprobs or [])
                    ]
                }
                for lp in choice.logprobs.content
            ]
        
        return GenerationResult(
            text=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            logprobs=logprobs_data,
            model_ref=self.model_ref,
            latency_ms=latency_ms,
        )


class AnthropicProvider(ModelProvider):
    """Anthropic API provider."""
    
    def __init__(self, config: ModelConfig, api_config: APIConfig):
        super().__init__(config, api_config)
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            api_key = self.api_config.get_key(ModelProviderType.ANTHROPIC)
            if not api_key:
                raise ValueError("Anthropic API key not configured")
            self._client = AsyncAnthropic(api_key=api_key)
        return self._client
    
    def _create_model_ref(self) -> ModelRef:
        return ModelRef(
            provider=ModelProviderType.ANTHROPIC,
            model_name=self.config.model_name,
            revision=self.config.revision,
            capabilities=[
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CHAT,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.VISION,
                ModelCapability.STREAMING,
            ],
            metadata={"temperature": self.config.temperature},
        )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, system_prompt=system_prompt, **kwargs)
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        import time
        start_time = time.time()
        
        request_kwargs = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        
        if system_prompt:
            request_kwargs["system"] = system_prompt
        
        response = await self.client.messages.create(**request_kwargs)
        
        latency_ms = (time.time() - start_time) * 1000
        
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
        
        return GenerationResult(
            text=text,
            finish_reason=response.stop_reason,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            model_ref=self.model_ref,
            latency_ms=latency_ms,
        )


class HuggingFaceProvider(ModelProvider):
    """HuggingFace Transformers provider for local inference."""
    
    def __init__(self, config: ModelConfig, api_config: APIConfig):
        super().__init__(config, api_config)
        self._model = None
        self._tokenizer = None
        self._instrumenter = None  # White-box instrumenter
    
    def _load_model(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            token = self.api_config.get_key(ModelProviderType.HUGGINGFACE)
            
            load_kwargs = {
                "device_map": self.config.device_map,
                "trust_remote_code": True,
            }
            
            if token:
                load_kwargs["token"] = token
            
            if self.config.torch_dtype == "auto":
                load_kwargs["torch_dtype"] = torch.float16
            elif self.config.torch_dtype:
                load_kwargs["torch_dtype"] = getattr(torch, self.config.torch_dtype)
            
            if self.config.load_in_8bit:
                load_kwargs["load_in_8bit"] = True
            elif self.config.load_in_4bit:
                load_kwargs["load_in_4bit"] = True
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                token=token,
                trust_remote_code=True,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **load_kwargs,
            )
    
    @property
    def model(self):
        self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        self._load_model()
        return self._tokenizer
    
    @property
    def instrumenter(self):
        """Get or create white-box instrumenter."""
        if self._instrumenter is None and self.config.instrumentation_mode == InstrumentationMode.WHITE_BOX:
            from risklab.models.whitebox import WhiteBoxInstrumenter, WhiteBoxConfig
            whitebox_config = WhiteBoxConfig(
                capture_activations=True,
                capture_attention=True,
                layers_to_capture=None,  # All layers
            )
            self._instrumenter = WhiteBoxInstrumenter(self.model, self.tokenizer, whitebox_config)
        return self._instrumenter
    
    def _create_model_ref(self) -> ModelRef:
        capabilities = [
            ModelCapability.TEXT_GENERATION,
            ModelCapability.CHAT,
            ModelCapability.LOGPROBS,
        ]
        
        return ModelRef(
            provider=ModelProviderType.HUGGINGFACE,
            model_name=self.config.model_name,
            revision=self.config.revision,
            capabilities=capabilities,
            metadata={
                "temperature": self.config.temperature,
                "device_map": self.config.device_map,
            },
        )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        import time
        import torch
        
        start_time = time.time()
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        
        gen_kwargs = {
            "max_new_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        # Enable logprobs for gray/white-box mode
        output_scores = self.config.instrumentation_mode != InstrumentationMode.BLACK_BOX
        if output_scores:
            gen_kwargs["output_scores"] = True
            gen_kwargs["return_dict_in_generate"] = True
        
        # Enable attention output for white-box mode
        output_attentions = self.config.instrumentation_mode == InstrumentationMode.WHITE_BOX
        if output_attentions:
            gen_kwargs["output_attentions"] = True
            gen_kwargs["output_hidden_states"] = True
        
        # Set up white-box instrumentation hooks
        activations_data = None
        attention_data = None
        
        if self.config.instrumentation_mode == InstrumentationMode.WHITE_BOX and self.instrumenter:
            self.instrumenter.setup_hooks()
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Capture white-box data
            if self.config.instrumentation_mode == InstrumentationMode.WHITE_BOX and self.instrumenter:
                activations = self.instrumenter.get_activations()
                attention_patterns = self.instrumenter.get_attention_patterns()
                
                if activations:
                    activations_data = {
                        "layers": [
                            {
                                "layer_idx": a.layer_idx,
                                "layer_name": a.layer_name,
                                "shape": a.shape,
                                "norm": a.norm(),
                                "mean": a.mean_activation(),
                            }
                            for a in activations
                        ],
                        "num_layers": len(activations),
                    }
                
                if attention_patterns:
                    from risklab.models.whitebox import analyze_attention_patterns
                    attention_data = analyze_attention_patterns(attention_patterns)
                    attention_data["num_layers"] = len(attention_patterns)
        finally:
            if self.config.instrumentation_mode == InstrumentationMode.WHITE_BOX and self.instrumenter:
                self.instrumenter.clear_hooks()
        
        latency_ms = (time.time() - start_time) * 1000
        
        if output_scores:
            generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Extract logprobs
            logprobs_data = []
            if outputs.scores:
                for i, score in enumerate(outputs.scores):
                    probs = torch.nn.functional.log_softmax(score[0], dim=-1)
                    token_id = generated_ids[i].item()
                    token = self.tokenizer.decode([token_id])
                    top_k_probs, top_k_ids = torch.topk(probs, 5)
                    logprobs_data.append({
                        "token": token,
                        "logprob": probs[token_id].item(),
                        "top_logprobs": [
                            {"token": self.tokenizer.decode([tid.item()]), "logprob": lp.item()}
                            for tid, lp in zip(top_k_ids, top_k_probs)
                        ]
                    })
        else:
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            logprobs_data = None
        
        return GenerationResult(
            text=text,
            finish_reason="stop",
            usage={
                "prompt_tokens": inputs.input_ids.shape[1],
                "completion_tokens": len(generated_ids),
                "total_tokens": inputs.input_ids.shape[1] + len(generated_ids),
            },
            logprobs=logprobs_data,
            token_ids=generated_ids.tolist() if hasattr(generated_ids, 'tolist') else list(generated_ids),
            activations=activations_data,
            attention_patterns=attention_data,
            model_ref=self.model_ref,
            latency_ms=latency_ms,
        )
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        # Format messages for chat
        if hasattr(self.tokenizer, "apply_chat_template"):
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback formatting
            prompt_parts = []
            if system_prompt:
                prompt_parts.append(f"System: {system_prompt}")
            for msg in messages:
                role = msg["role"].capitalize()
                prompt_parts.append(f"{role}: {msg['content']}")
            prompt_parts.append("Assistant:")
            prompt = "\n\n".join(prompt_parts)
        
        return await self.generate(prompt, **kwargs)
