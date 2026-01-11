"""
Episode definitions for evaluation scenarios.

An Episode represents a task-invariant interaction scenario containing
the core task, multiple framings, paraphrased variants, and metadata.
"""

import uuid
from typing import Optional, List, Dict, Any, Callable
from pydantic import BaseModel, Field

from risklab.scenarios.context import ContextMetadata
from risklab.scenarios.framing import Framing, FramingSet, FramingType, STANDARD_FRAMINGS


class ReferenceAnswer(BaseModel):
    """Reference answer or grading criteria for an episode."""
    answer: Optional[str] = None
    key_points: List[str] = Field(default_factory=list)
    unacceptable_elements: List[str] = Field(default_factory=list)
    grading_rubric: Optional[str] = None
    correctness_weight: float = Field(default=1.0, ge=0.0, le=1.0)


class EpisodeVariant(BaseModel):
    """A paraphrased or modified variant of the core task prompt."""
    variant_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    prompt: str
    variant_type: str = "paraphrase"  # paraphrase, simplified, technical, etc.
    notes: Optional[str] = None


class Episode(BaseModel):
    """
    A task-invariant interaction scenario for evaluation.
    
    Each episode contains:
    - A core task prompt
    - Multiple framings of that task
    - Paraphrased variants
    - Contextual metadata
    - Optional reference answers or grading logic
    """
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    
    # Core task
    core_prompt: str
    system_prompt: Optional[str] = None
    
    # Variants and framings
    variants: List[EpisodeVariant] = Field(default_factory=list)
    framings: List[Framing] = Field(default_factory=list)
    
    # Context and reference
    context: ContextMetadata = Field(default_factory=ContextMetadata)
    reference: Optional[ReferenceAnswer] = None
    
    # Categorization
    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def __init__(self, **data):
        super().__init__(**data)
        # Ensure at least standard framings are present
        if not self.framings:
            self.framings = [
                STANDARD_FRAMINGS[FramingType.NEUTRAL],
                STANDARD_FRAMINGS[FramingType.EVALUATED],
                STANDARD_FRAMINGS[FramingType.OVERSIGHT],
            ]
    
    def get_framed_prompt(
        self,
        framing_type: FramingType,
        variant_id: Optional[str] = None,
    ) -> str:
        """Get the prompt with a specific framing applied."""
        # Get base prompt (variant or core)
        base_prompt = self.core_prompt
        if variant_id:
            for variant in self.variants:
                if variant.variant_id == variant_id:
                    base_prompt = variant.prompt
                    break
        
        # Apply framing
        framing = self.get_framing(framing_type)
        if framing:
            return framing.apply_to_prompt(base_prompt)
        return base_prompt
    
    def get_framing(self, framing_type: FramingType) -> Optional[Framing]:
        """Get a specific framing by type."""
        for framing in self.framings:
            if framing.framing_type == framing_type:
                return framing
        return None
    
    def get_system_prompt(self, framing_type: Optional[FramingType] = None) -> Optional[str]:
        """Get system prompt, optionally modified by framing."""
        if framing_type:
            framing = self.get_framing(framing_type)
            if framing:
                return framing.get_system_prompt(self.system_prompt)
        return self.system_prompt
    
    def add_variant(
        self,
        prompt: str,
        variant_type: str = "paraphrase",
        notes: Optional[str] = None,
    ) -> EpisodeVariant:
        """Add a paraphrased variant of the core prompt."""
        variant = EpisodeVariant(
            prompt=prompt,
            variant_type=variant_type,
            notes=notes,
        )
        self.variants.append(variant)
        return variant
    
    def add_framing(self, framing: Framing) -> None:
        """Add a framing to this episode."""
        # Replace if same type exists
        self.framings = [f for f in self.framings if f.framing_type != framing.framing_type]
        self.framings.append(framing)
    
    def iter_all_combinations(self):
        """
        Iterate over all prompt/framing combinations.
        
        Yields tuples of (prompt, framing_type, variant_id, system_prompt)
        """
        prompts = [(self.core_prompt, None)]
        for variant in self.variants:
            prompts.append((variant.prompt, variant.variant_id))
        
        for prompt, variant_id in prompts:
            for framing in self.framings:
                framed_prompt = framing.apply_to_prompt(prompt)
                system_prompt = framing.get_system_prompt(self.system_prompt)
                yield framed_prompt, framing.framing_type, variant_id, system_prompt
    
    def get_combination_count(self) -> int:
        """Get total number of prompt/framing combinations."""
        prompt_count = 1 + len(self.variants)
        framing_count = len(self.framings)
        return prompt_count * framing_count


class EpisodeBuilder:
    """Builder pattern for constructing Episodes."""
    
    def __init__(self, name: str, core_prompt: str):
        self._name = name
        self._core_prompt = core_prompt
        self._description = ""
        self._system_prompt: Optional[str] = None
        self._variants: List[EpisodeVariant] = []
        self._framings: List[Framing] = []
        self._context = ContextMetadata()
        self._reference: Optional[ReferenceAnswer] = None
        self._tags: List[str] = []
        self._category: Optional[str] = None
        self._metadata: Dict[str, Any] = {}
    
    def description(self, desc: str) -> "EpisodeBuilder":
        self._description = desc
        return self
    
    def system_prompt(self, prompt: str) -> "EpisodeBuilder":
        self._system_prompt = prompt
        return self
    
    def add_variant(
        self,
        prompt: str,
        variant_type: str = "paraphrase",
    ) -> "EpisodeBuilder":
        self._variants.append(EpisodeVariant(
            prompt=prompt,
            variant_type=variant_type,
        ))
        return self
    
    def add_framing(self, framing: Framing) -> "EpisodeBuilder":
        self._framings.append(framing)
        return self
    
    def use_standard_framings(self) -> "EpisodeBuilder":
        self._framings = [
            STANDARD_FRAMINGS[FramingType.NEUTRAL],
            STANDARD_FRAMINGS[FramingType.EVALUATED],
            STANDARD_FRAMINGS[FramingType.OVERSIGHT],
        ]
        return self
    
    def context(self, ctx: ContextMetadata) -> "EpisodeBuilder":
        self._context = ctx
        return self
    
    def reference_answer(
        self,
        answer: Optional[str] = None,
        key_points: Optional[List[str]] = None,
        unacceptable: Optional[List[str]] = None,
    ) -> "EpisodeBuilder":
        self._reference = ReferenceAnswer(
            answer=answer,
            key_points=key_points or [],
            unacceptable_elements=unacceptable or [],
        )
        return self
    
    def tags(self, *tags: str) -> "EpisodeBuilder":
        self._tags.extend(tags)
        return self
    
    def category(self, cat: str) -> "EpisodeBuilder":
        self._category = cat
        return self
    
    def metadata(self, **kwargs) -> "EpisodeBuilder":
        self._metadata.update(kwargs)
        return self
    
    def build(self) -> Episode:
        return Episode(
            name=self._name,
            description=self._description,
            core_prompt=self._core_prompt,
            system_prompt=self._system_prompt,
            variants=self._variants,
            framings=self._framings if self._framings else None,
            context=self._context,
            reference=self._reference,
            tags=self._tags,
            category=self._category,
            metadata=self._metadata,
        )


class EpisodeRun(BaseModel):
    """Record of a single episode execution."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    episode_id: str
    framing_type: FramingType
    variant_id: Optional[str] = None
    
    # Input
    prompt: str
    system_prompt: Optional[str] = None
    
    # Output
    response: Optional[str] = None
    response_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Timing
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    latency_ms: Optional[float] = None
    
    # Model info
    model_ref_id: Optional[str] = None
