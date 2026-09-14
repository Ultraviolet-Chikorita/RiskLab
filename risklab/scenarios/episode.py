"""
Episode specification and execution records.

Episodes define prompts, optional variants, framing conditions, context, and
reference information used by RiskLab evaluation runs.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple

from pydantic import BaseModel, Field

from risklab.scenarios.context import ContextMetadata
from risklab.scenarios.framing import Framing, FramingType, STANDARD_FRAMINGS


class EpisodeVariant(BaseModel):
    """A controlled prompt variant within an episode."""

    prompt: str
    variant_type: str = "paraphrase"
    variant_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReferenceAnswer(BaseModel):
    """Optional reference information for evaluating an episode."""

    answer: Optional[str] = None
    key_points: List[str] = Field(default_factory=list)
    unacceptable_elements: List[str] = Field(default_factory=list)


class Episode(BaseModel):
    """A behavioral evaluation episode."""

    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str = ""
    core_prompt: str
    system_prompt: Optional[str] = None

    variants: List[EpisodeVariant] = Field(default_factory=list)
    framings: List[Framing] = Field(default_factory=list)
    context: ContextMetadata = Field(default_factory=ContextMetadata)
    reference: Optional[ReferenceAnswer] = None

    tags: List[str] = Field(default_factory=list)
    category: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_all_prompts(self) -> List[Tuple[str, str]]:
        """Return the core prompt and all controlled variants."""

        prompts = [(self.core_prompt, "core")]
        prompts.extend((variant.prompt, variant.variant_id) for variant in self.variants)
        return prompts

    def get_framed_prompt(self, framing_type: FramingType) -> str:
        """Apply one of this episode's framings to the core prompt."""

        for framing in self.framings:
            if framing.framing_type == framing_type:
                return framing.apply_to_prompt(self.core_prompt)
        return self.core_prompt

    def iter_evaluation_prompts(
        self,
    ) -> Iterator[Tuple[str, Optional[FramingType], str, Optional[str]]]:
        """Yield prompt/framing combinations for evaluation.

        An episode with no explicit framings still yields each prompt once. This
        keeps the empty-framing case useful rather than silently producing zero
        evaluations.
        """

        prompts = self.get_all_prompts()
        if not self.framings:
            for prompt, variant_id in prompts:
                yield prompt, None, variant_id, self.system_prompt
            return

        for prompt, variant_id in prompts:
            for framing in self.framings:
                yield (
                    framing.apply_to_prompt(prompt),
                    framing.framing_type,
                    variant_id,
                    framing.get_system_prompt(self.system_prompt),
                )

    def get_combination_count(self) -> int:
        """Return the number of prompt/framing evaluations this episode yields."""

        prompt_count = 1 + len(self.variants)
        framing_count = max(1, len(self.framings))
        return prompt_count * framing_count


class EpisodeBuilder:
    """Fluent builder for constructing :class:`Episode` objects."""

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
        self._variants.append(
            EpisodeVariant(prompt=prompt, variant_type=variant_type)
        )
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

    def metadata(self, **kwargs: Any) -> "EpisodeBuilder":
        self._metadata.update(kwargs)
        return self

    def build(self) -> Episode:
        """Build a valid episode, including the no-framing case."""

        return Episode(
            name=self._name,
            description=self._description,
            core_prompt=self._core_prompt,
            system_prompt=self._system_prompt,
            variants=list(self._variants),
            framings=list(self._framings),
            context=self._context,
            reference=self._reference,
            tags=list(self._tags),
            category=self._category,
            metadata=dict(self._metadata),
        )


class EpisodeRun(BaseModel):
    """Record of a single episode execution."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    episode_id: str
    variant_id: Optional[str] = None
    framing_type: Optional[FramingType] = None

    prompt: str
    system_prompt: Optional[str] = None
    response: str = ""

    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    latency_ms: Optional[float] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)
