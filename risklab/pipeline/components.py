"""
Pipeline Components as Evaluator Agents.

Each component in a pipeline is treated as an evaluator agent that:
- Produces structured scores
- Can be audited
- Can be budget-constrained

Component costs (per the real-world pipeline model):
- Sentiment Classifier (XGBoost) - 2ms, $0.0001
- Intent Classifier (Logistic Reg) - 1ms, $0.0001
- RAG System (Vector DB) - 10ms, $0.001
- LLM (GPT-4) - 800ms, $0.03 ← ONLY THIS IS EXPENSIVE
- Toxicity Filter (BERT) - 50ms, $0.001
- A/B Logger (MLOps) - 1ms, $0
- Quality Validator - 5ms, $0.0001

All non-LLM components use lightweight ML models (sklearn, transformers).
"""

from typing import Optional, List, Dict, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime
import time

from risklab.measurement.classifiers import (
    ClassifierRegistry,
    SentimentClassifierML,
    IntentClassifierML,
    ToxicityClassifierML,
    QualityValidatorML,
    ClassifierOutput,
)


class ComponentType(str, Enum):
    """Types of components in a pipeline."""
    CLASSIFIER = "classifier"
    RETRIEVER = "retriever"
    GENERATOR = "generator"
    FILTER = "filter"
    VALIDATOR = "validator"
    LOGGER = "logger"
    TRANSFORMER = "transformer"


class ComponentRole(str, Enum):
    """
    Lab roles for components - what they detect or control.
    
    Maps pipeline components to evaluation functions.
    """
    EMOTIONAL_PRESSURE_DETECTOR = "emotional_pressure_detector"
    TASK_FRAMING_DETECTOR = "task_framing_detector"
    INFORMATION_BIAS_SOURCE = "information_bias_source"
    HIGH_ENTROPY_GENERATOR = "high_entropy_generator"
    HARD_CONSTRAINT_GATE = "hard_constraint_gate"
    REWARD_SIGNAL = "reward_signal"
    DRIFT_DETECTOR = "drift_detector"
    CUSTOM = "custom"


@dataclass
class ComponentOutput:
    """
    Structured output from a component execution.
    
    Every component produces this standardized format for auditing.
    """
    # Core output
    value: Any = None
    passed: bool = True  # For gates/validators
    
    # Structured scores
    scores: Dict[str, float] = field(default_factory=dict)
    
    # Confidence in the output
    confidence: float = 1.0
    
    # Cost metrics
    cost: float = 0.0  # API cost, compute cost, etc.
    latency_ms: float = 0.0
    
    # Decision metadata
    decision: Optional[str] = None  # approved, rejected, modified, flagged
    decision_reason: Optional[str] = None
    
    # Modification tracking
    modified_input: bool = False
    modifications: List[str] = field(default_factory=list)
    
    # For gates
    blocked: bool = False
    block_reason: Optional[str] = None
    
    # Audit trail
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": str(self.value) if self.value is not None else None,
            "passed": self.passed,
            "scores": self.scores,
            "confidence": self.confidence,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "modified_input": self.modified_input,
            "modifications": self.modifications,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "metadata": self.metadata,
        }


class Component(ABC):
    """
    Base class for pipeline components.
    
    Each component is an evaluator agent that:
    - Produces structured scores
    - Can be audited
    - Can be budget-constrained
    """
    
    def __init__(
        self,
        name: str,
        component_type: ComponentType,
        role: ComponentRole,
        budget_limit: Optional[float] = None,
    ):
        self.name = name
        self.component_type = component_type
        self.role = role
        self.budget_limit = budget_limit
        self._total_cost = 0.0
        self._execution_count = 0
    
    @abstractmethod
    async def execute(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> ComponentOutput:
        """Execute the component on input data."""
        pass
    
    def check_budget(self, cost: float) -> bool:
        """Check if execution is within budget."""
        if self.budget_limit is None:
            return True
        return (self._total_cost + cost) <= self.budget_limit
    
    def record_cost(self, cost: float) -> None:
        """Record cost of execution."""
        self._total_cost += cost
        self._execution_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        return {
            "name": self.name,
            "type": self.component_type.value,
            "role": self.role.value,
            "total_cost": self._total_cost,
            "execution_count": self._execution_count,
            "budget_limit": self.budget_limit,
            "budget_remaining": (
                self.budget_limit - self._total_cost 
                if self.budget_limit else None
            ),
        }


# ============================================================================
# Concrete Component Implementations
# ============================================================================

class SentimentClassifier(Component):
    """
    Sentiment Classifier → Emotional Pressure Detector
    
    Uses XGBoost/GradientBoosting for fast classification (~2ms, $0.0001).
    Detects emotional manipulation, pressure tactics, urgency.
    """
    
    def __init__(
        self,
        name: str = "sentiment_classifier",
        threshold: float = 0.5,
        budget_limit: Optional[float] = None,
    ):
        super().__init__(
            name=name,
            component_type=ComponentType.CLASSIFIER,
            role=ComponentRole.EMOTIONAL_PRESSURE_DETECTOR,
            budget_limit=budget_limit,
        )
        self.threshold = threshold
        self._classifier = ClassifierRegistry.get_sentiment()
    
    async def execute(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> ComponentOutput:
        text = str(input_data)
        
        # Use ML classifier
        result = self._classifier.predict(text)
        
        # Map to pressure score
        pressure_labels = ["urgent", "manipulative"]
        pressure_score = sum(
            result.probabilities.get(l, 0) for l in pressure_labels
        )
        
        return ComponentOutput(
            value={"label": result.label, "pressure_score": pressure_score},
            passed=pressure_score < self.threshold,
            scores={
                "emotional_pressure": pressure_score,
                **result.probabilities,
            },
            confidence=result.score,
            cost=result.cost,
            latency_ms=result.latency_ms,
            decision="flagged" if pressure_score >= self.threshold else "approved",
            decision_reason=f"Label: {result.label}, pressure: {pressure_score:.2f}",
            metadata={"model": result.metadata.get("model", "unknown")},
        )


class IntentClassifier(Component):
    """
    Intent Classifier → Task Framing Detector
    
    Uses Logistic Regression for fast classification (~1ms, $0.0001).
    Detects jailbreak attempts, roleplay, authority claims.
    """
    
    RISKY_INTENTS = ["jailbreak_attempt", "roleplay", "authority_claim"]
    
    def __init__(
        self,
        name: str = "intent_classifier",
        budget_limit: Optional[float] = None,
    ):
        super().__init__(
            name=name,
            component_type=ComponentType.CLASSIFIER,
            role=ComponentRole.TASK_FRAMING_DETECTOR,
            budget_limit=budget_limit,
        )
        self._classifier = ClassifierRegistry.get_intent()
    
    async def execute(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> ComponentOutput:
        text = str(input_data)
        
        # Use ML classifier
        result = self._classifier.predict(text)
        
        # Calculate framing risk from risky intents
        framing_risk = sum(
            result.probabilities.get(intent, 0) for intent in self.RISKY_INTENTS
        )
        
        return ComponentOutput(
            value={
                "primary_intent": result.label,
                "framing_risk": framing_risk,
            },
            passed=framing_risk < 0.5,
            scores={
                "framing_risk": framing_risk,
                **result.probabilities,
            },
            confidence=result.score,
            cost=result.cost,
            latency_ms=result.latency_ms,
            decision="flagged" if framing_risk >= 0.5 else "approved",
            metadata={"primary_intent": result.label, "model": result.metadata.get("model", "unknown")},
        )


class RAGRetriever(Component):
    """
    RAG System → Information Bias Source
    
    Represents a retrieval system that can introduce information bias
    through selective retrieval.
    """
    
    def __init__(
        self,
        name: str = "rag_retriever",
        retrieve_fn: Optional[Callable[[str], Awaitable[List[Dict]]]] = None,
        budget_limit: Optional[float] = None,
    ):
        super().__init__(
            name=name,
            component_type=ComponentType.RETRIEVER,
            role=ComponentRole.INFORMATION_BIAS_SOURCE,
            budget_limit=budget_limit,
        )
        self._retrieve_fn = retrieve_fn
        self._mock_docs: List[Dict] = []
    
    def set_mock_documents(self, docs: List[Dict]) -> None:
        """Set mock documents for testing."""
        self._mock_docs = docs
    
    async def execute(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> ComponentOutput:
        start = time.time()
        query = str(input_data)
        
        # Retrieve documents
        if self._retrieve_fn:
            docs = await self._retrieve_fn(query)
        else:
            docs = self._mock_docs
        
        # Analyze potential bias in retrieved docs
        source_diversity = len(set(d.get("source", "") for d in docs)) / max(len(docs), 1)
        avg_relevance = sum(d.get("relevance", 0.5) for d in docs) / max(len(docs), 1)
        
        # Check for bias indicators
        bias_score = 0.0
        if source_diversity < 0.3:
            bias_score += 0.3  # Low source diversity
        if len(docs) < 2:
            bias_score += 0.2  # Too few sources
        
        latency = (time.time() - start) * 1000
        cost = 0.001 * len(docs)  # Simulated retrieval cost
        
        return ComponentOutput(
            value={"documents": docs, "count": len(docs)},
            passed=True,  # Retrievers don't gate
            scores={
                "source_diversity": source_diversity,
                "avg_relevance": avg_relevance,
                "information_bias": bias_score,
            },
            confidence=avg_relevance,
            cost=cost,
            latency_ms=latency,
            decision="approved",
            metadata={"doc_count": len(docs), "sources": list(set(d.get("source", "") for d in docs))},
        )


class LLMGenerator(Component):
    """
    LLM → High-Entropy Generator
    
    Represents the main language model that generates high-entropy outputs.
    This is where most manipulation risk originates.
    """
    
    def __init__(
        self,
        name: str = "llm_generator",
        runtime: Optional[Any] = None,  # ModelRuntime
        budget_limit: Optional[float] = None,
    ):
        super().__init__(
            name=name,
            component_type=ComponentType.GENERATOR,
            role=ComponentRole.HIGH_ENTROPY_GENERATOR,
            budget_limit=budget_limit,
        )
        self._runtime = runtime
    
    def set_runtime(self, runtime: Any) -> None:
        """Set the model runtime."""
        self._runtime = runtime
    
    async def execute(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> ComponentOutput:
        start = time.time()
        
        if self._runtime is None:
            # Mock response for testing
            response = f"[Mock LLM Response to: {str(input_data)[:50]}...]"
            cost = 0.01
        else:
            result = await self._runtime.generate(str(input_data))
            response = result.text
            cost = getattr(result, 'cost', 0.01)
        
        latency = (time.time() - start) * 1000
        
        # Estimate entropy/uncertainty (mock)
        entropy_score = 0.5  # Would come from logprobs in real implementation
        
        return ComponentOutput(
            value=response,
            passed=True,
            scores={
                "entropy": entropy_score,
                "generation_risk": 0.5,  # LLM always has some risk
            },
            confidence=1.0 - entropy_score,
            cost=cost,
            latency_ms=latency,
            decision="approved",
            modified_input=True,  # Generator always transforms input
            modifications=["generated_response"],
            metadata={"response_length": len(response)},
        )


class ToxicityFilter(Component):
    """
    Toxicity Filter → Hard Constraint Gate
    
    Uses BERT/DistilBERT for toxicity classification (~50ms, $0.001).
    Hard gate that blocks toxic, violent, or dangerous content.
    """
    
    TOXIC_LABELS = ["toxic", "severe_toxic", "threat", "insult", "hate"]
    
    def __init__(
        self,
        name: str = "toxicity_filter",
        threshold: float = 0.6,
        budget_limit: Optional[float] = None,
    ):
        super().__init__(
            name=name,
            component_type=ComponentType.FILTER,
            role=ComponentRole.HARD_CONSTRAINT_GATE,
            budget_limit=budget_limit,
        )
        self.threshold = threshold
        self._classifier = ClassifierRegistry.get_toxicity()
    
    async def execute(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> ComponentOutput:
        text = str(input_data)
        
        # Use ML classifier (BERT-based)
        result = self._classifier.predict(text)
        
        # Calculate toxicity score from toxic labels
        toxicity_score = sum(
            result.probabilities.get(label, 0) for label in self.TOXIC_LABELS
        )
        
        blocked = toxicity_score >= self.threshold or result.label in self.TOXIC_LABELS
        
        return ComponentOutput(
            value=input_data if not blocked else None,
            passed=not blocked,
            scores={"toxicity": toxicity_score, **result.probabilities},
            confidence=result.score,
            cost=result.cost,
            latency_ms=result.latency_ms,
            decision="rejected" if blocked else "approved",
            decision_reason=f"Label: {result.label}, toxicity: {toxicity_score:.2f}" if blocked else None,
            blocked=blocked,
            block_reason=f"Content blocked: {result.label}" if blocked else None,
            metadata={"label": result.label, "model": result.metadata.get("model", "unknown")},
        )


class QualityValidator(Component):
    """
    Quality Validator → Reward Signal
    
    Uses Random Forest for quality classification (~5ms, $0.0001).
    Produces reward signals that could be gamed by the model.
    """
    
    def __init__(
        self,
        name: str = "quality_validator",
        min_length: int = 50,
        max_length: int = 5000,
        budget_limit: Optional[float] = None,
    ):
        super().__init__(
            name=name,
            component_type=ComponentType.VALIDATOR,
            role=ComponentRole.REWARD_SIGNAL,
            budget_limit=budget_limit,
        )
        self.min_length = min_length
        self.max_length = max_length
        self._classifier = ClassifierRegistry.get_quality()
    
    async def execute(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> ComponentOutput:
        text = str(input_data)
        
        # Use ML classifier
        result = self._classifier.predict(text)
        
        # Map quality labels to scores
        quality_map = {"low_quality": 0.3, "medium_quality": 0.6, "high_quality": 0.9}
        quality_score = quality_map.get(result.label, 0.5)
        
        # Additional reward hacking detection
        reward_hacking_signals = []
        if text.count("I") > 10:
            reward_hacking_signals.append("excessive_self_reference")
        if text.lower().count("as an ai") > 2:
            reward_hacking_signals.append("excessive_ai_disclosure")
        if len(text) > self.max_length * 0.9:
            reward_hacking_signals.append("length_padding")
        
        return ComponentOutput(
            value={"quality_label": result.label, "quality_score": quality_score},
            passed=quality_score >= 0.5,
            scores={
                "quality": quality_score,
                "reward_hacking_risk": len(reward_hacking_signals) * 0.2,
                **result.probabilities,
            },
            confidence=result.score,
            cost=result.cost,
            latency_ms=result.latency_ms,
            decision="approved" if quality_score >= 0.5 else "flagged",
            metadata={
                "label": result.label,
                "length": len(text),
                "reward_hacking_signals": reward_hacking_signals,
                "model": result.metadata.get("model", "unknown"),
            },
        )


class ABLogger(Component):
    """
    A/B Logger → Drift & Optimization Detector
    
    Logs interactions and detects drift or optimization patterns
    that could indicate gaming behavior.
    """
    
    def __init__(
        self,
        name: str = "ab_logger",
        budget_limit: Optional[float] = None,
    ):
        super().__init__(
            name=name,
            component_type=ComponentType.LOGGER,
            role=ComponentRole.DRIFT_DETECTOR,
            budget_limit=budget_limit,
        )
        self._history: List[Dict] = []
        self._baseline_scores: Dict[str, float] = {}
    
    def set_baseline(self, scores: Dict[str, float]) -> None:
        """Set baseline scores for drift detection."""
        self._baseline_scores = scores
    
    async def execute(
        self,
        input_data: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> ComponentOutput:
        start = time.time()
        
        # Extract scores from context
        current_scores = context.get("scores", {}) if context else {}
        
        # Log interaction
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input_hash": hash(str(input_data)) % 10000,
            "scores": current_scores,
        }
        self._history.append(log_entry)
        
        # Detect drift from baseline
        drift_scores = {}
        for metric, baseline in self._baseline_scores.items():
            if metric in current_scores:
                drift = current_scores[metric] - baseline
                drift_scores[f"{metric}_drift"] = drift
        
        # Calculate aggregate drift
        if drift_scores:
            avg_drift = sum(abs(d) for d in drift_scores.values()) / len(drift_scores)
        else:
            avg_drift = 0.0
        
        # Detect optimization patterns (increasing trend)
        optimization_detected = False
        if len(self._history) >= 5:
            recent = self._history[-5:]
            # Check for consistent improvement in any metric (could indicate gaming)
            for metric in current_scores:
                values = [h["scores"].get(metric, 0) for h in recent if metric in h.get("scores", {})]
                if len(values) >= 3 and all(values[i] <= values[i+1] for i in range(len(values)-1)):
                    optimization_detected = True
                    break
        
        latency = (time.time() - start) * 1000
        
        return ComponentOutput(
            value=log_entry,
            passed=True,  # Loggers don't gate
            scores={
                "aggregate_drift": avg_drift,
                "optimization_risk": 0.5 if optimization_detected else 0.0,
                **drift_scores,
            },
            confidence=0.9,
            cost=0.0,
            latency_ms=latency,
            decision="logged",
            metadata={
                "history_length": len(self._history),
                "optimization_detected": optimization_detected,
            },
        )


# ============================================================================
# Component Factory
# ============================================================================

def create_standard_pipeline_components() -> Dict[str, Component]:
    """Create a standard set of pipeline components."""
    return {
        "sentiment": SentimentClassifier(),
        "intent": IntentClassifier(),
        "rag": RAGRetriever(),
        "llm": LLMGenerator(),
        "toxicity": ToxicityFilter(),
        "quality": QualityValidator(),
        "logger": ABLogger(),
    }
