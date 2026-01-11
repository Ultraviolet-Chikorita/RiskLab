"""
Component Pipeline for AI System Evaluation.

Provides a graph-based abstraction for evaluating multi-component AI systems,
treating each component as an evaluator agent that can be audited and constrained.

Key concepts:
- ComponentGraph: Formal pipeline representation (nodes + edges)
- Component: Evaluator agents (classifiers, generators, validators, etc.)
- PipelineExecutor: Runs the pipeline with tracing and budget control
- ComponentRiskConditioner: Risk = component × context × interaction
- EpisodePipelineRunner: Integrates pipeline with episode evaluation
"""

from risklab.pipeline.graph import (
    ComponentGraph,
    ComponentNode,
    Edge,
    EdgeType,
    ExecutionTrace,
    NodeExecution,
    PipelineResult,
)
from risklab.pipeline.components import (
    Component,
    ComponentType,
    ComponentRole,
    ComponentOutput,
    SentimentClassifier,
    IntentClassifier,
    RAGRetriever,
    LLMGenerator,
    ToxicityFilter,
    QualityValidator,
    ABLogger,
    create_standard_pipeline_components,
)
from risklab.pipeline.executor import (
    PipelineExecutor,
    ExecutionContext,
    ExecutionConfig,
    create_standard_pipeline,
)
from risklab.pipeline.risk import (
    ComponentRiskConditioner,
    ComponentRiskScore,
    ComponentRiskReport,
    InteractionRisk,
    InteractionType,
    condition_pipeline_risk,
)
from risklab.pipeline.episode_runner import (
    EpisodePipelineRunner,
    EpisodePipelineResult,
    ComponentBehavior,
    PipelineFramingComparison,
    run_episode_through_pipeline,
)

__all__ = [
    # Graph
    "ComponentGraph",
    "ComponentNode",
    "Edge",
    "EdgeType",
    "ExecutionTrace",
    "NodeExecution",
    "PipelineResult",
    # Components
    "Component",
    "ComponentType",
    "ComponentRole",
    "ComponentOutput",
    "SentimentClassifier",
    "IntentClassifier",
    "RAGRetriever",
    "LLMGenerator",
    "ToxicityFilter",
    "QualityValidator",
    "ABLogger",
    "create_standard_pipeline_components",
    # Executor
    "PipelineExecutor",
    "ExecutionContext",
    "ExecutionConfig",
    "create_standard_pipeline",
    # Risk
    "ComponentRiskConditioner",
    "ComponentRiskScore",
    "ComponentRiskReport",
    "InteractionRisk",
    "InteractionType",
    "condition_pipeline_risk",
    # Episode integration
    "EpisodePipelineRunner",
    "EpisodePipelineResult",
    "ComponentBehavior",
    "PipelineFramingComparison",
    "run_episode_through_pipeline",
]
