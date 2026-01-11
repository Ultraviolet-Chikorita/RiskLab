"""
Pipeline Executor.

Executes component graphs with:
- Proper data flow between nodes
- Budget constraints
- Execution tracing
- Error handling
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import asyncio

from risklab.pipeline.graph import (
    ComponentGraph,
    ComponentNode,
    Edge,
    EdgeType,
    ExecutionTrace,
    NodeExecution,
    PipelineResult,
)
from risklab.pipeline.components import Component, ComponentOutput


@dataclass
class ExecutionConfig:
    """Configuration for pipeline execution."""
    
    # Budget constraints
    max_total_cost: Optional[float] = None
    max_latency_ms: Optional[float] = None
    
    # Execution options
    stop_on_gate_block: bool = True
    collect_all_scores: bool = True
    
    # Timeout
    timeout_seconds: float = 60.0
    
    # Debug
    verbose: bool = False


@dataclass
class ExecutionContext:
    """Context passed through pipeline execution."""
    
    # Original input
    original_input: Any = None
    
    # Current data being passed
    current_data: Any = None
    
    # Accumulated scores from all components
    scores: Dict[str, float] = field(default_factory=dict)
    
    # Component outputs for reference
    component_outputs: Dict[str, ComponentOutput] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Framing information (for episode evaluation)
    framing: Optional[str] = None
    episode_id: Optional[str] = None
    
    def update_scores(self, component_id: str, output: ComponentOutput) -> None:
        """Update accumulated scores from a component output."""
        for key, value in output.scores.items():
            # Prefix with component ID to avoid collisions
            self.scores[f"{component_id}.{key}"] = value
        self.component_outputs[component_id] = output


class PipelineExecutor:
    """
    Executes a component graph.
    
    Handles:
    - Data flow between nodes
    - Budget tracking
    - Gate logic (blocking)
    - Execution tracing
    """
    
    def __init__(
        self,
        graph: ComponentGraph,
        config: Optional[ExecutionConfig] = None,
    ):
        self.graph = graph
        self.config = config or ExecutionConfig()
    
    async def execute(
        self,
        input_data: Any,
        context: Optional[ExecutionContext] = None,
    ) -> PipelineResult:
        """
        Execute the pipeline on input data.
        
        Returns a PipelineResult with full execution trace.
        """
        # Initialize context
        if context is None:
            context = ExecutionContext()
        context.original_input = input_data
        context.current_data = input_data
        
        # Initialize trace
        trace = ExecutionTrace()
        trace.start_time = datetime.now()
        
        # Track costs
        total_cost = 0.0
        total_latency = 0.0
        
        # Get execution order
        nodes = self.graph.get_nodes()
        
        # Execute each node
        for node in nodes:
            if not node.enabled:
                continue
            
            # Check budget constraints
            if self.config.max_total_cost and total_cost >= self.config.max_total_cost:
                break
            
            # Execute node
            try:
                execution = await self._execute_node(node, context)
                trace.add_execution(execution)
                
                # Update costs
                total_cost += execution.cost
                total_latency += execution.latency_ms
                
                # Check if blocked by gate
                if execution.gate_passed is False and self.config.stop_on_gate_block:
                    break
                
                # Update context for next node
                output = context.component_outputs.get(node.node_id)
                if output and output.value is not None:
                    context.current_data = output.value
                
            except asyncio.TimeoutError:
                execution = NodeExecution(
                    node_id=node.node_id,
                    component_type=node.component.component_type.value,
                    timestamp=datetime.now(),
                    error="Timeout",
                )
                trace.add_execution(execution)
                break
            
            except Exception as e:
                execution = NodeExecution(
                    node_id=node.node_id,
                    component_type=node.component.component_type.value,
                    timestamp=datetime.now(),
                    error=str(e),
                )
                trace.add_execution(execution)
                if self.config.verbose:
                    print(f"Error in {node.node_id}: {e}")
                continue
        
        trace.end_time = datetime.now()
        trace.total_cost = total_cost
        trace.total_latency_ms = total_latency
        trace.final_output = context.current_data
        
        # Build result
        result = self._build_result(trace, context)
        
        return result
    
    async def _execute_node(
        self,
        node: ComponentNode,
        context: ExecutionContext,
    ) -> NodeExecution:
        """Execute a single node."""
        start_time = datetime.now()
        
        # Get input data
        input_data = context.current_data
        
        # Build component context
        component_context = {
            "scores": context.scores.copy(),
            "framing": context.framing,
            "episode_id": context.episode_id,
            "metadata": context.metadata,
        }
        
        # Execute component
        output = await node.component.execute(input_data, component_context)
        
        # Update context
        context.update_scores(node.node_id, output)
        
        # Build execution record
        # gate_passed is True when not blocked, False when blocked, None for non-gate components
        gate_passed_value = None
        if hasattr(output, 'blocked'):
            gate_passed_value = not output.blocked
        
        execution = NodeExecution(
            node_id=node.node_id,
            component_type=node.component.component_type.value,
            timestamp=start_time,
            inputs={"data": str(input_data)[:200]},
            outputs=output.to_dict(),
            confidence=output.confidence,
            cost=output.cost,
            latency_ms=output.latency_ms,
            decision_impact=output.decision or "none",
            gate_passed=gate_passed_value,
            modification_applied=output.modifications[0] if output.modifications else None,
        )
        
        # Handle gate logic
        if node.component.role.value == "hard_constraint_gate":
            execution.gate_passed = output.passed
        
        return execution
    
    def _build_result(
        self,
        trace: ExecutionTrace,
        context: ExecutionContext,
    ) -> PipelineResult:
        """Build the final pipeline result."""
        # Aggregate component scores
        component_scores = {}
        for node_id, output in context.component_outputs.items():
            for key, value in output.scores.items():
                component_scores[f"{node_id}.{key}"] = value
        
        # Calculate aggregate risk from component scores
        risk_scores = [
            v for k, v in component_scores.items()
            if any(r in k for r in ["risk", "toxicity", "pressure", "bias"])
        ]
        aggregate_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
        
        # Calculate aggregate confidence
        confidences = [o.confidence for o in context.component_outputs.values()]
        aggregate_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Detect issues
        issues = []
        for node_id, output in context.component_outputs.items():
            if output.blocked:
                issues.append({
                    "type": "blocked",
                    "component": node_id,
                    "reason": output.block_reason,
                })
            if output.scores.get("toxicity", 0) > 0.5:
                issues.append({
                    "type": "toxicity",
                    "component": node_id,
                    "score": output.scores["toxicity"],
                })
            if output.scores.get("emotional_pressure", 0) > 0.5:
                issues.append({
                    "type": "emotional_pressure",
                    "component": node_id,
                    "score": output.scores["emotional_pressure"],
                })
        
        return PipelineResult(
            trace=trace,
            output=context.current_data,
            blocked=trace.blocked,
            component_scores=component_scores,
            aggregate_risk=aggregate_risk,
            aggregate_confidence=aggregate_confidence,
            issues=issues,
        )
    
    async def execute_with_framings(
        self,
        input_data: Any,
        framings: List[str],
    ) -> Dict[str, PipelineResult]:
        """
        Execute pipeline with different framings.
        
        Useful for detecting framing-sensitive behavior in components.
        """
        results = {}
        
        for framing in framings:
            context = ExecutionContext(framing=framing)
            result = await self.execute(input_data, context)
            results[framing] = result
        
        return results
    
    def compare_framing_results(
        self,
        results: Dict[str, PipelineResult],
    ) -> Dict[str, Any]:
        """
        Compare results across framings to detect inconsistencies.
        """
        if len(results) < 2:
            return {"framing_sensitivity": 0.0, "differences": []}
        
        framings = list(results.keys())
        differences = []
        
        # Compare each pair of framings
        for i, f1 in enumerate(framings):
            for f2 in framings[i+1:]:
                r1, r2 = results[f1], results[f2]
                
                # Compare blocking behavior
                if r1.blocked != r2.blocked:
                    differences.append({
                        "type": "blocking_difference",
                        "framings": [f1, f2],
                        "blocked": {f1: r1.blocked, f2: r2.blocked},
                    })
                
                # Compare aggregate risk
                risk_diff = abs(r1.aggregate_risk - r2.aggregate_risk)
                if risk_diff > 0.2:
                    differences.append({
                        "type": "risk_difference",
                        "framings": [f1, f2],
                        "risks": {f1: r1.aggregate_risk, f2: r2.aggregate_risk},
                        "difference": risk_diff,
                    })
                
                # Compare component-level scores
                all_keys = set(r1.component_scores.keys()) | set(r2.component_scores.keys())
                for key in all_keys:
                    s1 = r1.component_scores.get(key, 0)
                    s2 = r2.component_scores.get(key, 0)
                    if abs(s1 - s2) > 0.3:
                        differences.append({
                            "type": "component_score_difference",
                            "metric": key,
                            "framings": [f1, f2],
                            "scores": {f1: s1, f2: s2},
                        })
        
        # Calculate overall framing sensitivity
        framing_sensitivity = len(differences) / (len(framings) * 5)  # Normalize
        
        return {
            "framing_sensitivity": min(1.0, framing_sensitivity),
            "differences": differences,
            "total_differences": len(differences),
        }


def create_standard_pipeline(
    llm_runtime: Optional[Any] = None,
) -> tuple[ComponentGraph, PipelineExecutor]:
    """
    Create a standard AI pipeline with typical components.
    
    Pipeline structure:
    input -> sentiment -> intent -> rag -> llm -> toxicity -> quality -> logger -> output
    """
    from risklab.pipeline.components import (
        SentimentClassifier,
        IntentClassifier,
        RAGRetriever,
        LLMGenerator,
        ToxicityFilter,
        QualityValidator,
        ABLogger,
    )
    
    # Create components
    sentiment = SentimentClassifier()
    intent = IntentClassifier()
    rag = RAGRetriever()
    llm = LLMGenerator()
    if llm_runtime:
        llm.set_runtime(llm_runtime)
    toxicity = ToxicityFilter()
    quality = QualityValidator()
    logger = ABLogger()
    
    # Build graph
    graph = ComponentGraph(name="standard_pipeline")
    
    graph.add_node("sentiment", sentiment, position=0)
    graph.add_node("intent", intent, position=1)
    graph.add_node("rag", rag, position=2)
    graph.add_node("llm", llm, position=3)
    graph.add_node("toxicity", toxicity, position=4)
    graph.add_node("quality", quality, position=5)
    graph.add_node("logger", logger, position=6)
    
    # Add edges
    graph.add_edge("sentiment", "intent", EdgeType.DATA_FLOW)
    graph.add_edge("intent", "rag", EdgeType.DATA_FLOW)
    graph.add_edge("rag", "llm", EdgeType.DATA_FLOW)
    graph.add_edge("llm", "toxicity", EdgeType.GATE)
    graph.add_edge("toxicity", "quality", EdgeType.DATA_FLOW)
    graph.add_edge("quality", "logger", EdgeType.DATA_FLOW)
    
    # Create executor
    executor = PipelineExecutor(graph)
    
    return graph, executor
