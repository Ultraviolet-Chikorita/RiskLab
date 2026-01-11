"""
LangGraph-based evaluation workflow for multi-agent governance.

Implements the evaluation pipeline as a state machine with configurable nodes.
"""

from typing import Optional, List, Dict, Any, TypedDict, Annotated
from datetime import datetime
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from risklab.models.runtime import ModelRuntime
from risklab.scenarios.episode import Episode, EpisodeRun
from risklab.scenarios.framing import FramingType
from risklab.scenarios.context import ContextMetadata
from risklab.measurement.metrics import BehavioralMetrics
from risklab.measurement.signals import ManipulationSignals
from risklab.measurement.analyzers import CompositeAnalyzer
from risklab.measurement.comparators import FramingComparator, CrossFramingAnalysis
from risklab.risk.conditioner import RiskConditioner, RiskConditionedMetrics
from risklab.risk.thresholds import RiskThresholdManager, DecisionResult, DecisionOutcome
from risklab.governance.council import EvaluationCouncil, CouncilVerdict, CouncilConfig
from risklab.governance.judge import JudgeReport


class EvaluationState(TypedDict):
    """State for the evaluation graph."""
    # Input
    episode: Optional[Episode]
    model_runtime: Optional[Any]  # ModelRuntime
    target_response: Optional[str]
    target_prompt: Optional[str]
    framing_type: Optional[FramingType]
    
    # Episode runs (for multi-framing evaluation)
    episode_runs: Annotated[List[Dict], operator.add]
    
    # Metrics and signals
    behavioral_metrics: Optional[Dict]
    manipulation_signals: Optional[Dict]
    metrics_by_framing: Dict[str, Dict]
    
    # Risk assessment
    conditioned_metrics: Optional[Dict]
    decision_result: Optional[Dict]
    
    # Governance
    council_verdict: Optional[Dict]
    judge_reports: Annotated[List[Dict], operator.add]
    
    # Cross-framing analysis
    cross_framing_analysis: Optional[Dict]
    
    # Final output
    final_assessment: Optional[Dict]
    errors: Annotated[List[str], operator.add]
    
    # Control flow
    current_step: str
    completed_steps: Annotated[List[str], operator.add]


def create_initial_state(
    episode: Optional[Episode] = None,
    model_runtime: Optional[ModelRuntime] = None,
) -> EvaluationState:
    """Create initial state for evaluation."""
    return EvaluationState(
        episode=episode,
        model_runtime=model_runtime,
        target_response=None,
        target_prompt=None,
        framing_type=None,
        episode_runs=[],
        behavioral_metrics=None,
        manipulation_signals=None,
        metrics_by_framing={},
        conditioned_metrics=None,
        decision_result=None,
        council_verdict=None,
        judge_reports=[],
        cross_framing_analysis=None,
        final_assessment=None,
        errors=[],
        current_step="start",
        completed_steps=[],
    )


async def generate_response_node(state: EvaluationState) -> EvaluationState:
    """Generate model response for the current framing."""
    try:
        episode = state["episode"]
        runtime = state["model_runtime"]
        framing_type = state.get("framing_type") or FramingType.NEUTRAL
        
        if not episode or not runtime:
            return {
                **state,
                "errors": ["Missing episode or runtime"],
                "current_step": "error",
            }
        
        # Get framed prompt
        prompt = episode.get_framed_prompt(framing_type)
        system_prompt = episode.get_system_prompt(framing_type)
        
        # Generate response
        result = await runtime.generate(prompt, system_prompt)
        
        # Record run
        run = EpisodeRun(
            episode_id=episode.episode_id,
            framing_type=framing_type,
            prompt=prompt,
            system_prompt=system_prompt,
            response=result.text,
            latency_ms=result.latency_ms,
        )
        
        return {
            **state,
            "target_prompt": prompt,
            "target_response": result.text,
            "episode_runs": [run.model_dump()],
            "current_step": "response_generated",
            "completed_steps": ["generate_response"],
        }
    
    except Exception as e:
        return {
            **state,
            "errors": [f"Response generation failed: {str(e)}"],
            "current_step": "error",
        }


async def analyze_behavior_node(state: EvaluationState) -> EvaluationState:
    """Analyze behavioral metrics for the response."""
    try:
        response = state.get("target_response")
        prompt = state.get("target_prompt")
        episode = state.get("episode")
        runtime = state.get("model_runtime")
        framing_type = state.get("framing_type") or FramingType.NEUTRAL
        
        if not response:
            return {
                **state,
                "errors": ["No response to analyze"],
                "current_step": "error",
            }
        
        # Use composite analyzer
        analyzer = CompositeAnalyzer(runtime)
        result = await analyzer.analyze(response, prompt, episode)
        
        metrics_dict = result.metrics.model_dump()
        
        # Store metrics by framing
        metrics_by_framing = state.get("metrics_by_framing", {}).copy()
        metrics_by_framing[framing_type.value] = metrics_dict
        
        return {
            **state,
            "behavioral_metrics": metrics_dict,
            "metrics_by_framing": metrics_by_framing,
            "current_step": "behavior_analyzed",
            "completed_steps": ["analyze_behavior"],
        }
    
    except Exception as e:
        return {
            **state,
            "errors": [f"Behavior analysis failed: {str(e)}"],
            "current_step": "error",
        }


async def compute_signals_node(state: EvaluationState) -> EvaluationState:
    """Compute manipulation signals from metrics."""
    try:
        from risklab.measurement.signals import SignalComputer
        
        metrics_by_framing = state.get("metrics_by_framing", {})
        
        if not metrics_by_framing:
            return {
                **state,
                "errors": ["No metrics available for signal computation"],
                "current_step": "error",
            }
        
        # Reconstruct metrics objects
        framing_metrics = {}
        for framing_str, metrics_dict in metrics_by_framing.items():
            framing_type = FramingType(framing_str)
            metrics = BehavioralMetrics(**metrics_dict)
            framing_metrics[framing_type] = metrics
        
        # Compute signals
        computer = SignalComputer()
        signals = computer.compute_all_signals(framing_metrics)
        
        return {
            **state,
            "manipulation_signals": signals.model_dump(),
            "current_step": "signals_computed",
            "completed_steps": ["compute_signals"],
        }
    
    except Exception as e:
        return {
            **state,
            "errors": [f"Signal computation failed: {str(e)}"],
            "current_step": "error",
        }


async def condition_risk_node(state: EvaluationState) -> EvaluationState:
    """Apply risk conditioning to metrics."""
    try:
        metrics_dict = state.get("behavioral_metrics")
        signals_dict = state.get("manipulation_signals")
        episode = state.get("episode")
        
        if not metrics_dict:
            return {
                **state,
                "errors": ["No metrics for risk conditioning"],
                "current_step": "error",
            }
        
        # Get context
        context = episode.context if episode else ContextMetadata()
        
        # Apply conditioning
        conditioner = RiskConditioner()
        metrics = BehavioralMetrics(**metrics_dict)
        
        if signals_dict:
            signals = ManipulationSignals(**signals_dict)
            conditioned = conditioner.full_conditioning(metrics, signals, context)
        else:
            conditioned = conditioner.condition_metrics(metrics, context)
        
        return {
            **state,
            "conditioned_metrics": conditioned.model_dump(),
            "current_step": "risk_conditioned",
            "completed_steps": ["condition_risk"],
        }
    
    except Exception as e:
        return {
            **state,
            "errors": [f"Risk conditioning failed: {str(e)}"],
            "current_step": "error",
        }


async def threshold_decision_node(state: EvaluationState) -> EvaluationState:
    """Apply thresholds and produce decision."""
    try:
        conditioned_dict = state.get("conditioned_metrics")
        
        if not conditioned_dict:
            return {
                **state,
                "errors": ["No conditioned metrics for decision"],
                "current_step": "error",
            }
        
        conditioned = RiskConditionedMetrics(**conditioned_dict)
        
        # Apply thresholds
        threshold_mgr = RiskThresholdManager()
        decision = threshold_mgr.evaluate(conditioned)
        
        return {
            **state,
            "decision_result": decision.model_dump(),
            "current_step": "decision_made",
            "completed_steps": ["threshold_decision"],
        }
    
    except Exception as e:
        return {
            **state,
            "errors": [f"Threshold decision failed: {str(e)}"],
            "current_step": "error",
        }


async def council_evaluation_node(state: EvaluationState) -> EvaluationState:
    """Run multi-agent council evaluation."""
    try:
        response = state.get("target_response")
        prompt = state.get("target_prompt")
        episode = state.get("episode")
        runtime = state.get("model_runtime")
        
        if not response:
            return {
                **state,
                "errors": ["No response for council evaluation"],
                "current_step": "error",
            }
        
        # Create council
        config = CouncilConfig(num_judges=3)
        council = EvaluationCouncil(config, runtime)
        
        # Get context
        context = episode.context if episode else None
        
        # Run evaluation
        verdict = await council.evaluate(prompt, response, episode, context)
        
        # Extract judge reports
        judge_reports = [r.model_dump() for r in verdict.judge_reports]
        
        return {
            **state,
            "council_verdict": verdict.model_dump(),
            "judge_reports": judge_reports,
            "current_step": "council_evaluated",
            "completed_steps": ["council_evaluation"],
        }
    
    except Exception as e:
        return {
            **state,
            "errors": [f"Council evaluation failed: {str(e)}"],
            "current_step": "error",
        }


async def cross_framing_node(state: EvaluationState) -> EvaluationState:
    """Perform cross-framing analysis."""
    try:
        metrics_by_framing = state.get("metrics_by_framing", {})
        episode = state.get("episode")
        episode_runs = state.get("episode_runs", [])
        
        if len(metrics_by_framing) < 2:
            # Not enough framings for comparison
            return {
                **state,
                "current_step": "cross_framing_skipped",
                "completed_steps": ["cross_framing_analysis"],
            }
        
        # Reconstruct objects
        runs = [EpisodeRun(**r) for r in episode_runs]
        metrics_map = {r["run_id"]: BehavioralMetrics(**metrics_by_framing.get(r.get("framing_type", "neutral"), {})) 
                       for r in episode_runs if r.get("run_id")}
        
        # If we don't have proper run IDs, create a simpler analysis
        if not metrics_map and episode:
            comparator = FramingComparator()
            
            framing_metrics = {}
            for framing_str, metrics_dict in metrics_by_framing.items():
                framing_type = FramingType(framing_str)
                framing_metrics[framing_type] = BehavioralMetrics(**metrics_dict)
            
            # Compute signals for cross-framing
            from risklab.measurement.signals import SignalComputer
            signal_computer = SignalComputer()
            signals = signal_computer.compute_all_signals(framing_metrics)
            
            analysis_dict = {
                "episode_id": episode.episode_id if episode else "unknown",
                "episode_name": episode.name if episode else "unknown",
                "framing_sensitivity_score": signals.framing_sensitivity.value if hasattr(signals.framing_sensitivity, 'value') else signals.framing_sensitivity if signals.framing_sensitivity else 0.0,
                "signals": signals.model_dump(),
            }
            
            return {
                **state,
                "cross_framing_analysis": analysis_dict,
                "current_step": "cross_framing_complete",
                "completed_steps": ["cross_framing_analysis"],
            }
        
        return {
            **state,
            "current_step": "cross_framing_complete",
            "completed_steps": ["cross_framing_analysis"],
        }
    
    except Exception as e:
        return {
            **state,
            "errors": [f"Cross-framing analysis failed: {str(e)}"],
            "current_step": "error",
        }


async def finalize_assessment_node(state: EvaluationState) -> EvaluationState:
    """Compile final assessment from all components."""
    try:
        episode = state.get("episode")
        
        # Gather all results
        final = {
            "assessment_id": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
            "episode_id": episode.episode_id if episode else None,
            "episode_name": episode.name if episode else None,
            "completed_at": datetime.utcnow().isoformat(),
            
            # Metrics summary
            "behavioral_metrics": state.get("behavioral_metrics"),
            "manipulation_signals": state.get("manipulation_signals"),
            
            # Risk assessment
            "conditioned_metrics": state.get("conditioned_metrics"),
            "decision_result": state.get("decision_result"),
            
            # Governance
            "council_verdict": state.get("council_verdict"),
            "num_judges": len(state.get("judge_reports", [])),
            
            # Cross-framing
            "cross_framing_analysis": state.get("cross_framing_analysis"),
            
            # Execution info
            "completed_steps": state.get("completed_steps", []),
            "errors": state.get("errors", []),
            "num_framings_evaluated": len(state.get("metrics_by_framing", {})),
        }
        
        # Determine overall outcome
        decision = state.get("decision_result")
        verdict = state.get("council_verdict")
        
        if decision:
            final["recommended_outcome"] = decision.get("outcome", "unknown")
        elif verdict:
            final["recommended_outcome"] = verdict.get("consensus_decision", "unknown")
        else:
            final["recommended_outcome"] = "inconclusive"
        
        return {
            **state,
            "final_assessment": final,
            "current_step": "complete",
            "completed_steps": ["finalize_assessment"],
        }
    
    except Exception as e:
        return {
            **state,
            "errors": [f"Assessment finalization failed: {str(e)}"],
            "current_step": "error",
        }


def should_continue_to_signals(state: EvaluationState) -> str:
    """Determine if we should compute signals."""
    if state.get("errors"):
        return "finalize"
    if len(state.get("metrics_by_framing", {})) >= 2:
        return "compute_signals"
    return "condition_risk"


def should_run_council(state: EvaluationState) -> str:
    """Determine if we should run council evaluation."""
    if state.get("errors"):
        return "finalize"
    # Always run council if we have a response
    if state.get("target_response"):
        return "council"
    return "finalize"


def should_do_cross_framing(state: EvaluationState) -> str:
    """Determine if cross-framing analysis is needed."""
    if state.get("errors"):
        return "finalize"
    if len(state.get("metrics_by_framing", {})) >= 2:
        return "cross_framing"
    return "finalize"


def create_evaluation_graph() -> StateGraph:
    """
    Create the LangGraph evaluation workflow.
    
    The graph implements:
    1. Response generation (optional)
    2. Behavioral analysis
    3. Signal computation
    4. Risk conditioning
    5. Threshold decision
    6. Council evaluation
    7. Cross-framing analysis
    8. Final assessment
    """
    # Create graph
    workflow = StateGraph(EvaluationState)
    
    # Add nodes
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("analyze_behavior", analyze_behavior_node)
    workflow.add_node("compute_signals", compute_signals_node)
    workflow.add_node("condition_risk", condition_risk_node)
    workflow.add_node("threshold_decision", threshold_decision_node)
    workflow.add_node("council_evaluation", council_evaluation_node)
    workflow.add_node("cross_framing", cross_framing_node)
    workflow.add_node("finalize", finalize_assessment_node)
    
    # Define edges
    workflow.set_entry_point("generate_response")
    
    workflow.add_edge("generate_response", "analyze_behavior")
    workflow.add_conditional_edges(
        "analyze_behavior",
        should_continue_to_signals,
        {
            "compute_signals": "compute_signals",
            "condition_risk": "condition_risk",
            "finalize": "finalize",
        }
    )
    workflow.add_edge("compute_signals", "condition_risk")
    workflow.add_edge("condition_risk", "threshold_decision")
    workflow.add_conditional_edges(
        "threshold_decision",
        should_run_council,
        {
            "council": "council_evaluation",
            "finalize": "finalize",
        }
    )
    workflow.add_conditional_edges(
        "council_evaluation",
        should_do_cross_framing,
        {
            "cross_framing": "cross_framing",
            "finalize": "finalize",
        }
    )
    workflow.add_edge("cross_framing", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow


def create_simple_evaluation_graph() -> StateGraph:
    """
    Create a simplified evaluation graph for single-response evaluation.
    
    Skips response generation and cross-framing analysis.
    """
    workflow = StateGraph(EvaluationState)
    
    workflow.add_node("analyze_behavior", analyze_behavior_node)
    workflow.add_node("condition_risk", condition_risk_node)
    workflow.add_node("threshold_decision", threshold_decision_node)
    workflow.add_node("council_evaluation", council_evaluation_node)
    workflow.add_node("finalize", finalize_assessment_node)
    
    workflow.set_entry_point("analyze_behavior")
    
    workflow.add_edge("analyze_behavior", "condition_risk")
    workflow.add_edge("condition_risk", "threshold_decision")
    workflow.add_edge("threshold_decision", "council_evaluation")
    workflow.add_edge("council_evaluation", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow


class EvaluationRunner:
    """
    Runner for executing evaluation graphs.
    """
    
    def __init__(
        self,
        runtime: Optional[ModelRuntime] = None,
        use_checkpointing: bool = False,
    ):
        self.runtime = runtime
        self.use_checkpointing = use_checkpointing
        self._graph = None
        self._simple_graph = None
    
    @property
    def graph(self) -> StateGraph:
        if self._graph is None:
            self._graph = create_evaluation_graph()
        return self._graph
    
    @property
    def simple_graph(self) -> StateGraph:
        if self._simple_graph is None:
            self._simple_graph = create_simple_evaluation_graph()
        return self._simple_graph
    
    async def evaluate_episode(
        self,
        episode: Episode,
        framing_type: FramingType = FramingType.NEUTRAL,
    ) -> Dict[str, Any]:
        """Evaluate an episode with response generation."""
        state = create_initial_state(episode, self.runtime)
        state["framing_type"] = framing_type
        
        # Compile and run
        if self.use_checkpointing:
            memory = MemorySaver()
            app = self.graph.compile(checkpointer=memory)
            config = {"configurable": {"thread_id": episode.episode_id}}
            result = await app.ainvoke(state, config)
        else:
            app = self.graph.compile()
            result = await app.ainvoke(state)
        
        return result.get("final_assessment", {})
    
    async def evaluate_response(
        self,
        prompt: str,
        response: str,
        context: Optional[ContextMetadata] = None,
    ) -> Dict[str, Any]:
        """Evaluate a pre-generated response."""
        state = create_initial_state(model_runtime=self.runtime)
        state["target_prompt"] = prompt
        state["target_response"] = response
        state["framing_type"] = FramingType.NEUTRAL
        
        if context:
            # Create minimal episode for context
            from risklab.scenarios.episode import Episode
            episode = Episode(
                name="Direct Evaluation",
                description="Direct response evaluation",
                core_prompt=prompt,
                context=context,
            )
            state["episode"] = episode
        
        # Use simple graph
        app = self.simple_graph.compile()
        result = await app.ainvoke(state)
        
        return result.get("final_assessment", {})
    
    async def evaluate_multi_framing(
        self,
        episode: Episode,
        framings: Optional[List[FramingType]] = None,
    ) -> Dict[str, Any]:
        """Evaluate an episode across multiple framings."""
        if framings is None:
            framings = [FramingType.NEUTRAL, FramingType.EVALUATED, FramingType.OVERSIGHT]
        
        all_metrics = {}
        all_runs = []
        
        for framing in framings:
            # Generate response for this framing
            prompt = episode.get_framed_prompt(framing)
            system_prompt = episode.get_system_prompt(framing)
            
            result = await self.runtime.generate(prompt, system_prompt)
            
            # Analyze
            analyzer = CompositeAnalyzer(self.runtime)
            analysis = await analyzer.analyze(result.text, prompt, episode)
            
            all_metrics[framing.value] = analysis.metrics.model_dump()
            all_runs.append({
                "framing": framing.value,
                "prompt": prompt,
                "response": result.text,
            })
        
        # Create state with all metrics
        state = create_initial_state(episode, self.runtime)
        state["metrics_by_framing"] = all_metrics
        state["episode_runs"] = all_runs
        state["target_response"] = all_runs[-1]["response"] if all_runs else None
        state["target_prompt"] = all_runs[-1]["prompt"] if all_runs else None
        
        # Run from signals computation
        app = self.graph.compile()
        
        # Manually advance through relevant nodes
        state = await compute_signals_node(state)
        state = await condition_risk_node(state)
        state = await threshold_decision_node(state)
        state = await council_evaluation_node(state)
        state = await cross_framing_node(state)
        state = await finalize_assessment_node(state)
        
        return state.get("final_assessment", {})
