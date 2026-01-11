"""
Training-time monitoring for behavioral analysis.

Logs loss trajectories, evaluation metrics, and detects phase transitions.
"""

from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
from pydantic import BaseModel, Field
import numpy as np


class CheckpointMetrics(BaseModel):
    """Metrics captured at a training checkpoint."""
    checkpoint_id: str
    step: int
    epoch: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Loss metrics
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    
    # Standard evaluation metrics
    accuracy: Optional[float] = None
    perplexity: Optional[float] = None
    
    # Risk-adjusted behavior scores
    risk_scores: Dict[str, float] = Field(default_factory=dict)
    manipulation_scores: Dict[str, float] = Field(default_factory=dict)
    
    # Trigger-specific metrics
    trigger_response_rates: Dict[str, float] = Field(default_factory=dict)
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    def to_vector(self, metric_keys: List[str]) -> List[float]:
        """Convert specified metrics to a vector."""
        all_metrics = {
            "train_loss": self.train_loss,
            "eval_loss": self.eval_loss,
            "accuracy": self.accuracy,
            "perplexity": self.perplexity,
            **self.risk_scores,
            **self.manipulation_scores,
            **self.custom_metrics,
        }
        return [all_metrics.get(k, 0.0) or 0.0 for k in metric_keys]


class PhaseTransition(BaseModel):
    """Detected phase transition during training."""
    transition_id: str
    detected_at_step: int
    metric_name: str
    
    # Change characteristics
    change_magnitude: float
    change_direction: str  # increase, decrease
    
    # Statistical significance
    z_score: float
    is_significant: bool
    
    # Context
    before_value: float
    after_value: float
    window_size: int
    
    notes: Optional[str] = None


class TrainingMonitor:
    """
    Monitors training dynamics for behavioral changes.
    
    Detects sharp changes using change-point analysis.
    """
    
    def __init__(
        self,
        window_size: int = 10,
        significance_threshold: float = 2.0,
    ):
        self.window_size = window_size
        self.significance_threshold = significance_threshold
        
        self.checkpoints: List[CheckpointMetrics] = []
        self.transitions: List[PhaseTransition] = []
        self._metric_history: Dict[str, List[float]] = {}
    
    def log_checkpoint(self, metrics: CheckpointMetrics) -> List[PhaseTransition]:
        """
        Log a checkpoint and check for phase transitions.
        
        Returns any newly detected transitions.
        """
        self.checkpoints.append(metrics)
        
        # Update metric history
        self._update_history(metrics)
        
        # Check for transitions
        new_transitions = self._detect_transitions(metrics.step)
        self.transitions.extend(new_transitions)
        
        return new_transitions
    
    def _update_history(self, metrics: CheckpointMetrics) -> None:
        """Update metric history with new checkpoint."""
        metric_values = {
            "train_loss": metrics.train_loss,
            "eval_loss": metrics.eval_loss,
            "accuracy": metrics.accuracy,
            **metrics.risk_scores,
            **metrics.manipulation_scores,
        }
        
        for name, value in metric_values.items():
            if value is not None:
                if name not in self._metric_history:
                    self._metric_history[name] = []
                self._metric_history[name].append(value)
    
    def _detect_transitions(self, current_step: int) -> List[PhaseTransition]:
        """Detect phase transitions using change-point analysis."""
        transitions = []
        
        for metric_name, history in self._metric_history.items():
            if len(history) < self.window_size * 2:
                continue
            
            # Compare recent window to previous window
            recent = history[-self.window_size:]
            previous = history[-2*self.window_size:-self.window_size]
            
            recent_mean = np.mean(recent)
            previous_mean = np.mean(previous)
            
            # Combined standard deviation
            combined_std = np.std(previous + recent)
            if combined_std < 1e-8:
                continue
            
            # Z-score for the change
            z_score = abs(recent_mean - previous_mean) / combined_std
            
            if z_score > self.significance_threshold:
                direction = "increase" if recent_mean > previous_mean else "decrease"
                
                transition = PhaseTransition(
                    transition_id=f"{metric_name}_{current_step}",
                    detected_at_step=current_step,
                    metric_name=metric_name,
                    change_magnitude=abs(recent_mean - previous_mean),
                    change_direction=direction,
                    z_score=z_score,
                    is_significant=True,
                    before_value=previous_mean,
                    after_value=recent_mean,
                    window_size=self.window_size,
                )
                transitions.append(transition)
        
        return transitions
    
    def get_metric_trajectory(self, metric_name: str) -> List[float]:
        """Get the full trajectory for a metric."""
        return self._metric_history.get(metric_name, [])
    
    def get_risk_trajectory(self) -> Dict[str, List[float]]:
        """Get trajectories for all risk-related metrics."""
        return {
            k: v for k, v in self._metric_history.items()
            if "risk" in k.lower() or "manipulation" in k.lower()
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get monitoring summary."""
        return {
            "num_checkpoints": len(self.checkpoints),
            "num_transitions": len(self.transitions),
            "metrics_tracked": list(self._metric_history.keys()),
            "significant_transitions": [
                {
                    "step": t.detected_at_step,
                    "metric": t.metric_name,
                    "direction": t.change_direction,
                    "magnitude": t.change_magnitude,
                }
                for t in self.transitions if t.is_significant
            ],
        }
    
    def detect_concerning_patterns(self) -> List[Dict[str, Any]]:
        """Identify concerning patterns in training dynamics."""
        concerns = []
        
        # Check for risk score increases
        for metric, history in self._metric_history.items():
            if "risk" in metric.lower() and len(history) > 5:
                if history[-1] > history[0] * 1.5:
                    concerns.append({
                        "type": "risk_increase",
                        "metric": metric,
                        "initial": history[0],
                        "final": history[-1],
                        "increase_ratio": history[-1] / max(history[0], 0.01),
                    })
        
        # Check for sudden transitions
        for transition in self.transitions:
            if transition.z_score > 3.0:
                concerns.append({
                    "type": "sudden_transition",
                    "metric": transition.metric_name,
                    "step": transition.detected_at_step,
                    "z_score": transition.z_score,
                })
        
        # Check for diverging train/eval loss
        train_loss = self._metric_history.get("train_loss", [])
        eval_loss = self._metric_history.get("eval_loss", [])
        
        if len(train_loss) > 10 and len(eval_loss) > 10:
            recent_gap = eval_loss[-1] - train_loss[-1]
            initial_gap = eval_loss[0] - train_loss[0]
            
            if recent_gap > initial_gap * 2:
                concerns.append({
                    "type": "overfitting",
                    "initial_gap": initial_gap,
                    "recent_gap": recent_gap,
                })
        
        return concerns
    
    def export_for_visualization(self) -> Dict[str, Any]:
        """Export data for visualization."""
        steps = [c.step for c in self.checkpoints]
        
        return {
            "steps": steps,
            "metrics": {
                name: values 
                for name, values in self._metric_history.items()
            },
            "transitions": [
                {
                    "step": t.detected_at_step,
                    "metric": t.metric_name,
                    "z_score": t.z_score,
                }
                for t in self.transitions
            ],
        }


class TrainingCallback:
    """
    Callback interface for integration with training frameworks.
    """
    
    def __init__(self, monitor: TrainingMonitor):
        self.monitor = monitor
    
    def on_evaluate(
        self,
        step: int,
        metrics: Dict[str, float],
        **kwargs,
    ) -> None:
        """Called after each evaluation."""
        checkpoint = CheckpointMetrics(
            checkpoint_id=f"step_{step}",
            step=step,
            train_loss=metrics.get("train_loss"),
            eval_loss=metrics.get("eval_loss") or metrics.get("loss"),
            accuracy=metrics.get("accuracy"),
            perplexity=metrics.get("perplexity"),
            custom_metrics={
                k: v for k, v in metrics.items()
                if k not in ("train_loss", "eval_loss", "loss", "accuracy", "perplexity")
            },
        )
        
        transitions = self.monitor.log_checkpoint(checkpoint)
        
        if transitions:
            print(f"[TrainingMonitor] Detected {len(transitions)} phase transition(s) at step {step}")
            for t in transitions:
                print(f"  - {t.metric_name}: {t.change_direction} by {t.change_magnitude:.4f}")
    
    def on_train_end(self) -> Dict[str, Any]:
        """Called at the end of training."""
        concerns = self.monitor.detect_concerning_patterns()
        summary = self.monitor.get_summary()
        
        if concerns:
            print(f"[TrainingMonitor] {len(concerns)} concerning pattern(s) detected:")
            for c in concerns:
                print(f"  - {c['type']}: {c}")
        
        return {
            "summary": summary,
            "concerns": concerns,
        }
