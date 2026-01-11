"""
Metric Calibration Pipeline.

Provides:
- Human judgment calibration
- Ground truth alignment
- Confidence score calibration
- Metric correlation analysis
- Automated recalibration recommendations
"""

from typing import Optional, List, Dict, Any, Tuple, Callable
from datetime import datetime
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field
import numpy as np
from dataclasses import dataclass
import json


class CalibrationLabel(BaseModel):
    """Human-provided label for calibration."""
    text: str
    human_score: float  # 0-1
    human_label: str  # categorical
    confidence: float = 1.0
    annotator_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricCalibrationData(BaseModel):
    """Calibration data for a single metric."""
    metric_name: str
    predictions: List[float] = Field(default_factory=list)
    ground_truth: List[float] = Field(default_factory=list)
    confidences: List[float] = Field(default_factory=list)
    
    # Computed calibration metrics
    mean_absolute_error: Optional[float] = None
    correlation: Optional[float] = None
    bias: Optional[float] = None  # Systematic over/under prediction
    calibration_error: Optional[float] = None  # Expected calibration error
    
    def add_sample(
        self,
        prediction: float,
        ground_truth: float,
        confidence: float = 1.0
    ) -> None:
        """Add a calibration sample."""
        self.predictions.append(prediction)
        self.ground_truth.append(ground_truth)
        self.confidences.append(confidence)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute calibration metrics."""
        if len(self.predictions) < 5:
            return {"error": "insufficient_data"}
        
        preds = np.array(self.predictions)
        truth = np.array(self.ground_truth)
        
        # Mean absolute error
        self.mean_absolute_error = float(np.mean(np.abs(preds - truth)))
        
        # Correlation
        if np.std(preds) > 0 and np.std(truth) > 0:
            self.correlation = float(np.corrcoef(preds, truth)[0, 1])
        else:
            self.correlation = 0.0
        
        # Bias (mean difference)
        self.bias = float(np.mean(preds - truth))
        
        # Expected calibration error (ECE)
        self.calibration_error = self._compute_ece()
        
        return {
            "mae": self.mean_absolute_error,
            "correlation": self.correlation,
            "bias": self.bias,
            "ece": self.calibration_error
        }
    
    def _compute_ece(self, n_bins: int = 10) -> float:
        """Compute expected calibration error."""
        preds = np.array(self.predictions)
        truth = np.array(self.ground_truth)
        confs = np.array(self.confidences)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
            if np.sum(mask) > 0:
                bin_acc = np.mean(np.abs(preds[mask] - truth[mask]) < 0.1)
                bin_conf = np.mean(confs[mask])
                bin_size = np.sum(mask) / len(preds)
                ece += bin_size * np.abs(bin_acc - bin_conf)
        
        return float(ece)


class CalibrationResult(BaseModel):
    """Result of calibration analysis."""
    metric_name: str
    is_well_calibrated: bool
    calibration_quality: str  # excellent, good, fair, poor
    
    # Metrics
    mae: float
    correlation: float
    bias: float
    ece: float
    
    # Recommendations
    recommendations: List[str] = Field(default_factory=list)
    suggested_adjustment: Optional[Callable] = None
    
    # Sample size
    sample_count: int = 0


class CorrelationAnalysis(BaseModel):
    """Analysis of correlations between metrics."""
    metric_pairs: List[Tuple[str, str]] = Field(default_factory=list)
    correlations: Dict[str, float] = Field(default_factory=dict)
    redundant_pairs: List[Tuple[str, str, float]] = Field(default_factory=list)
    recommended_removals: List[str] = Field(default_factory=list)


class CalibrationPipeline:
    """
    Pipeline for calibrating metrics against human judgments.
    """
    
    CALIBRATION_THRESHOLDS = {
        "excellent": {"mae": 0.05, "correlation": 0.9, "ece": 0.05},
        "good": {"mae": 0.1, "correlation": 0.7, "ece": 0.1},
        "fair": {"mae": 0.2, "correlation": 0.5, "ece": 0.2},
        "poor": {"mae": float('inf'), "correlation": 0, "ece": float('inf')}
    }
    
    def __init__(self):
        self.metric_data: Dict[str, MetricCalibrationData] = {}
        self.labels: List[CalibrationLabel] = []
    
    def add_calibration_label(self, label: CalibrationLabel) -> None:
        """Add a human calibration label."""
        self.labels.append(label)
    
    def add_prediction(
        self,
        metric_name: str,
        prediction: float,
        ground_truth: float,
        confidence: float = 1.0
    ) -> None:
        """Add a prediction-ground truth pair for calibration."""
        if metric_name not in self.metric_data:
            self.metric_data[metric_name] = MetricCalibrationData(metric_name=metric_name)
        
        self.metric_data[metric_name].add_sample(prediction, ground_truth, confidence)
    
    def calibrate_metric(self, metric_name: str) -> CalibrationResult:
        """Calibrate a single metric."""
        if metric_name not in self.metric_data:
            raise ValueError(f"No calibration data for metric: {metric_name}")
        
        data = self.metric_data[metric_name]
        metrics = data.compute_metrics()
        
        if "error" in metrics:
            return CalibrationResult(
                metric_name=metric_name,
                is_well_calibrated=False,
                calibration_quality="unknown",
                mae=0, correlation=0, bias=0, ece=0,
                recommendations=["Need at least 5 samples for calibration"],
                sample_count=len(data.predictions)
            )
        
        # Determine calibration quality
        quality = self._determine_quality(metrics)
        is_well_calibrated = quality in ["excellent", "good"]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, data)
        
        return CalibrationResult(
            metric_name=metric_name,
            is_well_calibrated=is_well_calibrated,
            calibration_quality=quality,
            mae=metrics["mae"],
            correlation=metrics["correlation"],
            bias=metrics["bias"],
            ece=metrics["ece"],
            recommendations=recommendations,
            sample_count=len(data.predictions)
        )
    
    def _determine_quality(self, metrics: Dict[str, float]) -> str:
        """Determine calibration quality from metrics."""
        for quality, thresholds in self.CALIBRATION_THRESHOLDS.items():
            if (metrics["mae"] <= thresholds["mae"] and
                metrics["correlation"] >= thresholds["correlation"] and
                metrics["ece"] <= thresholds["ece"]):
                return quality
        return "poor"
    
    def _generate_recommendations(
        self,
        metrics: Dict[str, float],
        data: MetricCalibrationData
    ) -> List[str]:
        """Generate calibration recommendations."""
        recommendations = []
        
        # Check bias
        if abs(metrics["bias"]) > 0.1:
            direction = "over" if metrics["bias"] > 0 else "under"
            recommendations.append(
                f"Metric shows systematic {direction}prediction (bias: {metrics['bias']:.3f}). "
                f"Consider applying correction factor of {-metrics['bias']:.3f}"
            )
        
        # Check correlation
        if metrics["correlation"] < 0.5:
            recommendations.append(
                "Low correlation with ground truth suggests metric may not capture intended behavior. "
                "Consider revising metric definition or adding features."
            )
        
        # Check ECE
        if metrics["ece"] > 0.15:
            recommendations.append(
                "High calibration error indicates confidence scores are unreliable. "
                "Consider temperature scaling or Platt scaling for confidence calibration."
            )
        
        # Check sample size
        if len(data.predictions) < 50:
            recommendations.append(
                f"Only {len(data.predictions)} samples available. "
                "Recommend collecting more calibration data for reliable estimates."
            )
        
        if not recommendations:
            recommendations.append("Metric is well-calibrated. No adjustments needed.")
        
        return recommendations
    
    def calibrate_all(self) -> Dict[str, CalibrationResult]:
        """Calibrate all metrics with available data."""
        results = {}
        for metric_name in self.metric_data:
            results[metric_name] = self.calibrate_metric(metric_name)
        return results
    
    def analyze_correlations(self) -> CorrelationAnalysis:
        """Analyze correlations between metrics to identify redundancy."""
        analysis = CorrelationAnalysis()
        
        metric_names = list(self.metric_data.keys())
        if len(metric_names) < 2:
            return analysis
        
        # Compute pairwise correlations
        for i, name1 in enumerate(metric_names):
            for name2 in metric_names[i+1:]:
                data1 = self.metric_data[name1]
                data2 = self.metric_data[name2]
                
                # Need aligned data
                min_len = min(len(data1.predictions), len(data2.predictions))
                if min_len < 10:
                    continue
                
                preds1 = np.array(data1.predictions[:min_len])
                preds2 = np.array(data2.predictions[:min_len])
                
                if np.std(preds1) > 0 and np.std(preds2) > 0:
                    corr = float(np.corrcoef(preds1, preds2)[0, 1])
                    analysis.correlations[f"{name1}:{name2}"] = corr
                    analysis.metric_pairs.append((name1, name2))
                    
                    if abs(corr) > 0.85:
                        analysis.redundant_pairs.append((name1, name2, corr))
        
        # Recommend removals for highly correlated pairs
        seen = set()
        for m1, m2, corr in analysis.redundant_pairs:
            if m1 not in seen and m2 not in seen:
                # Keep the one with better calibration if available
                if m1 in self.metric_data and m2 in self.metric_data:
                    cal1 = self.calibrate_metric(m1)
                    cal2 = self.calibrate_metric(m2)
                    to_remove = m2 if cal1.correlation >= cal2.correlation else m1
                else:
                    to_remove = m2  # Default to removing second
                
                analysis.recommended_removals.append(to_remove)
                seen.add(to_remove)
        
        return analysis
    
    def create_adjustment_function(
        self,
        metric_name: str
    ) -> Callable[[float], float]:
        """Create an adjustment function based on calibration data."""
        if metric_name not in self.metric_data:
            return lambda x: x  # Identity function
        
        data = self.metric_data[metric_name]
        metrics = data.compute_metrics()
        
        if "error" in metrics:
            return lambda x: x
        
        bias = metrics.get("bias", 0)
        
        # Simple bias correction
        def adjust(value: float) -> float:
            adjusted = value - bias
            return max(0.0, min(1.0, adjusted))
        
        return adjust
    
    def save(self, path: Path) -> Path:
        """Save calibration data to file."""
        path = Path(path)
        
        data = {
            "metric_data": {
                name: {
                    "predictions": md.predictions,
                    "ground_truth": md.ground_truth,
                    "confidences": md.confidences
                }
                for name, md in self.metric_data.items()
            },
            "labels": [l.model_dump() for l in self.labels],
            "saved_at": datetime.utcnow().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return path
    
    @classmethod
    def load(cls, path: Path) -> "CalibrationPipeline":
        """Load calibration data from file."""
        with open(path) as f:
            data = json.load(f)
        
        pipeline = cls()
        
        for name, md in data.get("metric_data", {}).items():
            pipeline.metric_data[name] = MetricCalibrationData(
                metric_name=name,
                predictions=md["predictions"],
                ground_truth=md["ground_truth"],
                confidences=md["confidences"]
            )
        
        for label_data in data.get("labels", []):
            pipeline.labels.append(CalibrationLabel(**label_data))
        
        return pipeline


class HumanCalibrationInterface:
    """
    Interface for collecting human calibration labels.
    """
    
    def __init__(self, pipeline: CalibrationPipeline):
        self.pipeline = pipeline
        self.pending_items: List[Dict[str, Any]] = []
    
    def add_item_for_review(
        self,
        text: str,
        metric_predictions: Dict[str, float],
        context: Optional[Dict] = None
    ) -> str:
        """Add an item for human review."""
        item_id = f"review-{len(self.pending_items)}"
        
        self.pending_items.append({
            "id": item_id,
            "text": text,
            "predictions": metric_predictions,
            "context": context or {},
            "status": "pending"
        })
        
        return item_id
    
    def submit_label(
        self,
        item_id: str,
        labels: Dict[str, float],
        annotator_id: Optional[str] = None
    ) -> None:
        """Submit human labels for an item."""
        # Find the item
        item = None
        for i, pending in enumerate(self.pending_items):
            if pending["id"] == item_id:
                item = pending
                break
        
        if not item:
            raise ValueError(f"Item not found: {item_id}")
        
        # Add calibration data
        for metric_name, human_score in labels.items():
            if metric_name in item["predictions"]:
                self.pipeline.add_prediction(
                    metric_name=metric_name,
                    prediction=item["predictions"][metric_name],
                    ground_truth=human_score,
                    confidence=1.0
                )
        
        # Add label
        self.pipeline.add_calibration_label(CalibrationLabel(
            text=item["text"],
            human_score=np.mean(list(labels.values())),
            human_label=self._determine_label(labels),
            annotator_id=annotator_id,
            metadata={"predictions": item["predictions"], "labels": labels}
        ))
        
        # Update status
        item["status"] = "completed"
    
    def _determine_label(self, labels: Dict[str, float]) -> str:
        """Determine categorical label from scores."""
        avg = np.mean(list(labels.values()))
        if avg >= 0.8:
            return "safe"
        elif avg >= 0.6:
            return "acceptable"
        elif avg >= 0.4:
            return "concerning"
        else:
            return "unsafe"
    
    def get_pending_count(self) -> int:
        """Get number of pending items."""
        return sum(1 for item in self.pending_items if item["status"] == "pending")
    
    def generate_review_batch(self, batch_size: int = 10) -> List[Dict[str, Any]]:
        """Generate a batch of items for review."""
        pending = [item for item in self.pending_items if item["status"] == "pending"]
        return pending[:batch_size]


def create_calibration_report(
    results: Dict[str, CalibrationResult],
    correlation_analysis: CorrelationAnalysis
) -> str:
    """Generate a calibration report."""
    report = ["# Metric Calibration Report", ""]
    report.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    report.append("")
    
    # Summary
    well_calibrated = sum(1 for r in results.values() if r.is_well_calibrated)
    total = len(results)
    
    report.append("## Summary")
    report.append(f"- **Total Metrics:** {total}")
    report.append(f"- **Well Calibrated:** {well_calibrated} ({well_calibrated/total*100:.0f}%)")
    report.append(f"- **Needs Attention:** {total - well_calibrated}")
    report.append("")
    
    # Per-metric results
    report.append("## Metric Details")
    report.append("")
    
    for name, result in sorted(results.items(), key=lambda x: x[1].calibration_quality):
        emoji = "✅" if result.is_well_calibrated else "⚠️"
        report.append(f"### {emoji} {name}")
        report.append(f"- Quality: **{result.calibration_quality}**")
        report.append(f"- MAE: {result.mae:.3f}")
        report.append(f"- Correlation: {result.correlation:.3f}")
        report.append(f"- Bias: {result.bias:+.3f}")
        report.append(f"- ECE: {result.ece:.3f}")
        report.append(f"- Samples: {result.sample_count}")
        
        if result.recommendations:
            report.append("- **Recommendations:**")
            for rec in result.recommendations:
                report.append(f"  - {rec}")
        report.append("")
    
    # Correlation analysis
    if correlation_analysis.redundant_pairs:
        report.append("## Redundant Metric Pairs")
        report.append("")
        for m1, m2, corr in correlation_analysis.redundant_pairs:
            report.append(f"- {m1} ↔ {m2}: {corr:.3f}")
        
        if correlation_analysis.recommended_removals:
            report.append("")
            report.append("**Recommended for removal:** " + ", ".join(correlation_analysis.recommended_removals))
    
    return "\n".join(report)
