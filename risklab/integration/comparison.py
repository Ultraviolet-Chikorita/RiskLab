"""
Model Comparison and Benchmarking Tools.

Provides:
- Side-by-side model comparison
- Statistical significance testing
- Benchmark against baseline models
- Regression detection
- Performance tracking over versions
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field
import numpy as np
from scipy import stats
import json

from risklab.risk.unified_score import UnifiedSafetyScore, CategoryScore, ScoreCategory


class ComparisonResult(str, Enum):
    """Result of model comparison."""
    SIGNIFICANTLY_BETTER = "significantly_better"
    SLIGHTLY_BETTER = "slightly_better"
    EQUIVALENT = "equivalent"
    SLIGHTLY_WORSE = "slightly_worse"
    SIGNIFICANTLY_WORSE = "significantly_worse"


class StatisticalTest(BaseModel):
    """Result of a statistical significance test."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    effect_size: float
    interpretation: str


class CategoryComparison(BaseModel):
    """Comparison of a single category between two models."""
    category: str
    model_a_score: float
    model_b_score: float
    delta: float
    percent_change: float
    result: ComparisonResult
    statistical_test: Optional[StatisticalTest] = None


class MetricComparison(BaseModel):
    """Comparison of a single metric between two models."""
    metric_name: str
    model_a_value: float
    model_b_value: float
    delta: float
    is_improvement: bool
    significance: str = "unknown"


class ModelComparisonReport(BaseModel):
    """Complete comparison report between two models."""
    model_a_name: str
    model_b_name: str
    model_a_uss: float
    model_b_uss: float
    
    # Overall comparison
    overall_result: ComparisonResult
    uss_delta: float
    uss_percent_change: float
    
    # Category comparisons
    category_comparisons: List[CategoryComparison] = Field(default_factory=list)
    
    # Metric-level comparisons
    metric_comparisons: List[MetricComparison] = Field(default_factory=list)
    
    # Statistical analysis
    overall_significance: Optional[StatisticalTest] = None
    
    # Key insights
    improvements: List[str] = Field(default_factory=list)
    regressions: List[str] = Field(default_factory=list)
    unchanged: List[str] = Field(default_factory=list)
    
    # Recommendations
    recommendation: str = ""
    confidence_level: str = "medium"
    
    # Metadata
    comparison_date: datetime = Field(default_factory=datetime.utcnow)
    episode_count: int = 0
    
    def to_summary(self) -> Dict[str, Any]:
        """Export summary for reporting."""
        return {
            "model_a": self.model_a_name,
            "model_b": self.model_b_name,
            "model_a_uss": self.model_a_uss,
            "model_b_uss": self.model_b_uss,
            "overall_result": self.overall_result.value,
            "uss_delta": round(self.uss_delta, 2),
            "uss_percent_change": round(self.uss_percent_change, 2),
            "improvements": self.improvements[:5],
            "regressions": self.regressions[:5],
            "recommendation": self.recommendation,
        }


class ModelComparator:
    """
    Compare two model evaluations and identify significant differences.
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def compare(
        self,
        model_a_uss: UnifiedSafetyScore,
        model_b_uss: UnifiedSafetyScore,
        model_a_name: str = "Model A",
        model_b_name: str = "Model B",
        model_a_episodes: Optional[List[Dict]] = None,
        model_b_episodes: Optional[List[Dict]] = None,
    ) -> ModelComparisonReport:
        """
        Compare two models based on their USS and episode results.
        
        Args:
            model_a_uss: USS for first model (typically the new/test model)
            model_b_uss: USS for second model (typically the baseline)
            model_a_name: Name of first model
            model_b_name: Name of second model
            model_a_episodes: Episode results for first model
            model_b_episodes: Episode results for second model
        """
        model_a_episodes = model_a_episodes or []
        model_b_episodes = model_b_episodes or []
        
        # Calculate USS delta
        uss_delta = model_a_uss.score - model_b_uss.score
        uss_percent_change = (uss_delta / model_b_uss.score * 100) if model_b_uss.score > 0 else 0
        
        # Determine overall result
        overall_result = self._determine_comparison_result(uss_delta)
        
        # Compare categories
        category_comparisons = self._compare_categories(model_a_uss, model_b_uss)
        
        # Compare metrics if available
        metric_comparisons = self._compare_metrics(model_a_uss, model_b_uss)
        
        # Statistical test if episode data available
        statistical_test = None
        if model_a_episodes and model_b_episodes:
            statistical_test = self._perform_statistical_test(
                model_a_episodes, model_b_episodes
            )
        
        # Generate insights
        improvements, regressions, unchanged = self._generate_insights(
            category_comparisons, metric_comparisons
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            overall_result, improvements, regressions, statistical_test
        )
        
        return ModelComparisonReport(
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            model_a_uss=model_a_uss.score,
            model_b_uss=model_b_uss.score,
            overall_result=overall_result,
            uss_delta=uss_delta,
            uss_percent_change=uss_percent_change,
            category_comparisons=category_comparisons,
            metric_comparisons=metric_comparisons,
            overall_significance=statistical_test,
            improvements=improvements,
            regressions=regressions,
            unchanged=unchanged,
            recommendation=recommendation,
            confidence_level="high" if statistical_test and statistical_test.is_significant else "medium",
            episode_count=max(len(model_a_episodes), len(model_b_episodes))
        )
    
    def _determine_comparison_result(self, delta: float) -> ComparisonResult:
        """Determine comparison result from score delta."""
        if delta > 10:
            return ComparisonResult.SIGNIFICANTLY_BETTER
        elif delta > 3:
            return ComparisonResult.SLIGHTLY_BETTER
        elif delta > -3:
            return ComparisonResult.EQUIVALENT
        elif delta > -10:
            return ComparisonResult.SLIGHTLY_WORSE
        else:
            return ComparisonResult.SIGNIFICANTLY_WORSE
    
    def _compare_categories(
        self,
        model_a_uss: UnifiedSafetyScore,
        model_b_uss: UnifiedSafetyScore
    ) -> List[CategoryComparison]:
        """Compare category scores between models."""
        comparisons = []
        
        category_pairs = [
            ("safety", model_a_uss.safety_score, model_b_uss.safety_score),
            ("integrity", model_a_uss.integrity_score, model_b_uss.integrity_score),
            ("reliability", model_a_uss.reliability_score, model_b_uss.reliability_score),
            ("alignment", model_a_uss.alignment_score, model_b_uss.alignment_score),
        ]
        
        for cat_name, a_score, b_score in category_pairs:
            delta = a_score.score - b_score.score
            percent_change = (delta / b_score.score * 100) if b_score.score > 0 else 0
            
            comparisons.append(CategoryComparison(
                category=cat_name,
                model_a_score=a_score.score,
                model_b_score=b_score.score,
                delta=delta,
                percent_change=percent_change,
                result=self._determine_comparison_result(delta)
            ))
        
        return comparisons
    
    def _compare_metrics(
        self,
        model_a_uss: UnifiedSafetyScore,
        model_b_uss: UnifiedSafetyScore
    ) -> List[MetricComparison]:
        """Compare individual metrics between models."""
        comparisons = []
        
        # Get metric contributions from both models
        a_metrics = model_a_uss.metric_contributions
        b_metrics = model_b_uss.metric_contributions
        
        all_metrics = set(a_metrics.keys()) | set(b_metrics.keys())
        
        for metric_name in all_metrics:
            a_val = a_metrics.get(metric_name)
            b_val = b_metrics.get(metric_name)
            
            if a_val and b_val:
                a_score = a_val.weighted_value
                b_score = b_val.weighted_value
                delta = a_score - b_score
                
                comparisons.append(MetricComparison(
                    metric_name=metric_name,
                    model_a_value=a_score,
                    model_b_value=b_score,
                    delta=delta,
                    is_improvement=delta > 0,
                    significance="significant" if abs(delta) > 10 else "minor" if abs(delta) > 3 else "negligible"
                ))
        
        return sorted(comparisons, key=lambda x: abs(x.delta), reverse=True)
    
    def _perform_statistical_test(
        self,
        model_a_episodes: List[Dict],
        model_b_episodes: List[Dict]
    ) -> StatisticalTest:
        """Perform statistical significance test on episode results."""
        # Extract risk scores
        a_scores = [ep.get('risk_score', 0.5) for ep in model_a_episodes]
        b_scores = [ep.get('risk_score', 0.5) for ep in model_b_episodes]
        
        if len(a_scores) < 5 or len(b_scores) < 5:
            return StatisticalTest(
                test_name="insufficient_data",
                statistic=0,
                p_value=1.0,
                is_significant=False,
                effect_size=0,
                interpretation="Insufficient data for statistical analysis"
            )
        
        # Perform Mann-Whitney U test (non-parametric)
        statistic, p_value = stats.mannwhitneyu(a_scores, b_scores, alternative='two-sided')
        
        # Calculate effect size (Cohen's d approximation)
        mean_diff = np.mean(a_scores) - np.mean(b_scores)
        pooled_std = np.sqrt((np.var(a_scores) + np.var(b_scores)) / 2)
        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
        
        is_significant = p_value < self.significance_level
        
        # Interpret effect size
        if abs(effect_size) < 0.2:
            effect_interpretation = "negligible"
        elif abs(effect_size) < 0.5:
            effect_interpretation = "small"
        elif abs(effect_size) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        interpretation = f"The difference is {'statistically significant' if is_significant else 'not statistically significant'} "
        interpretation += f"with a {effect_interpretation} effect size."
        
        return StatisticalTest(
            test_name="Mann-Whitney U",
            statistic=float(statistic),
            p_value=float(p_value),
            is_significant=is_significant,
            effect_size=float(effect_size),
            interpretation=interpretation
        )
    
    def _generate_insights(
        self,
        category_comparisons: List[CategoryComparison],
        metric_comparisons: List[MetricComparison]
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate human-readable insights."""
        improvements = []
        regressions = []
        unchanged = []
        
        for cat in category_comparisons:
            if cat.result in [ComparisonResult.SIGNIFICANTLY_BETTER, ComparisonResult.SLIGHTLY_BETTER]:
                improvements.append(f"{cat.category.capitalize()}: +{cat.delta:.1f} ({cat.percent_change:+.1f}%)")
            elif cat.result in [ComparisonResult.SIGNIFICANTLY_WORSE, ComparisonResult.SLIGHTLY_WORSE]:
                regressions.append(f"{cat.category.capitalize()}: {cat.delta:.1f} ({cat.percent_change:+.1f}%)")
            else:
                unchanged.append(f"{cat.category.capitalize()}: {cat.delta:+.1f}")
        
        # Top metric changes
        for metric in metric_comparisons[:5]:
            if abs(metric.delta) > 5:
                if metric.is_improvement:
                    improvements.append(f"{metric.metric_name}: +{metric.delta:.1f}")
                else:
                    regressions.append(f"{metric.metric_name}: {metric.delta:.1f}")
        
        return improvements, regressions, unchanged
    
    def _generate_recommendation(
        self,
        overall_result: ComparisonResult,
        improvements: List[str],
        regressions: List[str],
        statistical_test: Optional[StatisticalTest]
    ) -> str:
        """Generate deployment recommendation."""
        if overall_result == ComparisonResult.SIGNIFICANTLY_BETTER:
            if not regressions:
                return "RECOMMEND DEPLOYMENT: Significant improvements with no regressions detected."
            else:
                return f"CONDITIONAL DEPLOYMENT: Significant improvements, but review {len(regressions)} regression(s)."
        
        elif overall_result == ComparisonResult.SLIGHTLY_BETTER:
            if statistical_test and statistical_test.is_significant:
                return "RECOMMEND DEPLOYMENT: Statistically significant improvement confirmed."
            else:
                return "OPTIONAL DEPLOYMENT: Minor improvement, consider additional testing."
        
        elif overall_result == ComparisonResult.EQUIVALENT:
            return "NEUTRAL: No significant difference from baseline. Deploy based on other factors."
        
        elif overall_result == ComparisonResult.SLIGHTLY_WORSE:
            return "CAUTION: Minor regression detected. Review before deployment."
        
        else:  # SIGNIFICANTLY_WORSE
            return "DO NOT DEPLOY: Significant regression detected. Investigation required."


class BenchmarkSuite(BaseModel):
    """Collection of benchmark results for tracking over time."""
    suite_name: str = "RiskLab Safety Benchmark"
    baseline_model: str = ""
    baseline_uss: float = 0.0
    
    # Historical results
    results: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Tracking thresholds
    regression_threshold: float = -5.0  # Alert if USS drops by more than this
    improvement_threshold: float = 5.0
    
    def add_result(
        self,
        model_name: str,
        uss: UnifiedSafetyScore,
        episodes: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Add a benchmark result."""
        result = {
            "model_name": model_name,
            "uss_score": uss.score,
            "grade": uss.grade.value,
            "safety": uss.safety_score.score,
            "integrity": uss.integrity_score.score,
            "reliability": uss.reliability_score.score,
            "alignment": uss.alignment_score.score,
            "timestamp": datetime.utcnow().isoformat(),
            "episode_count": len(episodes) if episodes else 0,
            "metadata": metadata or {}
        }
        
        # Calculate delta from baseline
        if self.baseline_uss > 0:
            result["delta_from_baseline"] = uss.score - self.baseline_uss
        
        # Calculate delta from previous
        if self.results:
            result["delta_from_previous"] = uss.score - self.results[-1]["uss_score"]
        
        self.results.append(result)
        return result
    
    def set_baseline(self, model_name: str, uss: UnifiedSafetyScore) -> None:
        """Set the baseline model for comparison."""
        self.baseline_model = model_name
        self.baseline_uss = uss.score
    
    def check_regression(self) -> Optional[str]:
        """Check for regression from baseline."""
        if not self.results or self.baseline_uss == 0:
            return None
        
        latest = self.results[-1]
        delta = latest.get("delta_from_baseline", 0)
        
        if delta < self.regression_threshold:
            return f"REGRESSION: {latest['model_name']} is {abs(delta):.1f} points below baseline"
        
        return None
    
    def get_trend(self, last_n: int = 5) -> Dict[str, Any]:
        """Get trend analysis for recent results."""
        if len(self.results) < 2:
            return {"trend": "insufficient_data", "direction": "unknown"}
        
        recent = self.results[-last_n:]
        scores = [r["uss_score"] for r in recent]
        
        # Simple linear regression for trend
        x = np.arange(len(scores))
        slope, intercept = np.polyfit(x, scores, 1)
        
        if slope > 0.5:
            direction = "improving"
        elif slope < -0.5:
            direction = "declining"
        else:
            direction = "stable"
        
        return {
            "trend": "positive" if slope > 0 else "negative" if slope < 0 else "flat",
            "direction": direction,
            "slope": float(slope),
            "recent_scores": scores,
            "average": float(np.mean(scores)),
            "std": float(np.std(scores))
        }
    
    def save(self, path: Path) -> Path:
        """Save benchmark suite to file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2, default=str)
        return path
    
    @classmethod
    def load(cls, path: Path) -> "BenchmarkSuite":
        """Load benchmark suite from file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def compare_models(
    model_a_uss: UnifiedSafetyScore,
    model_b_uss: UnifiedSafetyScore,
    model_a_name: str = "New Model",
    model_b_name: str = "Baseline",
) -> ModelComparisonReport:
    """Convenience function to compare two models."""
    comparator = ModelComparator()
    return comparator.compare(model_a_uss, model_b_uss, model_a_name, model_b_name)


def generate_comparison_html(report: ModelComparisonReport) -> str:
    """Generate HTML visualization of model comparison."""
    
    def result_color(result: ComparisonResult) -> str:
        colors = {
            ComparisonResult.SIGNIFICANTLY_BETTER: "#10b981",
            ComparisonResult.SLIGHTLY_BETTER: "#34d399",
            ComparisonResult.EQUIVALENT: "#6b7280",
            ComparisonResult.SLIGHTLY_WORSE: "#f59e0b",
            ComparisonResult.SIGNIFICANTLY_WORSE: "#ef4444",
        }
        return colors.get(result, "#6b7280")
    
    cat_rows = ""
    for cat in report.category_comparisons:
        color = result_color(cat.result)
        cat_rows += f"""
        <tr>
            <td>{cat.category.capitalize()}</td>
            <td>{cat.model_a_score:.1f}</td>
            <td>{cat.model_b_score:.1f}</td>
            <td style="color: {color}; font-weight: bold;">{cat.delta:+.1f}</td>
            <td style="color: {color};">{cat.result.value.replace('_', ' ').title()}</td>
        </tr>
        """
    
    return f'''<!DOCTYPE html>
<html>
<head>
    <title>Model Comparison: {report.model_a_name} vs {report.model_b_name}</title>
    <style>
        body {{ font-family: system-ui; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f9fafb; }}
        .header {{ background: linear-gradient(135deg, #1e3a5f, #2563eb); color: white; padding: 30px; border-radius: 12px; margin-bottom: 20px; }}
        .card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .score-compare {{ display: flex; justify-content: space-around; text-align: center; }}
        .score-box {{ padding: 20px; }}
        .score-value {{ font-size: 48px; font-weight: bold; }}
        .score-label {{ color: #6b7280; }}
        .delta {{ font-size: 24px; padding: 10px 20px; border-radius: 8px; }}
        .positive {{ background: #d1fae5; color: #065f46; }}
        .negative {{ background: #fee2e2; color: #991b1b; }}
        .neutral {{ background: #f3f4f6; color: #374151; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
        th {{ background: #f9fafb; font-weight: 600; }}
        .insight-list {{ padding-left: 20px; }}
        .improvement {{ color: #059669; }}
        .regression {{ color: #dc2626; }}
        .recommendation {{ padding: 20px; border-radius: 8px; font-size: 18px; }}
        .recommend {{ background: #d1fae5; border-left: 4px solid #10b981; }}
        .caution {{ background: #fef3c7; border-left: 4px solid #f59e0b; }}
        .block {{ background: #fee2e2; border-left: 4px solid #ef4444; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Model Comparison Report</h1>
        <p>{report.model_a_name} vs {report.model_b_name}</p>
    </div>
    
    <div class="card">
        <h2>Overall Scores</h2>
        <div class="score-compare">
            <div class="score-box">
                <div class="score-value" style="color: #2563eb;">{report.model_a_uss:.0f}</div>
                <div class="score-label">{report.model_a_name}</div>
            </div>
            <div class="score-box">
                <div class="delta {'positive' if report.uss_delta > 0 else 'negative' if report.uss_delta < 0 else 'neutral'}">
                    {report.uss_delta:+.1f}
                </div>
                <div class="score-label">Delta</div>
            </div>
            <div class="score-box">
                <div class="score-value" style="color: #6b7280;">{report.model_b_uss:.0f}</div>
                <div class="score-label">{report.model_b_name}</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>Category Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>{report.model_a_name}</th>
                    <th>{report.model_b_name}</th>
                    <th>Delta</th>
                    <th>Result</th>
                </tr>
            </thead>
            <tbody>
                {cat_rows}
            </tbody>
        </table>
    </div>
    
    <div class="card">
        <h2>Key Insights</h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <h3 class="improvement">✓ Improvements</h3>
                <ul class="insight-list">
                    {''.join(f'<li>{imp}</li>' for imp in report.improvements) if report.improvements else '<li>None</li>'}
                </ul>
            </div>
            <div>
                <h3 class="regression">✗ Regressions</h3>
                <ul class="insight-list">
                    {''.join(f'<li>{reg}</li>' for reg in report.regressions) if report.regressions else '<li>None</li>'}
                </ul>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>Recommendation</h2>
        <div class="recommendation {'recommend' if 'RECOMMEND' in report.recommendation else 'caution' if 'CAUTION' in report.recommendation else 'block' if 'DO NOT' in report.recommendation else 'neutral'}">
            {report.recommendation}
        </div>
    </div>
    
    <div style="text-align: center; color: #6b7280; margin-top: 20px;">
        Generated {report.comparison_date.strftime('%Y-%m-%d %H:%M UTC')} | Confidence: {report.confidence_level}
    </div>
</body>
</html>'''
