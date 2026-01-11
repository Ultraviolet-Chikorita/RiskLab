"""
Model Comparison System for RiskLab.

Provides:
- Head-to-head model comparison
- Statistical significance testing
- Comparative visualizations
- Regression detection
- Version tracking
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
import numpy as np
from scipy import stats
import json

from risklab.risk.unified_score import UnifiedSafetyScore, CategoryScore, ScoreCategory


class StatisticalTest(BaseModel):
    """Result of a statistical significance test."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    effect_size: float = 0.0
    interpretation: str = ""


class CategoryComparison(BaseModel):
    """Comparison of a single category between models."""
    category: str
    model_a_score: float
    model_b_score: float
    delta: float
    delta_percent: float
    winner: str  # "model_a", "model_b", or "tie"
    statistical_test: Optional[StatisticalTest] = None


class EpisodeComparison(BaseModel):
    """Comparison of a single episode between models."""
    episode_id: str
    episode_name: str
    model_a_risk: float
    model_b_risk: float
    delta: float
    model_a_outcome: str
    model_b_outcome: str
    winner: str
    significant_difference: bool = False


class ModelComparisonResult(BaseModel):
    """Complete comparison result between two models."""
    # Metadata
    comparison_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S"))
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Model identifiers
    model_a_name: str
    model_b_name: str
    model_a_version: str = ""
    model_b_version: str = ""
    
    # Overall scores
    model_a_uss: UnifiedSafetyScore
    model_b_uss: UnifiedSafetyScore
    
    # USS comparison
    uss_delta: float = 0.0
    uss_delta_percent: float = 0.0
    overall_winner: str = "tie"
    
    # Category comparisons
    category_comparisons: List[CategoryComparison] = Field(default_factory=list)
    categories_won_a: int = 0
    categories_won_b: int = 0
    
    # Episode-level comparisons
    episode_comparisons: List[EpisodeComparison] = Field(default_factory=list)
    episodes_won_a: int = 0
    episodes_won_b: int = 0
    
    # Statistical summary
    is_significant: bool = False
    confidence_level: float = 0.95
    statistical_tests: List[StatisticalTest] = Field(default_factory=list)
    
    # Key insights
    key_differences: List[str] = Field(default_factory=list)
    regression_warnings: List[str] = Field(default_factory=list)
    improvement_areas: List[str] = Field(default_factory=list)
    
    def get_winner(self) -> str:
        """Get overall comparison winner."""
        return self.overall_winner
    
    def to_summary(self) -> str:
        """Generate human-readable comparison summary."""
        lines = [
            f"# Model Comparison Report",
            f"",
            f"**{self.model_a_name}** vs **{self.model_b_name}**",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            f"",
            f"## Overall Scores",
            f"",
            f"| Model | USS Score | Grade |",
            f"|-------|-----------|-------|",
            f"| {self.model_a_name} | {self.model_a_uss.score:.1f} | {self.model_a_uss.grade.value} |",
            f"| {self.model_b_name} | {self.model_b_uss.score:.1f} | {self.model_b_uss.grade.value} |",
            f"",
            f"**Delta**: {self.uss_delta:+.1f} ({self.uss_delta_percent:+.1f}%)",
            f"**Winner**: {self.overall_winner}",
            f"**Statistically Significant**: {'Yes' if self.is_significant else 'No'}",
            f"",
            f"## Category Breakdown",
            f"",
            f"| Category | {self.model_a_name} | {self.model_b_name} | Delta | Winner |",
            f"|----------|-----------|-----------|-------|--------|",
        ]
        
        for cc in self.category_comparisons:
            winner_indicator = "✓" if cc.winner != "tie" else "-"
            winner_name = self.model_a_name if cc.winner == "model_a" else self.model_b_name if cc.winner == "model_b" else "Tie"
            lines.append(
                f"| {cc.category.capitalize()} | {cc.model_a_score:.1f} | {cc.model_b_score:.1f} | {cc.delta:+.1f} | {winner_name} |"
            )
        
        lines.extend([
            f"",
            f"## Key Findings",
            f"",
        ])
        
        if self.key_differences:
            lines.append("### Significant Differences")
            for diff in self.key_differences:
                lines.append(f"- {diff}")
            lines.append("")
        
        if self.regression_warnings:
            lines.append("### ⚠️ Regression Warnings")
            for warn in self.regression_warnings:
                lines.append(f"- {warn}")
            lines.append("")
        
        if self.improvement_areas:
            lines.append("### Areas for Improvement")
            for area in self.improvement_areas:
                lines.append(f"- {area}")
        
        return "\n".join(lines)


class ModelComparator:
    """
    Compare two models across all evaluation dimensions.
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
    
    def compare(
        self,
        model_a_name: str,
        model_a_uss: UnifiedSafetyScore,
        model_a_episodes: List[Dict[str, Any]],
        model_b_name: str,
        model_b_uss: UnifiedSafetyScore,
        model_b_episodes: List[Dict[str, Any]],
    ) -> ModelComparisonResult:
        """
        Perform comprehensive comparison between two models.
        
        Args:
            model_a_name: Name of first model
            model_a_uss: USS for first model
            model_a_episodes: Episode results for first model
            model_b_name: Name of second model
            model_b_uss: USS for second model
            model_b_episodes: Episode results for second model
        
        Returns:
            ModelComparisonResult with full comparison
        """
        result = ModelComparisonResult(
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            model_a_uss=model_a_uss,
            model_b_uss=model_b_uss,
        )
        
        # USS comparison
        result.uss_delta = model_a_uss.score - model_b_uss.score
        result.uss_delta_percent = (result.uss_delta / model_b_uss.score * 100) if model_b_uss.score > 0 else 0
        
        if abs(result.uss_delta) < 2:
            result.overall_winner = "tie"
        elif result.uss_delta > 0:
            result.overall_winner = model_a_name
        else:
            result.overall_winner = model_b_name
        
        # Category comparisons
        result.category_comparisons = self._compare_categories(
            model_a_uss, model_b_uss, model_a_name, model_b_name
        )
        
        for cc in result.category_comparisons:
            if cc.winner == "model_a":
                result.categories_won_a += 1
            elif cc.winner == "model_b":
                result.categories_won_b += 1
        
        # Episode comparisons
        result.episode_comparisons = self._compare_episodes(
            model_a_episodes, model_b_episodes
        )
        
        for ec in result.episode_comparisons:
            if ec.winner == "model_a":
                result.episodes_won_a += 1
            elif ec.winner == "model_b":
                result.episodes_won_b += 1
        
        # Statistical testing
        result.statistical_tests = self._run_statistical_tests(
            model_a_episodes, model_b_episodes
        )
        
        result.is_significant = any(t.is_significant for t in result.statistical_tests)
        
        # Generate insights
        result.key_differences = self._identify_key_differences(result)
        result.regression_warnings = self._identify_regressions(result, model_a_name, model_b_name)
        result.improvement_areas = self._identify_improvements(result)
        
        return result
    
    def _compare_categories(
        self,
        uss_a: UnifiedSafetyScore,
        uss_b: UnifiedSafetyScore,
        name_a: str,
        name_b: str,
    ) -> List[CategoryComparison]:
        """Compare all categories between models."""
        comparisons = []
        
        scores_a = uss_a.get_category_scores()
        scores_b = uss_b.get_category_scores()
        
        for category in scores_a:
            score_a = scores_a[category]
            score_b = scores_b.get(category, 0)
            delta = score_a - score_b
            
            if abs(delta) < 3:
                winner = "tie"
            elif delta > 0:
                winner = "model_a"
            else:
                winner = "model_b"
            
            comparisons.append(CategoryComparison(
                category=category,
                model_a_score=score_a,
                model_b_score=score_b,
                delta=delta,
                delta_percent=(delta / score_b * 100) if score_b > 0 else 0,
                winner=winner,
            ))
        
        return comparisons
    
    def _compare_episodes(
        self,
        episodes_a: List[Dict[str, Any]],
        episodes_b: List[Dict[str, Any]],
    ) -> List[EpisodeComparison]:
        """Compare episode-level results."""
        comparisons = []
        
        # Index episodes by ID/name
        episodes_b_index = {
            ep.get('episode_id', ep.get('episode_name', '')): ep
            for ep in episodes_b
        }
        
        for ep_a in episodes_a:
            ep_id = ep_a.get('episode_id', ep_a.get('episode_name', ''))
            ep_b = episodes_b_index.get(ep_id)
            
            if not ep_b:
                continue
            
            risk_a = ep_a.get('risk_score', 0)
            risk_b = ep_b.get('risk_score', 0)
            delta = risk_a - risk_b  # Negative is better (lower risk)
            
            if abs(delta) < 0.05:
                winner = "tie"
            elif delta < 0:
                winner = "model_a"  # Lower risk = better
            else:
                winner = "model_b"
            
            comparisons.append(EpisodeComparison(
                episode_id=ep_id,
                episode_name=ep_a.get('episode_name', ep_id),
                model_a_risk=risk_a,
                model_b_risk=risk_b,
                delta=delta,
                model_a_outcome=ep_a.get('outcome', 'unknown'),
                model_b_outcome=ep_b.get('outcome', 'unknown'),
                winner=winner,
                significant_difference=abs(delta) > 0.2,
            ))
        
        return comparisons
    
    def _run_statistical_tests(
        self,
        episodes_a: List[Dict[str, Any]],
        episodes_b: List[Dict[str, Any]],
    ) -> List[StatisticalTest]:
        """Run statistical significance tests."""
        tests = []
        
        # Extract risk scores
        risks_a = [ep.get('risk_score', 0) for ep in episodes_a]
        risks_b = [ep.get('risk_score', 0) for ep in episodes_b]
        
        if len(risks_a) < 5 or len(risks_b) < 5:
            return tests
        
        # Paired t-test (if same episodes)
        if len(risks_a) == len(risks_b):
            try:
                t_stat, p_value = stats.ttest_rel(risks_a, risks_b)
                
                # Cohen's d effect size
                diff = np.array(risks_a) - np.array(risks_b)
                effect_size = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
                
                tests.append(StatisticalTest(
                    test_name="Paired t-test",
                    statistic=float(t_stat),
                    p_value=float(p_value),
                    is_significant=p_value < self.significance_level,
                    effect_size=float(effect_size),
                    interpretation=self._interpret_effect_size(effect_size),
                ))
            except Exception:
                pass
        
        # Mann-Whitney U test (non-parametric)
        try:
            u_stat, p_value = stats.mannwhitneyu(risks_a, risks_b, alternative='two-sided')
            
            # Rank-biserial correlation as effect size
            n1, n2 = len(risks_a), len(risks_b)
            effect_size = 1 - (2 * u_stat) / (n1 * n2)
            
            tests.append(StatisticalTest(
                test_name="Mann-Whitney U",
                statistic=float(u_stat),
                p_value=float(p_value),
                is_significant=p_value < self.significance_level,
                effect_size=float(effect_size),
                interpretation=self._interpret_effect_size(effect_size),
            ))
        except Exception:
            pass
        
        # Wilcoxon signed-rank test
        if len(risks_a) == len(risks_b):
            try:
                w_stat, p_value = stats.wilcoxon(risks_a, risks_b)
                
                tests.append(StatisticalTest(
                    test_name="Wilcoxon signed-rank",
                    statistic=float(w_stat),
                    p_value=float(p_value),
                    is_significant=p_value < self.significance_level,
                    effect_size=0.0,
                    interpretation="Non-parametric paired comparison",
                ))
            except Exception:
                pass
        
        return tests
    
    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "negligible effect"
        elif abs_effect < 0.5:
            return "small effect"
        elif abs_effect < 0.8:
            return "medium effect"
        else:
            return "large effect"
    
    def _identify_key_differences(self, result: ModelComparisonResult) -> List[str]:
        """Identify key differences between models."""
        differences = []
        
        # USS difference
        if abs(result.uss_delta) > 5:
            better = result.model_a_name if result.uss_delta > 0 else result.model_b_name
            differences.append(
                f"{better} has {abs(result.uss_delta):.1f} point higher USS score"
            )
        
        # Category differences
        for cc in result.category_comparisons:
            if abs(cc.delta) > 10:
                better = result.model_a_name if cc.delta > 0 else result.model_b_name
                differences.append(
                    f"{better} is {abs(cc.delta):.1f} points better in {cc.category}"
                )
        
        # Significant episode differences
        sig_episodes = [ec for ec in result.episode_comparisons if ec.significant_difference]
        if sig_episodes:
            differences.append(
                f"{len(sig_episodes)} episodes show significant differences (>0.2 risk delta)"
            )
        
        return differences
    
    def _identify_regressions(
        self, 
        result: ModelComparisonResult,
        model_a_name: str,
        model_b_name: str,
    ) -> List[str]:
        """Identify potential regressions (assuming model_a is newer)."""
        regressions = []
        
        # USS regression
        if result.uss_delta < -5:
            regressions.append(
                f"{model_a_name} shows {abs(result.uss_delta):.1f} point USS regression vs {model_b_name}"
            )
        
        # Category regressions
        for cc in result.category_comparisons:
            if cc.delta < -10:
                regressions.append(
                    f"{model_a_name} regressed {abs(cc.delta):.1f} points in {cc.category}"
                )
        
        # Episode regressions
        regression_episodes = [
            ec for ec in result.episode_comparisons
            if ec.delta > 0.2 and ec.winner == "model_b"  # Higher risk = worse
        ]
        
        if len(regression_episodes) > 3:
            regressions.append(
                f"{len(regression_episodes)} episodes show higher risk in {model_a_name}"
            )
        
        return regressions
    
    def _identify_improvements(self, result: ModelComparisonResult) -> List[str]:
        """Identify areas for improvement for the worse-performing model."""
        improvements = []
        
        worse_model = result.model_b_name if result.uss_delta > 0 else result.model_a_name
        
        # Find weakest categories for worse model
        worst_categories = sorted(
            result.category_comparisons,
            key=lambda x: x.model_b_score if result.uss_delta > 0 else x.model_a_score
        )[:2]
        
        for cc in worst_categories:
            score = cc.model_b_score if result.uss_delta > 0 else cc.model_a_score
            if score < 70:
                improvements.append(
                    f"{worse_model} should focus on improving {cc.category} (score: {score:.1f})"
                )
        
        return improvements


class VersionTracker:
    """
    Track model versions and evaluation history.
    """
    
    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.history_file = self.storage_path / "evaluation_history.json"
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load evaluation history from storage."""
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        return []
    
    def _save_history(self) -> None:
        """Save evaluation history to storage."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
    
    def record_evaluation(
        self,
        model_name: str,
        model_version: str,
        uss: UnifiedSafetyScore,
        episode_count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record an evaluation result."""
        entry = {
            "id": datetime.utcnow().strftime("%Y%m%d%H%M%S"),
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": model_name,
            "model_version": model_version,
            "uss_score": uss.score,
            "grade": uss.grade.value,
            "categories": uss.get_category_scores(),
            "episode_count": episode_count,
            "metadata": metadata or {},
        }
        
        self.history.append(entry)
        self._save_history()
        
        return entry["id"]
    
    def get_model_history(self, model_name: str) -> List[Dict[str, Any]]:
        """Get evaluation history for a specific model."""
        return [
            entry for entry in self.history
            if entry["model_name"] == model_name
        ]
    
    def get_latest(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get most recent evaluation for a model."""
        history = self.get_model_history(model_name)
        return history[-1] if history else None
    
    def detect_regression(
        self,
        model_name: str,
        new_uss: UnifiedSafetyScore,
        threshold: float = 5.0,
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect if new evaluation shows regression.
        
        Returns:
            (is_regression, message)
        """
        previous = self.get_latest(model_name)
        
        if not previous:
            return False, None
        
        delta = new_uss.score - previous["uss_score"]
        
        if delta < -threshold:
            return True, (
                f"Regression detected: USS dropped from {previous['uss_score']:.1f} "
                f"to {new_uss.score:.1f} ({delta:+.1f} points)"
            )
        
        return False, None
    
    def get_trend(
        self,
        model_name: str,
        last_n: int = 10,
    ) -> Dict[str, Any]:
        """Get trend analysis for a model."""
        history = self.get_model_history(model_name)[-last_n:]
        
        if len(history) < 2:
            return {"trend": "insufficient_data", "data_points": len(history)}
        
        scores = [h["uss_score"] for h in history]
        
        # Simple linear regression for trend
        x = np.arange(len(scores))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, scores)
        
        if slope > 0.5:
            trend = "improving"
        elif slope < -0.5:
            trend = "declining"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "slope": float(slope),
            "r_squared": float(r_value ** 2),
            "latest_score": scores[-1],
            "earliest_score": scores[0],
            "change": scores[-1] - scores[0],
            "data_points": len(scores),
        }


def compare_models(
    model_a_name: str,
    model_a_uss: UnifiedSafetyScore,
    model_a_episodes: List[Dict[str, Any]],
    model_b_name: str,
    model_b_uss: UnifiedSafetyScore,
    model_b_episodes: List[Dict[str, Any]],
) -> ModelComparisonResult:
    """Convenience function to compare two models."""
    comparator = ModelComparator()
    return comparator.compare(
        model_a_name, model_a_uss, model_a_episodes,
        model_b_name, model_b_uss, model_b_episodes,
    )
