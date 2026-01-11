"""
Statistical Anomaly Detection for Evaluator Bias.

Compares LLM-based evaluations against rule-based signals to detect
systematic biases in LLM evaluators. When LLM judgments diverge from
what rule-based analysis would predict, that's a signal for investigation.

Key metrics:
1. Systematic divergence: Does the LLM consistently over/under-estimate risk?
2. Conditional bias: Does divergence correlate with scenario type?
3. Outlier detection: Which individual evaluations are anomalous?
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

import numpy as np

from risklab.scenarios.institutional import NormDomain
from risklab.scenarios.context import Domain, StakesLevel


class AnomalyType(str, Enum):
    """Types of detected anomalies."""
    SYSTEMATIC_OVERESTIMATE = "systematic_overestimate"
    SYSTEMATIC_UNDERESTIMATE = "systematic_underestimate"
    DOMAIN_CONDITIONAL_BIAS = "domain_conditional_bias"
    STAKES_CONDITIONAL_BIAS = "stakes_conditional_bias"
    NORM_CONDITIONAL_BIAS = "norm_conditional_bias"
    INDIVIDUAL_OUTLIER = "individual_outlier"
    INCONSISTENT_REASONING = "inconsistent_reasoning"


@dataclass
class EvaluationRecord:
    """Record of a single evaluation for anomaly analysis."""
    
    # Identifiers
    scenario_name: str
    response_hash: str  # For deduplication
    
    # Context
    domain: Domain = Domain.GENERAL
    stakes_level: StakesLevel = StakesLevel.MEDIUM
    norm_domain: Optional[NormDomain] = None
    
    # Rule-based signals
    rule_risk_score: float = 0.0
    rule_sycophancy: float = 0.0
    rule_omission: float = 0.0
    rule_hedging: float = 0.0
    rule_norm_bias: Optional[float] = None  # -1 to +1
    
    # LLM signals
    llm_risk_score: float = 0.0
    llm_sycophancy: float = 0.0
    llm_omission: float = 0.0
    llm_manipulation: float = 0.0
    llm_norm_bias: Optional[float] = None
    llm_confidence: float = 0.8
    
    # Metadata
    evaluator_model: str = ""
    timestamp: Optional[str] = None


@dataclass
class AnomalySignal:
    """A detected anomaly in evaluation patterns."""
    
    anomaly_type: AnomalyType
    severity: float  # 0-1
    description: str
    
    # Evidence
    metric: str  # Which metric showed the anomaly
    expected_value: float
    actual_value: float
    divergence: float
    
    # Context
    affected_scenarios: List[str] = field(default_factory=list)
    affected_domain: Optional[Domain] = None
    sample_size: int = 0
    
    # Statistical details
    p_value: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.anomaly_type.value,
            "severity": self.severity,
            "description": self.description,
            "metric": self.metric,
            "expected": self.expected_value,
            "actual": self.actual_value,
            "divergence": self.divergence,
            "sample_size": self.sample_size,
            "p_value": self.p_value,
        }


@dataclass
class AnomalyReport:
    """Complete anomaly detection report."""
    
    signals: List[AnomalySignal] = field(default_factory=list)
    
    # Summary statistics
    total_evaluations: int = 0
    systematic_bias: float = 0.0  # Overall LLM bias direction
    bias_variance: float = 0.0
    
    # Per-metric summaries
    metric_summaries: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Recommendations
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def add_signal(self, signal: AnomalySignal) -> None:
        self.signals.append(signal)
        
        # Update warnings based on severity
        if signal.severity > 0.7:
            self.warnings.append(f"HIGH: {signal.description}")
        elif signal.severity > 0.5:
            self.warnings.append(f"MEDIUM: {signal.description}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_evaluations": self.total_evaluations,
            "systematic_bias": self.systematic_bias,
            "n_anomalies": len(self.signals),
            "signals": [s.to_dict() for s in self.signals],
            "metric_summaries": self.metric_summaries,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }


class StatisticalAnomalyDetector:
    """
    Detects statistical anomalies in LLM evaluations.
    
    Core approach:
    1. Collect paired (rule_based, llm) evaluations
    2. Compute divergence statistics
    3. Test for systematic biases
    4. Identify outliers and conditional patterns
    """
    
    def __init__(
        self,
        min_samples: int = 10,
        outlier_threshold: float = 2.5,  # Standard deviations
        divergence_threshold: float = 0.2,  # Min divergence to flag
    ):
        self.min_samples = min_samples
        self.outlier_threshold = outlier_threshold
        self.divergence_threshold = divergence_threshold
        
        self.records: List[EvaluationRecord] = []
    
    def add_record(self, record: EvaluationRecord) -> None:
        """Add an evaluation record for analysis."""
        self.records.append(record)
    
    def add_evaluation(
        self,
        scenario_name: str,
        rule_assessment: Dict[str, Any],
        llm_assessment: Dict[str, Any],
        domain: Domain = Domain.GENERAL,
        stakes_level: StakesLevel = StakesLevel.MEDIUM,
        norm_domain: Optional[NormDomain] = None,
        evaluator_model: str = "",
    ) -> None:
        """Add an evaluation from assessment dicts."""
        import hashlib
        
        # Create hash for deduplication
        content = f"{scenario_name}:{rule_assessment}:{llm_assessment}"
        response_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        
        record = EvaluationRecord(
            scenario_name=scenario_name,
            response_hash=response_hash,
            domain=domain,
            stakes_level=stakes_level,
            norm_domain=norm_domain,
            # Rule-based
            rule_risk_score=rule_assessment.get("aggregate_risk_score", 0),
            rule_sycophancy=rule_assessment.get("sycophancy", {}).get("score", 0),
            rule_omission=rule_assessment.get("safety_disclosure", {}).get("omission_score", 0),
            rule_hedging=rule_assessment.get("hedging", {}).get("score", 0),
            rule_norm_bias=rule_assessment.get("norm_alignment", {}).get("directional_bias"),
            # LLM-based
            llm_risk_score=llm_assessment.get("risk_score", 0),
            llm_sycophancy=llm_assessment.get("sycophancy_score", 0),
            llm_omission=llm_assessment.get("omission_score", 0),
            llm_manipulation=llm_assessment.get("manipulation_score", 0),
            llm_norm_bias=llm_assessment.get("directional_bias"),
            llm_confidence=llm_assessment.get("confidence", 0.8),
            evaluator_model=evaluator_model,
        )
        
        self.records.append(record)
    
    def analyze(self) -> AnomalyReport:
        """Run full anomaly detection analysis."""
        report = AnomalyReport(total_evaluations=len(self.records))
        
        if len(self.records) < self.min_samples:
            report.warnings.append(
                f"Insufficient samples ({len(self.records)} < {self.min_samples})"
            )
            return report
        
        # 1. Systematic bias detection
        self._detect_systematic_bias(report)
        
        # 2. Per-metric analysis
        self._analyze_per_metric(report)
        
        # 3. Domain-conditional bias
        self._detect_domain_conditional_bias(report)
        
        # 4. Stakes-conditional bias
        self._detect_stakes_conditional_bias(report)
        
        # 5. Norm-conditional bias
        self._detect_norm_conditional_bias(report)
        
        # 6. Individual outliers
        self._detect_outliers(report)
        
        # Generate recommendations
        self._generate_recommendations(report)
        
        return report
    
    def _detect_systematic_bias(self, report: AnomalyReport) -> None:
        """Detect overall systematic over/under-estimation."""
        rule_risks = [r.rule_risk_score for r in self.records]
        llm_risks = [r.llm_risk_score for r in self.records]
        
        divergences = [l - r for l, r in zip(llm_risks, rule_risks)]
        
        mean_divergence = np.mean(divergences)
        std_divergence = np.std(divergences)
        
        report.systematic_bias = mean_divergence
        report.bias_variance = std_divergence ** 2
        
        # Test if mean divergence is significantly different from 0
        if len(divergences) > 1:
            se = std_divergence / np.sqrt(len(divergences))
            if se > 0:
                t_stat = mean_divergence / se
                # Rough p-value approximation
                p_value = 2 * (1 - min(0.9999, abs(t_stat) / 10))
            else:
                t_stat = 0
                p_value = 1.0
        else:
            t_stat = 0
            p_value = 1.0
        
        if abs(mean_divergence) > self.divergence_threshold:
            anomaly_type = (
                AnomalyType.SYSTEMATIC_OVERESTIMATE if mean_divergence > 0
                else AnomalyType.SYSTEMATIC_UNDERESTIMATE
            )
            
            direction = "overestimates" if mean_divergence > 0 else "underestimates"
            
            report.add_signal(AnomalySignal(
                anomaly_type=anomaly_type,
                severity=min(1.0, abs(mean_divergence) * 2),
                description=f"LLM evaluator systematically {direction} risk by {abs(mean_divergence):.3f} on average",
                metric="risk_score",
                expected_value=float(np.mean(rule_risks)),
                actual_value=float(np.mean(llm_risks)),
                divergence=mean_divergence,
                sample_size=len(self.records),
                p_value=p_value,
            ))
    
    def _analyze_per_metric(self, report: AnomalyReport) -> None:
        """Analyze divergence per metric."""
        metrics = [
            ("sycophancy", "rule_sycophancy", "llm_sycophancy"),
            ("omission", "rule_omission", "llm_omission"),
        ]
        
        for metric_name, rule_attr, llm_attr in metrics:
            rule_values = [getattr(r, rule_attr) for r in self.records]
            llm_values = [getattr(r, llm_attr) for r in self.records]
            
            divergences = [l - r for l, r in zip(llm_values, rule_values)]
            
            report.metric_summaries[metric_name] = {
                "mean_rule": float(np.mean(rule_values)),
                "mean_llm": float(np.mean(llm_values)),
                "mean_divergence": float(np.mean(divergences)),
                "std_divergence": float(np.std(divergences)),
                "correlation": float(np.corrcoef(rule_values, llm_values)[0, 1]) if len(rule_values) > 1 else 0,
            }
    
    def _detect_domain_conditional_bias(self, report: AnomalyReport) -> None:
        """Detect if bias varies by domain."""
        domain_divergences: Dict[Domain, List[float]] = defaultdict(list)
        
        for r in self.records:
            divergence = r.llm_risk_score - r.rule_risk_score
            domain_divergences[r.domain].append(divergence)
        
        # Compare domains
        domain_means = {}
        for domain, divs in domain_divergences.items():
            if len(divs) >= 3:
                domain_means[domain] = np.mean(divs)
        
        if len(domain_means) >= 2:
            overall_mean = np.mean(list(domain_means.values()))
            
            for domain, mean_div in domain_means.items():
                if abs(mean_div - overall_mean) > self.divergence_threshold:
                    direction = "overestimates" if mean_div > overall_mean else "underestimates"
                    
                    report.add_signal(AnomalySignal(
                        anomaly_type=AnomalyType.DOMAIN_CONDITIONAL_BIAS,
                        severity=min(1.0, abs(mean_div - overall_mean) * 2),
                        description=f"LLM {direction} risk in {domain.value} domain compared to other domains",
                        metric="risk_score",
                        expected_value=overall_mean,
                        actual_value=mean_div,
                        divergence=mean_div - overall_mean,
                        affected_domain=domain,
                        sample_size=len(domain_divergences[domain]),
                    ))
    
    def _detect_stakes_conditional_bias(self, report: AnomalyReport) -> None:
        """Detect if bias varies by stakes level."""
        stakes_divergences: Dict[StakesLevel, List[float]] = defaultdict(list)
        
        for r in self.records:
            divergence = r.llm_risk_score - r.rule_risk_score
            stakes_divergences[r.stakes_level].append(divergence)
        
        # Compare stakes levels
        stakes_means = {}
        for stakes, divs in stakes_divergences.items():
            if len(divs) >= 3:
                stakes_means[stakes] = np.mean(divs)
        
        if len(stakes_means) >= 2:
            overall_mean = np.mean(list(stakes_means.values()))
            
            for stakes, mean_div in stakes_means.items():
                if abs(mean_div - overall_mean) > self.divergence_threshold:
                    direction = "overestimates" if mean_div > overall_mean else "underestimates"
                    
                    report.add_signal(AnomalySignal(
                        anomaly_type=AnomalyType.STAKES_CONDITIONAL_BIAS,
                        severity=min(1.0, abs(mean_div - overall_mean) * 2),
                        description=f"LLM {direction} risk in {stakes.value}-stakes scenarios",
                        metric="risk_score",
                        expected_value=overall_mean,
                        actual_value=mean_div,
                        divergence=mean_div - overall_mean,
                        sample_size=len(stakes_divergences[stakes]),
                    ))
    
    def _detect_norm_conditional_bias(self, report: AnomalyReport) -> None:
        """Detect if LLM has bias toward particular institutional norms."""
        # Compare norm bias assessments where both are available
        norm_comparisons = [
            (r.rule_norm_bias, r.llm_norm_bias, r.norm_domain, r.scenario_name)
            for r in self.records
            if r.rule_norm_bias is not None and r.llm_norm_bias is not None
        ]
        
        if len(norm_comparisons) < 3:
            return
        
        rule_biases, llm_biases, domains, scenarios = zip(*norm_comparisons)
        
        divergences = [l - r for l, r in zip(llm_biases, rule_biases)]
        mean_divergence = np.mean(divergences)
        
        if abs(mean_divergence) > self.divergence_threshold:
            # Determine direction
            if mean_divergence > 0:
                direction = "toward Norm B (e.g., EU/stakeholder)"
            else:
                direction = "toward Norm A (e.g., US/shareholder)"
            
            report.add_signal(AnomalySignal(
                anomaly_type=AnomalyType.NORM_CONDITIONAL_BIAS,
                severity=min(1.0, abs(mean_divergence)),
                description=f"LLM evaluator shows systematic bias {direction} compared to rule-based assessment",
                metric="directional_bias",
                expected_value=float(np.mean(rule_biases)),
                actual_value=float(np.mean(llm_biases)),
                divergence=mean_divergence,
                sample_size=len(norm_comparisons),
            ))
    
    def _detect_outliers(self, report: AnomalyReport) -> None:
        """Detect individual evaluation outliers."""
        divergences = [r.llm_risk_score - r.rule_risk_score for r in self.records]
        
        mean_div = np.mean(divergences)
        std_div = np.std(divergences)
        
        if std_div == 0:
            return
        
        outlier_scenarios = []
        
        for r, div in zip(self.records, divergences):
            z_score = (div - mean_div) / std_div
            
            if abs(z_score) > self.outlier_threshold:
                outlier_scenarios.append(r.scenario_name)
        
        if outlier_scenarios:
            report.add_signal(AnomalySignal(
                anomaly_type=AnomalyType.INDIVIDUAL_OUTLIER,
                severity=min(1.0, len(outlier_scenarios) / 10),
                description=f"{len(outlier_scenarios)} evaluations are statistical outliers (>{self.outlier_threshold} std from mean)",
                metric="risk_score",
                expected_value=mean_div,
                actual_value=0,  # N/A
                divergence=0,
                affected_scenarios=outlier_scenarios[:10],
                sample_size=len(self.records),
            ))
    
    def _generate_recommendations(self, report: AnomalyReport) -> None:
        """Generate actionable recommendations based on findings."""
        for signal in report.signals:
            if signal.anomaly_type == AnomalyType.SYSTEMATIC_OVERESTIMATE:
                report.recommendations.append(
                    "Consider adjusting LLM evaluator temperature or prompting to reduce false positives"
                )
            elif signal.anomaly_type == AnomalyType.SYSTEMATIC_UNDERESTIMATE:
                report.recommendations.append(
                    "LLM may be missing risks - consider stricter evaluation rubric"
                )
            elif signal.anomaly_type == AnomalyType.DOMAIN_CONDITIONAL_BIAS:
                report.recommendations.append(
                    f"Use domain-specific calibration for {signal.affected_domain.value if signal.affected_domain else 'affected'} domain"
                )
            elif signal.anomaly_type == AnomalyType.NORM_CONDITIONAL_BIAS:
                report.recommendations.append(
                    "LLM evaluator may share institutional biases with evaluated model - use cross-family validation"
                )
            elif signal.anomaly_type == AnomalyType.INDIVIDUAL_OUTLIER:
                report.recommendations.append(
                    "Review flagged outlier scenarios for potential labeling errors or edge cases"
                )
        
        # General recommendations based on report state
        if report.systematic_bias != 0:
            report.recommendations.append(
                f"Apply bias correction of {-report.systematic_bias:.3f} to LLM risk scores"
            )
        
        if not report.signals:
            report.recommendations.append(
                "No significant anomalies detected - LLM and rule-based evaluators are well-calibrated"
            )


def compute_divergence_summary(
    rule_assessments: List[Dict[str, Any]],
    llm_assessments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Quick summary of rule vs LLM divergence without full anomaly detection.
    """
    if not rule_assessments or not llm_assessments:
        return {"error": "No assessments provided"}
    
    n = min(len(rule_assessments), len(llm_assessments))
    
    rule_risks = [r.get("aggregate_risk_score", r.get("risk_score", 0)) for r in rule_assessments[:n]]
    llm_risks = [l.get("risk_score", 0) for l in llm_assessments[:n]]
    
    divergences = [l - r for l, r in zip(llm_risks, rule_risks)]
    
    return {
        "n_comparisons": n,
        "mean_rule_risk": float(np.mean(rule_risks)),
        "mean_llm_risk": float(np.mean(llm_risks)),
        "mean_divergence": float(np.mean(divergences)),
        "std_divergence": float(np.std(divergences)),
        "max_divergence": float(max(divergences, key=abs)),
        "correlation": float(np.corrcoef(rule_risks, llm_risks)[0, 1]) if n > 1 else 0,
        "llm_bias_direction": "overestimates" if np.mean(divergences) > 0 else "underestimates",
    }
