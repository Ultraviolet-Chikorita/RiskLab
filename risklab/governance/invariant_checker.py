"""
Design Invariants Checker for RiskLab.

Enforces all hard rules from the vision document:
1. No silent type coercion
2. No float without semantics
3. No risk score without provenance
4. No LLM as sole authority
5. Every decision must be explainable post-hoc
6. Visualization never invents data

Run this checker before accepting any evaluation result.
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import warnings


class InvariantType(str, Enum):
    """Types of design invariants."""
    TYPE_SAFETY = "type_safety"
    PROVENANCE = "provenance"
    MULTI_EVALUATOR = "multi_evaluator"
    EXPLAINABILITY = "explainability"
    DATA_INTEGRITY = "data_integrity"


class InvariantViolation(BaseModel):
    """A single invariant violation."""
    invariant: InvariantType
    severity: str  # error, warning, info
    message: str
    location: str = ""
    evidence: List[str] = Field(default_factory=list)
    suggested_fix: str = ""


class InvariantCheckResult(BaseModel):
    """Result of invariant checking."""
    passed: bool = True
    checked_at: datetime = Field(default_factory=datetime.utcnow)
    violations: List[InvariantViolation] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Summary counts
    error_count: int = 0
    warning_count: int = 0
    
    def add_violation(self, violation: InvariantViolation) -> None:
        self.violations.append(violation)
        if violation.severity == "error":
            self.error_count += 1
            self.passed = False
        elif violation.severity == "warning":
            self.warning_count += 1


class InvariantChecker:
    """
    Checks all design invariants before accepting evaluation results.
    
    Usage:
        checker = InvariantChecker()
        result = checker.check_all(episode_data, decision, audit_trail)
        if not result.passed:
            # Handle violations
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Args:
            strict_mode: If True, warnings become errors
        """
        self.strict_mode = strict_mode
    
    def check_all(
        self,
        episode_data: Dict[str, Any],
        decision: Optional[Dict[str, Any]] = None,
        audit_trail: Optional[Dict[str, Any]] = None,
    ) -> InvariantCheckResult:
        """
        Run all invariant checks.
        
        Returns comprehensive result with all violations found.
        """
        result = InvariantCheckResult()
        
        # 1. Check type safety
        self._check_type_safety(episode_data, result)
        
        # 2. Check provenance
        self._check_provenance(episode_data, result)
        
        # 3. Check multi-evaluator requirement
        self._check_multi_evaluator(episode_data, decision, result)
        
        # 4. Check explainability
        self._check_explainability(decision, audit_trail, result)
        
        # 5. Check data integrity
        self._check_data_integrity(episode_data, result)
        
        return result
    
    def _check_type_safety(
        self,
        episode_data: Dict[str, Any],
        result: InvariantCheckResult,
    ) -> None:
        """
        Check: No silent type coercion, No float without semantics
        """
        # Check metrics have proper structure
        metrics = episode_data.get("metrics_by_framing", {})
        
        for framing, framing_metrics in metrics.items():
            if isinstance(framing_metrics, dict):
                metrics_data = framing_metrics.get("metrics", framing_metrics)
                
                for metric_name, metric_value in metrics_data.items():
                    # Check for raw floats
                    if isinstance(metric_value, (int, float)):
                        result.add_violation(InvariantViolation(
                            invariant=InvariantType.TYPE_SAFETY,
                            severity="warning" if not self.strict_mode else "error",
                            message=f"Raw float value for metric '{metric_name}'",
                            location=f"metrics_by_framing.{framing}.{metric_name}",
                            evidence=[f"Value: {metric_value}"],
                            suggested_fix="Use MetricResult or SemanticScore with provenance",
                        ))
                    
                    # Check for proper MetricResult structure
                    elif isinstance(metric_value, dict):
                        if "value" in metric_value and "provenance" not in metric_value:
                            # Has value but no provenance - partial compliance
                            if "confidence" not in metric_value and "evidence" not in metric_value:
                                result.add_violation(InvariantViolation(
                                    invariant=InvariantType.TYPE_SAFETY,
                                    severity="info",
                                    message=f"Metric '{metric_name}' missing confidence/evidence",
                                    location=f"metrics_by_framing.{framing}.{metric_name}",
                                    suggested_fix="Include confidence and evidence for full auditability",
                                ))
    
    def _check_provenance(
        self,
        episode_data: Dict[str, Any],
        result: InvariantCheckResult,
    ) -> None:
        """
        Check: No risk score without provenance
        """
        # Check decision provenance
        decision = episode_data.get("decision", {})
        if decision:
            if "provenance" not in decision:
                result.add_violation(InvariantViolation(
                    invariant=InvariantType.PROVENANCE,
                    severity="error",
                    message="Decision missing provenance",
                    location="decision.provenance",
                    suggested_fix="Add ProvenanceRecord to decision",
                ))
            
            provenance = decision.get("provenance", {})
            if provenance:
                # Check provenance completeness
                required_fields = ["evaluator_type", "evaluator_id", "timestamp"]
                for field in required_fields:
                    if field not in provenance:
                        result.add_violation(InvariantViolation(
                            invariant=InvariantType.PROVENANCE,
                            severity="warning",
                            message=f"Provenance missing '{field}'",
                            location=f"decision.provenance.{field}",
                        ))
        
        # Check conditioning provenance
        conditioning = episode_data.get("conditioning", {})
        if conditioning and "conditioned_metrics" in conditioning:
            for metric_name, metric_data in conditioning.get("conditioned_metrics", {}).items():
                if isinstance(metric_data, dict):
                    if "risk_adjusted_value" in metric_data and "provenance" not in metric_data:
                        result.add_violation(InvariantViolation(
                            invariant=InvariantType.PROVENANCE,
                            severity="warning",
                            message=f"Conditioned metric '{metric_name}' missing provenance",
                            location=f"conditioning.conditioned_metrics.{metric_name}",
                        ))
        
        # Check aggregated scores have parent references
        signals = episode_data.get("signals", {})
        if isinstance(signals, dict):
            for signal_name, signal_data in signals.get("signals", {}).items():
                if isinstance(signal_data, dict):
                    if "derived_from" not in signal_data and "contributing_metrics" not in signal_data:
                        result.add_violation(InvariantViolation(
                            invariant=InvariantType.PROVENANCE,
                            severity="info",
                            message=f"Signal '{signal_name}' missing derivation info",
                            location=f"signals.signals.{signal_name}",
                            suggested_fix="Include derived_from or contributing_metrics",
                        ))
    
    def _check_multi_evaluator(
        self,
        episode_data: Dict[str, Any],
        decision: Optional[Dict[str, Any]],
        result: InvariantCheckResult,
    ) -> None:
        """
        Check: No LLM as sole authority
        """
        decision_data = decision or episode_data.get("decision", {})
        provenance = decision_data.get("provenance", {})
        
        if provenance:
            evaluator_type = provenance.get("evaluator_type", "")
            
            # Check if LLM is sole decision maker
            if evaluator_type == "llm_judge":
                # Check if council was used
                council = episode_data.get("council", {})
                if not council:
                    result.add_violation(InvariantViolation(
                        invariant=InvariantType.MULTI_EVALUATOR,
                        severity="error",
                        message="LLM as sole authority without council",
                        location="decision.provenance.evaluator_type",
                        evidence=[f"Evaluator type: {evaluator_type}"],
                        suggested_fix="Use evaluation council with multiple judges",
                    ))
        
        # Check evaluator diversity in metrics
        metrics = episode_data.get("metrics_by_framing", {})
        evaluator_types = set()
        
        for framing, framing_data in metrics.items():
            if isinstance(framing_data, dict):
                for evaluator in framing_data.get("evaluators_used", []):
                    if ":" in str(evaluator):
                        eval_type = str(evaluator).split(":")[0]
                        evaluator_types.add(eval_type)
        
        if len(evaluator_types) == 1 and "llm_judge" in evaluator_types:
            result.add_violation(InvariantViolation(
                invariant=InvariantType.MULTI_EVALUATOR,
                severity="warning",
                message="All metrics from single evaluator type (LLM)",
                location="metrics_by_framing",
                evidence=[f"Evaluator types: {evaluator_types}"],
                suggested_fix="Include rule-based or ML evaluators for diversity",
            ))
        
        # Check council has multiple judges
        council = episode_data.get("council", {})
        if council:
            judge_reports = council.get("judge_reports", [])
            if len(judge_reports) < 2:
                result.add_violation(InvariantViolation(
                    invariant=InvariantType.MULTI_EVALUATOR,
                    severity="warning",
                    message=f"Council has only {len(judge_reports)} judge(s)",
                    location="council.judge_reports",
                    suggested_fix="Council should have at least 2-3 independent judges",
                ))
    
    def _check_explainability(
        self,
        decision: Optional[Dict[str, Any]],
        audit_trail: Optional[Dict[str, Any]],
        result: InvariantCheckResult,
    ) -> None:
        """
        Check: Every decision must be explainable post-hoc
        """
        decision_data = decision or {}
        
        if decision_data:
            # Check decision has explanation
            if "primary_factors" not in decision_data and "reasoning" not in decision_data:
                result.add_violation(InvariantViolation(
                    invariant=InvariantType.EXPLAINABILITY,
                    severity="warning",
                    message="Decision missing explanation factors",
                    location="decision.primary_factors",
                    suggested_fix="Include primary_factors or reasoning in decision",
                ))
        
        # Check audit trail completeness
        if audit_trail:
            # Check decision chain is present
            if "decision_chain" not in audit_trail or not audit_trail["decision_chain"]:
                result.add_violation(InvariantViolation(
                    invariant=InvariantType.EXPLAINABILITY,
                    severity="warning",
                    message="Audit trail missing decision chain",
                    location="audit_trail.decision_chain",
                    suggested_fix="Record each decision step in decision_chain",
                ))
            
            # Check scores have audit entries
            scores = audit_trail.get("scores", {})
            for score_name, score_data in scores.items():
                if isinstance(score_data, dict):
                    if "provenance" not in score_data:
                        result.add_violation(InvariantViolation(
                            invariant=InvariantType.EXPLAINABILITY,
                            severity="info",
                            message=f"Audit trail score '{score_name}' missing provenance",
                            location=f"audit_trail.scores.{score_name}",
                        ))
    
    def _check_data_integrity(
        self,
        episode_data: Dict[str, Any],
        result: InvariantCheckResult,
    ) -> None:
        """
        Check: Visualization never invents data (data completeness)
        """
        # Check required fields are present
        required_fields = [
            ("responses", "No responses recorded"),
            ("metrics_by_framing", "No metrics recorded"),
            ("context", "No context recorded"),
        ]
        
        for field, message in required_fields:
            if field not in episode_data or not episode_data[field]:
                result.add_violation(InvariantViolation(
                    invariant=InvariantType.DATA_INTEGRITY,
                    severity="error",
                    message=message,
                    location=field,
                    suggested_fix=f"Ensure {field} is populated before finalizing",
                ))
        
        # Check responses exist for each framing in metrics
        responses = episode_data.get("responses", {})
        metrics = episode_data.get("metrics_by_framing", {})
        
        for framing in metrics.keys():
            if framing not in responses:
                result.add_violation(InvariantViolation(
                    invariant=InvariantType.DATA_INTEGRITY,
                    severity="warning",
                    message=f"Metrics exist for framing '{framing}' but no response",
                    location=f"responses.{framing}",
                    suggested_fix="Ensure response is recorded for each framing",
                ))
        
        # Check high-stakes episodes have council
        context = episode_data.get("context", {})
        stakes = context.get("stakes_level", "medium")
        
        if stakes in ["high", "critical"]:
            if "council" not in episode_data or not episode_data.get("council"):
                result.add_violation(InvariantViolation(
                    invariant=InvariantType.DATA_INTEGRITY,
                    severity="warning" if stakes == "high" else "error",
                    message=f"{stakes.capitalize()}-stakes episode missing council evaluation",
                    location="council",
                    suggested_fix="High/critical stakes episodes require council evaluation",
                ))


def check_invariants(
    episode_data: Dict[str, Any],
    strict: bool = False,
) -> InvariantCheckResult:
    """
    Convenience function to check all invariants on episode data.
    
    Args:
        episode_data: The episode record to check
        strict: If True, warnings become errors
    
    Returns:
        InvariantCheckResult with all violations found
    """
    checker = InvariantChecker(strict_mode=strict)
    return checker.check_all(
        episode_data,
        decision=episode_data.get("decision"),
        audit_trail=episode_data.get("audit_trail"),
    )


def enforce_invariants(
    episode_data: Dict[str, Any],
    strict: bool = True,
) -> None:
    """
    Enforce invariants - raises exception if violations found.
    
    Args:
        episode_data: The episode record to check
        strict: If True, treat warnings as errors too
    
    Raises:
        ValueError: If any violations are found
    """
    result = check_invariants(episode_data, strict=strict)
    
    if not result.passed:
        violations_text = "\n".join([
            f"  - [{v.severity.upper()}] {v.message} at {v.location}"
            for v in result.violations
            if v.severity == "error" or strict
        ])
        raise ValueError(
            f"Design invariant violations found:\n{violations_text}\n"
            f"Total: {result.error_count} errors, {result.warning_count} warnings"
        )


def validate_before_export(
    episodes: List[Dict[str, Any]],
    strict: bool = False,
) -> Tuple[bool, List[InvariantCheckResult]]:
    """
    Validate all episodes before export.
    
    Args:
        episodes: List of episode records
        strict: If True, warnings become errors
    
    Returns:
        (all_passed, list of results per episode)
    """
    results = []
    all_passed = True
    
    for episode in episodes:
        result = check_invariants(episode, strict=strict)
        results.append(result)
        if not result.passed:
            all_passed = False
    
    return all_passed, results


# Report generation
def generate_invariant_report(results: List[InvariantCheckResult]) -> str:
    """Generate human-readable invariant check report."""
    lines = [
        "# Design Invariant Check Report",
        f"Checked at: {datetime.utcnow().isoformat()}",
        "",
    ]
    
    total_errors = sum(r.error_count for r in results)
    total_warnings = sum(r.warning_count for r in results)
    passed = sum(1 for r in results if r.passed)
    
    lines.extend([
        f"## Summary",
        f"- Episodes checked: {len(results)}",
        f"- Passed: {passed}",
        f"- Failed: {len(results) - passed}",
        f"- Total errors: {total_errors}",
        f"- Total warnings: {total_warnings}",
        "",
    ])
    
    if total_errors > 0 or total_warnings > 0:
        lines.append("## Violations by Type")
        
        by_type: Dict[str, List[InvariantViolation]] = {}
        for r in results:
            for v in r.violations:
                if v.invariant.value not in by_type:
                    by_type[v.invariant.value] = []
                by_type[v.invariant.value].append(v)
        
        for inv_type, violations in by_type.items():
            lines.append(f"\n### {inv_type.replace('_', ' ').title()}")
            for v in violations[:10]:  # Limit to 10 per type
                lines.append(f"- [{v.severity}] {v.message}")
                if v.suggested_fix:
                    lines.append(f"  Fix: {v.suggested_fix}")
        
        if sum(len(v) for v in by_type.values()) > 10 * len(by_type):
            lines.append("\n(Additional violations truncated)")
    
    return "\n".join(lines)
