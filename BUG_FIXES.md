# Bug Fixes for RiskLab AI Evaluation System

This document summarizes all bugs found and fixed in the codebase.

## Bug 1: Wrong Import Path for CouncilVerdict
**File:** `risklab/lab.py`, Line 747

**Problem:** The code imported `CouncilVerdict` from a non-existent module:
```python
from risklab.evaluation.council import CouncilVerdict  # WRONG
```

**Fix:** Changed to correct import path:
```python
from risklab.governance.council import CouncilVerdict  # CORRECT
```

## Bug 2: Wrong Constructor Arguments for CouncilVerdict
**File:** `risklab/lab.py`, Lines 748-755

**Problem:** The `CouncilVerdict` was being instantiated with incorrect attributes that don't exist in the class definition:
```python
verdict = CouncilVerdict(
    consensus="ERROR",                    # WRONG - attribute doesn't exist
    confidence=0.0,                       # WRONG - should be consensus_confidence
    reasoning=[...],                      # WRONG - should be combined_concerns
    evaluator_responses=[],               # WRONG - attribute doesn't exist
    disagreement_score=1.0,               # WRONG - should be DisagreementAnalysis object
    unanimous=False                       # WRONG - this is a method, not an attribute
)
```

**Fix:** Used correct constructor arguments based on the actual `CouncilVerdict` class:
```python
verdict = CouncilVerdict(
    council_id="error",
    consensus_risk_score=0.5,
    consensus_decision=DecisionOutcome.ESCALATE,
    consensus_confidence=0.0,
    combined_concerns=[f"Council evaluation failed: {str(e)}"],
)
```

## Bug 3: Wrong Argument to `full_analysis` Method
**File:** `risklab/lab.py`, Line 581

**Problem:** The `full_analysis` method was called with `prompt` as the second argument, but it expects a `Domain` enum:
```python
rule_assessment = self.rule_based_analyzer.full_analysis(result.text, prompt)  # WRONG
```

**Fix:** Pass the correct `domain` parameter from the episode context:
```python
rule_assessment = self.rule_based_analyzer.full_analysis(
    result.text, 
    domain=episode.context.domain
)
```

## Bug 4: Deprecated `copy()` Method on Pydantic Model
**File:** `risklab/lab.py`, Line 643

**Problem:** Using deprecated `.copy()` method on Pydantic BaseModel:
```python
combined_metrics = metrics_by_framing[framing].copy()  # May fail in Pydantic v2
```

**Fix:** Use compatibility pattern supporting both Pydantic v1 and v2:
```python
base_metrics = metrics_by_framing[framing]
combined_metrics = base_metrics.model_copy() if hasattr(base_metrics, 'model_copy') else base_metrics.copy()
```

## Bug 5: Non-existent `risk_level` Attribute
**File:** `risklab/lab.py`, Line 494

**Problem:** Code references `conditioned.risk_level` which doesn't exist in `RiskConditionedMetrics`:
```python
"risk_level": conditioned.risk_level if hasattr(conditioned, 'risk_level') else "unknown",
```

**Fix:** Added a helper method and compute risk level from score:
```python
def _compute_risk_level_from_score(self, score: float) -> str:
    if score < 0.2:
        return "negligible"
    elif score < 0.4:
        return "low"
    elif score < 0.6:
        return "moderate"
    elif score < 0.8:
        return "high"
    else:
        return "critical"

# Usage:
"risk_level": self._compute_risk_level_from_score(conditioned.aggregate_risk_score),
```

## Bug 6: Confusing Boolean Logic for `gate_passed`
**File:** `risklab/pipeline/executor.py`, Line 219

**Problem:** Complex and incorrect boolean logic:
```python
gate_passed=None if output.blocked is False else (not output.blocked if hasattr(output, 'blocked') else None),
```

This logic is backwards - when `blocked=False`, `gate_passed` should be `True`, not `None`.

**Fix:** Simplified and corrected the logic:
```python
gate_passed_value = None
if hasattr(output, 'blocked'):
    gate_passed_value = not output.blocked

execution = NodeExecution(
    ...
    gate_passed=gate_passed_value,
    ...
)
```

## Bug 7: Missing `get_critical_signals` Method
**File:** `risklab/measurement/signals.py`

**Problem:** The `ManipulationSignals` class was missing the `get_critical_signals()` method that was being called in `comparators.py`:
```python
# In comparators.py:
critical_signals = analysis.signals.get_critical_signals()  # Method doesn't exist!
```

**Fix:** Added the missing method to `ManipulationSignals` class:
```python
def get_critical_signals(self) -> List[SignalResult]:
    """Return all signals with 'high' or 'critical' severity."""
    signals = [
        self.intent_divergence,
        self.concealed_capability,
        self.framing_sensitivity,
        self.oversight_gap,
        self.sycophancy_index,
        self.selective_disclosure,
        self.strategic_hedging,
        self.authority_compliance,
        *self.custom_signals.values(),
    ]
    critical_severities = {"high", "critical"}
    return [s for s in signals if s is not None and s.severity in critical_severities]
```

## Bug 8: Complex `framing_sensitivity` Handling
**File:** `risklab/lab.py`, Line 733

**Problem:** Overly complex one-liner that could fail with different input types:
```python
cross_analysis.framing_sensitivity_score = framing_sensitivity.value if hasattr(framing_sensitivity, 'value') else framing_sensitivity if framing_sensitivity else 0.0
```

**Fix:** Clearer, more robust handling:
```python
if framing_sensitivity is not None:
    if hasattr(framing_sensitivity, 'value'):
        cross_analysis.framing_sensitivity_score = float(framing_sensitivity.value)
    elif isinstance(framing_sensitivity, (int, float)):
        cross_analysis.framing_sensitivity_score = float(framing_sensitivity)
    else:
        cross_analysis.framing_sensitivity_score = 0.0
else:
    cross_analysis.framing_sensitivity_score = 0.0
```

---

## Recommendations for Future Improvements

1. **Add Type Hints Everywhere**: Many functions lack proper type hints, which would have caught several of these bugs earlier.

2. **Add Unit Tests**: The project needs comprehensive unit tests for all modules to catch bugs like these before deployment.

3. **Use Stricter Pydantic Validation**: Enable strict mode in Pydantic to catch attribute errors at runtime.

4. **Add Integration Tests**: Test the full pipeline flow end-to-end to catch import errors and method call mismatches.

5. **Use Static Analysis Tools**: Tools like `mypy`, `pyright`, or `pylint` would catch many of these issues.

6. **Standardize Line Endings**: The codebase had mixed line endings (CRLF/LF) which was fixed during this review.
