"""
Common utilities and base classes for the RiskLab package.

Consolidates duplicated patterns across the codebase.
"""

from typing import Optional, Dict, Any, TypeVar, Generic
from enum import Enum
from pathlib import Path
import io
import base64
import json
import re


# === Severity/Risk Level Utilities ===

class RiskLevel(str, Enum):
    """Unified risk/severity levels used across the package."""
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


def compute_risk_level(score: float) -> RiskLevel:
    """
    Compute risk level from a 0-1 score.
    
    Used by SignalResult.compute_severity, BiasRiskLevel, threshold logic, etc.
    """
    if score < 0.2:
        return RiskLevel.NEGLIGIBLE
    elif score < 0.4:
        return RiskLevel.LOW
    elif score < 0.6:
        return RiskLevel.MODERATE
    elif score < 0.8:
        return RiskLevel.HIGH
    else:
        return RiskLevel.CRITICAL


def risk_level_to_severity(level: RiskLevel) -> str:
    """Convert RiskLevel to legacy severity string for backward compatibility."""
    mapping = {
        RiskLevel.NEGLIGIBLE: "low",
        RiskLevel.LOW: "low",
        RiskLevel.MODERATE: "medium",
        RiskLevel.HIGH: "high",
        RiskLevel.CRITICAL: "critical",
    }
    return mapping.get(level, "low")


def severity_to_risk_level(severity: str) -> RiskLevel:
    """Convert legacy severity string to RiskLevel."""
    mapping = {
        "low": RiskLevel.LOW,
        "medium": RiskLevel.MODERATE,
        "high": RiskLevel.HIGH,
        "critical": RiskLevel.CRITICAL,
    }
    return mapping.get(severity.lower(), RiskLevel.LOW)


# === JSON Parsing Utilities ===

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from text that may contain other content.
    
    Handles LLM outputs that wrap JSON in markdown or include preamble.
    """
    if not text:
        return None
    
    # Try to find JSON object
    json_start = text.find('{')
    json_end = text.rfind('}') + 1
    
    if json_start >= 0 and json_end > json_start:
        json_str = text[json_start:json_end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON in code blocks
    code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    return None


def safe_get_float(data: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """Safely get a float value from a dict, handling various input types."""
    value = data.get(key, default)
    if isinstance(value, dict):
        value = value.get("score", value.get("value", default))
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# === Matplotlib Rendering Utilities ===

def render_figure_to_output(
    fig,
    output_path: Optional[Path] = None,
    dpi: int = 150,
    close_after: bool = True,
) -> Optional[str]:
    """
    Render a matplotlib figure to file or base64 string.
    
    Args:
        fig: matplotlib Figure object
        output_path: Path to save the figure. If None, returns base64.
        dpi: Resolution for output
        close_after: Whether to close the figure after rendering
    
    Returns:
        None if saved to file, base64 string otherwise
    """
    import matplotlib.pyplot as plt
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        if close_after:
            plt.close(fig)
        return None
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        if close_after:
            plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')


# === Enum Mapping Utilities ===

T = TypeVar('T')


class EnumMapper(Generic[T]):
    """
    Generic mapper between enum values and attribute names.
    
    Consolidates the repeated get_metric/set_metric patterns.
    """
    
    def __init__(self, mapping: Dict[T, str]):
        """
        Args:
            mapping: Dict mapping enum values to attribute names
        """
        self.enum_to_attr = mapping
        self.attr_to_enum = {v: k for k, v in mapping.items()}
    
    def get_attr_name(self, enum_val: T) -> Optional[str]:
        """Get attribute name for an enum value."""
        return self.enum_to_attr.get(enum_val)
    
    def get_enum(self, attr_name: str) -> Optional[T]:
        """Get enum value for an attribute name."""
        return self.attr_to_enum.get(attr_name)
    
    def get_value(self, obj: Any, enum_val: T) -> Any:
        """Get attribute value from object using enum."""
        attr_name = self.get_attr_name(enum_val)
        if attr_name:
            return getattr(obj, attr_name, None)
        return None
    
    def set_value(self, obj: Any, enum_val: T, value: Any) -> bool:
        """Set attribute value on object using enum."""
        attr_name = self.get_attr_name(enum_val)
        if attr_name:
            setattr(obj, attr_name, value)
            return True
        return False


# === Text Similarity Utilities ===

def compute_word_overlap_similarity(text1: str, text2: str) -> float:
    """
    Compute Jaccard similarity using word overlap.
    
    Used by NormAlignmentComputer._compute_similarity and other modules.
    """
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union)


# === Color Mapping Utilities ===

RISK_LEVEL_COLORS = {
    RiskLevel.NEGLIGIBLE: "#27ae60",
    RiskLevel.LOW: "#27ae60",
    RiskLevel.MODERATE: "#f39c12",
    RiskLevel.HIGH: "#e67e22",
    RiskLevel.CRITICAL: "#e74c3c",
}

RISK_LEVEL_COLORS_BY_STRING = {
    "negligible": "#27ae60",
    "low": "#27ae60",
    "medium": "#f39c12",
    "moderate": "#f39c12",
    "high": "#e67e22",
    "critical": "#e74c3c",
}


def get_risk_color(level: str) -> str:
    """Get color for a risk level string."""
    return RISK_LEVEL_COLORS_BY_STRING.get(level.lower(), "#666")


DECISION_OUTCOME_COLORS = {
    "acceptable": "#27ae60",
    "monitor": "#3498db",
    "mitigated": "#f39c12",
    "escalate": "#e67e22",
    "block": "#e74c3c",
}


def get_decision_outcome_color(outcome) -> str:
    """Get color for a decision outcome (enum or string)."""
    if hasattr(outcome, 'value'):
        outcome = outcome.value
    return DECISION_OUTCOME_COLORS.get(str(outcome).lower(), "#666")


# === Dataclass Serialization Utilities ===

def dataclass_to_dict(obj, exclude_none: bool = True) -> Dict[str, Any]:
    """
    Convert a dataclass to dict, handling nested structures.
    
    Consolidates the many to_dict() implementations.
    """
    from dataclasses import fields, is_dataclass
    
    if not is_dataclass(obj):
        return {}
    
    result = {}
    for field in fields(obj):
        value = getattr(obj, field.name)
        
        if exclude_none and value is None:
            continue
        
        # Handle nested dataclasses
        if is_dataclass(value):
            result[field.name] = dataclass_to_dict(value, exclude_none)
        # Handle enums
        elif isinstance(value, Enum):
            result[field.name] = value.value
        # Handle lists of dataclasses
        elif isinstance(value, list):
            result[field.name] = [
                dataclass_to_dict(item, exclude_none) if is_dataclass(item)
                else item.value if isinstance(item, Enum)
                else item
                for item in value
            ]
        # Handle dicts
        elif isinstance(value, dict):
            result[field.name] = {
                k: dataclass_to_dict(v, exclude_none) if is_dataclass(v)
                else v.value if isinstance(v, Enum)
                else v
                for k, v in value.items()
            }
        else:
            result[field.name] = value
    
    return result
