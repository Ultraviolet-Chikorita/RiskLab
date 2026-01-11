"""
Context metadata for evaluation scenarios.
"""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Domain(str, Enum):
    """Domain classification for scenarios."""
    HEALTH = "health"
    FINANCE = "finance"
    LEGAL = "legal"
    EDUCATION = "education"
    TECHNOLOGY = "technology"
    PERSONAL = "personal"
    PROFESSIONAL = "professional"
    POLITICAL = "political"
    SAFETY_CRITICAL = "safety_critical"
    GENERAL = "general"


class StakesLevel(str, Enum):
    """Stakes level classification."""
    LOW = "low"           # Minor consequences
    MEDIUM = "medium"     # Moderate consequences
    HIGH = "high"         # Significant consequences
    CRITICAL = "critical" # Severe or irreversible consequences


class VulnerabilityLevel(str, Enum):
    """User vulnerability classification."""
    LOW = "low"           # Standard user
    MEDIUM = "medium"     # Some vulnerability factors
    HIGH = "high"         # Significant vulnerability
    PROTECTED = "protected"  # Special protection required


class InteractionHorizon(str, Enum):
    """Interaction structure classification."""
    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"
    ONGOING = "ongoing"
    EPISODIC = "episodic"


class SensitivityMarker(str, Enum):
    """Sensitivity markers for scenarios."""
    PII = "pii"                           # Personal identifiable information
    MEDICAL = "medical"                   # Medical/health information
    FINANCIAL = "financial"               # Financial information
    MINOR = "minor"                       # Involves minors
    MENTAL_HEALTH = "mental_health"       # Mental health related
    DISCRIMINATION = "discrimination"     # Potential discrimination
    SAFETY = "safety"                     # Physical safety implications
    LEGAL_LIABILITY = "legal_liability"   # Legal liability concerns
    REGULATORY = "regulatory"             # Regulatory compliance
    CONFIDENTIAL = "confidential"         # Confidential information


class ContextMetadata(BaseModel):
    """
    Structured context metadata for evaluation scenarios.
    
    Used downstream for risk weighting and decision logic.
    """
    domain: Domain = Domain.GENERAL
    stakes_level: StakesLevel = StakesLevel.MEDIUM
    vulnerability_level: VulnerabilityLevel = VulnerabilityLevel.LOW
    interaction_horizon: InteractionHorizon = InteractionHorizon.SINGLE_TURN
    sensitivity_markers: List[SensitivityMarker] = Field(default_factory=list)
    
    # Additional context
    user_context: Optional[str] = None
    institutional_context: Optional[str] = None
    regulatory_context: Optional[str] = None
    
    # Custom tags for extensibility
    custom_tags: Dict[str, Any] = Field(default_factory=dict)
    
    def get_risk_multiplier(self) -> float:
        """
        Calculate a base risk multiplier from context factors.
        
        Returns a value >= 1.0 that increases with risk factors.
        """
        multiplier = 1.0
        
        # Stakes multiplier
        stakes_map = {
            StakesLevel.LOW: 1.0,
            StakesLevel.MEDIUM: 1.5,
            StakesLevel.HIGH: 2.5,
            StakesLevel.CRITICAL: 4.0,
        }
        multiplier *= stakes_map.get(self.stakes_level, 1.0)
        
        # Vulnerability multiplier
        vulnerability_map = {
            VulnerabilityLevel.LOW: 1.0,
            VulnerabilityLevel.MEDIUM: 1.3,
            VulnerabilityLevel.HIGH: 1.8,
            VulnerabilityLevel.PROTECTED: 2.5,
        }
        multiplier *= vulnerability_map.get(self.vulnerability_level, 1.0)
        
        # Domain multiplier
        domain_map = {
            Domain.GENERAL: 1.0,
            Domain.EDUCATION: 1.1,
            Domain.TECHNOLOGY: 1.1,
            Domain.PERSONAL: 1.2,
            Domain.PROFESSIONAL: 1.2,
            Domain.POLITICAL: 1.5,
            Domain.FINANCE: 1.6,
            Domain.LEGAL: 1.7,
            Domain.HEALTH: 1.8,
            Domain.SAFETY_CRITICAL: 2.5,
        }
        multiplier *= domain_map.get(self.domain, 1.0)
        
        # Sensitivity marker adjustment
        if self.sensitivity_markers:
            sensitivity_boost = 1.0 + (0.1 * len(self.sensitivity_markers))
            multiplier *= min(sensitivity_boost, 1.5)  # Cap at 50% boost
        
        return multiplier
    
    def has_sensitivity(self, marker: SensitivityMarker) -> bool:
        """Check if a specific sensitivity marker is present."""
        return marker in self.sensitivity_markers
    
    def to_description(self) -> str:
        """Generate a human-readable description of the context."""
        parts = [
            f"Domain: {self.domain.value}",
            f"Stakes: {self.stakes_level.value}",
            f"Vulnerability: {self.vulnerability_level.value}",
            f"Interaction: {self.interaction_horizon.value}",
        ]
        
        if self.sensitivity_markers:
            markers = ", ".join(m.value for m in self.sensitivity_markers)
            parts.append(f"Sensitivities: {markers}")
        
        return " | ".join(parts)


class ContextBuilder:
    """Builder pattern for constructing ContextMetadata."""
    
    def __init__(self):
        self._domain = Domain.GENERAL
        self._stakes = StakesLevel.MEDIUM
        self._vulnerability = VulnerabilityLevel.LOW
        self._horizon = InteractionHorizon.SINGLE_TURN
        self._sensitivities: List[SensitivityMarker] = []
        self._custom_tags: Dict[str, Any] = {}
    
    def domain(self, domain: Domain) -> "ContextBuilder":
        self._domain = domain
        return self
    
    def stakes(self, level: StakesLevel) -> "ContextBuilder":
        self._stakes = level
        return self
    
    def vulnerability(self, level: VulnerabilityLevel) -> "ContextBuilder":
        self._vulnerability = level
        return self
    
    def horizon(self, horizon: InteractionHorizon) -> "ContextBuilder":
        self._horizon = horizon
        return self
    
    def add_sensitivity(self, *markers: SensitivityMarker) -> "ContextBuilder":
        self._sensitivities.extend(markers)
        return self
    
    def tag(self, key: str, value: Any) -> "ContextBuilder":
        self._custom_tags[key] = value
        return self
    
    def build(self) -> ContextMetadata:
        return ContextMetadata(
            domain=self._domain,
            stakes_level=self._stakes,
            vulnerability_level=self._vulnerability,
            interaction_horizon=self._horizon,
            sensitivity_markers=list(set(self._sensitivities)),
            custom_tags=self._custom_tags,
        )


# Predefined context presets for common scenarios
CONTEXT_PRESETS = {
    "medical_advice": ContextBuilder()
        .domain(Domain.HEALTH)
        .stakes(StakesLevel.HIGH)
        .add_sensitivity(SensitivityMarker.MEDICAL)
        .build(),
    
    "financial_advice": ContextBuilder()
        .domain(Domain.FINANCE)
        .stakes(StakesLevel.HIGH)
        .add_sensitivity(SensitivityMarker.FINANCIAL)
        .build(),
    
    "legal_advice": ContextBuilder()
        .domain(Domain.LEGAL)
        .stakes(StakesLevel.HIGH)
        .add_sensitivity(SensitivityMarker.LEGAL_LIABILITY)
        .build(),
    
    "mental_health": ContextBuilder()
        .domain(Domain.HEALTH)
        .stakes(StakesLevel.CRITICAL)
        .vulnerability(VulnerabilityLevel.HIGH)
        .add_sensitivity(SensitivityMarker.MENTAL_HEALTH, SensitivityMarker.MEDICAL)
        .build(),
    
    "child_interaction": ContextBuilder()
        .domain(Domain.EDUCATION)
        .vulnerability(VulnerabilityLevel.PROTECTED)
        .add_sensitivity(SensitivityMarker.MINOR)
        .build(),
    
    "safety_critical": ContextBuilder()
        .domain(Domain.SAFETY_CRITICAL)
        .stakes(StakesLevel.CRITICAL)
        .add_sensitivity(SensitivityMarker.SAFETY)
        .build(),
    
    "general_assistance": ContextBuilder()
        .domain(Domain.GENERAL)
        .stakes(StakesLevel.LOW)
        .build(),
}
