"""
Latent Bias & Manipulability Model.

Fits a latent factor model where each model version has:
- bias vector: which institutional norms it defaults to
- susceptibility vector: which framings/pressures move it

This enables tracking of:
- bias evolution over model versions
- drift in susceptibility
- vulnerability patterns
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field

from risklab.scenarios.institutional import (
    NormDomain,
    InstitutionalRegime,
    PressureType,
)
from risklab.measurement.norm_alignment import (
    NormAlignmentScore,
    NormStabilityCurve,
    InstitutionalBiasProfile,
)


@dataclass
class BiasVector:
    """
    Latent bias vector representing default institutional leanings.
    
    Each dimension corresponds to a norm domain, with values from -1 (Norm A)
    to +1 (Norm B).
    """
    
    model_id: str
    model_version: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Bias by norm domain
    domain_biases: Dict[str, float] = field(default_factory=dict)
    
    # Aggregate bias direction
    mean_bias: float = 0.0
    bias_consistency: float = 0.0  # How consistent across domains
    
    # Factor loadings (from factor analysis)
    latent_factors: Optional[np.ndarray] = None
    
    def compute_aggregates(self) -> None:
        """Compute aggregate statistics."""
        if self.domain_biases:
            values = list(self.domain_biases.values())
            self.mean_bias = float(np.mean(values))
            self.bias_consistency = 1 - float(np.std(values))
    
    def similarity_to(self, other: 'BiasVector') -> float:
        """Compute similarity to another bias vector."""
        common_domains = set(self.domain_biases.keys()) & set(other.domain_biases.keys())
        
        if not common_domains:
            return 0.0
        
        self_values = [self.domain_biases[d] for d in common_domains]
        other_values = [other.domain_biases[d] for d in common_domains]
        
        # Cosine similarity
        dot = np.dot(self_values, other_values)
        norm = np.linalg.norm(self_values) * np.linalg.norm(other_values)
        
        if norm == 0:
            return 1.0
        
        return float(dot / norm)
    
    def to_array(self, domains: List[str]) -> np.ndarray:
        """Convert to numpy array for a given domain ordering."""
        return np.array([self.domain_biases.get(d, 0.0) for d in domains])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "timestamp": self.timestamp.isoformat(),
            "domain_biases": self.domain_biases,
            "mean_bias": self.mean_bias,
            "bias_consistency": self.bias_consistency,
        }


@dataclass
class SusceptibilityVector:
    """
    Latent susceptibility vector representing vulnerability to different pressures.
    
    Each dimension corresponds to a pressure type, with values from 0 (immune)
    to 1 (highly susceptible).
    """
    
    model_id: str
    model_version: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Susceptibility by pressure type
    pressure_susceptibility: Dict[str, float] = field(default_factory=dict)
    
    # Aggregate susceptibility
    mean_susceptibility: float = 0.0
    max_susceptibility: float = 0.0
    most_vulnerable_to: Optional[str] = None
    
    # Factor loadings
    latent_factors: Optional[np.ndarray] = None
    
    def compute_aggregates(self) -> None:
        """Compute aggregate statistics."""
        if self.pressure_susceptibility:
            values = list(self.pressure_susceptibility.values())
            self.mean_susceptibility = float(np.mean(values))
            self.max_susceptibility = float(np.max(values))
            
            max_idx = np.argmax(values)
            self.most_vulnerable_to = list(self.pressure_susceptibility.keys())[max_idx]
    
    def to_array(self, pressures: List[str]) -> np.ndarray:
        """Convert to numpy array for a given pressure ordering."""
        return np.array([self.pressure_susceptibility.get(p, 0.0) for p in pressures])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "timestamp": self.timestamp.isoformat(),
            "pressure_susceptibility": self.pressure_susceptibility,
            "mean_susceptibility": self.mean_susceptibility,
            "max_susceptibility": self.max_susceptibility,
            "most_vulnerable_to": self.most_vulnerable_to,
        }


@dataclass
class ModelLatentProfile:
    """
    Complete latent profile for a model version.
    
    Combines bias and susceptibility vectors with metadata.
    """
    
    model_id: str
    model_version: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    bias_vector: Optional[BiasVector] = None
    susceptibility_vector: Optional[SusceptibilityVector] = None
    
    # Training metadata
    training_data_hash: Optional[str] = None
    fine_tuning_description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "timestamp": self.timestamp.isoformat(),
            "bias": self.bias_vector.to_dict() if self.bias_vector else None,
            "susceptibility": self.susceptibility_vector.to_dict() if self.susceptibility_vector else None,
        }


class LatentBiasModel:
    """
    Probabilistic latent variable model for bias and susceptibility.
    
    Fits a factor model where:
    - Observed: alignment scores across scenarios, shift amounts under pressure
    - Latent: bias factors, susceptibility factors
    
    This enables:
    - Dimensionality reduction for interpretability
    - Tracking changes over model versions
    - Predicting behavior in new scenarios
    """
    
    def __init__(
        self,
        n_bias_factors: int = 3,
        n_susceptibility_factors: int = 2,
    ):
        self.n_bias_factors = n_bias_factors
        self.n_susceptibility_factors = n_susceptibility_factors
        
        # Learned parameters
        self.bias_loadings: Optional[np.ndarray] = None  # [n_domains, n_factors]
        self.susceptibility_loadings: Optional[np.ndarray] = None  # [n_pressures, n_factors]
        
        # Domain/pressure ordering
        self.domains: List[str] = []
        self.pressures: List[str] = []
        
        # Fitted profiles
        self.profiles: Dict[str, ModelLatentProfile] = {}
    
    def fit(
        self,
        profiles: List[InstitutionalBiasProfile],
    ) -> Dict[str, Any]:
        """
        Fit the latent model to a collection of bias profiles.
        
        Uses PCA/factor analysis to extract latent factors.
        """
        if not profiles:
            return {"error": "No profiles provided"}
        
        # Collect all domains and pressures
        all_domains = set()
        all_pressures = set()
        
        for profile in profiles:
            all_domains.update(profile.domain_biases.keys())
            all_pressures.update(profile.pressure_effectiveness.keys())
        
        self.domains = sorted(list(all_domains))
        self.pressures = sorted(list(all_pressures))
        
        # Build observation matrices
        n_models = len(profiles)
        n_domains = len(self.domains)
        n_pressures = len(self.pressures)
        
        bias_matrix = np.zeros((n_models, n_domains))
        susceptibility_matrix = np.zeros((n_models, n_pressures))
        
        for i, profile in enumerate(profiles):
            for j, domain in enumerate(self.domains):
                bias_matrix[i, j] = profile.domain_biases.get(domain, 0.0)
            
            for j, pressure in enumerate(self.pressures):
                susceptibility_matrix[i, j] = profile.pressure_effectiveness.get(pressure, 0.0)
        
        # Fit factor models using SVD (simple approach)
        # Bias factors
        if n_domains > 0 and n_models > 1:
            bias_centered = bias_matrix - np.mean(bias_matrix, axis=0)
            try:
                U, S, Vt = np.linalg.svd(bias_centered, full_matrices=False)
                n_factors = min(self.n_bias_factors, len(S))
                self.bias_loadings = Vt[:n_factors].T  # [n_domains, n_factors]
            except np.linalg.LinAlgError:
                self.bias_loadings = np.eye(n_domains, self.n_bias_factors)
        
        # Susceptibility factors
        if n_pressures > 0 and n_models > 1:
            susc_centered = susceptibility_matrix - np.mean(susceptibility_matrix, axis=0)
            try:
                U, S, Vt = np.linalg.svd(susc_centered, full_matrices=False)
                n_factors = min(self.n_susceptibility_factors, len(S))
                self.susceptibility_loadings = Vt[:n_factors].T
            except np.linalg.LinAlgError:
                self.susceptibility_loadings = np.eye(n_pressures, self.n_susceptibility_factors)
        
        # Create latent profiles for each model
        for i, profile in enumerate(profiles):
            latent_profile = self._create_latent_profile(
                profile,
                bias_matrix[i] if n_domains > 0 else None,
                susceptibility_matrix[i] if n_pressures > 0 else None,
            )
            self.profiles[profile.model_id] = latent_profile
        
        return {
            "n_models": n_models,
            "n_domains": n_domains,
            "n_pressures": n_pressures,
            "n_bias_factors": self.n_bias_factors,
            "n_susceptibility_factors": self.n_susceptibility_factors,
        }
    
    def _create_latent_profile(
        self,
        profile: InstitutionalBiasProfile,
        bias_obs: Optional[np.ndarray],
        susc_obs: Optional[np.ndarray],
    ) -> ModelLatentProfile:
        """Create latent profile from observations."""
        # Bias vector
        bias_vec = BiasVector(
            model_id=profile.model_id,
            model_version="v1",  # Default
            domain_biases=profile.domain_biases.copy(),
        )
        
        if bias_obs is not None and self.bias_loadings is not None:
            # Project onto latent factors
            bias_vec.latent_factors = bias_obs @ self.bias_loadings
        
        bias_vec.compute_aggregates()
        
        # Susceptibility vector
        susc_vec = SusceptibilityVector(
            model_id=profile.model_id,
            model_version="v1",
            pressure_susceptibility=profile.pressure_effectiveness.copy(),
        )
        
        if susc_obs is not None and self.susceptibility_loadings is not None:
            susc_vec.latent_factors = susc_obs @ self.susceptibility_loadings
        
        susc_vec.compute_aggregates()
        
        return ModelLatentProfile(
            model_id=profile.model_id,
            model_version="v1",
            bias_vector=bias_vec,
            susceptibility_vector=susc_vec,
        )
    
    def get_profile(self, model_id: str) -> Optional[ModelLatentProfile]:
        """Get latent profile for a model."""
        return self.profiles.get(model_id)
    
    def compute_drift(
        self,
        profile_a: ModelLatentProfile,
        profile_b: ModelLatentProfile,
    ) -> Dict[str, Any]:
        """
        Compute drift between two model profiles.
        
        Returns metrics on how bias and susceptibility changed.
        """
        result = {
            "model_a": profile_a.model_id,
            "model_b": profile_b.model_id,
        }
        
        # Bias drift
        if profile_a.bias_vector and profile_b.bias_vector:
            bias_a = profile_a.bias_vector
            bias_b = profile_b.bias_vector
            
            result["bias_similarity"] = bias_a.similarity_to(bias_b)
            result["mean_bias_shift"] = bias_b.mean_bias - bias_a.mean_bias
            
            # Per-domain drift
            domain_drifts = {}
            common = set(bias_a.domain_biases.keys()) & set(bias_b.domain_biases.keys())
            for d in common:
                domain_drifts[d] = bias_b.domain_biases[d] - bias_a.domain_biases[d]
            result["domain_drifts"] = domain_drifts
        
        # Susceptibility drift
        if profile_a.susceptibility_vector and profile_b.susceptibility_vector:
            susc_a = profile_a.susceptibility_vector
            susc_b = profile_b.susceptibility_vector
            
            result["susceptibility_change"] = susc_b.mean_susceptibility - susc_a.mean_susceptibility
            result["max_susceptibility_change"] = susc_b.max_susceptibility - susc_a.max_susceptibility
            
            # Per-pressure drift
            pressure_drifts = {}
            common = set(susc_a.pressure_susceptibility.keys()) & set(susc_b.pressure_susceptibility.keys())
            for p in common:
                pressure_drifts[p] = susc_b.pressure_susceptibility[p] - susc_a.pressure_susceptibility[p]
            result["pressure_drifts"] = pressure_drifts
        
        return result
    
    def predict_behavior(
        self,
        model_id: str,
        domain: str,
        pressure: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Predict model behavior in a new scenario.
        
        Uses latent factors to extrapolate from observed behavior.
        """
        profile = self.profiles.get(model_id)
        if not profile:
            return {"error": f"Unknown model: {model_id}"}
        
        result = {"model_id": model_id, "domain": domain}
        
        # Predict baseline bias
        if profile.bias_vector:
            if domain in profile.bias_vector.domain_biases:
                result["predicted_bias"] = profile.bias_vector.domain_biases[domain]
            else:
                # Extrapolate from similar domains (use mean for simplicity)
                result["predicted_bias"] = profile.bias_vector.mean_bias
                result["extrapolated"] = True
        
        # Predict susceptibility
        if pressure and profile.susceptibility_vector:
            if pressure in profile.susceptibility_vector.pressure_susceptibility:
                result["predicted_shift"] = profile.susceptibility_vector.pressure_susceptibility[pressure]
            else:
                result["predicted_shift"] = profile.susceptibility_vector.mean_susceptibility
                result["extrapolated"] = True
        
        return result


class ModelEvolutionTracker:
    """
    Tracks bias and susceptibility evolution across model versions.
    
    Enables analysis of:
    - Whether training reduces bias or redistributes it
    - How susceptibility changes with fine-tuning
    - Regression detection for safety-relevant properties
    """
    
    def __init__(self):
        self.version_profiles: Dict[str, List[ModelLatentProfile]] = {}  # model_id -> versions
        self.latent_model = LatentBiasModel()
    
    def add_version(
        self,
        model_id: str,
        version: str,
        profile: InstitutionalBiasProfile,
    ) -> None:
        """Add a new model version."""
        if model_id not in self.version_profiles:
            self.version_profiles[model_id] = []
        
        # Create latent profile
        bias_vec = BiasVector(
            model_id=model_id,
            model_version=version,
            domain_biases=profile.domain_biases.copy(),
        )
        bias_vec.compute_aggregates()
        
        susc_vec = SusceptibilityVector(
            model_id=model_id,
            model_version=version,
            pressure_susceptibility=profile.pressure_effectiveness.copy(),
        )
        susc_vec.compute_aggregates()
        
        latent_profile = ModelLatentProfile(
            model_id=model_id,
            model_version=version,
            bias_vector=bias_vec,
            susceptibility_vector=susc_vec,
        )
        
        self.version_profiles[model_id].append(latent_profile)
    
    def get_evolution(self, model_id: str) -> Dict[str, Any]:
        """Get evolution data for a model."""
        versions = self.version_profiles.get(model_id, [])
        
        if not versions:
            return {"error": f"No versions for {model_id}"}
        
        evolution = {
            "model_id": model_id,
            "n_versions": len(versions),
            "versions": [],
            "bias_trend": [],
            "susceptibility_trend": [],
        }
        
        for v in versions:
            version_data = {
                "version": v.model_version,
                "timestamp": v.timestamp.isoformat(),
            }
            
            if v.bias_vector:
                version_data["mean_bias"] = v.bias_vector.mean_bias
                evolution["bias_trend"].append(v.bias_vector.mean_bias)
            
            if v.susceptibility_vector:
                version_data["mean_susceptibility"] = v.susceptibility_vector.mean_susceptibility
                evolution["susceptibility_trend"].append(v.susceptibility_vector.mean_susceptibility)
            
            evolution["versions"].append(version_data)
        
        # Compute trends
        if len(evolution["bias_trend"]) > 1:
            # Linear trend
            x = np.arange(len(evolution["bias_trend"]))
            y = np.array(evolution["bias_trend"])
            slope = np.polyfit(x, y, 1)[0]
            evolution["bias_slope"] = float(slope)
            evolution["bias_improving"] = abs(slope) < 0.01 or (abs(y[-1]) < abs(y[0]))
        
        if len(evolution["susceptibility_trend"]) > 1:
            x = np.arange(len(evolution["susceptibility_trend"]))
            y = np.array(evolution["susceptibility_trend"])
            slope = np.polyfit(x, y, 1)[0]
            evolution["susceptibility_slope"] = float(slope)
            evolution["susceptibility_improving"] = slope < 0  # Lower is better
        
        return evolution
    
    def detect_regression(
        self,
        model_id: str,
        threshold: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Detect safety regressions between versions.
        
        Returns warnings if bias or susceptibility increased significantly.
        """
        versions = self.version_profiles.get(model_id, [])
        
        if len(versions) < 2:
            return {"regressions": [], "warnings": []}
        
        regressions = []
        warnings = []
        
        for i in range(1, len(versions)):
            prev = versions[i-1]
            curr = versions[i]
            
            # Check bias regression
            if prev.bias_vector and curr.bias_vector:
                prev_magnitude = abs(prev.bias_vector.mean_bias)
                curr_magnitude = abs(curr.bias_vector.mean_bias)
                
                if curr_magnitude > prev_magnitude + threshold:
                    regressions.append({
                        "type": "bias_increase",
                        "from_version": prev.model_version,
                        "to_version": curr.model_version,
                        "change": curr_magnitude - prev_magnitude,
                    })
            
            # Check susceptibility regression
            if prev.susceptibility_vector and curr.susceptibility_vector:
                prev_susc = prev.susceptibility_vector.mean_susceptibility
                curr_susc = curr.susceptibility_vector.mean_susceptibility
                
                if curr_susc > prev_susc + threshold:
                    regressions.append({
                        "type": "susceptibility_increase",
                        "from_version": prev.model_version,
                        "to_version": curr.model_version,
                        "change": curr_susc - prev_susc,
                    })
        
        # Generate warnings
        for reg in regressions:
            if reg["type"] == "bias_increase":
                warnings.append(
                    f"Bias magnitude increased by {reg['change']:.3f} from "
                    f"{reg['from_version']} to {reg['to_version']}"
                )
            elif reg["type"] == "susceptibility_increase":
                warnings.append(
                    f"Susceptibility increased by {reg['change']:.3f} from "
                    f"{reg['from_version']} to {reg['to_version']}"
                )
        
        return {"regressions": regressions, "warnings": warnings}
