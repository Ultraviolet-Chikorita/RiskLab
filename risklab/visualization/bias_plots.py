"""
Institutional Bias Visualizations.

Provides clear, legible visuals for:
1. Institutional Bias Compass - 2D plot of norm alignment
2. Manipulability Radar - Spider chart of pressure vulnerability
3. Drift Over Time - Evolution of bias/susceptibility across versions
4. Norm Stability Curves - How alignment changes under pressure
"""

from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass

import numpy as np

from risklab.scenarios.institutional import PressureType, NormDomain
from risklab.utils import render_figure_to_output
from risklab.measurement.norm_alignment import (
    NormAlignmentScore,
    NormStabilityCurve,
    InstitutionalBiasProfile,
)
from risklab.measurement.latent_bias_model import (
    BiasVector,
    SusceptibilityVector,
    ModelLatentProfile,
)


def _get_matplotlib():
    """Lazy import matplotlib."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    return plt, mpatches


class InstitutionalBiasCompass:
    """
    2D compass visualization of institutional bias.
    
    Axes: Norm A ↔ Norm B
    Points: model versions or scenarios
    Error bars: uncertainty
    
    Intuitive even for non-technical audiences.
    """
    
    def __init__(
        self,
        title: str = "Institutional Bias Compass",
        norm_a_label: str = "Norm A",
        norm_b_label: str = "Norm B",
        figsize: Tuple[int, int] = (10, 10),
    ):
        self.title = title
        self.norm_a_label = norm_a_label
        self.norm_b_label = norm_b_label
        self.figsize = figsize
        
        self.data_points: List[Dict[str, Any]] = []
    
    def add_model(
        self,
        model_id: str,
        bias_x: float,  # -1 (Norm A) to +1 (Norm B) on x-axis
        bias_y: float,  # Second dimension (e.g., different domain pair)
        uncertainty_x: float = 0.0,
        uncertainty_y: float = 0.0,
        color: Optional[str] = None,
        marker: str = 'o',
    ) -> None:
        """Add a model point to the compass."""
        self.data_points.append({
            "model_id": model_id,
            "x": bias_x,
            "y": bias_y,
            "xerr": uncertainty_x,
            "yerr": uncertainty_y,
            "color": color,
            "marker": marker,
        })
    
    def add_from_profile(
        self,
        profile: InstitutionalBiasProfile,
        x_domain: str,
        y_domain: str,
        color: Optional[str] = None,
    ) -> None:
        """Add a model from its bias profile."""
        bias_x = profile.domain_biases.get(x_domain, 0.0)
        bias_y = profile.domain_biases.get(y_domain, 0.0)
        
        # Estimate uncertainty from variance if available
        uncertainty = np.sqrt(profile.bias_variance) if profile.bias_variance else 0.1
        
        self.add_model(
            model_id=profile.model_id,
            bias_x=bias_x,
            bias_y=bias_y,
            uncertainty_x=uncertainty,
            uncertainty_y=uncertainty,
            color=color,
        )
    
    def render(self, output_path: Optional[Path] = None) -> Optional[str]:
        """
        Render the compass plot.
        
        Returns base64 encoded PNG if no output path.
        """
        plt, mpatches = _get_matplotlib()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Draw compass background
        circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
        ax.add_patch(circle)
        
        # Draw axes
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        
        # Draw quadrant labels
        ax.text(0.7, 0.7, "Norm B\n(both)", ha='center', va='center', fontsize=10, alpha=0.5)
        ax.text(-0.7, 0.7, "Mixed\n(Y:B, X:A)", ha='center', va='center', fontsize=10, alpha=0.5)
        ax.text(-0.7, -0.7, "Norm A\n(both)", ha='center', va='center', fontsize=10, alpha=0.5)
        ax.text(0.7, -0.7, "Mixed\n(Y:A, X:B)", ha='center', va='center', fontsize=10, alpha=0.5)
        
        # Plot data points
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.data_points)))
        
        for i, point in enumerate(self.data_points):
            color = point.get("color") or colors[i]
            
            ax.errorbar(
                point["x"], point["y"],
                xerr=point["xerr"], yerr=point["yerr"],
                fmt=point["marker"],
                color=color,
                markersize=12,
                capsize=5,
                label=point["model_id"],
            )
        
        # Labels and title
        ax.set_xlabel(f"← {self.norm_a_label}  |  {self.norm_b_label} →", fontsize=12)
        ax.set_ylabel(f"← {self.norm_a_label}  |  {self.norm_b_label} →", fontsize=12)
        ax.set_title(self.title, fontsize=14, fontweight='bold')
        
        # Set limits
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        
        # Legend
        if len(self.data_points) <= 10:
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        
        plt.tight_layout()
        
        return render_figure_to_output(fig, output_path)


class ManipulabilityRadar:
    """
    Spider/radar chart showing vulnerability to different pressure types.
    
    Spokes: persuasion mechanisms (authority, efficiency, risk, etc.)
    Radius: degree of shift under that pressure
    
    Shows HOW manipulation works, not just THAT it exists.
    """
    
    def __init__(
        self,
        title: str = "Manipulability Radar",
        figsize: Tuple[int, int] = (10, 10),
    ):
        self.title = title
        self.figsize = figsize
        
        self.models: Dict[str, Dict[str, float]] = {}
    
    def add_model(
        self,
        model_id: str,
        pressure_susceptibility: Dict[str, float],
    ) -> None:
        """Add a model's susceptibility profile."""
        self.models[model_id] = pressure_susceptibility
    
    def add_from_stability(
        self,
        model_id: str,
        stability_curve: NormStabilityCurve,
    ) -> None:
        """Add a model from its stability curve."""
        susceptibility = {}
        
        for point in stability_curve.pressure_points:
            susceptibility[point.pressure_type.value] = abs(point.shift_from_baseline)
        
        self.models[model_id] = susceptibility
    
    def add_from_profile(
        self,
        profile: InstitutionalBiasProfile,
    ) -> None:
        """Add a model from its bias profile."""
        self.models[profile.model_id] = profile.pressure_effectiveness.copy()
    
    def render(self, output_path: Optional[Path] = None) -> Optional[str]:
        """Render the radar chart."""
        plt, _ = _get_matplotlib()
        
        if not self.models:
            return None
        
        # Get all pressure types
        all_pressures = set()
        for model_data in self.models.values():
            all_pressures.update(model_data.keys())
        
        pressures = sorted(list(all_pressures))
        n_pressures = len(pressures)
        
        if n_pressures < 3:
            return None  # Need at least 3 spokes
        
        # Compute angles
        angles = np.linspace(0, 2 * np.pi, n_pressures, endpoint=False).tolist()
        angles += angles[:1]  # Close the polygon
        
        fig, ax = plt.subplots(figsize=self.figsize, subplot_kw=dict(polar=True))
        
        # Plot each model
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.models)))
        
        for i, (model_id, susceptibility) in enumerate(self.models.items()):
            values = [susceptibility.get(p, 0) for p in pressures]
            values += values[:1]  # Close the polygon
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_id, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Labels
        ax.set_xticks(angles[:-1])
        # Format pressure type labels
        labels = [p.replace('_', ' ').title() for p in pressures]
        ax.set_xticklabels(labels, fontsize=10)
        
        ax.set_ylim(0, 1)
        ax.set_title(self.title, fontsize=14, fontweight='bold', pad=20)
        
        # Legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        
        return render_figure_to_output(fig, output_path)


class BiasEvolutionPlot:
    """
    Line plot showing bias and susceptibility evolution across model versions.
    
    Shows:
    - Whether training reduces bias or redistributes it
    - How susceptibility changes over time
    - Regression detection
    """
    
    def __init__(
        self,
        title: str = "Bias & Susceptibility Evolution",
        figsize: Tuple[int, int] = (12, 6),
    ):
        self.title = title
        self.figsize = figsize
        
        self.versions: List[str] = []
        self.bias_values: List[float] = []
        self.susceptibility_values: List[float] = []
        self.timestamps: List[str] = []
    
    def add_version(
        self,
        version: str,
        bias: float,
        susceptibility: float,
        timestamp: Optional[str] = None,
    ) -> None:
        """Add a model version data point."""
        self.versions.append(version)
        self.bias_values.append(bias)
        self.susceptibility_values.append(susceptibility)
        self.timestamps.append(timestamp or version)
    
    def add_from_evolution(self, evolution_data: Dict[str, Any]) -> None:
        """Add data from ModelEvolutionTracker output."""
        for v in evolution_data.get("versions", []):
            self.add_version(
                version=v.get("version", ""),
                bias=abs(v.get("mean_bias", 0)),  # Use magnitude
                susceptibility=v.get("mean_susceptibility", 0),
                timestamp=v.get("timestamp"),
            )
    
    def render(self, output_path: Optional[Path] = None) -> Optional[str]:
        """Render the evolution plot."""
        plt, _ = _get_matplotlib()
        
        if len(self.versions) < 2:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        x = np.arange(len(self.versions))
        
        # Bias plot
        ax1.plot(x, self.bias_values, 'b-o', linewidth=2, markersize=8)
        ax1.fill_between(x, 0, self.bias_values, alpha=0.3)
        ax1.set_xlabel("Version", fontsize=11)
        ax1.set_ylabel("Bias Magnitude", fontsize=11)
        ax1.set_title("Bias Over Versions", fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.versions, rotation=45, ha='right')
        ax1.set_ylim(0, max(self.bias_values) * 1.2 if self.bias_values else 1)
        
        # Add trend line
        if len(x) > 1:
            z = np.polyfit(x, self.bias_values, 1)
            p = np.poly1d(z)
            ax1.plot(x, p(x), 'r--', alpha=0.5, label=f'Trend (slope: {z[0]:.3f})')
            ax1.legend()
        
        # Susceptibility plot
        ax2.plot(x, self.susceptibility_values, 'g-o', linewidth=2, markersize=8)
        ax2.fill_between(x, 0, self.susceptibility_values, alpha=0.3, color='green')
        ax2.set_xlabel("Version", fontsize=11)
        ax2.set_ylabel("Mean Susceptibility", fontsize=11)
        ax2.set_title("Susceptibility Over Versions", fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.versions, rotation=45, ha='right')
        ax2.set_ylim(0, max(self.susceptibility_values) * 1.2 if self.susceptibility_values else 1)
        
        # Add trend line
        if len(x) > 1:
            z = np.polyfit(x, self.susceptibility_values, 1)
            p = np.poly1d(z)
            ax2.plot(x, p(x), 'r--', alpha=0.5, label=f'Trend (slope: {z[0]:.3f})')
            ax2.legend()
        
        plt.suptitle(self.title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return render_figure_to_output(fig, output_path)


class NormStabilityPlot:
    """
    Plot showing how model alignment changes under different pressures.
    
    X-axis: Pressure type or intensity
    Y-axis: Alignment with original norm (1 = unchanged, 0 = fully shifted)
    """
    
    def __init__(
        self,
        title: str = "Norm Stability Under Pressure",
        figsize: Tuple[int, int] = (12, 6),
    ):
        self.title = title
        self.figsize = figsize
        
        self.curves: Dict[str, NormStabilityCurve] = {}
    
    def add_curve(
        self,
        model_id: str,
        curve: NormStabilityCurve,
    ) -> None:
        """Add a stability curve for a model."""
        self.curves[model_id] = curve
    
    def render(self, output_path: Optional[Path] = None) -> Optional[str]:
        """Render the stability plot."""
        plt, _ = _get_matplotlib()
        
        if not self.curves:
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(self.curves)))
        
        for i, (model_id, curve) in enumerate(self.curves.items()):
            if not curve.pressure_points:
                continue
            
            # Extract data
            pressures = []
            stability_values = []
            
            for point in curve.pressure_points:
                pressures.append(point.pressure_type.value.replace('_', ' ').title())
                # Stability = 1 - shift
                stability_values.append(1 - abs(point.shift_from_baseline))
            
            # Sort by stability
            sorted_data = sorted(zip(pressures, stability_values), key=lambda x: -x[1])
            pressures, stability_values = zip(*sorted_data)
            
            x = np.arange(len(pressures))
            
            ax.bar(x + i * 0.25, stability_values, width=0.25, 
                   label=model_id, color=colors[i], alpha=0.8)
        
        # Reference line for "stable"
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Stability threshold')
        
        ax.set_xlabel("Pressure Type", fontsize=11)
        ax.set_ylabel("Stability (1 = unchanged)", fontsize=11)
        ax.set_title(self.title, fontsize=14, fontweight='bold')
        
        if self.curves:
            first_curve = list(self.curves.values())[0]
            if first_curve.pressure_points:
                pressures = [p.pressure_type.value.replace('_', ' ').title() 
                           for p in first_curve.pressure_points]
                ax.set_xticks(np.arange(len(pressures)) + 0.25 * (len(self.curves) - 1) / 2)
                ax.set_xticklabels(pressures, rotation=45, ha='right')
        
        ax.set_ylim(0, 1.1)
        ax.legend()
        
        plt.tight_layout()
        
        return render_figure_to_output(fig, output_path)


class DomainBiasHeatmap:
    """
    Heatmap showing bias across different norm domains.
    
    Rows: Models
    Columns: Norm domains
    Color: Direction and magnitude of bias
    """
    
    def __init__(
        self,
        title: str = "Institutional Bias by Domain",
        figsize: Tuple[int, int] = (14, 8),
    ):
        self.title = title
        self.figsize = figsize
        
        self.models: List[str] = []
        self.domains: List[str] = []
        self.bias_matrix: List[List[float]] = []
    
    def add_profile(self, profile: InstitutionalBiasProfile) -> None:
        """Add a model's bias profile."""
        self.models.append(profile.model_id)
        
        # Ensure consistent domain ordering
        if not self.domains:
            self.domains = sorted(profile.domain_biases.keys())
        
        row = [profile.domain_biases.get(d, 0.0) for d in self.domains]
        self.bias_matrix.append(row)
    
    def render(self, output_path: Optional[Path] = None) -> Optional[str]:
        """Render the heatmap."""
        plt, _ = _get_matplotlib()
        import matplotlib.colors as mcolors
        
        if not self.models or not self.domains:
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        data = np.array(self.bias_matrix)
        
        # Diverging colormap (blue = Norm A, red = Norm B)
        cmap = plt.cm.RdBu_r
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        
        im = ax.imshow(data, cmap=cmap, norm=norm, aspect='auto')
        
        # Labels
        ax.set_xticks(np.arange(len(self.domains)))
        ax.set_yticks(np.arange(len(self.models)))
        
        domain_labels = [d.replace('_', ' ').title() for d in self.domains]
        ax.set_xticklabels(domain_labels, rotation=45, ha='right')
        ax.set_yticklabels(self.models)
        
        # Add values in cells
        for i in range(len(self.models)):
            for j in range(len(self.domains)):
                value = data[i, j]
                color = 'white' if abs(value) > 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=color, fontsize=9)
        
        # Colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("← Norm A  |  Norm B →", rotation=-90, va="bottom")
        
        ax.set_title(self.title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        return render_figure_to_output(fig, output_path)


def generate_institutional_bias_report(
    profiles: List[InstitutionalBiasProfile],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Generate complete set of institutional bias visualizations.
    
    Args:
        profiles: List of bias profiles to visualize
        output_dir: Directory to save plots
    
    Returns:
        Dict with paths to generated plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {"plots": []}
    
    if not profiles:
        return results
    
    # 1. Domain bias heatmap
    heatmap = DomainBiasHeatmap()
    for profile in profiles:
        heatmap.add_profile(profile)
    
    heatmap_path = output_dir / "domain_bias_heatmap.png"
    heatmap.render(heatmap_path)
    results["plots"].append({"name": "Domain Bias Heatmap", "path": str(heatmap_path)})
    
    # 2. Manipulability radar (for each model)
    for profile in profiles:
        if profile.pressure_effectiveness:
            radar = ManipulabilityRadar(title=f"Manipulability: {profile.model_id}")
            radar.add_from_profile(profile)
            
            radar_path = output_dir / f"manipulability_radar_{profile.model_id}.png"
            radar.render(radar_path)
            results["plots"].append({
                "name": f"Manipulability Radar: {profile.model_id}",
                "path": str(radar_path),
            })
    
    # 3. Bias compass (if we have at least 2 domains)
    if profiles and len(profiles[0].domain_biases) >= 2:
        domains = list(profiles[0].domain_biases.keys())[:2]
        
        compass = InstitutionalBiasCompass(
            norm_a_label=domains[0].replace('_', ' ').title(),
            norm_b_label=domains[1].replace('_', ' ').title() if len(domains) > 1 else "Domain 2",
        )
        
        for profile in profiles:
            compass.add_from_profile(profile, domains[0], domains[1] if len(domains) > 1 else domains[0])
        
        compass_path = output_dir / "bias_compass.png"
        compass.render(compass_path)
        results["plots"].append({"name": "Bias Compass", "path": str(compass_path)})
    
    return results
