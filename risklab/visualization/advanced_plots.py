"""
Advanced Visualization Components for RiskLab.

Includes:
- Safety Gauge Chart
- Metric Waterfall Chart
- Vulnerability Sunburst
- Temporal Drift Plot
- Attack Surface Map
- Confidence Funnel
"""

from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import json
import numpy as np

from risklab.risk.unified_score import UnifiedSafetyScore, CategoryScore


class SafetyGauge:
    """
    Circular gauge visualization showing USS with color zones.
    """
    
    def __init__(
        self,
        score: float,
        grade: str,
        confidence_interval: Tuple[float, float],
        history: Optional[List[float]] = None,
    ):
        self.score = score
        self.grade = grade
        self.confidence_interval = confidence_interval
        self.history = history or []
    
    def to_plotly_figure(self) -> Dict[str, Any]:
        """Generate Plotly figure configuration."""
        return {
            "data": [{
                "type": "indicator",
                "mode": "gauge+number+delta",
                "value": self.score,
                "title": {"text": f"Safety Score<br><span style='font-size:0.6em'>Grade: {self.grade}</span>"},
                "delta": {
                    "reference": self.history[-1] if self.history else self.score,
                    "increasing": {"color": "#10b981"},
                    "decreasing": {"color": "#ef4444"},
                },
                "gauge": {
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": self._get_score_color()},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "steps": [
                        {"range": [0, 60], "color": "#fecaca"},
                        {"range": [60, 70], "color": "#fef3c7"},
                        {"range": [70, 80], "color": "#d1fae5"},
                        {"range": [80, 100], "color": "#a7f3d0"},
                    ],
                    "threshold": {
                        "line": {"color": "#1f2937", "width": 4},
                        "thickness": 0.75,
                        "value": self.score,
                    },
                },
            }],
            "layout": {
                "width": 400,
                "height": 300,
                "margin": {"t": 50, "b": 50, "l": 50, "r": 50},
                "paper_bgcolor": "transparent",
                "font": {"color": "#374151"},
            },
        }
    
    def _get_score_color(self) -> str:
        if self.score >= 80:
            return "#10b981"
        elif self.score >= 70:
            return "#f59e0b"
        elif self.score >= 60:
            return "#f97316"
        else:
            return "#ef4444"
    
    def to_html(self) -> str:
        """Generate standalone HTML with embedded gauge."""
        fig = self.to_plotly_figure()
        
        return f'''
<div id="safety-gauge"></div>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<script>
Plotly.newPlot('safety-gauge', {json.dumps(fig['data'])}, {json.dumps(fig['layout'])});
</script>
'''


class MetricWaterfall:
    """
    Waterfall chart showing how each metric contributes to final score.
    """
    
    def __init__(self, uss: UnifiedSafetyScore):
        self.uss = uss
    
    def get_waterfall_data(self) -> List[Dict[str, Any]]:
        """Calculate waterfall data from USS."""
        data = []
        
        # Start from 100 (perfect score)
        running_total = 100.0
        
        # Add category contributions
        categories = [
            ("Safety", self.uss.safety_score),
            ("Integrity", self.uss.integrity_score),
            ("Reliability", self.uss.reliability_score),
            ("Alignment", self.uss.alignment_score),
        ]
        
        for name, cat_score in categories:
            # Calculate impact: how much this category deducts from perfect
            expected_contribution = cat_score.weight * 100
            actual_contribution = cat_score.score * cat_score.weight
            impact = actual_contribution - expected_contribution
            
            data.append({
                "name": name,
                "contribution": impact,
                "running_total": running_total + impact,
                "category_score": cat_score.score,
                "weight": cat_score.weight,
            })
            running_total += impact
        
        return data
    
    def to_plotly_figure(self) -> Dict[str, Any]:
        """Generate Plotly waterfall figure."""
        data = self.get_waterfall_data()
        
        x_labels = ["Start"] + [d["name"] for d in data] + ["Final"]
        y_values = [100] + [d["contribution"] for d in data] + [0]
        
        # Calculate measure types (relative vs total)
        measures = ["total"] + ["relative"] * len(data) + ["total"]
        
        # Text labels
        text = [f"{100:.0f}"] + [f"{d['contribution']:+.1f}" for d in data] + [f"{self.uss.score:.0f}"]
        
        # Colors based on positive/negative
        colors = ["#6b7280"]  # Start
        for d in data:
            if d["contribution"] >= 0:
                colors.append("#10b981")  # Green for positive
            else:
                colors.append("#ef4444")  # Red for negative
        colors.append("#3b82f6")  # Final
        
        return {
            "data": [{
                "type": "waterfall",
                "orientation": "v",
                "measure": measures,
                "x": x_labels,
                "y": y_values,
                "text": text,
                "textposition": "outside",
                "connector": {"line": {"color": "rgb(63, 63, 63)"}},
                "decreasing": {"marker": {"color": "#ef4444"}},
                "increasing": {"marker": {"color": "#10b981"}},
                "totals": {"marker": {"color": "#3b82f6"}},
            }],
            "layout": {
                "title": "Score Breakdown by Category",
                "showlegend": False,
                "xaxis": {"title": "Category"},
                "yaxis": {"title": "Score Impact", "range": [0, 110]},
                "paper_bgcolor": "transparent",
                "plot_bgcolor": "transparent",
            },
        }


class VulnerabilitySunburst:
    """
    Hierarchical sunburst showing Domain → Stakes → Episode → Metric.
    """
    
    def __init__(self, episodes: List[Dict[str, Any]]):
        self.episodes = episodes
    
    def build_hierarchy(self) -> Dict[str, Any]:
        """Build hierarchical data structure."""
        # Group by domain -> stakes -> episode
        hierarchy = {}
        
        for ep in self.episodes:
            domain = ep.get("domain", "general")
            stakes = ep.get("stakes_level", "medium")
            name = ep.get("episode_name", "unknown")
            risk = ep.get("risk_score", 0)
            
            if domain not in hierarchy:
                hierarchy[domain] = {}
            if stakes not in hierarchy[domain]:
                hierarchy[domain][stakes] = []
            
            hierarchy[domain][stakes].append({
                "name": name,
                "risk": risk,
            })
        
        return hierarchy
    
    def to_plotly_figure(self) -> Dict[str, Any]:
        """Generate Plotly sunburst figure."""
        hierarchy = self.build_hierarchy()
        
        ids = []
        labels = []
        parents = []
        values = []
        colors = []
        
        # Root
        ids.append("root")
        labels.append("All")
        parents.append("")
        values.append(0)
        colors.append("#6b7280")
        
        for domain, stakes_data in hierarchy.items():
            # Domain level
            domain_id = domain
            ids.append(domain_id)
            labels.append(domain.capitalize())
            parents.append("root")
            
            domain_risks = []
            
            for stakes, episodes in stakes_data.items():
                # Stakes level
                stakes_id = f"{domain}-{stakes}"
                ids.append(stakes_id)
                labels.append(stakes.capitalize())
                parents.append(domain_id)
                
                stakes_risk = np.mean([e["risk"] for e in episodes])
                values.append(len(episodes))
                colors.append(self._risk_color(stakes_risk))
                domain_risks.append(stakes_risk)
                
                for ep in episodes:
                    # Episode level
                    ep_id = f"{stakes_id}-{ep['name'][:20]}"
                    ids.append(ep_id)
                    labels.append(ep["name"][:15])
                    parents.append(stakes_id)
                    values.append(1)
                    colors.append(self._risk_color(ep["risk"]))
            
            domain_avg = np.mean(domain_risks) if domain_risks else 0
            values[ids.index(domain_id)] = len([e for s in stakes_data.values() for e in s])
            colors[ids.index(domain_id)] = self._risk_color(domain_avg)
        
        return {
            "data": [{
                "type": "sunburst",
                "ids": ids,
                "labels": labels,
                "parents": parents,
                "values": values,
                "marker": {"colors": colors},
                "branchvalues": "total",
            }],
            "layout": {
                "title": "Vulnerability Hierarchy",
                "margin": {"t": 50, "b": 0, "l": 0, "r": 0},
            },
        }
    
    def _risk_color(self, risk: float) -> str:
        if risk < 0.3:
            return "#10b981"
        elif risk < 0.5:
            return "#f59e0b"
        elif risk < 0.7:
            return "#f97316"
        else:
            return "#ef4444"


class TemporalDriftPlot:
    """
    Line chart tracking metric stability over conversation turns.
    """
    
    def __init__(self, turn_data: List[Dict[str, float]]):
        """
        Args:
            turn_data: List of dicts with metric names as keys
        """
        self.turn_data = turn_data
    
    def to_plotly_figure(self) -> Dict[str, Any]:
        """Generate Plotly multi-line figure."""
        if not self.turn_data:
            return {"data": [], "layout": {"title": "No data"}}
        
        # Get all metric names
        metrics = list(self.turn_data[0].keys())
        turns = list(range(1, len(self.turn_data) + 1))
        
        traces = []
        colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
        
        for i, metric in enumerate(metrics[:5]):  # Limit to 5 metrics
            values = [td.get(metric, 0) for td in self.turn_data]
            traces.append({
                "type": "scatter",
                "mode": "lines+markers",
                "name": metric,
                "x": turns,
                "y": values,
                "line": {"color": colors[i % len(colors)]},
            })
        
        return {
            "data": traces,
            "layout": {
                "title": "Metric Drift Over Conversation",
                "xaxis": {"title": "Turn Number"},
                "yaxis": {"title": "Metric Value", "range": [0, 1]},
                "legend": {"x": 1.02, "y": 1},
                "hovermode": "x unified",
            },
        }


class ConfidenceFunnel:
    """
    Funnel chart showing confidence narrowing through evaluation pipeline.
    """
    
    def __init__(
        self,
        ml_confidence: float,
        rule_confidence: float,
        llm_confidence: float,
        council_confidence: float,
    ):
        self.stages = [
            ("ML Classifiers", ml_confidence),
            ("Rule-Based Analysis", rule_confidence),
            ("LLM Evaluation", llm_confidence),
            ("Council Consensus", council_confidence),
        ]
    
    def to_plotly_figure(self) -> Dict[str, Any]:
        """Generate Plotly funnel figure."""
        return {
            "data": [{
                "type": "funnel",
                "y": [s[0] for s in self.stages],
                "x": [s[1] * 100 for s in self.stages],
                "textinfo": "value+percent initial",
                "marker": {
                    "color": ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"],
                },
            }],
            "layout": {
                "title": "Confidence Through Evaluation Pipeline",
                "funnelmode": "stack",
            },
        }


class AttackSurfaceMap:
    """
    Network graph showing vulnerability clusters and attack paths.
    """
    
    def __init__(self, vulnerabilities: List[Dict[str, Any]]):
        """
        Args:
            vulnerabilities: List of dicts with 'name', 'category', 'severity', 'connections'
        """
        self.vulnerabilities = vulnerabilities
    
    def to_plotly_figure(self) -> Dict[str, Any]:
        """Generate Plotly network graph."""
        # Build node positions (circular layout)
        n = len(self.vulnerabilities)
        if n == 0:
            return {"data": [], "layout": {"title": "No vulnerabilities"}}
        
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        x_pos = np.cos(angles)
        y_pos = np.sin(angles)
        
        # Node sizes based on severity
        severity_sizes = {"critical": 40, "high": 30, "medium": 20, "low": 15}
        sizes = [severity_sizes.get(v.get("severity", "medium"), 20) for v in self.vulnerabilities]
        
        # Node colors based on category
        category_colors = {
            "safety": "#ef4444",
            "integrity": "#f59e0b",
            "reliability": "#3b82f6",
            "alignment": "#8b5cf6",
        }
        colors = [category_colors.get(v.get("category", ""), "#6b7280") for v in self.vulnerabilities]
        
        # Build edges
        edge_x = []
        edge_y = []
        
        for i, v in enumerate(self.vulnerabilities):
            for conn_name in v.get("connections", []):
                # Find connected node
                for j, v2 in enumerate(self.vulnerabilities):
                    if v2.get("name") == conn_name:
                        edge_x.extend([x_pos[i], x_pos[j], None])
                        edge_y.extend([y_pos[i], y_pos[j], None])
                        break
        
        return {
            "data": [
                # Edges
                {
                    "type": "scatter",
                    "x": edge_x,
                    "y": edge_y,
                    "mode": "lines",
                    "line": {"width": 1, "color": "#9ca3af"},
                    "hoverinfo": "none",
                },
                # Nodes
                {
                    "type": "scatter",
                    "x": x_pos.tolist(),
                    "y": y_pos.tolist(),
                    "mode": "markers+text",
                    "marker": {
                        "size": sizes,
                        "color": colors,
                        "line": {"width": 2, "color": "white"},
                    },
                    "text": [v.get("name", "")[:10] for v in self.vulnerabilities],
                    "textposition": "top center",
                    "hoverinfo": "text",
                    "hovertext": [
                        f"{v.get('name')}<br>Category: {v.get('category')}<br>Severity: {v.get('severity')}"
                        for v in self.vulnerabilities
                    ],
                },
            ],
            "layout": {
                "title": "Attack Surface Map",
                "showlegend": False,
                "hovermode": "closest",
                "xaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
                "yaxis": {"showgrid": False, "zeroline": False, "showticklabels": False},
            },
        }


class ComparisonRadar:
    """
    Side-by-side radar charts for model comparison.
    """
    
    def __init__(
        self,
        model_a_name: str,
        model_a_scores: Dict[str, float],
        model_b_name: str,
        model_b_scores: Dict[str, float],
    ):
        self.model_a_name = model_a_name
        self.model_a_scores = model_a_scores
        self.model_b_name = model_b_name
        self.model_b_scores = model_b_scores
    
    def to_plotly_figure(self) -> Dict[str, Any]:
        """Generate overlaid radar chart."""
        categories = list(self.model_a_scores.keys())
        
        # Close the radar by repeating first value
        values_a = [self.model_a_scores[c] for c in categories] + [self.model_a_scores[categories[0]]]
        values_b = [self.model_b_scores[c] for c in categories] + [self.model_b_scores[categories[0]]]
        categories = categories + [categories[0]]
        
        return {
            "data": [
                {
                    "type": "scatterpolar",
                    "r": values_a,
                    "theta": [c.capitalize() for c in categories],
                    "fill": "toself",
                    "fillcolor": "rgba(59, 130, 246, 0.2)",
                    "line": {"color": "#3b82f6"},
                    "name": self.model_a_name,
                },
                {
                    "type": "scatterpolar",
                    "r": values_b,
                    "theta": [c.capitalize() for c in categories],
                    "fill": "toself",
                    "fillcolor": "rgba(239, 68, 68, 0.2)",
                    "line": {"color": "#ef4444"},
                    "name": self.model_b_name,
                },
            ],
            "layout": {
                "title": f"{self.model_a_name} vs {self.model_b_name}",
                "polar": {
                    "radialaxis": {"visible": True, "range": [0, 100]},
                },
                "showlegend": True,
                "legend": {"x": 1.1, "y": 1},
            },
        }


def generate_all_advanced_plots(
    uss: UnifiedSafetyScore,
    episodes: List[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Generate all advanced visualization plots.
    
    Returns dict of plot name to file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plots = {}
    
    # Safety Gauge
    gauge = SafetyGauge(
        score=uss.score,
        grade=uss.grade.value,
        confidence_interval=uss.confidence_interval,
    )
    gauge_path = output_dir / "safety_gauge.json"
    gauge_path.write_text(json.dumps(gauge.to_plotly_figure(), indent=2))
    plots["safety_gauge"] = gauge_path
    
    # Metric Waterfall
    waterfall = MetricWaterfall(uss)
    waterfall_path = output_dir / "metric_waterfall.json"
    waterfall_path.write_text(json.dumps(waterfall.to_plotly_figure(), indent=2))
    plots["metric_waterfall"] = waterfall_path
    
    # Vulnerability Sunburst
    sunburst = VulnerabilitySunburst(episodes)
    sunburst_path = output_dir / "vulnerability_sunburst.json"
    sunburst_path.write_text(json.dumps(sunburst.to_plotly_figure(), indent=2))
    plots["vulnerability_sunburst"] = sunburst_path
    
    # Confidence Funnel (with placeholder values)
    funnel = ConfidenceFunnel(
        ml_confidence=0.7,
        rule_confidence=0.75,
        llm_confidence=0.85,
        council_confidence=uss.confidence_interval[1] / 100,
    )
    funnel_path = output_dir / "confidence_funnel.json"
    funnel_path.write_text(json.dumps(funnel.to_plotly_figure(), indent=2))
    plots["confidence_funnel"] = funnel_path
    
    return plots
