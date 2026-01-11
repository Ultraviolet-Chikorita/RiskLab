"""
Unified Dashboard System for RiskLab AI Evaluation.

Provides:
- Single-pane-of-glass dashboard
- Interactive visualizations
- Cross-filtering and drill-down capabilities
- Model comparison views
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
import json
import numpy as np

from risklab.risk.unified_score import UnifiedSafetyScore, CategoryScore, ScoreCategory


class EpisodeSummary(BaseModel):
    """Summary of a single episode evaluation."""
    episode_id: str
    episode_name: str
    domain: str
    stakes_level: str
    risk_score: float
    outcome: str
    top_concerns: List[str] = Field(default_factory=list)
    framing_used: str = "neutral"


class SignalSummary(BaseModel):
    """Summary of a manipulation signal."""
    signal_name: str
    value: float
    severity: str
    episode_id: str
    evidence: List[str] = Field(default_factory=list)


class TimelinePoint(BaseModel):
    """Historical data point for trend visualization."""
    timestamp: datetime
    uss_score: float
    grade: str
    model_version: Optional[str] = None
    evaluation_id: Optional[str] = None
    episode_count: int = 0


class WaterfallItem(BaseModel):
    """Item for metric waterfall visualization."""
    metric_name: str
    category: str
    contribution: float  # Positive or negative contribution to score
    cumulative: float    # Running total
    is_positive: bool
    raw_value: float


class RadarDataPoint(BaseModel):
    """Data point for radar/spider chart."""
    category: str
    score: float
    benchmark: Optional[float] = None  # Optional comparison value


class HeatmapCell(BaseModel):
    """Cell data for heatmap visualization."""
    row_label: str
    col_label: str
    value: float
    count: int = 1
    episode_ids: List[str] = Field(default_factory=list)


class ComparisonData(BaseModel):
    """Data for model comparison."""
    model_a_name: str
    model_b_name: str
    model_a_uss: UnifiedSafetyScore
    model_b_uss: UnifiedSafetyScore
    category_deltas: Dict[str, float] = Field(default_factory=dict)
    significant_differences: List[str] = Field(default_factory=list)
    
    def compute_deltas(self) -> None:
        """Compute differences between models."""
        a_cats = self.model_a_uss.get_category_scores()
        b_cats = self.model_b_uss.get_category_scores()
        
        for cat in a_cats:
            delta = a_cats[cat] - b_cats[cat]
            self.category_deltas[cat] = delta
            if abs(delta) > 10:
                winner = self.model_a_name if delta > 0 else self.model_b_name
                self.significant_differences.append(
                    f"{cat.capitalize()}: {winner} better by {abs(delta):.1f} points"
                )


class DashboardData(BaseModel):
    """
    Complete data package for the unified dashboard.
    
    Contains all data needed to render the full dashboard with
    cross-filtering and drill-down capabilities.
    """
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    model_identifier: str = "unknown"
    evaluation_id: str = ""
    
    # Top-level scores
    uss: UnifiedSafetyScore
    
    # Visualization data
    radar_data: List[RadarDataPoint] = Field(default_factory=list)
    waterfall_data: List[WaterfallItem] = Field(default_factory=list)
    heatmap_data: List[HeatmapCell] = Field(default_factory=list)
    timeline_data: List[TimelinePoint] = Field(default_factory=list)
    
    # Issue tracking
    high_risk_episodes: List[EpisodeSummary] = Field(default_factory=list)
    critical_signals: List[SignalSummary] = Field(default_factory=list)
    
    # Comparison data (optional)
    comparison: Optional[ComparisonData] = None
    
    # Drill-down data
    episodes_by_domain: Dict[str, List[EpisodeSummary]] = Field(default_factory=dict)
    episodes_by_stakes: Dict[str, List[EpisodeSummary]] = Field(default_factory=dict)
    
    # Statistics
    total_episodes: int = 0
    pass_rate: float = 0.0
    avg_confidence: float = 0.0
    
    def add_episode(self, episode: EpisodeSummary) -> None:
        """Add an episode and update indices."""
        if episode.risk_score > 0.5:
            self.high_risk_episodes.append(episode)
        
        # Index by domain
        if episode.domain not in self.episodes_by_domain:
            self.episodes_by_domain[episode.domain] = []
        self.episodes_by_domain[episode.domain].append(episode)
        
        # Index by stakes
        if episode.stakes_level not in self.episodes_by_stakes:
            self.episodes_by_stakes[episode.stakes_level] = []
        self.episodes_by_stakes[episode.stakes_level].append(episode)
        
        self.total_episodes += 1
    
    def build_radar_data(self) -> None:
        """Build radar chart data from USS."""
        self.radar_data = [
            RadarDataPoint(category="Safety", score=self.uss.safety_score.score),
            RadarDataPoint(category="Integrity", score=self.uss.integrity_score.score),
            RadarDataPoint(category="Reliability", score=self.uss.reliability_score.score),
            RadarDataPoint(category="Alignment", score=self.uss.alignment_score.score),
        ]
    
    def build_waterfall_data(self) -> None:
        """Build waterfall chart data from metric contributions."""
        items = []
        cumulative = 100.0  # Start from max
        
        # Sort by contribution magnitude
        sorted_contribs = sorted(
            self.uss.metric_contributions.items(),
            key=lambda x: abs(x[1].contribution_to_uss),
            reverse=True
        )
        
        for metric_name, contrib in sorted_contribs[:15]:  # Top 15
            # Contribution is how much it reduces from perfect score
            impact = (100 - contrib.weighted_value) * (self.uss.get_category_scores().get(
                contrib.category.value, 25) / 100)
            
            is_positive = contrib.weighted_value >= 75
            cumulative -= impact if not is_positive else 0
            
            items.append(WaterfallItem(
                metric_name=metric_name,
                category=contrib.category.value,
                contribution=-impact if not is_positive else impact * 0.1,
                cumulative=cumulative,
                is_positive=is_positive,
                raw_value=contrib.raw_value,
            ))
        
        self.waterfall_data = items
    
    def build_heatmap_data(self) -> None:
        """Build domain×stakes heatmap data."""
        cells = []
        
        for domain, episodes in self.episodes_by_domain.items():
            stakes_groups: Dict[str, List[EpisodeSummary]] = {}
            for ep in episodes:
                if ep.stakes_level not in stakes_groups:
                    stakes_groups[ep.stakes_level] = []
                stakes_groups[ep.stakes_level].append(ep)
            
            for stakes, eps in stakes_groups.items():
                avg_risk = np.mean([e.risk_score for e in eps])
                cells.append(HeatmapCell(
                    row_label=domain,
                    col_label=stakes,
                    value=avg_risk,
                    count=len(eps),
                    episode_ids=[e.episode_id for e in eps],
                ))
        
        self.heatmap_data = cells
    
    def compute_statistics(self) -> None:
        """Compute aggregate statistics."""
        if self.high_risk_episodes:
            all_episodes = []
            for eps in self.episodes_by_domain.values():
                all_episodes.extend(eps)
            
            passing = sum(1 for e in all_episodes if e.risk_score < 0.5)
            self.pass_rate = passing / len(all_episodes) if all_episodes else 1.0
    
    def to_plotly_json(self) -> str:
        """Export as Plotly-compatible JSON for frontend rendering."""
        return json.dumps(self.model_dump(), default=str, indent=2)
    
    def to_react_props(self) -> Dict[str, Any]:
        """Export as React component props."""
        return {
            'uss': {
                'score': self.uss.score,
                'grade': self.uss.grade.value,
                'confidence': self.uss.confidence_interval,
            },
            'categories': self.uss.get_category_scores(),
            'radar': [{'category': r.category, 'score': r.score} for r in self.radar_data],
            'waterfall': [w.model_dump() for w in self.waterfall_data],
            'heatmap': [h.model_dump() for h in self.heatmap_data],
            'timeline': [{'date': t.timestamp.isoformat(), 'score': t.uss_score} for t in self.timeline_data],
            'issues': [e.model_dump() for e in self.high_risk_episodes[:20]],
            'signals': [s.model_dump() for s in self.critical_signals[:10]],
            'stats': {
                'totalEpisodes': self.total_episodes,
                'passRate': self.pass_rate,
                'avgConfidence': self.avg_confidence,
            },
        }


class DashboardBuilder:
    """
    Builds dashboard data from evaluation results.
    """
    
    def __init__(self):
        self.timeline_history: List[TimelinePoint] = []
    
    def build(
        self,
        uss: UnifiedSafetyScore,
        episodes: List[Dict[str, Any]],
        model_identifier: str = "unknown",
        evaluation_id: str = "",
        comparison_uss: Optional[UnifiedSafetyScore] = None,
        comparison_model: Optional[str] = None,
    ) -> DashboardData:
        """
        Build complete dashboard data from evaluation results.
        
        Args:
            uss: Computed Unified Safety Score
            episodes: List of episode evaluation results
            model_identifier: Name/version of evaluated model
            evaluation_id: Unique ID for this evaluation
            comparison_uss: Optional USS for comparison model
            comparison_model: Name of comparison model
        """
        dashboard = DashboardData(
            uss=uss,
            model_identifier=model_identifier,
            evaluation_id=evaluation_id,
        )
        
        # Process episodes
        for ep_data in episodes:
            episode = EpisodeSummary(
                episode_id=ep_data.get('episode_id', ''),
                episode_name=ep_data.get('episode_name', ep_data.get('name', '')),
                domain=ep_data.get('domain', ep_data.get('context', {}).get('domain', 'general')),
                stakes_level=ep_data.get('stakes_level', ep_data.get('context', {}).get('stakes_level', 'medium')),
                risk_score=ep_data.get('risk_score', ep_data.get('full_evaluation', {}).get('decision', {}).get('score', 0)),
                outcome=ep_data.get('outcome', ep_data.get('full_evaluation', {}).get('decision', {}).get('outcome', 'unknown')),
                top_concerns=ep_data.get('concerns', [])[:5],
                framing_used=ep_data.get('framing', 'neutral'),
            )
            dashboard.add_episode(episode)
            
            # Extract signals if present
            signals = ep_data.get('signals', {})
            for signal_name, signal_data in signals.items():
                if isinstance(signal_data, dict) and signal_data.get('severity') in ['high', 'critical']:
                    dashboard.critical_signals.append(SignalSummary(
                        signal_name=signal_name,
                        value=signal_data.get('value', 0),
                        severity=signal_data.get('severity', 'unknown'),
                        episode_id=episode.episode_id,
                        evidence=signal_data.get('evidence', [])[:3],
                    ))
        
        # Build visualization data
        dashboard.build_radar_data()
        dashboard.build_waterfall_data()
        dashboard.build_heatmap_data()
        dashboard.compute_statistics()
        
        # Add to timeline
        self.timeline_history.append(TimelinePoint(
            timestamp=datetime.utcnow(),
            uss_score=uss.score,
            grade=uss.grade.value,
            model_version=model_identifier,
            evaluation_id=evaluation_id,
            episode_count=dashboard.total_episodes,
        ))
        dashboard.timeline_data = self.timeline_history[-50:]  # Keep last 50
        
        # Add comparison if provided
        if comparison_uss and comparison_model:
            comparison = ComparisonData(
                model_a_name=model_identifier,
                model_b_name=comparison_model,
                model_a_uss=uss,
                model_b_uss=comparison_uss,
            )
            comparison.compute_deltas()
            dashboard.comparison = comparison
        
        return dashboard


class DashboardRenderer:
    """
    Renders dashboard data to various output formats.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def render_html(self, dashboard: DashboardData, filename: str = "dashboard.html") -> Path:
        """Render interactive HTML dashboard."""
        
        props = dashboard.to_react_props()
        props_json = json.dumps(props, default=str)
        
        html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RiskLab Safety Dashboard - {dashboard.model_identifier}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
        }}
        .dashboard {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 1px solid #333;
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2.5em;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        .score-display {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 40px;
            margin: 30px 0;
        }}
        .main-score {{
            width: 200px;
            height: 200px;
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: conic-gradient(
                {'#10b981' if dashboard.uss.score >= 80 else '#f59e0b' if dashboard.uss.score >= 60 else '#ef4444'} 
                {dashboard.uss.score * 3.6}deg, 
                #333 0deg
            );
            position: relative;
        }}
        .main-score::before {{
            content: '';
            position: absolute;
            width: 160px;
            height: 160px;
            border-radius: 50%;
            background: #1a1a2e;
        }}
        .main-score .score-value {{
            position: relative;
            z-index: 1;
            font-size: 3em;
            font-weight: bold;
        }}
        .main-score .score-grade {{
            position: relative;
            z-index: 1;
            font-size: 1.5em;
            color: #888;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }}
        .card {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .card h3 {{
            font-size: 1.1em;
            color: #888;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .category-bars {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        .category-bar {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .category-label {{
            width: 100px;
            font-size: 0.9em;
        }}
        .bar-container {{
            flex: 1;
            height: 24px;
            background: #333;
            border-radius: 12px;
            overflow: hidden;
        }}
        .bar-fill {{
            height: 100%;
            border-radius: 12px;
            transition: width 0.5s ease;
        }}
        .bar-value {{
            width: 50px;
            text-align: right;
            font-weight: bold;
        }}
        .issues-list {{
            max-height: 300px;
            overflow-y: auto;
        }}
        .issue-item {{
            padding: 12px;
            margin-bottom: 8px;
            background: rgba(239, 68, 68, 0.1);
            border-left: 3px solid #ef4444;
            border-radius: 0 8px 8px 0;
        }}
        .issue-item.warning {{
            background: rgba(245, 158, 11, 0.1);
            border-left-color: #f59e0b;
        }}
        .issue-name {{
            font-weight: bold;
            margin-bottom: 4px;
        }}
        .issue-meta {{
            font-size: 0.85em;
            color: #888;
        }}
        .chart-container {{
            height: 300px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }}
        .stat-box {{
            text-align: center;
            padding: 15px;
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
        }}
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #00d4ff;
        }}
        .stat-label {{
            font-size: 0.8em;
            color: #888;
            margin-top: 5px;
        }}
        .confidence-bar {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-top: 10px;
        }}
        .confidence-range {{
            flex: 1;
            height: 8px;
            background: #333;
            border-radius: 4px;
            position: relative;
        }}
        .confidence-fill {{
            position: absolute;
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            border-radius: 4px;
        }}
        @media (max-width: 900px) {{
            .grid {{ grid-template-columns: 1fr; }}
            .score-display {{ flex-direction: column; }}
        }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>🛡️ RiskLab Safety Dashboard</h1>
            <p>Model: {dashboard.model_identifier} | Generated: {dashboard.generated_at.strftime('%Y-%m-%d %H:%M UTC')}</p>
        </div>
        
        <div class="score-display">
            <div class="main-score">
                <div class="score-value">{dashboard.uss.score:.0f}</div>
                <div class="score-grade">{dashboard.uss.grade.value}</div>
            </div>
            <div class="score-details">
                <h2>Unified Safety Score</h2>
                <div class="confidence-bar">
                    <span>{dashboard.uss.confidence_interval[0]:.0f}</span>
                    <div class="confidence-range">
                        <div class="confidence-fill" style="left: {dashboard.uss.confidence_interval[0]}%; width: {dashboard.uss.confidence_interval[1] - dashboard.uss.confidence_interval[0]}%;"></div>
                    </div>
                    <span>{dashboard.uss.confidence_interval[1]:.0f}</span>
                </div>
                <p style="color: #888; margin-top: 15px;">95% Confidence Interval</p>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>Category Breakdown</h3>
                <div class="category-bars">
                    <div class="category-bar">
                        <span class="category-label">Safety</span>
                        <div class="bar-container">
                            <div class="bar-fill" style="width: {dashboard.uss.safety_score.score}%; background: {'#10b981' if dashboard.uss.safety_score.score >= 80 else '#f59e0b' if dashboard.uss.safety_score.score >= 60 else '#ef4444'};"></div>
                        </div>
                        <span class="bar-value">{dashboard.uss.safety_score.score:.0f}</span>
                    </div>
                    <div class="category-bar">
                        <span class="category-label">Integrity</span>
                        <div class="bar-container">
                            <div class="bar-fill" style="width: {dashboard.uss.integrity_score.score}%; background: {'#10b981' if dashboard.uss.integrity_score.score >= 80 else '#f59e0b' if dashboard.uss.integrity_score.score >= 60 else '#ef4444'};"></div>
                        </div>
                        <span class="bar-value">{dashboard.uss.integrity_score.score:.0f}</span>
                    </div>
                    <div class="category-bar">
                        <span class="category-label">Reliability</span>
                        <div class="bar-container">
                            <div class="bar-fill" style="width: {dashboard.uss.reliability_score.score}%; background: {'#10b981' if dashboard.uss.reliability_score.score >= 80 else '#f59e0b' if dashboard.uss.reliability_score.score >= 60 else '#ef4444'};"></div>
                        </div>
                        <span class="bar-value">{dashboard.uss.reliability_score.score:.0f}</span>
                    </div>
                    <div class="category-bar">
                        <span class="category-label">Alignment</span>
                        <div class="bar-container">
                            <div class="bar-fill" style="width: {dashboard.uss.alignment_score.score}%; background: {'#10b981' if dashboard.uss.alignment_score.score >= 80 else '#f59e0b' if dashboard.uss.alignment_score.score >= 60 else '#ef4444'};"></div>
                        </div>
                        <span class="bar-value">{dashboard.uss.alignment_score.score:.0f}</span>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3>Statistics</h3>
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-value">{dashboard.total_episodes}</div>
                        <div class="stat-label">Episodes</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{dashboard.pass_rate*100:.0f}%</div>
                        <div class="stat-label">Pass Rate</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{len(dashboard.high_risk_episodes)}</div>
                        <div class="stat-label">High Risk</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3>Radar Chart</h3>
                <div id="radar-chart" class="chart-container"></div>
            </div>
            
            <div class="card">
                <h3>Domain Risk Heatmap</h3>
                <div id="heatmap-chart" class="chart-container"></div>
            </div>
            
            <div class="card" style="grid-column: span 2;">
                <h3>Top Concerns</h3>
                <div class="issues-list">
                    {''.join("""
                    <div class="issue-item {'warning' if ep.risk_score < 0.7 else ''}">
                        <div class="issue-name">{ep.episode_name[:50]}</div>
                        <div class="issue-meta">Domain: {ep.domain} | Stakes: {ep.stakes_level} | Risk: {round(ep.risk_score, 2)}</div>
                    </div>
                    """ for ep in sorted(dashboard.high_risk_episodes, key=lambda x: x.risk_score, reverse=True)[:10])}
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const data = {props_json};
        
        // Radar chart
        Plotly.newPlot('radar-chart', [{{
            type: 'scatterpolar',
            r: data.radar.map(d => d.score),
            theta: data.radar.map(d => d.category),
            fill: 'toself',
            fillcolor: 'rgba(0, 212, 255, 0.2)',
            line: {{ color: '#00d4ff' }},
            marker: {{ size: 8, color: '#00d4ff' }}
        }}], {{
            polar: {{
                radialaxis: {{ visible: true, range: [0, 100], tickfont: {{ color: '#888' }} }},
                angularaxis: {{ tickfont: {{ color: '#888' }} }},
                bgcolor: 'transparent'
            }},
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            margin: {{ t: 30, b: 30, l: 60, r: 60 }},
            showlegend: false
        }}, {{ responsive: true }});
        
        // Heatmap
        const domains = [...new Set(data.heatmap.map(h => h.row_label))];
        const stakes = ['low', 'medium', 'high', 'critical'];
        const heatmapValues = domains.map(d => 
            stakes.map(s => {{
                const cell = data.heatmap.find(h => h.row_label === d && h.col_label === s);
                return cell ? cell.value : null;
            }})
        );
        
        Plotly.newPlot('heatmap-chart', [{{
            type: 'heatmap',
            z: heatmapValues,
            x: stakes,
            y: domains,
            colorscale: [[0, '#10b981'], [0.5, '#f59e0b'], [1, '#ef4444']],
            showscale: true,
            colorbar: {{ tickfont: {{ color: '#888' }} }}
        }}], {{
            paper_bgcolor: 'transparent',
            plot_bgcolor: 'transparent',
            margin: {{ t: 30, b: 50, l: 100, r: 30 }},
            xaxis: {{ tickfont: {{ color: '#888' }} }},
            yaxis: {{ tickfont: {{ color: '#888' }} }}
        }}, {{ responsive: true }});
    </script>
</body>
</html>'''
        
        path = self.output_dir / filename
        path.write_text(html, encoding='utf-8')
        return path
    
    def render_json(self, dashboard: DashboardData, filename: str = "dashboard_data.json") -> Path:
        """Export dashboard data as JSON."""
        path = self.output_dir / filename
        path.write_text(dashboard.to_plotly_json(), encoding='utf-8')
        return path


def build_dashboard(
    uss: UnifiedSafetyScore,
    episodes: List[Dict[str, Any]],
    model_identifier: str = "unknown",
    output_dir: Optional[Path] = None,
) -> Tuple[DashboardData, Optional[Path]]:
    """
    Convenience function to build and optionally render dashboard.
    
    Returns dashboard data and path to HTML file if output_dir provided.
    """
    builder = DashboardBuilder()
    dashboard = builder.build(uss, episodes, model_identifier)
    
    html_path = None
    if output_dir:
        renderer = DashboardRenderer(output_dir)
        html_path = renderer.render_html(dashboard)
        renderer.render_json(dashboard)
    
    return dashboard, html_path
