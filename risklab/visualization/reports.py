"""
Report generation for the Risk-Conditioned AI Evaluation Lab.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
import json

from risklab.visualization.cards import ScenarioNarrativeCard, CardGenerator
from risklab.risk.aggregator import AggregatedRiskReport
from risklab.risk.thresholds import DecisionOutcome
from risklab.utils import get_risk_color, get_decision_outcome_color


class ReportGenerator:
    """
    Base report generator.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_json_report(
        self,
        data: Dict[str, Any],
        filename: str = "report.json",
    ) -> Path:
        """Generate JSON report."""
        path = self.output_dir / filename
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return path
    
    def generate_summary_text(
        self,
        report: AggregatedRiskReport,
        filename: str = "summary.txt",
    ) -> Path:
        """Generate plain text summary."""
        lines = [
            "=" * 70,
            "RISK-CONDITIONED AI EVALUATION LAB - SUMMARY REPORT",
            "=" * 70,
            "",
            f"Report ID: {report.report_id}",
            f"Model: {report.model_identifier or 'Unknown'}",
            f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "-" * 70,
            "OVERALL ASSESSMENT",
            "-" * 70,
            "",
            f"Risk Level: {report.overall_risk_level.upper()}",
            f"Deployment Recommendation: {report.deployment_recommendation.value.upper()}",
            f"Confidence Interval: {report.confidence_interval[0]:.2f} - {report.confidence_interval[1]:.2f}",
            "",
            "-" * 70,
            "STATISTICS",
            "-" * 70,
            "",
            f"Episodes Evaluated: {report.total_episodes}",
            f"Total Framings: {report.total_framings}",
            f"Mean Risk Score: {report.mean_risk_score:.3f}",
            f"Max Risk Score: {report.max_risk_score:.3f}",
            f"Standard Deviation: {report.std_risk_score:.3f}",
            "",
        ]
        
        # Outcome distribution
        lines.extend([
            "-" * 70,
            "OUTCOME DISTRIBUTION",
            "-" * 70,
            "",
        ])
        for outcome, count in report.outcome_distribution.items():
            pct = count / report.total_episodes * 100 if report.total_episodes > 0 else 0
            lines.append(f"  {outcome}: {count} ({pct:.1f}%)")
        lines.append("")
        
        # Risk by domain
        if report.risk_by_domain:
            lines.extend([
                "-" * 70,
                "RISK BY DOMAIN",
                "-" * 70,
                "",
            ])
            for domain, risk in sorted(report.risk_by_domain.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {domain}: {risk:.3f}")
            lines.append("")
        
        # Key findings
        if report.key_findings:
            lines.extend([
                "-" * 70,
                "KEY FINDINGS",
                "-" * 70,
                "",
            ])
            for finding in report.key_findings:
                lines.append(f"  • {finding}")
            lines.append("")
        
        # Recommendations
        if report.recommended_actions:
            lines.extend([
                "-" * 70,
                "RECOMMENDED ACTIONS",
                "-" * 70,
                "",
            ])
            for action in report.recommended_actions:
                lines.append(f"  • {action}")
            lines.append("")
        
        # Limitations
        if report.limitations:
            lines.extend([
                "-" * 70,
                "LIMITATIONS",
                "-" * 70,
                "",
            ])
            for limitation in report.limitations:
                lines.append(f"  • {limitation}")
            lines.append("")
        
        lines.append("=" * 70)
        
        path = self.output_dir / filename
        path.write_text("\n".join(lines), encoding='utf-8')
        return path


class HTMLReportGenerator(ReportGenerator):
    """
    Generate comprehensive HTML reports.
    """
    
    def generate_full_report(
        self,
        report: AggregatedRiskReport,
        cards: Optional[List[ScenarioNarrativeCard]] = None,
        plots: Optional[Dict[str, Path]] = None,
        filename: str = "report.html",
    ) -> Path:
        """Generate full HTML report with all components."""
        
        risk_color = self._get_risk_color(report.overall_risk_level)
        action_color = self._get_action_color(report.deployment_recommendation)
        
        # Build outcome distribution chart data
        outcome_data = json.dumps(report.outcome_distribution, default=str)
        domain_data = json.dumps(report.risk_by_domain, default=str)
        
        # Embed episode data if available
        episode_data = ""
        if hasattr(report, 'episode_details') and report.episode_details:
            episode_data = json.dumps(report.episode_details, default=str)
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Risk-Conditioned AI Evaluation Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50, #3498db);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 24px;
        }}
        .header h1 {{ margin: 0 0 10px 0; }}
        .header .meta {{ opacity: 0.9; font-size: 14px; }}
        .card {{
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .card h2 {{
            margin-top: 0;
            padding-bottom: 12px;
            border-bottom: 2px solid #eee;
            color: #2c3e50;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .metric-box {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 4px;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        .risk-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
        }}
        .finding-list, .action-list {{
            list-style: none;
            padding: 0;
        }}
        .finding-list li, .action-list li {{
            padding: 12px 16px;
            margin-bottom: 8px;
            background: #f8f9fa;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }}
        .action-list li {{ border-left-color: #27ae60; }}
        .limitation-list li {{ border-left-color: #95a5a6; }}
        .chart-container {{
            height: 300px;
            margin: 16px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Risk-Conditioned AI Evaluation Report</h1>
        <div class="meta">
            Report ID: {report.report_id} | 
            Model: {report.model_identifier or 'Unknown'} | 
            Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}
        </div>
    </div>
    
    <div class="card">
        <h2>Executive Summary</h2>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-value" style="color: {risk_color};">{report.overall_risk_level.upper()}</div>
                <div class="metric-label">Overall Risk Level</div>
            </div>
            <div class="metric-box">
                <div class="metric-value" style="color: {action_color};">{report.deployment_recommendation.value.upper()}</div>
                <div class="metric-label">Recommendation</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{report.mean_risk_score:.2f}</div>
                <div class="metric-label">Mean Risk Score</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{report.total_episodes}</div>
                <div class="metric-label">Episodes Evaluated</div>
            </div>
        </div>
    </div>
    
    <div class="card">
        <h2>Key Findings</h2>
        <ul class="finding-list">
            {''.join(f'<li>{f}</li>' for f in report.key_findings) if report.key_findings else '<li>No significant findings</li>'}
        </ul>
    </div>
    
    <div class="card">
        <h2>Outcome Distribution</h2>
        <div id="outcome-chart" class="chart-container"></div>
    </div>
    
    <div class="card">
        <h2>Risk by Domain</h2>
        <div id="domain-chart" class="chart-container"></div>
    </div>
    
    <div class="card">
        <h2>Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Episodes</td><td>{report.total_episodes}</td></tr>
            <tr><td>Total Framings</td><td>{report.total_framings}</td></tr>
            <tr><td>Mean Risk Score</td><td>{report.mean_risk_score:.4f}</td></tr>
            <tr><td>Median Risk Score</td><td>{report.median_risk_score:.4f}</td></tr>
            <tr><td>Max Risk Score</td><td>{report.max_risk_score:.4f}</td></tr>
            <tr><td>Standard Deviation</td><td>{report.std_risk_score:.4f}</td></tr>
            <tr><td>Confidence Interval (95%)</td><td>{report.confidence_interval[0]:.3f} - {report.confidence_interval[1]:.3f}</td></tr>
        </table>
    </div>
    
    <div class="card">
        <h2>Recommended Actions</h2>
        <ul class="action-list">
            {''.join(f'<li>{a}</li>' for a in report.recommended_actions) if report.recommended_actions else '<li>No specific actions required</li>'}
        </ul>
    </div>
    
    <div class="card">
        <h2>Episode Details</h2>
        <div id="episodes-table"></div>
    </div>
    
    <div class="card">
        <h2>Limitations</h2>
        <ul class="finding-list limitation-list">
            {''.join(f'<li>{l}</li>' for l in report.limitations) if report.limitations else '<li>Standard limitations apply</li>'}
        </ul>
    </div>
    
    <div class="footer">
        Risk-Conditioned AI Evaluation Lab | Report generated automatically
    </div>
    
    <script>
        // Outcome distribution chart
        var outcomeData = {outcome_data};
        var outcomes = Object.keys(outcomeData);
        var counts = Object.values(outcomeData);
        var colors = {{
            'acceptable': '#27ae60',
            'monitor': '#3498db',
            'mitigated': '#f39c12',
            'escalate': '#e67e22',
            'block': '#e74c3c'
        }};
        
        Plotly.newPlot('outcome-chart', [{{
            x: outcomes,
            y: counts,
            type: 'bar',
            marker: {{ color: outcomes.map(o => colors[o] || '#666') }}
        }}], {{
            margin: {{ t: 20, b: 40 }},
            yaxis: {{ title: 'Count' }}
        }});
        
        // Domain risk chart
        var domainData = {domain_data};
        var domains = Object.keys(domainData);
        var risks = Object.values(domainData);
        
        Plotly.newPlot('domain-chart', [{{
            x: risks,
            y: domains,
            type: 'bar',
            orientation: 'h',
            marker: {{ color: risks.map(r => r > 0.5 ? '#e74c3c' : r > 0.3 ? '#f39c12' : '#27ae60') }}
        }}], {{
            margin: {{ t: 20, l: 100, r: 20 }},
            xaxis: {{ title: 'Risk Score', range: [0, 1] }}
        }});
        
        // Episodes table
        const episodeData = {episode_data};
        if (episodeData && episodeData.length > 0) {{
            const episodes = episodeData;
            const tableData = episodes.map(ep => {{
                const ml = ep.full_evaluation.ml_classifiers || {{}};
                return {{
                    name: ep.episode_name,
                    domain: ep.context.domain,
                    stakes: ep.context.stakes_level,
                    risk: ep.full_evaluation.decision.score.toFixed(3),
                    outcome: ep.full_evaluation.decision.outcome,
                    sentiment: ml.sentiment?.label || 'N/A',
                    intent: ml.intent?.label || 'N/A',
                    toxicity: ml.toxicity?.label || 'N/A',
                    quality: ml.quality?.label || 'N/A'
                }};
            }});
            
            const columns = [
                {{name: 'Episode', field: 'name'}},
                {{name: 'Domain', field: 'domain'}},
                {{name: 'Stakes', field: 'stakes'}},
                {{name: 'Risk Score', field: 'risk'}},
                {{name: 'Outcome', field: 'outcome'}},
                {{name: 'Sentiment', field: 'sentiment'}},
                {{name: 'Intent', field: 'intent'}},
                {{name: 'Toxicity', field: 'toxicity'}},
                {{name: 'Quality', field: 'quality'}}
            ];
            
            // Create simple table
            let tableHTML = '<table><thead><tr>' + 
                columns.map(col => `<th>${{col.name}}</th>`).join('') + 
                '</tr></thead><tbody>';
            
            tableData.forEach(row => {{
                tableHTML += '<tr>' + columns.map(col => `<td>${{row[col.field]}}</td>`).join('') + '</tr>';
            }});
            
            tableHTML += '</tbody></table>';
            document.getElementById('episodes-table').innerHTML = tableHTML;
        }} else {{
            document.getElementById('episodes-table').innerHTML = '<p>Episode details not available</p>';
        }}
    </script>
</body>
</html>
        """
        
        path = self.output_dir / filename
        path.write_text(html, encoding='utf-8')
        return path
    
    def _get_risk_color(self, level: str) -> str:
        return get_risk_color(level)
    
    def _get_action_color(self, action: DecisionOutcome) -> str:
        return get_decision_outcome_color(action)


def generate_complete_report(
    report: AggregatedRiskReport,
    output_dir: Path,
    include_cards: bool = True,
    evaluations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Path]:
    """
    Generate a complete report package.
    
    Returns dict of generated file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generated = {}
    
    # JSON report
    gen = ReportGenerator(output_dir)
    generated['json'] = gen.generate_json_report(
        report.to_summary_dict(),
        "report.json"
    )
    
    # Detailed episode data JSON
    if hasattr(report, 'episode_details') and report.episode_details:
        episodes_json = output_dir / "episodes.json"
        with open(episodes_json, 'w', encoding='utf-8') as f:
            json.dump(report.episode_details, f, indent=2, ensure_ascii=False, default=str)
        generated['episodes'] = episodes_json
        
        # Generate interactive episode viewer
        from risklab.visualization.episode_viewer import generate_episode_viewer
        episode_viewer = generate_episode_viewer(output_dir, episodes_json)
        generated['episode_viewer'] = episode_viewer
    
    # Text summary
    generated['summary'] = gen.generate_summary_text(report, "summary.txt")
    
    # HTML report
    html_gen = HTMLReportGenerator(output_dir)
    generated['html'] = html_gen.generate_full_report(report, filename="report.html")
    
    # Scenario cards
    if include_cards and evaluations:
        card_gen = CardGenerator()
        cards = card_gen.generate_batch(evaluations)
        cards_path = output_dir / "scenario_cards.html"
        card_gen.export_cards_html(cards, cards_path, "Scenario Evaluation Cards")
        generated['cards'] = cards_path
    
    return generated