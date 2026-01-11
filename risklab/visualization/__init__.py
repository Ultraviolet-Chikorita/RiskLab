"""
Visualization and decision output for the Risk-Conditioned AI Evaluation Lab.
"""

from risklab.visualization.plots import (
    FramingDeltaHeatmap,
    RiskScoreLadder,
    OversightGapPlot,
    EvaluatorDisagreementMatrix,
)
from risklab.visualization.cards import ScenarioNarrativeCard, CardGenerator
from risklab.visualization.reports import ReportGenerator, HTMLReportGenerator
from risklab.visualization.bias_plots import (
    InstitutionalBiasCompass,
    ManipulabilityRadar,
    BiasEvolutionPlot,
    NormStabilityPlot,
    DomainBiasHeatmap,
    generate_institutional_bias_report,
)
from risklab.visualization.dashboard import (
    DashboardData,
    DashboardBuilder,
    DashboardRenderer,
    EpisodeSummary,
    SignalSummary,
    TimelinePoint,
    WaterfallItem,
    RadarDataPoint,
    HeatmapCell,
    ComparisonData,
    build_dashboard,
)
from risklab.visualization.advanced_plots import (
    SafetyGauge,
    MetricWaterfall,
    VulnerabilitySunburst,
    TemporalDriftPlot,
    ConfidenceFunnel,
    AttackSurfaceMap,
    ComparisonRadar,
    generate_all_advanced_plots,
)

__all__ = [
    "FramingDeltaHeatmap",
    "RiskScoreLadder",
    "OversightGapPlot",
    "EvaluatorDisagreementMatrix",
    "ScenarioNarrativeCard",
    "CardGenerator",
    "ReportGenerator",
    "HTMLReportGenerator",
    # Institutional bias visualizations
    "InstitutionalBiasCompass",
    "ManipulabilityRadar",
    "BiasEvolutionPlot",
    "NormStabilityPlot",
    "DomainBiasHeatmap",
    "generate_institutional_bias_report",
    # Dashboard
    "DashboardData",
    "DashboardBuilder",
    "DashboardRenderer",
    "EpisodeSummary",
    "SignalSummary",
    "TimelinePoint",
    "WaterfallItem",
    "RadarDataPoint",
    "HeatmapCell",
    "ComparisonData",
    "build_dashboard",
    # Advanced plots
    "SafetyGauge",
    "MetricWaterfall",
    "VulnerabilitySunburst",
    "TemporalDriftPlot",
    "ConfidenceFunnel",
    "AttackSurfaceMap",
    "ComparisonRadar",
    "generate_all_advanced_plots",
]
