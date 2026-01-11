"""
Real-Time Monitoring System for AI Safety.

Provides:
- Live evaluation streaming
- Anomaly detection
- Alert generation
- Dashboard websocket support
- Metric aggregation over time windows
"""

import asyncio
from typing import Optional, List, Dict, Any, Callable, AsyncIterator
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
from pydantic import BaseModel, Field
import numpy as np
import json
import logging

from risklab.risk.unified_score import UnifiedSafetyScore, USSComputer
from risklab.measurement.metrics import BehavioralMetrics
from risklab.measurement.signals import ManipulationSignals


logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of alerts."""
    SCORE_DROP = "score_drop"
    ANOMALY_DETECTED = "anomaly_detected"
    THRESHOLD_BREACH = "threshold_breach"
    TREND_CHANGE = "trend_change"
    HIGH_RISK_EPISODE = "high_risk_episode"
    SYSTEM_ERROR = "system_error"


class Alert(BaseModel):
    """A monitoring alert."""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Context
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold: Optional[float] = None
    episode_id: Optional[str] = None
    
    # Status
    acknowledged: bool = False
    resolved: bool = False
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MetricWindow(BaseModel):
    """Rolling window for metric aggregation."""
    metric_name: str
    window_size: int = 100
    values: List[float] = Field(default_factory=list)
    timestamps: List[datetime] = Field(default_factory=list)
    
    # Statistics
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    trend: float = 0.0  # Slope of recent values
    
    def add_value(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Add a value to the window."""
        self.values.append(value)
        self.timestamps.append(timestamp or datetime.utcnow())
        
        # Maintain window size
        if len(self.values) > self.window_size:
            self.values.pop(0)
            self.timestamps.pop(0)
        
        self._update_stats()
    
    def _update_stats(self) -> None:
        """Update window statistics."""
        if len(self.values) < 2:
            return
        
        arr = np.array(self.values)
        self.mean = float(np.mean(arr))
        self.std = float(np.std(arr))
        self.min_val = float(np.min(arr))
        self.max_val = float(np.max(arr))
        
        # Compute trend (slope)
        x = np.arange(len(self.values))
        slope, _ = np.polyfit(x, arr, 1)
        self.trend = float(slope)
    
    def is_anomaly(self, value: float, threshold_std: float = 3.0) -> bool:
        """Check if a value is anomalous."""
        if len(self.values) < 10:
            return False
        
        z_score = abs(value - self.mean) / max(self.std, 0.001)
        return z_score > threshold_std
    
    def get_percentile(self, value: float) -> float:
        """Get percentile rank of a value."""
        if not self.values:
            return 50.0
        return float(np.percentile(np.array(self.values) <= value, 100))


class MonitoringConfig(BaseModel):
    """Configuration for monitoring system."""
    # Thresholds
    min_uss_threshold: float = 60.0
    score_drop_threshold: float = 10.0  # Alert if USS drops by this much
    anomaly_std_threshold: float = 3.0
    
    # Windows
    window_size: int = 100
    trend_window: int = 20
    
    # Alerts
    enable_alerts: bool = True
    alert_cooldown_seconds: int = 60  # Minimum time between same alert type
    
    # Callbacks
    alert_webhook_url: Optional[str] = None


class EvaluationEvent(BaseModel):
    """A single evaluation event for streaming."""
    event_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Evaluation results
    uss_score: float
    grade: str
    category_scores: Dict[str, float] = Field(default_factory=dict)
    
    # Episode info
    episode_id: Optional[str] = None
    episode_name: Optional[str] = None
    domain: Optional[str] = None
    
    # Alerts triggered
    alerts: List[Alert] = Field(default_factory=list)
    
    # Raw data
    metrics: Optional[Dict[str, float]] = None
    signals: Optional[Dict[str, float]] = None


class MonitoringState(BaseModel):
    """Current state of the monitoring system."""
    is_running: bool = False
    start_time: Optional[datetime] = None
    
    # Counters
    total_evaluations: int = 0
    total_alerts: int = 0
    alerts_by_severity: Dict[str, int] = Field(default_factory=dict)
    
    # Current metrics
    current_uss: float = 0.0
    uss_trend: str = "stable"
    
    # Recent history
    recent_events: List[EvaluationEvent] = Field(default_factory=list)
    active_alerts: List[Alert] = Field(default_factory=list)


class RealTimeMonitor:
    """
    Real-time monitoring system for AI safety evaluation.
    
    Features:
    - Streaming evaluation results
    - Anomaly detection
    - Alert generation
    - Trend analysis
    - WebSocket support for dashboards
    """
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.state = MonitoringState()
        
        # Metric windows
        self.windows: Dict[str, MetricWindow] = {
            "uss": MetricWindow(metric_name="uss", window_size=self.config.window_size),
            "safety": MetricWindow(metric_name="safety", window_size=self.config.window_size),
            "integrity": MetricWindow(metric_name="integrity", window_size=self.config.window_size),
            "reliability": MetricWindow(metric_name="reliability", window_size=self.config.window_size),
            "alignment": MetricWindow(metric_name="alignment", window_size=self.config.window_size),
        }
        
        # Alert tracking
        self.alert_history: deque = deque(maxlen=1000)
        self.last_alert_times: Dict[str, datetime] = {}
        
        # USS computer
        self.uss_computer = USSComputer()
        
        # Event subscribers
        self.subscribers: List[Callable[[EvaluationEvent], None]] = []
        self.async_subscribers: List[Callable[[EvaluationEvent], Any]] = []
        
        # Alert handlers
        self.alert_handlers: List[Callable[[Alert], None]] = []
    
    def subscribe(self, callback: Callable[[EvaluationEvent], None]) -> None:
        """Subscribe to evaluation events."""
        self.subscribers.append(callback)
    
    def subscribe_async(self, callback: Callable[[EvaluationEvent], Any]) -> None:
        """Subscribe to evaluation events (async)."""
        self.async_subscribers.append(callback)
    
    def on_alert(self, handler: Callable[[Alert], None]) -> None:
        """Register an alert handler."""
        self.alert_handlers.append(handler)
    
    def start(self) -> None:
        """Start the monitoring system."""
        self.state.is_running = True
        self.state.start_time = datetime.utcnow()
        logger.info("Monitoring system started")
    
    def stop(self) -> None:
        """Stop the monitoring system."""
        self.state.is_running = False
        logger.info("Monitoring system stopped")
    
    async def process_evaluation(
        self,
        uss: UnifiedSafetyScore,
        episode_id: Optional[str] = None,
        episode_name: Optional[str] = None,
        domain: Optional[str] = None,
        metrics: Optional[BehavioralMetrics] = None,
        signals: Optional[ManipulationSignals] = None,
    ) -> EvaluationEvent:
        """
        Process an evaluation result and generate events/alerts.
        """
        timestamp = datetime.utcnow()
        event_id = f"evt-{self.state.total_evaluations}"
        
        # Update windows
        self.windows["uss"].add_value(uss.score, timestamp)
        self.windows["safety"].add_value(uss.safety_score.score, timestamp)
        self.windows["integrity"].add_value(uss.integrity_score.score, timestamp)
        self.windows["reliability"].add_value(uss.reliability_score.score, timestamp)
        self.windows["alignment"].add_value(uss.alignment_score.score, timestamp)
        
        # Check for alerts
        alerts = []
        if self.config.enable_alerts:
            alerts = self._check_alerts(uss, episode_id)
        
        # Create event
        event = EvaluationEvent(
            event_id=event_id,
            timestamp=timestamp,
            uss_score=uss.score,
            grade=uss.grade.value,
            category_scores=uss.get_category_scores(),
            episode_id=episode_id,
            episode_name=episode_name,
            domain=domain,
            alerts=alerts,
            metrics=metrics.to_dict() if metrics else None,
            signals=signals.to_dict() if signals else None,
        )
        
        # Update state
        self.state.total_evaluations += 1
        self.state.current_uss = uss.score
        self.state.uss_trend = self._determine_trend()
        
        # Keep recent events
        self.state.recent_events.append(event)
        if len(self.state.recent_events) > 100:
            self.state.recent_events.pop(0)
        
        # Update active alerts
        self.state.active_alerts = [a for a in self.state.active_alerts if not a.resolved]
        self.state.active_alerts.extend(alerts)
        
        # Notify subscribers
        for subscriber in self.subscribers:
            try:
                subscriber(event)
            except Exception as e:
                logger.error(f"Subscriber error: {e}")
        
        for subscriber in self.async_subscribers:
            try:
                await subscriber(event)
            except Exception as e:
                logger.error(f"Async subscriber error: {e}")
        
        # Handle alerts
        for alert in alerts:
            self.state.total_alerts += 1
            self.state.alerts_by_severity[alert.severity.value] = \
                self.state.alerts_by_severity.get(alert.severity.value, 0) + 1
            
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler error: {e}")
        
        return event
    
    def _check_alerts(
        self,
        uss: UnifiedSafetyScore,
        episode_id: Optional[str]
    ) -> List[Alert]:
        """Check for alert conditions."""
        alerts = []
        
        # Threshold breach
        if uss.score < self.config.min_uss_threshold:
            if self._can_alert("threshold_breach"):
                alerts.append(Alert(
                    alert_id=f"alert-{len(self.alert_history)}",
                    alert_type=AlertType.THRESHOLD_BREACH,
                    severity=AlertSeverity.ERROR if uss.score < 50 else AlertSeverity.WARNING,
                    message=f"USS score {uss.score:.1f} below threshold {self.config.min_uss_threshold}",
                    metric_name="uss",
                    current_value=uss.score,
                    threshold=self.config.min_uss_threshold,
                    episode_id=episode_id
                ))
        
        # Score drop
        uss_window = self.windows["uss"]
        if len(uss_window.values) > 10:
            recent_avg = np.mean(uss_window.values[-10:])
            older_avg = np.mean(uss_window.values[-20:-10]) if len(uss_window.values) > 20 else recent_avg
            drop = older_avg - recent_avg
            
            if drop > self.config.score_drop_threshold:
                if self._can_alert("score_drop"):
                    alerts.append(Alert(
                        alert_id=f"alert-{len(self.alert_history)}",
                        alert_type=AlertType.SCORE_DROP,
                        severity=AlertSeverity.WARNING,
                        message=f"USS dropped by {drop:.1f} points over recent evaluations",
                        metric_name="uss",
                        current_value=recent_avg,
                        metadata={"drop": drop, "recent_avg": recent_avg, "older_avg": older_avg}
                    ))
        
        # Anomaly detection
        for metric_name, window in self.windows.items():
            current = getattr(uss, f"{metric_name}_score", None)
            if current is None:
                current = uss if metric_name == "uss" else None
            
            if current:
                value = current.score if hasattr(current, 'score') else current
                if window.is_anomaly(value, self.config.anomaly_std_threshold):
                    if self._can_alert(f"anomaly_{metric_name}"):
                        alerts.append(Alert(
                            alert_id=f"alert-{len(self.alert_history)}",
                            alert_type=AlertType.ANOMALY_DETECTED,
                            severity=AlertSeverity.WARNING,
                            message=f"Anomalous {metric_name} score: {value:.1f} (mean: {window.mean:.1f}, std: {window.std:.1f})",
                            metric_name=metric_name,
                            current_value=value,
                            metadata={"mean": window.mean, "std": window.std}
                        ))
        
        # High-risk episode
        if uss.score < 50 and episode_id:
            alerts.append(Alert(
                alert_id=f"alert-{len(self.alert_history)}",
                alert_type=AlertType.HIGH_RISK_EPISODE,
                severity=AlertSeverity.ERROR,
                message=f"High-risk episode detected: {episode_id}",
                episode_id=episode_id,
                current_value=uss.score,
                metadata={"concerns": uss.top_concerns[:3]}
            ))
        
        # Record alerts
        for alert in alerts:
            self.alert_history.append(alert)
            self.last_alert_times[alert.alert_type.value] = datetime.utcnow()
        
        return alerts
    
    def _can_alert(self, alert_key: str) -> bool:
        """Check if we can send an alert (cooldown check)."""
        last_time = self.last_alert_times.get(alert_key)
        if not last_time:
            return True
        
        cooldown = timedelta(seconds=self.config.alert_cooldown_seconds)
        return datetime.utcnow() - last_time > cooldown
    
    def _determine_trend(self) -> str:
        """Determine current USS trend."""
        uss_window = self.windows["uss"]
        if len(uss_window.values) < 5:
            return "stable"
        
        if uss_window.trend > 0.5:
            return "improving"
        elif uss_window.trend < -0.5:
            return "declining"
        else:
            return "stable"
    
    def get_dashboard_state(self) -> Dict[str, Any]:
        """Get current state for dashboard rendering."""
        return {
            "is_running": self.state.is_running,
            "uptime_seconds": (datetime.utcnow() - self.state.start_time).total_seconds() if self.state.start_time else 0,
            "total_evaluations": self.state.total_evaluations,
            "current_uss": self.state.current_uss,
            "uss_trend": self.state.uss_trend,
            "total_alerts": self.state.total_alerts,
            "alerts_by_severity": self.state.alerts_by_severity,
            "active_alert_count": len(self.state.active_alerts),
            "windows": {
                name: {
                    "mean": w.mean,
                    "std": w.std,
                    "min": w.min_val,
                    "max": w.max_val,
                    "trend": w.trend,
                    "count": len(w.values)
                }
                for name, w in self.windows.items()
            },
            "recent_scores": self.windows["uss"].values[-20:],
            "recent_alerts": [a.model_dump() for a in self.state.active_alerts[-5:]],
        }
    
    async def stream_events(self) -> AsyncIterator[EvaluationEvent]:
        """Stream evaluation events (for WebSocket support)."""
        last_index = 0
        while self.state.is_running:
            current_len = len(self.state.recent_events)
            if current_len > last_index:
                for event in self.state.recent_events[last_index:]:
                    yield event
                last_index = current_len
            await asyncio.sleep(0.1)


class MonitoringDashboardServer:
    """
    Simple WebSocket server for real-time dashboard updates.
    """
    
    def __init__(self, monitor: RealTimeMonitor, port: int = 8765):
        self.monitor = monitor
        self.port = port
        self.clients: List[Any] = []
    
    async def handler(self, websocket, path):
        """Handle WebSocket connections."""
        self.clients.append(websocket)
        try:
            # Send initial state
            await websocket.send(json.dumps({
                "type": "state",
                "data": self.monitor.get_dashboard_state()
            }, default=str))
            
            # Stream updates
            async for event in self.monitor.stream_events():
                message = json.dumps({
                    "type": "event",
                    "data": event.model_dump()
                }, default=str)
                await websocket.send(message)
        finally:
            self.clients.remove(websocket)
    
    async def broadcast_alert(self, alert: Alert) -> None:
        """Broadcast an alert to all connected clients."""
        message = json.dumps({
            "type": "alert",
            "data": alert.model_dump()
        }, default=str)
        
        for client in self.clients:
            try:
                await client.send(message)
            except Exception:
                pass


def create_monitor(
    min_threshold: float = 60.0,
    enable_alerts: bool = True
) -> RealTimeMonitor:
    """Create a configured monitoring instance."""
    config = MonitoringConfig(
        min_uss_threshold=min_threshold,
        enable_alerts=enable_alerts
    )
    return RealTimeMonitor(config)
