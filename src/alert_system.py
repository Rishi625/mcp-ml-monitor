"""
Alert and Recommendation System

Manages alerts based on drift detection and performance monitoring.
Provides actionable recommendations for model maintenance.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
import logging

from .drift_detector import DriftReport, DriftSeverity
from .performance_monitor import PerformanceReport, PerformanceStatus

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of alerts"""
    DATA_DRIFT = "data_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    MODEL_HEALTH = "model_health"
    RETRAINING_RECOMMENDED = "retraining_recommended"
    SYSTEM_WARNING = "system_warning"


class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Individual alert"""
    alert_id: str
    alert_type: AlertType
    priority: AlertPriority
    title: str
    message: str
    details: Dict[str, Any]
    timestamp: str
    acknowledged: bool = False
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "type": self.alert_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved
        }


@dataclass
class RetrainingRecommendation:
    """Retraining recommendation with supporting evidence"""
    should_retrain: bool
    urgency: str  # "immediate", "soon", "scheduled", "not_needed"
    reasons: List[str]
    affected_features: List[str]
    estimated_impact: str
    suggested_actions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_retrain": self.should_retrain,
            "urgency": self.urgency,
            "reasons": self.reasons,
            "affected_features": self.affected_features,
            "estimated_impact": self.estimated_impact,
            "suggested_actions": self.suggested_actions
        }


@dataclass
class HealthSummary:
    """Overall model health summary"""
    overall_status: str  # "healthy", "warning", "critical"
    drift_status: str
    performance_status: str
    last_check: str
    active_alerts: int
    recommendations: List[str]
    retraining: RetrainingRecommendation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "drift_status": self.drift_status,
            "performance_status": self.performance_status,
            "last_check": self.last_check,
            "active_alerts": self.active_alerts,
            "recommendations": self.recommendations,
            "retraining": self.retraining.to_dict()
        }


class AlertSystem:
    """
    Manages alerts and recommendations for ML model monitoring.
    
    Features:
    - Alert generation from drift and performance reports
    - Alert deduplication and cooldown
    - Retraining recommendations
    - Health summary generation
    """
    
    def __init__(
        self,
        alert_cooldown: int = 3600,
        retrain_threshold_drift: float = 0.2,
        retrain_threshold_performance: float = 0.08
    ):
        self.alert_cooldown = alert_cooldown  # seconds
        self.retrain_threshold_drift = retrain_threshold_drift
        self.retrain_threshold_performance = retrain_threshold_performance
        
        # Alert storage
        self.alerts: List[Alert] = []
        self.alert_history: Dict[str, datetime] = {}  # For cooldown tracking
        
        # Alert counter for IDs
        self.alert_counter = 0
        
        # Callbacks for alert handling
        self.alert_handlers: List[Callable[[Alert], None]] = []
    
    def register_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register a callback for new alerts"""
        self.alert_handlers.append(handler)
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        self.alert_counter += 1
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"ALERT-{timestamp}-{self.alert_counter:04d}"
    
    def _check_cooldown(self, alert_key: str) -> bool:
        """Check if alert is in cooldown period"""
        if alert_key not in self.alert_history:
            return False
        
        last_alert = self.alert_history[alert_key]
        cooldown_end = last_alert + timedelta(seconds=self.alert_cooldown)
        
        return datetime.utcnow() < cooldown_end
    
    def _create_alert(
        self,
        alert_type: AlertType,
        priority: AlertPriority,
        title: str,
        message: str,
        details: Dict[str, Any],
        alert_key: Optional[str] = None
    ) -> Optional[Alert]:
        """Create and store an alert"""
        # Check cooldown
        if alert_key and self._check_cooldown(alert_key):
            logger.debug(f"Alert '{alert_key}' in cooldown, skipping")
            return None
        
        alert = Alert(
            alert_id=self._generate_alert_id(),
            alert_type=alert_type,
            priority=priority,
            title=title,
            message=message,
            details=details,
            timestamp=datetime.utcnow().isoformat()
        )
        
        self.alerts.append(alert)
        
        if alert_key:
            self.alert_history[alert_key] = datetime.utcnow()
        
        # Trigger handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        logger.info(f"Alert created: {alert.title} ({alert.priority.value})")
        return alert
    
    def process_drift_report(self, report: DriftReport) -> List[Alert]:
        """Process drift report and generate alerts"""
        alerts = []
        
        # Overall drift alert
        if report.overall_severity == DriftSeverity.CRITICAL:
            alert = self._create_alert(
                alert_type=AlertType.DATA_DRIFT,
                priority=AlertPriority.CRITICAL,
                title="Critical Data Drift Detected",
                message=f"{report.drifted_features}/{report.total_features} features showing significant drift",
                details={
                    "drift_percentage": report.drift_percentage,
                    "drifted_features": [r.feature_name for r in report.feature_results if r.is_drifted]
                },
                alert_key="drift_critical"
            )
            if alert:
                alerts.append(alert)
                
        elif report.overall_severity == DriftSeverity.WARNING:
            alert = self._create_alert(
                alert_type=AlertType.DATA_DRIFT,
                priority=AlertPriority.HIGH,
                title="Data Drift Warning",
                message=f"Moderate drift detected in {report.drifted_features} features",
                details={
                    "drift_percentage": report.drift_percentage,
                    "drifted_features": [r.feature_name for r in report.feature_results if r.is_drifted]
                },
                alert_key="drift_warning"
            )
            if alert:
                alerts.append(alert)
        
        # Individual feature alerts for critical features
        for result in report.feature_results:
            if result.severity == DriftSeverity.CRITICAL:
                alert = self._create_alert(
                    alert_type=AlertType.DATA_DRIFT,
                    priority=AlertPriority.HIGH,
                    title=f"Critical Drift: {result.feature_name}",
                    message=f"PSI={result.psi_score:.3f}, KS p-value={result.ks_pvalue:.4f}",
                    details=result.details,
                    alert_key=f"drift_feature_{result.feature_name}"
                )
                if alert:
                    alerts.append(alert)
        
        return alerts
    
    def process_performance_report(self, report: PerformanceReport) -> List[Alert]:
        """Process performance report and generate alerts"""
        alerts = []
        
        if report.status == PerformanceStatus.CRITICAL:
            alert = self._create_alert(
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                priority=AlertPriority.CRITICAL,
                title="Critical Performance Degradation",
                message=f"Model accuracy dropped by {abs(report.metric_deltas.get('accuracy', 0))*100:.1f}%",
                details={
                    "current_metrics": report.current_metrics.to_dict(),
                    "baseline_metrics": report.baseline_metrics,
                    "deltas": report.metric_deltas
                },
                alert_key="performance_critical"
            )
            if alert:
                alerts.append(alert)
                
        elif report.status == PerformanceStatus.DEGRADED:
            alert = self._create_alert(
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                priority=AlertPriority.HIGH,
                title="Performance Degradation Detected",
                message=f"F1 score: {report.current_metrics.f1:.3f} (baseline: {report.baseline_metrics.get('f1', 0):.3f})",
                details={
                    "current_metrics": report.current_metrics.to_dict(),
                    "deltas": report.metric_deltas
                },
                alert_key="performance_warning"
            )
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def generate_retraining_recommendation(
        self,
        drift_report: Optional[DriftReport] = None,
        performance_report: Optional[PerformanceReport] = None
    ) -> RetrainingRecommendation:
        """Generate retraining recommendation based on all available data"""
        reasons = []
        affected_features = []
        should_retrain = False
        urgency = "not_needed"
        
        # Analyze drift
        if drift_report:
            if drift_report.overall_severity == DriftSeverity.CRITICAL:
                should_retrain = True
                urgency = "immediate"
                reasons.append(f"Critical drift detected in {drift_report.drifted_features} features")
                affected_features.extend([
                    r.feature_name for r in drift_report.feature_results if r.is_drifted
                ])
            elif drift_report.overall_severity == DriftSeverity.WARNING:
                should_retrain = True
                if urgency != "immediate":
                    urgency = "soon"
                reasons.append(f"Moderate drift detected ({drift_report.drift_percentage:.1f}% of features)")
                affected_features.extend([
                    r.feature_name for r in drift_report.feature_results if r.is_drifted
                ])
        
        # Analyze performance
        if performance_report:
            accuracy_delta = performance_report.metric_deltas.get('accuracy', 0)
            
            if abs(accuracy_delta) > self.retrain_threshold_performance:
                should_retrain = True
                if accuracy_delta < -0.15:
                    urgency = "immediate"
                elif urgency != "immediate":
                    urgency = "soon"
                reasons.append(f"Performance degraded by {abs(accuracy_delta)*100:.1f}%")
            
            if performance_report.trend == "declining" and should_retrain:
                reasons.append("Performance showing declining trend")
        
        # Generate suggested actions
        suggested_actions = []
        if should_retrain:
            suggested_actions.append("Collect recent production data for retraining")
            suggested_actions.append("Validate data quality and preprocessing pipeline")
            if affected_features:
                suggested_actions.append(f"Investigate feature engineering for: {', '.join(affected_features[:5])}")
            suggested_actions.append("Run A/B test with retrained model before full deployment")
        else:
            suggested_actions.append("Continue monitoring")
            suggested_actions.append("Schedule routine retraining check in 1 week")
        
        # Estimate impact
        if urgency == "immediate":
            estimated_impact = "High - Model predictions may be unreliable"
        elif urgency == "soon":
            estimated_impact = "Medium - Prediction quality degrading"
        elif should_retrain:
            estimated_impact = "Low - Minor performance impact"
        else:
            estimated_impact = "None - Model performing as expected"
        
        return RetrainingRecommendation(
            should_retrain=should_retrain,
            urgency=urgency,
            reasons=reasons if reasons else ["No issues detected"],
            affected_features=affected_features[:10],  # Limit to top 10
            estimated_impact=estimated_impact,
            suggested_actions=suggested_actions
        )
    
    def generate_health_summary(
        self,
        drift_report: Optional[DriftReport] = None,
        performance_report: Optional[PerformanceReport] = None
    ) -> HealthSummary:
        """Generate comprehensive health summary"""
        # Determine statuses
        drift_status = "unknown"
        if drift_report:
            drift_status = drift_report.overall_severity.value
        
        performance_status = "unknown"
        if performance_report:
            performance_status = performance_report.status.value
        
        # Determine overall status
        if drift_status == "critical" or performance_status == "critical":
            overall_status = "critical"
        elif drift_status == "warning" or performance_status in ["warning", "degraded"]:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        # Count active alerts
        active_alerts = len([a for a in self.alerts if not a.resolved])
        
        # Compile recommendations
        recommendations = []
        if drift_report:
            recommendations.extend(drift_report.recommendations[:3])
        if performance_report:
            recommendations.extend(performance_report.recommendations[:3])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for r in recommendations:
            if r not in seen:
                seen.add(r)
                unique_recommendations.append(r)
        
        # Generate retraining recommendation
        retraining = self.generate_retraining_recommendation(drift_report, performance_report)
        
        return HealthSummary(
            overall_status=overall_status,
            drift_status=drift_status,
            performance_status=performance_status,
            last_check=datetime.utcnow().isoformat(),
            active_alerts=active_alerts,
            recommendations=unique_recommendations[:5],
            retraining=retraining
        )
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all unresolved alerts"""
        return [a for a in self.alerts if not a.resolved]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark alert as resolved"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                return True
        return False

