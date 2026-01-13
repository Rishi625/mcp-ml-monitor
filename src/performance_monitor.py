"""
Performance Monitoring Module

Tracks model performance metrics over time and detects degradation.
- Accuracy, Precision, Recall, F1-Score tracking
- Rolling window analysis
- Trend detection
"""

import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime
import logging

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix
)

logger = logging.getLogger(__name__)


class PerformanceStatus(Enum):
    """Status levels for model performance"""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"


@dataclass
class MetricSnapshot:
    """Snapshot of metrics at a point in time"""
    timestamp: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    sample_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "sample_count": self.sample_count
        }


@dataclass
class PerformanceReport:
    """Complete performance monitoring report"""
    current_metrics: MetricSnapshot
    baseline_metrics: Dict[str, float]
    metric_deltas: Dict[str, float]
    status: PerformanceStatus
    trend: str  # "improving", "stable", "declining"
    predictions_analyzed: int
    recommendations: List[str]
    historical_metrics: List[MetricSnapshot] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_metrics": self.current_metrics.to_dict(),
            "baseline_metrics": {k: round(v, 4) for k, v in self.baseline_metrics.items()},
            "metric_deltas": {k: round(v, 4) for k, v in self.metric_deltas.items()},
            "status": self.status.value,
            "trend": self.trend,
            "predictions_analyzed": self.predictions_analyzed,
            "recommendations": self.recommendations,
            "historical_metrics": [m.to_dict() for m in self.historical_metrics[-10:]]
        }


class PerformanceMonitor:
    """
    Monitors ML model performance in production.
    
    Tracks:
    - Real-time accuracy, precision, recall, F1
    - Performance trends over time
    - Degradation detection
    """
    
    def __init__(
        self,
        baseline_accuracy: float = 0.0,
        baseline_precision: float = 0.0,
        baseline_recall: float = 0.0,
        baseline_f1: float = 0.0,
        warning_threshold: float = 0.05,
        critical_threshold: float = 0.10,
        window_size: int = 1000,
        snapshot_interval: int = 100
    ):
        self.baseline = {
            'accuracy': baseline_accuracy,
            'precision': baseline_precision,
            'recall': baseline_recall,
            'f1': baseline_f1
        }
        
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.window_size = window_size
        self.snapshot_interval = snapshot_interval
        
        # Rolling window storage
        self.predictions: deque = deque(maxlen=window_size)
        self.actuals: deque = deque(maxlen=window_size)
        
        # Historical snapshots
        self.snapshots: List[MetricSnapshot] = []
        self.prediction_count = 0
        
    def set_baseline(
        self, 
        accuracy: float, 
        precision: float, 
        recall: float, 
        f1: float
    ) -> None:
        """Set baseline metrics from training/validation"""
        self.baseline = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        logger.info(f"Baseline metrics set: {self.baseline}")
    
    def set_baseline_from_data(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate and set baseline from validation data"""
        metrics = self._calculate_metrics(y_true, y_pred)
        self.set_baseline(
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1=metrics['f1']
        )
        return metrics
    
    def record_predictions(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> None:
        """
        Record batch of predictions for monitoring.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        for true, pred in zip(y_true, y_pred):
            self.predictions.append(pred)
            self.actuals.append(true)
            self.prediction_count += 1
        
        # Take snapshot at intervals
        if self.prediction_count % self.snapshot_interval == 0 and len(self.actuals) >= 50:
            self._take_snapshot()
    
    def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate all performance metrics"""
        # Handle edge cases
        if len(y_true) == 0 or len(y_pred) == 0:
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Use 'weighted' average for multiclass, handle binary case
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        average = 'binary' if len(unique_classes) <= 2 else 'weighted'
        
        precision = precision_score(y_true, y_pred, average=average, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    def _take_snapshot(self) -> None:
        """Take a snapshot of current metrics"""
        if len(self.actuals) < 10:
            return
            
        y_true = np.array(list(self.actuals))
        y_pred = np.array(list(self.predictions))
        
        metrics = self._calculate_metrics(y_true, y_pred)
        
        snapshot = MetricSnapshot(
            timestamp=datetime.utcnow().isoformat(),
            accuracy=metrics['accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1=metrics['f1'],
            sample_count=len(y_true)
        )
        
        self.snapshots.append(snapshot)
        logger.debug(f"Snapshot taken: {snapshot.to_dict()}")
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current metrics from rolling window"""
        if len(self.actuals) < 10:
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        y_true = np.array(list(self.actuals))
        y_pred = np.array(list(self.predictions))
        
        return self._calculate_metrics(y_true, y_pred)
    
    def _detect_trend(self) -> str:
        """Detect performance trend from recent snapshots"""
        if len(self.snapshots) < 3:
            return "stable"
        
        # Use last 5 snapshots
        recent = self.snapshots[-5:]
        accuracies = [s.accuracy for s in recent]
        
        # Simple linear regression for trend
        x = np.arange(len(accuracies))
        slope, _ = np.polyfit(x, accuracies, 1)
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"
    
    def _determine_status(self, deltas: Dict[str, float]) -> PerformanceStatus:
        """Determine overall status based on metric deltas"""
        # Check if any metric has critically degraded
        for metric, delta in deltas.items():
            if delta < -self.critical_threshold:
                return PerformanceStatus.CRITICAL
        
        # Check for warning level degradation
        for metric, delta in deltas.items():
            if delta < -self.warning_threshold:
                return PerformanceStatus.DEGRADED
        
        # Check for minor degradation
        avg_delta = np.mean(list(deltas.values()))
        if avg_delta < -0.02:
            return PerformanceStatus.WARNING
        
        return PerformanceStatus.HEALTHY
    
    def _generate_recommendations(
        self, 
        status: PerformanceStatus, 
        deltas: Dict[str, float],
        trend: str
    ) -> List[str]:
        """Generate recommendations based on performance analysis - ALL DYNAMIC"""
        recommendations = []
        
        if status == PerformanceStatus.CRITICAL:
            recommendations.append("CRITICAL: Model retraining required immediately")
            recommendations.append("Consider rolling back to previous model version")
            recommendations.append("Investigate recent data changes or upstream issues")
        elif status == PerformanceStatus.DEGRADED:
            recommendations.append("WARNING: Schedule model retraining soon")
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Review recent predictions for error patterns")
        elif status == PerformanceStatus.WARNING:
            recommendations.append("NOTICE: Minor performance decline detected")
            recommendations.append("Continue monitoring and prepare retraining pipeline")
        
        # Metric-specific recommendations - DYNAMICALLY generated based on actual deltas
        if deltas.get('recall', 0) < -0.05:
            recommendations.append("Low recall: Consider adjusting classification threshold")
        if deltas.get('precision', 0) < -0.05:
            recommendations.append("Low precision: Review false positive cases")
        
        if trend == "declining":
            recommendations.append("Trend: Performance declining over time")
        elif trend == "improving":
            recommendations.append("Trend: Performance improving")
        
        if not recommendations:
            recommendations.append("Model performing within expected parameters")
        
        return recommendations
    
    def generate_report(self) -> PerformanceReport:
        """Generate comprehensive performance report"""
        current = self.get_current_metrics()
        
        # Calculate deltas from baseline
        deltas = {
            metric: current[metric] - self.baseline[metric]
            for metric in current.keys()
        }
        
        # Determine status and trend
        status = self._determine_status(deltas)
        trend = self._detect_trend()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(status, deltas, trend)
        
        current_snapshot = MetricSnapshot(
            timestamp=datetime.utcnow().isoformat(),
            accuracy=current['accuracy'],
            precision=current['precision'],
            recall=current['recall'],
            f1=current['f1'],
            sample_count=len(self.actuals)
        )
        
        return PerformanceReport(
            current_metrics=current_snapshot,
            baseline_metrics=self.baseline,
            metric_deltas=deltas,
            status=status,
            trend=trend,
            predictions_analyzed=self.prediction_count,
            recommendations=recommendations,
            historical_metrics=self.snapshots.copy()
        )
    
    def get_confusion_matrix(self) -> Optional[np.ndarray]:
        """Get confusion matrix for current window"""
        if len(self.actuals) < 10:
            return None
        
        y_true = np.array(list(self.actuals))
        y_pred = np.array(list(self.predictions))
        
        return confusion_matrix(y_true, y_pred)

