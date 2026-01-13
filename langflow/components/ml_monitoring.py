"""
ML Monitoring Components for LangFlow Pipeline

Custom components for:
- Drift detection (statistical tests)
- Anomaly detection (Isolation Forest)
- Time-series forecasting
- Performance analysis
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Severity(Enum):
    NONE = "none"
    LOW = "low"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DriftResult:
    feature: str
    ks_statistic: float
    ks_pvalue: float
    psi_score: float
    wasserstein_distance: float
    severity: Severity
    is_drifted: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "ks_statistic": round(self.ks_statistic, 4),
            "ks_pvalue": round(self.ks_pvalue, 6),
            "psi_score": round(self.psi_score, 4),
            "wasserstein_distance": round(self.wasserstein_distance, 4),
            "severity": self.severity.value,
            "is_drifted": self.is_drifted
        }


class DriftDetectionComponent:
    """
    LangFlow Component: Drift Detection
    
    Detects data drift using multiple statistical methods:
    - Kolmogorov-Smirnov test
    - Population Stability Index (PSI)
    - Wasserstein distance
    """
    
    display_name = "Drift Detection"
    description = "Detect data drift using statistical tests"
    
    def __init__(
        self,
        ks_threshold: float = 0.05,
        psi_warning: float = 0.1,
        psi_critical: float = 0.2,
        n_bins: int = 10
    ):
        self.ks_threshold = ks_threshold
        self.psi_warning = psi_warning
        self.psi_critical = psi_critical
        self.n_bins = n_bins
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_stats: Dict[str, Dict] = {}
    
    def set_reference(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Set reference distribution from training data"""
        self.reference_data = data.copy()
        
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                values = data[col].dropna().values
                self.reference_stats[col] = {
                    'values': values,
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        return {
            "status": "reference_set",
            "features": list(self.reference_stats.keys()),
            "samples": len(data)
        }
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate Population Stability Index"""
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        
        if min_val == max_val:
            return 0.0
        
        bins = np.linspace(min_val, max_val, self.n_bins + 1)
        
        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)
        
        # Add small value to avoid division by zero
        epsilon = 0.0001
        ref_props = (ref_counts + epsilon) / (len(reference) + epsilon * self.n_bins)
        cur_props = (cur_counts + epsilon) / (len(current) + epsilon * self.n_bins)
        
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        return float(psi)
    
    def detect(
        self,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Detect drift in current data compared to reference"""
        
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        if features is None:
            features = [col for col in current_data.columns 
                       if col in self.reference_stats]
        
        results = []
        drifted_count = 0
        max_severity = Severity.NONE
        
        for feature in features:
            if feature not in self.reference_stats:
                continue
            
            ref_values = self.reference_stats[feature]['values']
            cur_values = current_data[feature].dropna().values
            
            if len(cur_values) < 10:
                continue
            
            # KS test
            ks_stat, ks_pvalue = stats.ks_2samp(ref_values, cur_values)
            
            # PSI
            psi = self._calculate_psi(ref_values, cur_values)
            
            # Wasserstein
            w_dist = wasserstein_distance(ref_values, cur_values)
            value_range = max(ref_values.max(), cur_values.max()) - min(ref_values.min(), cur_values.min())
            if value_range > 0:
                w_dist = w_dist / value_range
            
            # Determine severity
            if psi >= self.psi_critical or ks_pvalue < self.ks_threshold:
                severity = Severity.CRITICAL
                is_drifted = True
            elif psi >= self.psi_warning:
                severity = Severity.WARNING
                is_drifted = True
            elif psi > 0.05:
                severity = Severity.LOW
                is_drifted = False
            else:
                severity = Severity.NONE
                is_drifted = False
            
            if is_drifted:
                drifted_count += 1
            
            if severity.value > max_severity.value:
                max_severity = severity
            
            results.append(DriftResult(
                feature=feature,
                ks_statistic=float(ks_stat),
                ks_pvalue=float(ks_pvalue),
                psi_score=float(psi),
                wasserstein_distance=float(w_dist),
                severity=severity,
                is_drifted=is_drifted
            ))
        
        return {
            "total_features": len(results),
            "drifted_features": drifted_count,
            "drift_percentage": round(drifted_count / len(results) * 100, 1) if results else 0,
            "overall_severity": max_severity.value,
            "feature_results": [r.to_dict() for r in results],
            "timestamp": datetime.now().isoformat()
        }


class AnomalyDetectionComponent:
    """
    LangFlow Component: Anomaly Detection
    
    Detects anomalies using Isolation Forest algorithm.
    """
    
    display_name = "Anomaly Detection"
    description = "Detect anomalies using Isolation Forest"
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None
        self.is_fitted = False
    
    def fit(self, data: pd.DataFrame, features: Optional[List[str]] = None) -> Dict[str, Any]:
        """Fit the anomaly detection model on normal data"""
        
        if features:
            X = data[features].values
        else:
            X = data.select_dtypes(include=[np.number]).values
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state
        )
        self.model.fit(X)
        self.is_fitted = True
        
        return {
            "status": "fitted",
            "samples": len(data),
            "features": features or list(data.select_dtypes(include=[np.number]).columns)
        }
    
    def detect(
        self,
        data: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Detect anomalies in new data"""
        
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        if features:
            X = data[features].values
        else:
            X = data.select_dtypes(include=[np.number]).values
        
        # Predict: -1 for anomaly, 1 for normal
        predictions = self.model.predict(X)
        scores = self.model.decision_function(X)
        
        anomaly_indices = np.where(predictions == -1)[0].tolist()
        anomaly_count = len(anomaly_indices)
        
        return {
            "total_samples": len(data),
            "anomaly_count": anomaly_count,
            "anomaly_percentage": round(anomaly_count / len(data) * 100, 2),
            "anomaly_indices": anomaly_indices[:100],  # Limit to first 100
            "severity": "critical" if anomaly_count / len(data) > 0.2 else 
                       "warning" if anomaly_count / len(data) > 0.1 else "low",
            "anomaly_scores": {
                "min": float(scores.min()),
                "max": float(scores.max()),
                "mean": float(scores.mean())
            },
            "timestamp": datetime.now().isoformat()
        }


class TimeSeriesForecastComponent:
    """
    LangFlow Component: Time Series Forecasting
    
    Forecasts metric trends using exponential smoothing.
    """
    
    display_name = "Metric Forecasting"
    description = "Forecast metric trends"
    
    def __init__(self, forecast_periods: int = 24):
        self.forecast_periods = forecast_periods
    
    def forecast(
        self,
        data: pd.DataFrame,
        metric_column: str,
        time_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """Forecast future metric values"""
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        
        values = data[metric_column].dropna().values
        
        if len(values) < 10:
            return {"error": "Insufficient data for forecasting"}
        
        # Fit exponential smoothing
        model = ExponentialSmoothing(
            values,
            trend='add',
            seasonal=None
        ).fit()
        
        # Forecast
        forecast = model.forecast(self.forecast_periods)
        
        # Calculate trend
        trend_direction = "increasing" if forecast[-1] > values[-1] else "decreasing"
        trend_magnitude = abs(forecast[-1] - values[-1]) / (values[-1] + 0.001) * 100
        
        return {
            "current_value": float(values[-1]),
            "forecast_values": [float(v) for v in forecast],
            "forecast_periods": self.forecast_periods,
            "trend_direction": trend_direction,
            "trend_magnitude_pct": round(trend_magnitude, 2),
            "alert": trend_magnitude > 20,  # Alert if >20% change expected
            "timestamp": datetime.now().isoformat()
        }


class PerformanceAnalysisComponent:
    """
    LangFlow Component: Performance Analysis
    
    Analyzes model performance metrics and detects degradation.
    """
    
    display_name = "Performance Analysis"
    description = "Analyze model performance metrics"
    
    def __init__(
        self,
        warning_threshold: float = 0.05,
        critical_threshold: float = 0.10
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.baseline: Dict[str, float] = {}
    
    def set_baseline(
        self,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float
    ) -> Dict[str, Any]:
        """Set baseline performance metrics"""
        self.baseline = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        return {"status": "baseline_set", "baseline": self.baseline}
    
    def analyze(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze current performance against baseline"""
        
        current = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }
        
        # Calculate deltas
        deltas = {}
        if self.baseline:
            deltas = {k: current[k] - self.baseline[k] for k in current}
        
        # Determine status
        status = "healthy"
        for metric, delta in deltas.items():
            if delta < -self.critical_threshold:
                status = "critical"
                break
            elif delta < -self.warning_threshold:
                status = "warning"
        
        return {
            "current_metrics": current,
            "baseline_metrics": self.baseline,
            "deltas": {k: round(v, 4) for k, v in deltas.items()},
            "status": status,
            "samples_analyzed": len(y_true),
            "timestamp": datetime.now().isoformat()
        }


# LangFlow component registration
LANGFLOW_COMPONENTS = {
    "drift_detection": DriftDetectionComponent,
    "anomaly_detection": AnomalyDetectionComponent,
    "time_series_forecast": TimeSeriesForecastComponent,
    "performance_analysis": PerformanceAnalysisComponent
}

