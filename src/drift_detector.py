"""
Data Drift Detection Module

Implements multiple statistical methods for detecting data drift:
- Kolmogorov-Smirnov Test (KS Test)
- Population Stability Index (PSI)
- Wasserstein Distance (Earth Mover's Distance)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DriftSeverity(Enum):
    """Severity levels for drift detection"""
    NONE = "none"
    LOW = "low"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class FeatureDriftResult:
    """Result of drift detection for a single feature"""
    feature_name: str
    ks_statistic: float
    ks_pvalue: float
    psi_score: float
    wasserstein_dist: float
    severity: DriftSeverity
    is_drifted: bool
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    """Complete drift detection report"""
    total_features: int
    drifted_features: int
    feature_results: List[FeatureDriftResult]
    overall_severity: DriftSeverity
    drift_percentage: float
    recommendations: List[str]
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_features": self.total_features,
            "drifted_features": self.drifted_features,
            "drift_percentage": round(self.drift_percentage, 2),
            "overall_severity": self.overall_severity.value,
            "feature_results": [
                {
                    "feature": r.feature_name,
                    "ks_statistic": round(r.ks_statistic, 4),
                    "ks_pvalue": round(r.ks_pvalue, 4),
                    "psi_score": round(r.psi_score, 4),
                    "wasserstein_distance": round(r.wasserstein_dist, 4),
                    "severity": r.severity.value,
                    "is_drifted": r.is_drifted
                }
                for r in self.feature_results
            ],
            "recommendations": self.recommendations,
            "timestamp": self.timestamp
        }


class DriftDetector:
    """
    Detects data drift using multiple statistical methods.
    
    Methods:
    - Kolmogorov-Smirnov Test: Non-parametric test comparing distributions
    - PSI (Population Stability Index): Measures shift in distributions
    - Wasserstein Distance: Measures the "work" to transform one distribution to another
    """
    
    def __init__(
        self,
        ks_threshold: float = 0.05,
        psi_warning: float = 0.1,
        psi_critical: float = 0.2,
        wasserstein_threshold: float = 0.15,
        n_bins: int = 10
    ):
        self.ks_threshold = ks_threshold
        self.psi_warning = psi_warning
        self.psi_critical = psi_critical
        self.wasserstein_threshold = wasserstein_threshold
        self.n_bins = n_bins
        
        # Store reference distributions
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_stats: Dict[str, Dict] = {}
        
    def set_reference(self, data: pd.DataFrame) -> None:
        """
        Set the reference (training) data distribution.
        
        Args:
            data: Training data to use as reference
        """
        self.reference_data = data.copy()
        
        # Compute and store statistics for each feature
        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                self.reference_stats[col] = {
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'median': data[col].median(),
                    'values': data[col].dropna().values
                }
        
        logger.info(f"Reference data set with {len(data)} samples, {len(self.reference_stats)} numeric features")
    
    def calculate_ks_test(self, reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test.
        
        Returns:
            Tuple of (statistic, p-value)
        """
        statistic, pvalue = stats.ks_2samp(reference, current)
        return float(statistic), float(pvalue)
    
    def calculate_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI measures the shift in distribution between reference and current data.
        - PSI < 0.1: No significant shift
        - 0.1 <= PSI < 0.2: Moderate shift (warning)
        - PSI >= 0.2: Significant shift (action required)
        
        Returns:
            PSI score
        """
        # Create bins based on reference data
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        
        # Handle edge case where all values are the same
        if min_val == max_val:
            return 0.0
        
        bins = np.linspace(min_val, max_val, self.n_bins + 1)
        
        # Calculate bin proportions
        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)
        
        # Convert to proportions (avoid division by zero)
        ref_props = (ref_counts + 0.0001) / (len(reference) + 0.0001 * self.n_bins)
        cur_props = (cur_counts + 0.0001) / (len(current) + 0.0001 * self.n_bins)
        
        # Calculate PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        
        return float(psi)
    
    def calculate_wasserstein(self, reference: np.ndarray, current: np.ndarray) -> float:
        """
        Calculate Wasserstein distance (Earth Mover's Distance).
        
        Measures the minimum "work" required to transform one distribution into another.
        Normalized by the range of values.
        
        Returns:
            Normalized Wasserstein distance
        """
        distance = wasserstein_distance(reference, current)
        
        # Normalize by the range
        value_range = max(reference.max(), current.max()) - min(reference.min(), current.min())
        if value_range > 0:
            distance = distance / value_range
        
        return float(distance)
    
    def detect_feature_drift(self, feature_name: str, current_values: np.ndarray) -> FeatureDriftResult:
        """
        Detect drift for a single feature.
        
        Args:
            feature_name: Name of the feature
            current_values: Current production values
            
        Returns:
            FeatureDriftResult with all metrics
        """
        if feature_name not in self.reference_stats:
            raise ValueError(f"Feature '{feature_name}' not in reference data")
        
        reference_values = self.reference_stats[feature_name]['values']
        
        # Calculate all metrics
        ks_stat, ks_pvalue = self.calculate_ks_test(reference_values, current_values)
        psi_score = self.calculate_psi(reference_values, current_values)
        wasserstein_dist = self.calculate_wasserstein(reference_values, current_values)
        
        # Determine severity
        severity = DriftSeverity.NONE
        is_drifted = False
        
        if psi_score >= self.psi_critical or ks_pvalue < self.ks_threshold:
            severity = DriftSeverity.CRITICAL
            is_drifted = True
        elif psi_score >= self.psi_warning or wasserstein_dist > self.wasserstein_threshold:
            severity = DriftSeverity.WARNING
            is_drifted = True
        elif psi_score > 0.05:
            severity = DriftSeverity.LOW
        
        return FeatureDriftResult(
            feature_name=feature_name,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pvalue,
            psi_score=psi_score,
            wasserstein_dist=wasserstein_dist,
            severity=severity,
            is_drifted=is_drifted,
            details={
                'reference_mean': self.reference_stats[feature_name]['mean'],
                'reference_std': self.reference_stats[feature_name]['std'],
                'current_mean': float(np.mean(current_values)),
                'current_std': float(np.std(current_values)),
            }
        )
    
    def detect_drift(
        self, 
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> DriftReport:
        """
        Detect drift across all features.
        
        Args:
            current_data: Current production data
            features: Optional list of features to check (default: all)
            
        Returns:
            DriftReport with complete analysis
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")
        
        # Determine features to analyze
        if features is None:
            features = [col for col in current_data.columns 
                       if col in self.reference_stats and pd.api.types.is_numeric_dtype(current_data[col])]
        
        results = []
        drifted_count = 0
        max_severity = DriftSeverity.NONE
        
        for feature in features:
            if feature not in current_data.columns:
                logger.warning(f"Feature '{feature}' not in current data, skipping")
                continue
                
            current_values = current_data[feature].dropna().values
            
            if len(current_values) < 10:
                logger.warning(f"Feature '{feature}' has too few values ({len(current_values)}), skipping")
                continue
            
            result = self.detect_feature_drift(feature, current_values)
            results.append(result)
            
            if result.is_drifted:
                drifted_count += 1
            
            # Track maximum severity
            severity_order = [DriftSeverity.NONE, DriftSeverity.LOW, DriftSeverity.WARNING, DriftSeverity.CRITICAL]
            if severity_order.index(result.severity) > severity_order.index(max_severity):
                max_severity = result.severity
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results, max_severity)
        
        from datetime import datetime
        
        return DriftReport(
            total_features=len(results),
            drifted_features=drifted_count,
            feature_results=results,
            overall_severity=max_severity,
            drift_percentage=(drifted_count / len(results) * 100) if results else 0,
            recommendations=recommendations,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def _generate_recommendations(
        self, 
        results: List[FeatureDriftResult], 
        severity: DriftSeverity
    ) -> List[str]:
        """Generate actionable recommendations based on drift analysis"""
        recommendations = []
        
        if severity == DriftSeverity.CRITICAL:
            recommendations.append("CRITICAL: Immediate model retraining recommended")
            recommendations.append("Investigate data pipeline for upstream changes")
        elif severity == DriftSeverity.WARNING:
            recommendations.append("WARNING: Monitor closely and prepare for retraining")
            recommendations.append("Validate model predictions manually on recent data")
        
        # Feature-specific recommendations - DYNAMICALLY generated from actual results
        critical_features = [r for r in results if r.severity == DriftSeverity.CRITICAL]
        if critical_features:
            feature_names = [f.feature_name for f in critical_features[:5]]
            recommendations.append(f"Critical drift detected in: {', '.join(feature_names)}")
        
        # High PSI features - DYNAMICALLY computed
        high_psi = [r for r in results if r.psi_score > 0.15]
        if high_psi:
            recommendations.append(f"{len(high_psi)} features show significant distribution shift (PSI > 0.15)")
        
        if not recommendations:
            recommendations.append("No significant drift detected. Model health is good.")
        
        return recommendations

