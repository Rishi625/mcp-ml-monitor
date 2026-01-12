"""
Unit tests for Drift Detector
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.drift_detector import DriftDetector, DriftSeverity


class TestDriftDetector:
    """Tests for DriftDetector class"""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance"""
        return DriftDetector(
            ks_threshold=0.05,
            psi_warning=0.1,
            psi_critical=0.2
        )
    
    @pytest.fixture
    def reference_data(self):
        """Create reference dataset"""
        np.random.seed(42)
        return pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.normal(5, 2, 1000),
            'feature_3': np.random.uniform(0, 10, 1000)
        })
    
    def test_set_reference(self, detector, reference_data):
        """Test setting reference data"""
        detector.set_reference(reference_data)
        
        assert detector.reference_data is not None
        assert len(detector.reference_stats) == 3
        assert 'feature_1' in detector.reference_stats
    
    def test_no_drift_detection(self, detector, reference_data):
        """Test detection when no drift exists"""
        detector.set_reference(reference_data)
        
        # Similar data should not show drift
        np.random.seed(43)  # Different seed, same distribution
        current_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 500),
            'feature_2': np.random.normal(5, 2, 500),
            'feature_3': np.random.uniform(0, 10, 500)
        })
        
        report = detector.detect_drift(current_data)
        
        # Should have no or minimal drift
        assert report.drifted_features <= 1
        assert report.overall_severity in [DriftSeverity.NONE, DriftSeverity.LOW]
    
    def test_drift_detection(self, detector, reference_data):
        """Test detection when drift exists"""
        detector.set_reference(reference_data)
        
        # Significantly shifted data
        np.random.seed(42)
        current_data = pd.DataFrame({
            'feature_1': np.random.normal(3, 1, 500),  # Shifted mean
            'feature_2': np.random.normal(5, 5, 500),  # Increased variance
            'feature_3': np.random.uniform(5, 15, 500)  # Shifted range
        })
        
        report = detector.detect_drift(current_data)
        
        # Should detect drift
        assert report.drifted_features >= 2
        assert report.overall_severity in [DriftSeverity.WARNING, DriftSeverity.CRITICAL]
    
    def test_psi_calculation(self, detector):
        """Test PSI calculation"""
        # Identical distributions should have PSI ≈ 0
        reference = np.random.normal(0, 1, 1000)
        current = np.random.normal(0, 1, 1000)
        
        psi = detector.calculate_psi(reference, current)
        assert psi < 0.1  # Should be low
        
        # Very different distributions should have high PSI
        shifted = np.random.normal(5, 1, 1000)
        psi_shifted = detector.calculate_psi(reference, shifted)
        assert psi_shifted > 0.2  # Should be high
    
    def test_ks_test(self, detector):
        """Test Kolmogorov-Smirnov test"""
        np.random.seed(42)
        reference = np.random.normal(0, 1, 1000)
        similar = np.random.normal(0, 1, 1000)
        different = np.random.normal(5, 1, 1000)
        
        # Similar distributions
        stat, pvalue = detector.calculate_ks_test(reference, similar)
        assert pvalue > 0.05  # Should not reject null hypothesis
        
        # Different distributions
        stat, pvalue = detector.calculate_ks_test(reference, different)
        assert pvalue < 0.05  # Should reject null hypothesis
    
    def test_recommendations_generated(self, detector, reference_data):
        """Test that recommendations are generated"""
        detector.set_reference(reference_data)
        
        # Create drifted data
        current_data = pd.DataFrame({
            'feature_1': np.random.normal(10, 1, 500),
            'feature_2': np.random.normal(20, 5, 500),
            'feature_3': np.random.uniform(50, 100, 500)
        })
        
        report = detector.detect_drift(current_data)
        
        # Should have recommendations
        assert len(report.recommendations) > 0


class TestDriftReport:
    """Tests for DriftReport serialization"""
    
    def test_to_dict(self):
        """Test report serialization"""
        from src.drift_detector import DriftReport, FeatureDriftResult, DriftSeverity
        
        result = FeatureDriftResult(
            feature_name="test_feature",
            ks_statistic=0.15,
            ks_pvalue=0.001,
            psi_score=0.25,
            wasserstein_dist=0.18,
            severity=DriftSeverity.CRITICAL,
            is_drifted=True
        )
        
        report = DriftReport(
            total_features=10,
            drifted_features=3,
            feature_results=[result],
            overall_severity=DriftSeverity.CRITICAL,
            drift_percentage=30.0,
            recommendations=["Retrain model"]
        )
        
        d = report.to_dict()
        
        assert d['total_features'] == 10
        assert d['drifted_features'] == 3
        assert d['overall_severity'] == 'critical'
        assert len(d['feature_results']) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

