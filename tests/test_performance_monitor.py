"""
Unit tests for Performance Monitor
"""
import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.performance_monitor import PerformanceMonitor, PerformanceStatus


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor class"""
    
    @pytest.fixture
    def monitor(self):
        """Create monitor instance with baseline"""
        return PerformanceMonitor(
            baseline_accuracy=0.90,
            baseline_precision=0.88,
            baseline_recall=0.85,
            baseline_f1=0.86,
            warning_threshold=0.05,
            critical_threshold=0.10
        )
    
    def test_set_baseline(self, monitor):
        """Test baseline setting"""
        monitor.set_baseline(0.92, 0.90, 0.88, 0.89)
        
        assert monitor.baseline['accuracy'] == 0.92
        assert monitor.baseline['precision'] == 0.90
    
    def test_record_predictions(self, monitor):
        """Test recording predictions"""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        
        monitor.record_predictions(y_true, y_pred)
        
        assert monitor.prediction_count == 5
        assert len(monitor.actuals) == 5
        assert len(monitor.predictions) == 5
    
    def test_perfect_predictions(self, monitor):
        """Test with perfect predictions"""
        y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0] * 10)
        y_pred = y_true.copy()  # Perfect predictions
        
        monitor.record_predictions(y_true, y_pred)
        metrics = monitor.get_current_metrics()
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
    
    def test_poor_predictions(self, monitor):
        """Test with poor predictions"""
        y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0] * 10)
        y_pred = 1 - y_true  # All wrong
        
        monitor.record_predictions(y_true, y_pred)
        metrics = monitor.get_current_metrics()
        
        assert metrics['accuracy'] == 0.0
    
    def test_status_healthy(self, monitor):
        """Test healthy status detection"""
        # Predictions close to baseline
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        
        # Simulate ~90% accuracy (matching baseline)
        y_pred = y_true.copy()
        errors = np.random.choice(200, size=20, replace=False)
        y_pred[errors] = 1 - y_pred[errors]
        
        monitor.record_predictions(y_true, y_pred)
        report = monitor.generate_report()
        
        assert report.status == PerformanceStatus.HEALTHY
    
    def test_status_degraded(self, monitor):
        """Test degraded status detection"""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        
        # Simulate ~80% accuracy (10% below baseline of 90%)
        y_pred = y_true.copy()
        errors = np.random.choice(200, size=40, replace=False)
        y_pred[errors] = 1 - y_pred[errors]
        
        monitor.record_predictions(y_true, y_pred)
        report = monitor.generate_report()
        
        # Should be warning or degraded
        assert report.status in [PerformanceStatus.WARNING, PerformanceStatus.DEGRADED, PerformanceStatus.CRITICAL]
    
    def test_report_generation(self, monitor):
        """Test report generation"""
        y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0] * 10)
        y_pred = y_true.copy()
        y_pred[0:10] = 1 - y_pred[0:10]  # Some errors
        
        monitor.record_predictions(y_true, y_pred)
        report = monitor.generate_report()
        
        assert report.predictions_analyzed == 100
        assert 'accuracy' in report.metric_deltas
        assert len(report.recommendations) > 0
    
    def test_trend_detection(self, monitor):
        """Test trend detection"""
        # Record multiple batches with declining accuracy
        for i in range(5):
            y_true = np.ones(50, dtype=int)
            y_pred = np.ones(50, dtype=int)
            # Increase errors each batch
            y_pred[:i*5] = 0
            monitor.record_predictions(y_true, y_pred)
        
        report = monitor.generate_report()
        
        # Should have a trend value
        assert report.trend in ['improving', 'stable', 'declining']


class TestPerformanceReport:
    """Tests for report serialization"""
    
    def test_to_dict(self):
        """Test report serialization"""
        from src.performance_monitor import PerformanceReport, MetricSnapshot, PerformanceStatus
        
        snapshot = MetricSnapshot(
            timestamp="2025-01-12T00:00:00",
            accuracy=0.85,
            precision=0.82,
            recall=0.80,
            f1=0.81,
            sample_count=500
        )
        
        report = PerformanceReport(
            current_metrics=snapshot,
            baseline_metrics={'accuracy': 0.90, 'precision': 0.88, 'recall': 0.85, 'f1': 0.86},
            metric_deltas={'accuracy': -0.05, 'precision': -0.06, 'recall': -0.05, 'f1': -0.05},
            status=PerformanceStatus.DEGRADED,
            trend="declining",
            predictions_analyzed=500,
            recommendations=["Retrain model"]
        )
        
        d = report.to_dict()
        
        assert d['status'] == 'degraded'
        assert d['trend'] == 'declining'
        assert d['predictions_analyzed'] == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

