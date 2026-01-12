"""
Configuration for MCP ML Model Monitoring Agent
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
import os


@dataclass
class DriftConfig:
    """Configuration for drift detection"""
    # Kolmogorov-Smirnov test threshold
    ks_threshold: float = 0.05
    
    # Population Stability Index thresholds
    psi_warning_threshold: float = 0.1
    psi_critical_threshold: float = 0.2
    
    # Wasserstein distance threshold (relative)
    wasserstein_threshold: float = 0.15
    
    # Minimum samples required for drift detection
    min_samples: int = 100
    
    # Features to monitor (empty = all features)
    monitored_features: List[str] = field(default_factory=list)


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring"""
    # Baseline metrics (set during training)
    baseline_accuracy: float = 0.0
    baseline_precision: float = 0.0
    baseline_recall: float = 0.0
    baseline_f1: float = 0.0
    
    # Degradation thresholds (relative drop)
    accuracy_warning_threshold: float = 0.05  # 5% drop
    accuracy_critical_threshold: float = 0.10  # 10% drop
    
    # Monitoring window (number of predictions)
    monitoring_window: int = 1000
    
    # Minimum predictions before alerting
    min_predictions: int = 100


@dataclass 
class AlertConfig:
    """Configuration for alert system"""
    # Enable/disable alerts
    alerts_enabled: bool = True
    
    # Severity levels
    enable_critical_alerts: bool = True
    enable_warning_alerts: bool = True
    
    # Retraining recommendation thresholds
    retrain_on_critical_drift: bool = True
    retrain_on_performance_drop: float = 0.08  # 8% performance drop
    
    # Alert cooldown (seconds between alerts for same issue)
    alert_cooldown: int = 3600  # 1 hour


@dataclass
class MonitorConfig:
    """Main configuration for the monitoring agent"""
    model_name: str = "default_model"
    model_version: str = "1.0.0"
    
    drift: DriftConfig = field(default_factory=DriftConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    @classmethod
    def from_json(cls, path: str) -> "MonitorConfig":
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            model_name=data.get('model_name', 'default_model'),
            model_version=data.get('model_version', '1.0.0'),
            drift=DriftConfig(**data.get('drift', {})),
            performance=PerformanceConfig(**data.get('performance', {})),
            alerts=AlertConfig(**data.get('alerts', {})),
            log_level=data.get('log_level', 'INFO'),
            log_file=data.get('log_file')
        )
    
    def to_json(self, path: str) -> None:
        """Save configuration to JSON file"""
        data = {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'drift': {
                'ks_threshold': self.drift.ks_threshold,
                'psi_warning_threshold': self.drift.psi_warning_threshold,
                'psi_critical_threshold': self.drift.psi_critical_threshold,
                'wasserstein_threshold': self.drift.wasserstein_threshold,
                'min_samples': self.drift.min_samples,
                'monitored_features': self.drift.monitored_features,
            },
            'performance': {
                'baseline_accuracy': self.performance.baseline_accuracy,
                'baseline_precision': self.performance.baseline_precision,
                'baseline_recall': self.performance.baseline_recall,
                'baseline_f1': self.performance.baseline_f1,
                'accuracy_warning_threshold': self.performance.accuracy_warning_threshold,
                'accuracy_critical_threshold': self.performance.accuracy_critical_threshold,
                'monitoring_window': self.performance.monitoring_window,
                'min_predictions': self.performance.min_predictions,
            },
            'alerts': {
                'alerts_enabled': self.alerts.alerts_enabled,
                'enable_critical_alerts': self.alerts.enable_critical_alerts,
                'enable_warning_alerts': self.alerts.enable_warning_alerts,
                'retrain_on_critical_drift': self.alerts.retrain_on_critical_drift,
                'retrain_on_performance_drop': self.alerts.retrain_on_performance_drop,
                'alert_cooldown': self.alerts.alert_cooldown,
            },
            'log_level': self.log_level,
            'log_file': self.log_file,
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# Default configuration
DEFAULT_CONFIG = MonitorConfig()

