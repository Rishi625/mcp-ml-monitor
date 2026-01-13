"""
LangFlow ML Monitoring Pipeline Orchestrator

Main entry point for running the monitoring pipeline.
Can be used standalone or integrated with LangFlow.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path

import pandas as pd
import numpy as np

# Import components
from components.data_ingestion import (
    CSVIngestionComponent,
    DataValidatorComponent,
    StreamIngestionComponent
)
from components.ml_monitoring import (
    DriftDetectionComponent,
    AnomalyDetectionComponent,
    PerformanceAnalysisComponent,
    TimeSeriesForecastComponent
)
from components.intelligent_response import (
    RootCauseAnalysisComponent,
    RemedySuggestionComponent,
    CodeGeneratorComponent,
    AnsiblePlaybookComponent
)
from components.integrations import (
    SlackAlertComponent,
    GitHubIssueComponent,
    PrometheusConnectorComponent
)
from components.advanced_features import (
    MultiModelComparisonComponent,
    ABTestingCoordinator,
    AutomatedRetrainingPipeline
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLMonitoringPipeline:
    """
    Orchestrates the complete ML monitoring pipeline.
    
    Flow:
    1. Ingest data (batch or streaming)
    2. Validate data quality
    3. Detect drift and anomalies
    4. Analyze performance
    5. Generate root cause analysis
    6. Suggest remedies
    7. Send alerts and create issues
    8. Trigger retraining if needed
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.drift_detector = DriftDetectionComponent(
            ks_threshold=self.config.get('ks_threshold', 0.05),
            psi_warning=self.config.get('psi_warning', 0.1),
            psi_critical=self.config.get('psi_critical', 0.2)
        )
        
        self.anomaly_detector = AnomalyDetectionComponent(
            contamination=self.config.get('contamination', 0.1)
        )
        
        self.performance_analyzer = PerformanceAnalysisComponent(
            warning_threshold=self.config.get('warning_threshold', 0.05),
            critical_threshold=self.config.get('critical_threshold', 0.10)
        )
        
        self.root_cause_analyzer = RootCauseAnalysisComponent()
        self.remedy_suggester = RemedySuggestionComponent()
        self.code_generator = CodeGeneratorComponent()
        
        self.retraining_pipeline = AutomatedRetrainingPipeline()
        
        # State
        self.reference_set = False
        self.baseline_set = False
        self.last_run: Optional[Dict[str, Any]] = None
    
    def set_reference_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Set reference data for drift detection"""
        result = self.drift_detector.set_reference(data)
        self.anomaly_detector.fit(data)
        self.reference_set = True
        logger.info(f"Reference data set: {result}")
        return result
    
    def set_baseline_metrics(
        self,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float
    ) -> Dict[str, Any]:
        """Set baseline performance metrics"""
        result = self.performance_analyzer.set_baseline(accuracy, precision, recall, f1)
        self.baseline_set = True
        logger.info(f"Baseline metrics set: {result}")
        return result
    
    def run(
        self,
        current_data: pd.DataFrame,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        send_alerts: bool = True
    ) -> Dict[str, Any]:
        """Run the complete monitoring pipeline"""
        
        if not self.reference_set:
            raise ValueError("Reference data not set. Call set_reference_data() first.")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "status": "running"
        }
        
        # 1. Data Validation
        validator = DataValidatorComponent()
        validation = validator.validate(current_data.to_dict(orient='records'))
        results["validation"] = validation.dict() if hasattr(validation, 'dict') else str(validation)
        
        # 2. Drift Detection
        drift_report = self.drift_detector.detect(current_data)
        results["drift"] = drift_report
        logger.info(f"Drift detected: {drift_report['drifted_features']}/{drift_report['total_features']} features")
        
        # 3. Anomaly Detection
        anomaly_report = self.anomaly_detector.detect(current_data)
        results["anomalies"] = anomaly_report
        logger.info(f"Anomalies detected: {anomaly_report['anomaly_count']} samples")
        
        # 4. Performance Analysis (if predictions provided)
        if y_true is not None and y_pred is not None:
            performance_report = self.performance_analyzer.analyze(y_true, y_pred)
            results["performance"] = performance_report
            logger.info(f"Performance status: {performance_report['status']}")
        else:
            performance_report = {"status": "not_analyzed", "current_metrics": {}, "deltas": {}}
            results["performance"] = performance_report
        
        # 5. Root Cause Analysis
        analysis = self.root_cause_analyzer.analyze(drift_report, performance_report)
        results["root_cause"] = analysis
        
        # 6. Remedy Suggestions
        remedies = self.remedy_suggester.suggest(drift_report, performance_report, anomaly_report)
        results["remedies"] = remedies
        
        # 7. Evaluate Retraining Triggers
        if self.config.get('model_id'):
            retrain_decision = self.retraining_pipeline.evaluate_triggers(
                model_id=self.config['model_id'],
                drift_score=drift_report.get('drift_percentage', 0) / 100,
                performance_delta=performance_report.get('deltas', {}).get('accuracy', 0)
            )
            results["retraining"] = retrain_decision
        
        # 8. Determine overall status
        if drift_report.get('overall_severity') == 'critical' or performance_report.get('status') == 'critical':
            results["status"] = "critical"
        elif drift_report.get('overall_severity') == 'warning' or performance_report.get('status') in ['warning', 'degraded']:
            results["status"] = "warning"
        else:
            results["status"] = "healthy"
        
        self.last_run = results
        return results
    
    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate a comprehensive report from the last run"""
        
        if self.last_run is None:
            return {"error": "No monitoring run executed yet"}
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "status": self.last_run.get("status"),
                "drift_severity": self.last_run.get("drift", {}).get("overall_severity"),
                "performance_status": self.last_run.get("performance", {}).get("status"),
                "anomaly_count": self.last_run.get("anomalies", {}).get("anomaly_count", 0)
            },
            "details": self.last_run,
            "recommendations": self.last_run.get("remedies", {}).get("remedies", [])
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Report saved to {output_path}")
        
        return report


def create_demo_pipeline():
    """Create a demo pipeline with synthetic data"""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=15,
        random_state=42
    )
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    train_df = pd.DataFrame(X, columns=feature_names)
    
    # Split for training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create pipeline
    pipeline = MLMonitoringPipeline({
        'model_id': 'demo_model',
        'ks_threshold': 0.05,
        'psi_warning': 0.1,
        'psi_critical': 0.2
    })
    
    # Set reference
    pipeline.set_reference_data(pd.DataFrame(X_train, columns=feature_names))
    
    # Set baseline
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    y_pred_train = model.predict(X_train)
    pipeline.set_baseline_metrics(
        accuracy=accuracy_score(y_train, y_pred_train),
        precision=precision_score(y_train, y_pred_train),
        recall=recall_score(y_train, y_pred_train),
        f1=f1_score(y_train, y_pred_train)
    )
    
    # Simulate production data with drift
    production_df = pd.DataFrame(X_test, columns=feature_names)
    # Add drift to some features
    for col in feature_names[:5]:
        production_df[col] = production_df[col] * 1.5 + 0.5
    
    # Run pipeline
    y_pred = model.predict(production_df.values)
    results = pipeline.run(
        current_data=production_df,
        y_true=y_test,
        y_pred=y_pred
    )
    
    # Generate report
    report = pipeline.generate_report("monitoring_report.json")
    
    return pipeline, report


if __name__ == "__main__":
    print("=" * 60)
    print("  LangFlow ML Monitoring Pipeline - Demo")
    print("=" * 60)
    
    pipeline, report = create_demo_pipeline()
    
    print(f"\nPipeline Status: {report['summary']['status'].upper()}")
    print(f"Drift Severity: {report['summary']['drift_severity']}")
    print(f"Performance Status: {report['summary']['performance_status']}")
    print(f"Anomalies Detected: {report['summary']['anomaly_count']}")
    
    print("\nRecommendations:")
    for rec in report['recommendations'][:5]:
        print(f"  - {rec}")
    
    print(f"\nFull report saved to: monitoring_report.json")

