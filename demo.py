"""
Demo Script for MCP ML Model Monitoring Agent

This script demonstrates the full capabilities of the monitoring system:
1. Simulates training data (reference)
2. Simulates production data with drift
3. Shows drift detection results
4. Demonstrates performance monitoring
5. Generates health reports and recommendations

"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Import our monitoring components
from src.drift_detector import DriftDetector
from src.performance_monitor import PerformanceMonitor
from src.alert_system import AlertSystem


def print_header(title: str):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def create_synthetic_data():
    """Create synthetic dataset for demonstration"""
    print_header("1. Creating Synthetic ML Dataset")
    
    # Generate classification dataset
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=15,
        n_redundant=3,
        n_clusters_per_class=2,
        random_state=42
    )
    
    # Convert to DataFrame with feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"Created dataset with {len(df)} samples, {len(feature_names)} features")
    print(f"Target distribution: {dict(pd.Series(y).value_counts())}")
    
    return df, feature_names


def simulate_data_drift(df: pd.DataFrame, drift_features: int = 5):
    """Simulate data drift in production data"""
    print_header("2. Simulating Production Data with Drift")
    
    production_df = df.copy()
    
    # Select random features to drift
    numeric_cols = [c for c in df.columns if c != 'target']
    drifted_cols = np.random.choice(numeric_cols, size=drift_features, replace=False)
    
    print(f"Introducing drift in features: {list(drifted_cols)}")
    
    for col in drifted_cols:
        # Shift mean and increase variance
        shift = np.random.uniform(0.5, 2.0) * production_df[col].std()
        scale = np.random.uniform(1.2, 1.8)
        
        production_df[col] = production_df[col] * scale + shift
    
    print(f"Created production dataset with drift in {drift_features} features")
    
    return production_df, list(drifted_cols)


def train_model(train_df: pd.DataFrame):
    """Train a simple model for demonstration"""
    print_header("3. Training Classification Model")
    
    X = train_df.drop('target', axis=1)
    y = train_df['target']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Get baseline metrics
    y_pred_val = model.predict(X_val)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    baseline = {
        'accuracy': accuracy_score(y_val, y_pred_val),
        'precision': precision_score(y_val, y_pred_val),
        'recall': recall_score(y_val, y_pred_val),
        'f1': f1_score(y_val, y_pred_val)
    }
    
    print(f"Model trained successfully")
    print(f"  Baseline Accuracy:  {baseline['accuracy']:.4f}")
    print(f"  Baseline Precision: {baseline['precision']:.4f}")
    print(f"  Baseline Recall:    {baseline['recall']:.4f}")
    print(f"  Baseline F1:        {baseline['f1']:.4f}")
    
    return model, baseline, X_train


def run_drift_detection(reference_data: pd.DataFrame, production_data: pd.DataFrame):
    """Run drift detection and show results"""
    print_header("4. Running Data Drift Detection")
    
    # Initialize detector
    detector = DriftDetector(
        ks_threshold=0.05,
        psi_warning=0.1,
        psi_critical=0.2,
        wasserstein_threshold=0.15
    )
    
    # Set reference (training) data
    detector.set_reference(reference_data)
    print("Reference data loaded")
    
    # Detect drift - ALL RESULTS ARE DYNAMICALLY COMPUTED
    report = detector.detect_drift(production_data)
    
    print(f"\nDRIFT DETECTION RESULTS:")
    print(f"  Total Features Analyzed: {report.total_features}")
    print(f"  Features with Drift:     {report.drifted_features}")
    print(f"  Drift Percentage:        {report.drift_percentage:.1f}%")
    print(f"  Overall Severity:        {report.overall_severity.value.upper()}")
    
    print(f"\nFeature-Level Results:")
    print("-" * 70)
    print(f"{'Feature':<15} {'KS Stat':<10} {'PSI':<10} {'Wasserstein':<12} {'Status':<10}")
    print("-" * 70)
    
    for result in sorted(report.feature_results, key=lambda x: x.psi_score, reverse=True)[:10]:
        status = "DRIFT" if result.is_drifted else "OK"
        print(f"{result.feature_name:<15} {result.ks_statistic:<10.4f} {result.psi_score:<10.4f} {result.wasserstein_dist:<12.4f} {status:<10}")
    
    print("\nRECOMMENDATIONS (Dynamically Generated):")
    for rec in report.recommendations:
        print(f"  - {rec}")
    
    return report


def run_performance_monitoring(model, production_data: pd.DataFrame, baseline: dict):
    """Run performance monitoring with simulated degradation"""
    print_header("5. Running Performance Monitoring")
    
    # Initialize monitor
    monitor = PerformanceMonitor(
        baseline_accuracy=baseline['accuracy'],
        baseline_precision=baseline['precision'],
        baseline_recall=baseline['recall'],
        baseline_f1=baseline['f1'],
        warning_threshold=0.05,
        critical_threshold=0.10
    )
    
    print("Performance monitor initialized with baseline metrics")
    
    # Get predictions on production data (which has drift - so performance will degrade)
    X_prod = production_data.drop('target', axis=1)
    y_true = production_data['target'].values
    y_pred = model.predict(X_prod)
    
    # Record predictions
    monitor.record_predictions(y_true, y_pred)
    
    # Generate report - ALL METRICS ARE DYNAMICALLY COMPUTED
    report = monitor.generate_report()
    
    print(f"\nPERFORMANCE MONITORING RESULTS:")
    print(f"  Predictions Analyzed: {report.predictions_analyzed}")
    print(f"  Status:              {report.status.value.upper()}")
    print(f"  Trend:               {report.trend}")
    
    print(f"\nCurrent vs Baseline Metrics:")
    print("-" * 50)
    print(f"{'Metric':<15} {'Current':<12} {'Baseline':<12} {'Delta':<12}")
    print("-" * 50)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        current = getattr(report.current_metrics, metric)
        baseline_val = report.baseline_metrics[metric]
        delta = report.metric_deltas[metric]
        delta_str = f"{delta:+.4f}"
        print(f"{metric.capitalize():<15} {current:<12.4f} {baseline_val:<12.4f} {delta_str:<12}")
    
    print("\nRECOMMENDATIONS (Dynamically Generated):")
    for rec in report.recommendations:
        print(f"  - {rec}")
    
    return report


def run_health_check(drift_report, performance_report):
    """Generate comprehensive health summary"""
    print_header("6. Model Health Summary & Retraining Recommendation")
    
    alert_system = AlertSystem()
    
    # Process reports to generate alerts - ALERTS ARE DYNAMICALLY GENERATED
    drift_alerts = alert_system.process_drift_report(drift_report)
    perf_alerts = alert_system.process_performance_report(performance_report)
    
    # Generate health summary
    summary = alert_system.generate_health_summary(drift_report, performance_report)
    
    print(f"\nMODEL HEALTH SUMMARY:")
    print(f"  Overall Status:     {summary.overall_status.upper()}")
    print(f"  Drift Status:       {summary.drift_status}")
    print(f"  Performance Status: {summary.performance_status}")
    print(f"  Active Alerts:      {summary.active_alerts}")
    print(f"  Last Check:         {summary.last_check}")
    
    print(f"\nACTIVE ALERTS ({len(drift_alerts) + len(perf_alerts)}):")
    for alert in (drift_alerts + perf_alerts):
        priority_label = alert.priority.value.upper()
        print(f"  ({priority_label}) {alert.title}")
        print(f"     {alert.message}")
    
    print(f"\nRETRAINING RECOMMENDATION:")
    retrain = summary.retraining
    print(f"  Should Retrain:    {'YES' if retrain.should_retrain else 'NO'}")
    print(f"  Urgency:           {retrain.urgency.upper()}")
    print(f"  Estimated Impact:  {retrain.estimated_impact}")
    
    if retrain.reasons:
        print(f"\n  Reasons:")
        for reason in retrain.reasons:
            print(f"    - {reason}")
    
    if retrain.affected_features:
        print(f"\n  Affected Features: {', '.join(retrain.affected_features[:5])}")
    
    print(f"\n  Suggested Actions:")
    for action in retrain.suggested_actions:
        print(f"    -> {action}")
    
    return summary


def generate_json_report(drift_report, performance_report, health_summary):
    """Generate JSON report for integration"""
    print_header("7. Generating JSON Report")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_name": "demo_model",
        "version": "1.0.0",
        "drift_analysis": drift_report.to_dict(),
        "performance_analysis": performance_report.to_dict(),
        "health_summary": health_summary.to_dict()
    }
    
    # Save to file
    output_file = "monitoring_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to: {output_file}")
    print(f"\nReport Preview:")
    print(json.dumps({
        "overall_status": report["health_summary"]["overall_status"],
        "drift_percentage": report["drift_analysis"]["drift_percentage"],
        "current_accuracy": report["performance_analysis"]["current_metrics"]["accuracy"],
        "should_retrain": report["health_summary"]["retraining"]["should_retrain"],
        "urgency": report["health_summary"]["retraining"]["urgency"]
    }, indent=2))
    
    return report


def main():
    """Run the complete demo"""
    print("\n" + "=" * 60)
    print("  MCP ML MODEL MONITORING AGENT - DEMO")
    print("  Demonstrating Data Drift Detection & Performance Monitoring")
    print("=" * 60)
    
    # Step 1: Create synthetic data
    train_df, feature_names = create_synthetic_data()
    
    # Step 2: Simulate production data with drift
    production_df, drifted_features = simulate_data_drift(train_df, drift_features=5)
    
    # Step 3: Train model
    model, baseline, X_train = train_model(train_df)
    
    # Step 4: Run drift detection
    drift_report = run_drift_detection(
        reference_data=X_train,
        production_data=production_df.drop('target', axis=1)
    )
    
    # Step 5: Run performance monitoring
    performance_report = run_performance_monitoring(model, production_df, baseline)
    
    # Step 6: Generate health summary
    health_summary = run_health_check(drift_report, performance_report)
    
    # Step 7: Generate JSON report
    json_report = generate_json_report(drift_report, performance_report, health_summary)
    
    print_header("DEMO COMPLETE")
    print("""
    Successfully demonstrated:
       - Data drift detection using KS-test, PSI, and Wasserstein distance
       - Performance monitoring with accuracy/precision/recall/F1 tracking
       - Automated alerting based on severity thresholds
       - Retraining recommendations with urgency levels
       - JSON report generation for integration
    
    Output: monitoring_report.json
    
    Ready for production use with MCP integration!
    """)


if __name__ == "__main__":
    main()
