"""
Intelligent Response Components for LangFlow ML Monitoring Pipeline

LLM-powered components for:
- Root cause analysis
- Automated remedy suggestion
- Code generation for fixes
- Ansible playbook creation
- SHAP explainability reports
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import json
from string import Template


class RootCauseAnalysisComponent:
    """
    LangFlow Component: LLM Root Cause Analysis
    
    Uses LLM to analyze drift and performance issues and identify root causes.
    """
    
    display_name = "Root Cause Analysis"
    description = "LLM-powered root cause analysis for ML issues"
    
    ANALYSIS_PROMPT = Template("""You are an ML Operations expert analyzing a production ML system.

## Current Issue
$issue_type: $issue_summary

## Drift Analysis
$drift_data

## Performance Metrics
$performance_data

## Recent Changes
$recent_changes

Based on this information, provide:

1. **Root Cause Analysis**: What is likely causing this issue?
2. **Contributing Factors**: What other factors may be involved?
3. **Impact Assessment**: How does this affect production predictions?
4. **Immediate Actions**: What should be done right now?
5. **Long-term Fixes**: What systemic changes are needed?

Be specific and actionable. Reference specific features and metrics.""")

    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client
    
    def analyze(
        self,
        drift_report: Dict[str, Any],
        performance_report: Dict[str, Any],
        recent_changes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Perform root cause analysis using LLM"""
        
        # Determine issue type
        if drift_report.get('overall_severity') == 'critical':
            issue_type = "Critical Data Drift"
            issue_summary = f"{drift_report.get('drifted_features', 0)} features showing significant distribution shift"
        elif performance_report.get('status') == 'critical':
            issue_type = "Performance Degradation"
            issue_summary = f"Accuracy dropped by {abs(performance_report.get('deltas', {}).get('accuracy', 0))*100:.1f}%"
        else:
            issue_type = "Monitoring Alert"
            issue_summary = "Potential issues detected requiring attention"
        
        # Format data for prompt
        drift_data = self._format_drift_data(drift_report)
        perf_data = self._format_performance_data(performance_report)
        changes = "\n".join(f"- {c}" for c in (recent_changes or ["No recent changes logged"]))
        
        prompt = self.ANALYSIS_PROMPT.substitute(
            issue_type=issue_type,
            issue_summary=issue_summary,
            drift_data=drift_data,
            performance_data=perf_data,
            recent_changes=changes
        )
        
        # Call LLM (mock response if no client)
        if self.llm_client:
            analysis = self._call_llm(prompt)
        else:
            analysis = self._generate_rule_based_analysis(drift_report, performance_report)
        
        return {
            "issue_type": issue_type,
            "issue_summary": issue_summary,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
    
    def _format_drift_data(self, report: Dict[str, Any]) -> str:
        lines = [
            f"- Overall Severity: {report.get('overall_severity', 'unknown')}",
            f"- Drifted Features: {report.get('drifted_features', 0)}/{report.get('total_features', 0)}",
            f"- Drift Percentage: {report.get('drift_percentage', 0)}%"
        ]
        
        for feat in report.get('feature_results', [])[:5]:
            if feat.get('is_drifted'):
                lines.append(f"  - {feat['feature']}: PSI={feat['psi_score']:.3f}")
        
        return "\n".join(lines)
    
    def _format_performance_data(self, report: Dict[str, Any]) -> str:
        current = report.get('current_metrics', {})
        deltas = report.get('deltas', {})
        
        lines = [f"- Status: {report.get('status', 'unknown')}"]
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            if metric in current:
                delta = deltas.get(metric, 0)
                lines.append(f"- {metric.capitalize()}: {current[metric]:.4f} (delta: {delta:+.4f})")
        
        return "\n".join(lines)
    
    def _generate_rule_based_analysis(
        self,
        drift_report: Dict[str, Any],
        performance_report: Dict[str, Any]
    ) -> str:
        """Generate analysis without LLM using rules"""
        analysis = []
        
        # Analyze drift
        if drift_report.get('overall_severity') == 'critical':
            drifted = [f['feature'] for f in drift_report.get('feature_results', []) if f.get('is_drifted')]
            analysis.append(f"ROOT CAUSE: Significant data drift in features: {', '.join(drifted[:5])}")
            analysis.append("LIKELY CAUSE: Upstream data pipeline changes or seasonal shifts")
            analysis.append("IMMEDIATE ACTION: Investigate recent ETL changes and data source modifications")
        
        # Analyze performance
        if performance_report.get('status') in ['critical', 'warning']:
            analysis.append("PERFORMANCE IMPACT: Model predictions may be unreliable")
            analysis.append("RECOMMENDATION: Prepare retraining with recent production data")
        
        if not analysis:
            analysis.append("No critical issues detected. Continue monitoring.")
        
        return "\n".join(analysis)
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API"""
        # Implementation depends on LLM client (OpenAI, etc.)
        response = self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content


class RemedySuggestionComponent:
    """
    LangFlow Component: Automated Remedy Suggestion
    
    Generates specific remediation steps based on detected issues.
    """
    
    display_name = "Remedy Suggestion"
    description = "Generate automated remediation suggestions"
    
    def __init__(self):
        self.remedy_templates = {
            'drift_critical': [
                "Trigger immediate model retraining pipeline",
                "Enable shadow mode for current model",
                "Activate fallback model if available",
                "Alert on-call ML engineer",
                "Begin root cause investigation"
            ],
            'drift_warning': [
                "Schedule retraining within 24-48 hours",
                "Increase monitoring frequency",
                "Prepare training data refresh",
                "Review feature engineering pipeline"
            ],
            'performance_critical': [
                "Roll back to previous model version",
                "Enable model circuit breaker",
                "Switch to rule-based fallback",
                "Escalate to ML team lead"
            ],
            'performance_warning': [
                "Monitor closely for next 24 hours",
                "Prepare retraining pipeline",
                "Review recent prediction samples"
            ],
            'anomaly_detected': [
                "Quarantine anomalous data points",
                "Investigate data source integrity",
                "Check for upstream system issues"
            ]
        }
    
    def suggest(
        self,
        drift_report: Dict[str, Any],
        performance_report: Dict[str, Any],
        anomaly_report: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate remediation suggestions"""
        
        remedies = []
        urgency = "low"
        
        # Check drift
        drift_severity = drift_report.get('overall_severity', 'none')
        if drift_severity == 'critical':
            remedies.extend(self.remedy_templates['drift_critical'])
            urgency = "immediate"
        elif drift_severity == 'warning':
            remedies.extend(self.remedy_templates['drift_warning'])
            if urgency != "immediate":
                urgency = "soon"
        
        # Check performance
        perf_status = performance_report.get('status', 'healthy')
        if perf_status == 'critical':
            remedies.extend(self.remedy_templates['performance_critical'])
            urgency = "immediate"
        elif perf_status in ['warning', 'degraded']:
            remedies.extend(self.remedy_templates['performance_warning'])
            if urgency != "immediate":
                urgency = "soon"
        
        # Check anomalies
        if anomaly_report and anomaly_report.get('severity') == 'critical':
            remedies.extend(self.remedy_templates['anomaly_detected'])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_remedies = []
        for r in remedies:
            if r not in seen:
                seen.add(r)
                unique_remedies.append(r)
        
        return {
            "urgency": urgency,
            "remedies": unique_remedies,
            "total_actions": len(unique_remedies),
            "timestamp": datetime.now().isoformat()
        }


class CodeGeneratorComponent:
    """
    LangFlow Component: Code Generation for Fixes
    
    Generates Python code snippets for common fixes.
    """
    
    display_name = "Code Generator"
    description = "Generate code snippets for ML fixes"
    
    def generate_retraining_script(
        self,
        model_name: str,
        data_source: str,
        features: List[str],
        target: str
    ) -> str:
        """Generate model retraining script"""
        
        return f'''"""
Auto-generated retraining script for {model_name}
Generated at: {datetime.now().isoformat()}
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load data
print("Loading data from {data_source}...")
data = pd.read_parquet("{data_source}")

# Prepare features
features = {features}
target = "{target}"

X = data[features]
y = data[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
model_path = f"{model_name}_retrained.joblib"
joblib.dump(model, model_path)
print(f"\\nModel saved to {{model_path}}")
'''

    def generate_feature_fix(
        self,
        feature_name: str,
        issue_type: str,
        current_stats: Dict[str, float],
        reference_stats: Dict[str, float]
    ) -> str:
        """Generate feature preprocessing fix"""
        
        return f'''"""
Feature fix for: {feature_name}
Issue: {issue_type}
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

def fix_{feature_name}(df):
    """
    Apply fix for {feature_name} drift
    
    Reference stats:
    - Mean: {reference_stats.get('mean', 'N/A')}
    - Std: {reference_stats.get('std', 'N/A')}
    
    Current stats:
    - Mean: {current_stats.get('mean', 'N/A')}
    - Std: {current_stats.get('std', 'N/A')}
    """
    
    # Option 1: Re-normalize to reference distribution
    ref_mean = {reference_stats.get('mean', 0)}
    ref_std = {reference_stats.get('std', 1)}
    
    df['{feature_name}_normalized'] = (df['{feature_name}'] - df['{feature_name}'].mean()) / df['{feature_name}'].std()
    df['{feature_name}_aligned'] = df['{feature_name}_normalized'] * ref_std + ref_mean
    
    return df

# Apply fix
# df = fix_{feature_name}(df)
'''


class AnsiblePlaybookComponent:
    """
    LangFlow Component: Ansible Playbook Creator
    
    Generates Ansible playbooks for automated remediation.
    """
    
    display_name = "Ansible Playbook"
    description = "Generate Ansible playbooks for remediation"
    
    def generate_rollback_playbook(
        self,
        model_name: str,
        previous_version: str,
        namespace: str = "ml-models"
    ) -> str:
        """Generate model rollback playbook"""
        
        return f'''---
# Auto-generated Ansible playbook for model rollback
# Model: {model_name}
# Target Version: {previous_version}
# Generated: {datetime.now().isoformat()}

- name: Rollback ML Model {model_name}
  hosts: ml_servers
  become: yes
  vars:
    model_name: "{model_name}"
    target_version: "{previous_version}"
    namespace: "{namespace}"
    model_registry: "s3://ml-models/registry"
  
  tasks:
    - name: Check current model version
      shell: kubectl get deployment {{{{ model_name }}}} -n {{{{ namespace }}}} -o jsonpath='{{{{.spec.template.spec.containers[0].image}}}}'
      register: current_version
      
    - name: Log current version
      debug:
        msg: "Current version: {{{{ current_version.stdout }}}}"
    
    - name: Download previous model version
      aws_s3:
        bucket: ml-models
        object: "registry/{{{{ model_name }}}}/{{{{ target_version }}}}/model.tar.gz"
        dest: "/tmp/{{{{ model_name }}}}_{{{{ target_version }}}}.tar.gz"
        mode: get
    
    - name: Update Kubernetes deployment
      kubernetes.core.k8s:
        state: present
        definition:
          apiVersion: apps/v1
          kind: Deployment
          metadata:
            name: "{{{{ model_name }}}}"
            namespace: "{{{{ namespace }}}}"
          spec:
            replicas: 3
            template:
              spec:
                containers:
                  - name: model-server
                    image: "ml-registry/{{{{ model_name }}}}:{{{{ target_version }}}}"
    
    - name: Wait for rollout
      shell: kubectl rollout status deployment/{{{{ model_name }}}} -n {{{{ namespace }}}} --timeout=300s
      
    - name: Verify model health
      uri:
        url: "http://{{{{ model_name }}}}.{{{{ namespace }}}}.svc.cluster.local/health"
        method: GET
        return_content: yes
      register: health_check
      until: health_check.status == 200
      retries: 10
      delay: 5
    
    - name: Send notification
      slack:
        token: "{{{{ lookup('env', 'SLACK_TOKEN') }}}}"
        channel: "#ml-ops"
        msg: "Model {{{{ model_name }}}} rolled back to version {{{{ target_version }}}}"
'''

    def generate_retraining_playbook(
        self,
        model_name: str,
        training_config: Dict[str, Any]
    ) -> str:
        """Generate model retraining playbook"""
        
        return f'''---
# Auto-generated Ansible playbook for model retraining
# Model: {model_name}
# Generated: {datetime.now().isoformat()}

- name: Retrain ML Model {model_name}
  hosts: training_servers
  become: yes
  vars:
    model_name: "{model_name}"
    training_config: {json.dumps(training_config, indent=6)}
    
  tasks:
    - name: Create training job directory
      file:
        path: "/opt/ml-training/{{{{ model_name }}}}"
        state: directory
        mode: '0755'
    
    - name: Pull latest training data
      shell: |
        aws s3 sync s3://ml-data/{{{{ model_name }}}}/latest/ /opt/ml-training/{{{{ model_name }}}}/data/
    
    - name: Start training job
      shell: |
        python /opt/ml-training/train.py \\
          --model-name {{{{ model_name }}}} \\
          --data-path /opt/ml-training/{{{{ model_name }}}}/data/ \\
          --output-path /opt/ml-training/{{{{ model_name }}}}/output/
      async: 3600
      poll: 60
      register: training_result
    
    - name: Validate trained model
      shell: |
        python /opt/ml-training/validate.py \\
          --model-path /opt/ml-training/{{{{ model_name }}}}/output/model.joblib
      register: validation
    
    - name: Upload model to registry
      aws_s3:
        bucket: ml-models
        object: "registry/{{{{ model_name }}}}/{{{{ ansible_date_time.iso8601 }}}}/model.tar.gz"
        src: "/opt/ml-training/{{{{ model_name }}}}/output/model.tar.gz"
        mode: put
      when: validation.rc == 0
    
    - name: Notify completion
      slack:
        token: "{{{{ lookup('env', 'SLACK_TOKEN') }}}}"
        channel: "#ml-ops"
        msg: "Model {{{{ model_name }}}} retraining completed successfully"
'''


class SHAPExplainerComponent:
    """
    LangFlow Component: SHAP Explainability
    
    Generates SHAP-based explainability reports.
    """
    
    display_name = "SHAP Explainer"
    description = "Generate SHAP explainability reports"
    
    def explain(
        self,
        model: Any,
        X: Any,
        feature_names: List[str],
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """Generate SHAP explanations"""
        import shap
        
        # Sample data if large
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices] if hasattr(X, '__getitem__') else X.iloc[indices]
        else:
            X_sample = X
        
        # Create explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Handle multi-class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        importance_dict = {
            feature_names[i]: float(feature_importance[i])
            for i in range(len(feature_names))
        }
        
        # Sort by importance
        sorted_importance = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        return {
            "feature_importance": sorted_importance,
            "top_features": list(sorted_importance.keys())[:10],
            "samples_analyzed": len(X_sample),
            "explanation_type": "TreeExplainer",
            "timestamp": datetime.now().isoformat()
        }


# LangFlow component registration
LANGFLOW_COMPONENTS = {
    "root_cause_analysis": RootCauseAnalysisComponent,
    "remedy_suggestion": RemedySuggestionComponent,
    "code_generator": CodeGeneratorComponent,
    "ansible_playbook": AnsiblePlaybookComponent,
    "shap_explainer": SHAPExplainerComponent
}

