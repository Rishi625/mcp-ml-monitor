# MCP Agent for ML Model Monitoring & Drift Detection

## Problem Statement

ML models degrade silently in production. Data distributions shift, feature relationships change, and model performance drops—often without any immediate signals. By the time accuracy metrics fall noticeably, significant business impact has already occurred.

**This agent solves**: Continuous, automated monitoring that detects drift and performance degradation *before* it impacts production outcomes.

---

## What is MCP (Model Context Protocol)?

MCP is a protocol that allows **AI assistants to invoke external tools**. Instead of just generating text, an AI can call specialized functions to perform real computations.

### MCP Flow in This Project

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MCP ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   USER/AI ASSISTANT                                                      │
│        │                                                                 │
│        │  "Is my fraud model still performing well?"                     │
│        │                                                                 │
│        ▼                                                                 │
│   ┌─────────────────────────────────────────┐                           │
│   │         MCP SERVER (mcp_server.py)      │                           │
│   │                                         │                           │
│   │   Exposes Tools:                        │                           │
│   │   ├── set_reference_data                │                           │
│   │   ├── detect_drift                      │                           │
│   │   ├── record_predictions                │                           │
│   │   ├── get_performance_report            │                           │
│   │   ├── get_health_summary                │                           │
│   │   └── get_retraining_recommendation     │                           │
│   │                                         │                           │
│   └──────────────────┬──────────────────────┘                           │
│                      │                                                   │
│                      │  Tool calls with JSON input/output                │
│                      ▼                                                   │
│   ┌─────────────────────────────────────────┐                           │
│   │           MONITORING ENGINE             │                           │
│   │                                         │                           │
│   │   DriftDetector ──► Statistical Tests   │                           │
│   │   PerformanceMonitor ──► Metrics        │                           │
│   │   AlertSystem ──► Recommendations       │                           │
│   │                                         │                           │
│   └──────────────────┬──────────────────────┘                           │
│                      │                                                   │
│                      ▼                                                   │
│              Structured JSON Response                                    │
│              (drift report, alerts, actions)                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### MCP Tool Definitions

Each tool is defined with a schema that the AI assistant understands:

```python
Tool(
    name="detect_drift",
    description="Detect data drift in production data compared to reference",
    inputSchema={
        "type": "object",
        "properties": {
            "data": {"type": "array", "description": "Production data"},
            "features": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["data"]
    }
)
```

When an AI calls `detect_drift`, the server:
1. Receives production data as JSON
2. Runs statistical tests against stored reference data
3. Returns structured drift report with severity and recommendations

---

## Core Components

### 1. Drift Detector (`drift_detector.py`)

Detects when production data distribution differs from training data.

**Statistical Methods:**

| Method | What It Measures | Threshold |
|--------|------------------|-----------|
| **KS-Test** | Whether two samples come from same distribution | p-value < 0.05 |
| **PSI** | Magnitude of distribution shift | < 0.1 OK, 0.1-0.2 Warning, > 0.2 Critical |
| **Wasserstein** | "Work" to transform one distribution to another | Normalized distance > 0.15 |

**How PSI is Calculated:**

```python
def calculate_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
    # Bin both distributions
    bins = np.linspace(min_val, max_val, self.n_bins + 1)
    ref_props = histogram(reference, bins) / len(reference)
    cur_props = histogram(current, bins) / len(current)
    
    # PSI = Σ (cur - ref) * ln(cur / ref)
    psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
    return psi
```

### 2. Performance Monitor (`performance_monitor.py`)

Tracks model predictions against ground truth over a rolling window.

**Metrics Tracked:**
- Accuracy, Precision, Recall, F1-Score
- Trend detection (improving / stable / declining)
- Delta from baseline

**Status Determination:**

```python
def _determine_status(self, deltas: Dict[str, float]) -> PerformanceStatus:
    for metric, delta in deltas.items():
        if delta < -self.critical_threshold:  # e.g., -10%
            return PerformanceStatus.CRITICAL
        if delta < -self.warning_threshold:   # e.g., -5%
            return PerformanceStatus.DEGRADED
    return PerformanceStatus.HEALTHY
```

### 3. Alert System (`alert_system.py`)

Generates actionable alerts and retraining recommendations.

**Recommendation Logic:**

```python
def generate_retraining_recommendation(self, drift_report, performance_report):
    reasons = []
    urgency = "not_needed"
    
    # Check drift severity
    if drift_report.overall_severity == DriftSeverity.CRITICAL:
        urgency = "immediate"
        reasons.append(f"Critical drift in {drift_report.drifted_features} features")
    
    # Check performance drop
    if abs(performance_report.metric_deltas['accuracy']) > 0.08:
        urgency = "immediate" if accuracy_drop > 0.15 else "soon"
        reasons.append(f"Performance degraded by {accuracy_drop*100:.1f}%")
    
    return RetrainingRecommendation(
        should_retrain=urgency != "not_needed",
        urgency=urgency,
        reasons=reasons,
        suggested_actions=[...]
    )
```

---

## Data Flow

```
TRAINING PHASE                          PRODUCTION PHASE
──────────────                          ────────────────

Historical Data                         Live Data Stream
      │                                       │
      ▼                                       ▼
┌─────────────┐                        ┌─────────────────┐
│ Train Model │                        │ Model Inference │
└──────┬──────┘                        └────────┬────────┘
       │                                        │
       │ Baseline metrics                       │ Predictions
       │ Reference distribution                 │ Feature values
       │                                        │
       ▼                                        ▼
┌──────────────────────────────────────────────────────────┐
│                    MCP MONITORING AGENT                   │
│                                                           │
│  ┌─────────────────┐    ┌──────────────────────────────┐ │
│  │ Reference Store │    │ Compare:                      │ │
│  │                 │◄───│  - Distribution (KS, PSI)     │ │
│  │ - Feature stats │    │  - Predictions vs Baseline    │ │
│  │ - Baseline perf │    │                               │ │
│  └─────────────────┘    └──────────────┬───────────────┘ │
│                                        │                  │
│                                        ▼                  │
│                         ┌──────────────────────────────┐ │
│                         │ Alert Generation:            │ │
│                         │  - Severity classification   │ │
│                         │  - Affected features         │ │
│                         │  - Retraining urgency        │ │
│                         └──────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

---

## Usage

### Standalone Demo

```bash
cd mcp-ml-monitor
pip install -r requirements.txt
python demo.py
```

### MCP Server Integration

Add to your MCP client configuration:

```json
{
  "mcpServers": {
    "ml-monitor": {
      "command": "python",
      "args": ["-m", "src.mcp_server"],
      "cwd": "/path/to/mcp-ml-monitor"
    }
  }
}
```

### Programmatic Usage

```python
from src.drift_detector import DriftDetector
from src.performance_monitor import PerformanceMonitor
from src.alert_system import AlertSystem

# Initialize
detector = DriftDetector(psi_critical=0.2)
detector.set_reference(training_data)

# Monitor production
drift_report = detector.detect_drift(production_data)

if drift_report.overall_severity.value == "critical":
    print(f"Drift in: {[r.feature_name for r in drift_report.feature_results if r.is_drifted]}")
```

---

## Output Example

```json
{
  "health_summary": {
    "overall_status": "critical",
    "drift_status": "critical",
    "performance_status": "healthy",
    "retraining": {
      "should_retrain": true,
      "urgency": "immediate",
      "reasons": ["Critical drift detected in 5 features"],
      "affected_features": ["feature_3", "feature_5", "feature_7"],
      "suggested_actions": [
        "Collect recent production data for retraining",
        "Validate data quality and preprocessing pipeline"
      ]
    }
  }
}
```

---

## Project Structure

```
mcp-ml-monitor/
├── src/
│   ├── config.py              # Threshold configurations
│   ├── drift_detector.py      # KS-test, PSI, Wasserstein
│   ├── performance_monitor.py # Metrics tracking
│   ├── alert_system.py        # Recommendations
│   └── mcp_server.py          # MCP protocol server
├── tests/
├── demo.py                    # End-to-end demonstration
├── config.json
└── requirements.txt
```

---

## Why This Matters

| Without Monitoring | With This Agent |
|--------------------|-----------------|
| Model fails silently | Drift detected in real-time |
| Manual metric checks | Automated 24/7 monitoring |
| Reactive retraining | Proactive recommendations |
| Unknown feature impact | Specific features identified |
| Delayed response | Severity-based urgency |
