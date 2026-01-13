# MCP Agent for ML Model Monitoring & Drift Detection

## Problem Statement

ML models degrade silently in production. Data distributions shift, feature relationships change, and model performance drops without immediate signals. This agent provides continuous automated monitoring that detects issues before they impact production.

---

## How the MCP Agent Works

### What is MCP?

MCP (Model Context Protocol) allows AI assistants (Claude, ChatGPT) to call external tools. Instead of just generating text, the AI can invoke real functions that perform computations.

### Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MCP ML MONITORING AGENT                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  USER / AI ASSISTANT                                                     │
│       │                                                                  │
│       │  "Is my model still working well?"                               │
│       ▼                                                                  │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    MCP SERVER (mcp_server.py)                      │  │
│  │                                                                    │  │
│  │  Exposes tools that AI can call:                                   │  │
│  │    - set_reference_data      - detect_drift                        │  │
│  │    - record_predictions      - get_performance_report              │  │
│  │    - get_health_summary      - get_retraining_recommendation       │  │
│  └────────────────────────────────┬──────────────────────────────────┘  │
│                                   │                                      │
│                                   ▼                                      │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐ │
│  │ DRIFT DETECTOR │  │  PERFORMANCE   │  │      ALERT SYSTEM          │ │
│  │                │  │    MONITOR     │  │                            │ │
│  │ - KS-Test      │  │ - Accuracy     │  │ - Severity Classification  │ │
│  │ - PSI Score    │  │ - Precision    │  │ - Retraining Decisions     │ │
│  │ - Wasserstein  │  │ - Recall, F1   │  │ - Actionable Suggestions   │ │
│  └────────────────┘  └────────────────┘  └────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### How Each Component Works

#### 1. Drift Detector (`src/drift_detector.py`)

**Purpose**: Detect when production data differs from training data.

**Process**:
```
Training Data (Reference)     Production Data (Current)
        │                              │
        └──────────┬───────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Statistical Tests  │
        │                     │
        │  1. KS-Test         │─── p-value < 0.05? → DRIFT
        │  2. PSI Score       │─── PSI > 0.2? → CRITICAL
        │  3. Wasserstein     │─── Distance > 0.15? → WARNING
        └─────────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │  Per-Feature Report │
        │                     │
        │  feature_1: OK      │
        │  feature_2: DRIFT   │
        │  feature_3: DRIFT   │
        └─────────────────────┘
```

**Statistical Methods**:

| Method | What It Measures | Formula |
|--------|------------------|---------|
| KS-Test | Whether two samples come from same distribution | Max difference between cumulative distributions |
| PSI | Magnitude of distribution shift | Σ (current - reference) × ln(current/reference) |
| Wasserstein | Minimum "work" to transform one distribution to another | Earth Mover's Distance |

#### 2. Performance Monitor (`src/performance_monitor.py`)

**Purpose**: Track model accuracy over time and detect degradation.

**Process**:
```
Model Predictions          Ground Truth
(y_pred)                   (y_true)
    │                          │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Calculate Metrics   │
    │                      │
    │  Accuracy   = 0.92   │
    │  Precision  = 0.89   │
    │  Recall     = 0.91   │
    │  F1-Score   = 0.90   │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Compare to Baseline │
    │                      │
    │  Baseline: 0.95      │
    │  Current:  0.92      │
    │  Delta:   -0.03      │──── Drop > 5%? → WARNING
    │                      │──── Drop > 10%? → CRITICAL
    └──────────────────────┘
```

#### 3. Alert System (`src/alert_system.py`)

**Purpose**: Generate actionable recommendations based on analysis.

**Decision Logic**:
```
Drift Report + Performance Report
              │
              ▼
    ┌─────────────────────────────────────┐
    │  Evaluate Conditions                │
    │                                     │
    │  IF drift_severity == CRITICAL      │
    │     OR performance_drop > 10%       │
    │  THEN urgency = IMMEDIATE           │
    │                                     │
    │  IF drift_severity == WARNING       │
    │     OR performance_drop > 5%        │
    │  THEN urgency = SOON                │
    │                                     │
    │  ELSE no retraining needed          │
    └─────────────────────────────────────┘
              │
              ▼
    ┌─────────────────────────────────────┐
    │  Generate Recommendations           │
    │                                     │
    │  - Collect recent production data   │
    │  - Validate preprocessing pipeline  │
    │  - Run A/B test before deployment   │
    └─────────────────────────────────────┘
```

---

## How LangFlow Works (Advanced Version)

LangFlow provides a visual interface to build the same monitoring pipeline with additional enterprise features.

### LangFlow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LANGFLOW VISUAL PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   DATA INGESTION              MONITORING                 INTELLIGENT         │
│   ─────────────               ──────────                 RESPONSE            │
│                                                          ───────────         │
│   ┌─────────┐                 ┌─────────┐               ┌─────────┐         │
│   │  CSV    │────────────────►│ Drift   │──────────────►│ Root    │         │
│   │  JSON   │                 │ Detect  │               │ Cause   │         │
│   │  API    │                 └────┬────┘               │ (LLM)   │         │
│   └────┬────┘                      │                    └────┬────┘         │
│        │                           │                         │              │
│        ▼                           ▼                         ▼              │
│   ┌─────────┐                 ┌─────────┐               ┌─────────┐         │
│   │ Data    │                 │ Anomaly │               │ Remedy  │         │
│   │ Valid   │                 │ Detect  │               │ Suggest │         │
│   └─────────┘                 └────┬────┘               └────┬────┘         │
│                                    │                         │              │
│                                    ▼                         ▼              │
│                               ┌─────────┐               ┌─────────┐         │
│                               │ Perf    │               │ Code    │         │
│                               │ Monitor │               │ Gen     │         │
│                               └────┬────┘               └────┬────┘         │
│                                    │                         │              │
│   INTEGRATIONS                     │                         │              │
│   ────────────                     │                         │              │
│                                    ▼                         ▼              │
│   ┌─────────┐                 ┌─────────┐               ┌─────────┐         │
│   │ Slack   │◄────────────────│ Alert   │◄──────────────│ Ansible │         │
│   │ K8s     │                 │ System  │               │ Playbook│         │
│   │ GitHub  │                 └─────────┘               └─────────┘         │
│   │ Prom    │                                                               │
│   └─────────┘                                                               │
│                                                                              │
│   ADVANCED FEATURES                                                          │
│   ─────────────────                                                          │
│                                                                              │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│   │ Multi-Model │  │ A/B Testing │  │ Auto        │  │ Federated   │        │
│   │ Comparison  │  │ Coordinator │  │ Retraining  │  │ Learning    │        │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### LangFlow Component Details

#### Data Ingestion Components

| Component | Purpose | How It Works |
|-----------|---------|--------------|
| CSV Ingestion | Load CSV files | Parses file, infers dtypes, returns DataFrame |
| API Ingestion | Fetch from REST API | Makes HTTP request, extracts data from JSON path |
| Stream Ingestion | Handle streaming | Buffers records, outputs windows when size reached |
| Data Validator | Check data quality | Validates schema, required columns, value ranges |

#### Monitoring Components

| Component | Purpose | How It Works |
|-----------|---------|--------------|
| Drift Detection | Compare distributions | Runs KS-test, PSI, Wasserstein on each feature |
| Anomaly Detection | Find outliers | Trains Isolation Forest, marks anomalous samples |
| Performance Monitor | Track metrics | Calculates accuracy/F1, compares to baseline |
| Time Series Forecast | Predict trends | Uses exponential smoothing to forecast metrics |

#### Intelligent Response Components

| Component | Purpose | How It Works |
|-----------|---------|--------------|
| Root Cause Analysis | Diagnose issues | Sends drift + performance data to LLM for analysis |
| Remedy Suggestion | Recommend fixes | Maps severity to action templates |
| Code Generator | Create fix scripts | Generates Python retraining code from templates |
| Ansible Playbook | Automate rollback | Creates K8s deployment/rollback playbooks |
| SHAP Explainer | Model explainability | Calculates SHAP values, ranks feature importance |

#### Integration Components

| Component | Purpose | How It Works |
|-----------|---------|--------------|
| Kubernetes | Manage deployments | Uses K8s API to scale, rollback deployments |
| Prometheus | Push metrics | Sends metrics to Pushgateway for Grafana |
| Slack | Send alerts | Posts formatted messages to Slack channels |
| GitHub | Create issues | Opens issues with drift/performance details |

#### Advanced Features

| Component | Purpose | How It Works |
|-----------|---------|--------------|
| Multi-Model Compare | Compare versions | Ranks models by accuracy, latency, drift score |
| A/B Testing | Statistical testing | Runs t-test, calculates Cohen's d for significance |
| Auto Retraining | Trigger retraining | Evaluates thresholds, initiates pipeline if exceeded |
| Federated Learning | Distributed training | Coordinates model updates across nodes, aggregates with FedAvg |

---

## Output Validation

The `monitoring_report.json` output is correct. Here's what it shows:

```
Drift Analysis:
├── Total Features: 20
├── Drifted Features: 5 (25%)
├── Severity: CRITICAL
└── Affected: feature_3, feature_5, feature_7, feature_10, feature_16

Performance:
├── Current Accuracy: 0.974
├── Baseline Accuracy: 0.937
├── Delta: +0.037 (improving)
└── Status: HEALTHY

Recommendation:
├── Should Retrain: YES
├── Urgency: IMMEDIATE
└── Reason: Critical drift in 5 features
```

**Key Insight**: Even though performance improved (+3.7%), the system correctly flags CRITICAL status because 25% of features have drifted significantly (PSI > 0.2). This is important because:
- Current performance may be misleading (test data still similar to training)
- Future predictions on truly drifted data will degrade
- Proactive retraining prevents future failures

---

## Quick Start

```bash
cd mcp-ml-monitor
pip install -r requirements.txt
python demo.py
```

---

## Project Structure

```
mcp-ml-monitor/
├── src/                        # Core MCP Agent
│   ├── drift_detector.py       # Statistical drift detection
│   ├── performance_monitor.py  # Metric tracking
│   ├── alert_system.py         # Recommendation engine
│   └── mcp_server.py           # MCP protocol server
│
├── langflow/                   # Advanced LangFlow Version
│   ├── components/
│   │   ├── data_ingestion.py   # CSV, API, Streaming
│   │   ├── ml_monitoring.py    # Drift, Anomaly, Forecast
│   │   ├── intelligent_response.py  # LLM, Code Gen
│   │   ├── integrations.py     # Slack, K8s, GitHub
│   │   └── advanced_features.py # A/B, Federated
│   ├── flows/
│   │   └── ml_monitoring_flow.json  # Import to LangFlow
│   └── orchestrator.py         # Pipeline runner
│
├── demo.py                     # Run this for demo
└── requirements.txt
```
