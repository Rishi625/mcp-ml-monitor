"""
Advanced Features for LangFlow ML Monitoring Pipeline

Enterprise-grade ML operations:
- Multi-model comparison
- A/B testing coordinator
- Automated retraining pipeline
- Federated learning coordinator
"""

from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from collections import defaultdict


@dataclass
class ModelMetrics:
    """Metrics for a single model"""
    model_id: str
    version: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    latency_ms: float
    throughput_qps: float
    drift_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "latency_ms": round(self.latency_ms, 2),
            "throughput_qps": round(self.throughput_qps, 2),
            "drift_score": round(self.drift_score, 4),
            "timestamp": self.timestamp
        }


class MultiModelComparisonComponent:
    """
    LangFlow Component: Multi-Model Comparison Dashboard
    
    Compare multiple model versions side-by-side.
    """
    
    display_name = "Multi-Model Comparison"
    description = "Compare multiple models side-by-side"
    
    def __init__(self):
        self.models: Dict[str, List[ModelMetrics]] = defaultdict(list)
    
    def register_model(
        self,
        model_id: str,
        version: str,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Register a model with its metrics"""
        
        model_metrics = ModelMetrics(
            model_id=model_id,
            version=version,
            accuracy=metrics.get('accuracy', 0),
            precision=metrics.get('precision', 0),
            recall=metrics.get('recall', 0),
            f1=metrics.get('f1', 0),
            latency_ms=metrics.get('latency_ms', 0),
            throughput_qps=metrics.get('throughput_qps', 0),
            drift_score=metrics.get('drift_score', 0)
        )
        
        self.models[model_id].append(model_metrics)
        
        return {
            "status": "registered",
            "model_id": model_id,
            "version": version
        }
    
    def compare(self, model_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare all registered models"""
        
        if model_ids:
            models_to_compare = {k: v for k, v in self.models.items() if k in model_ids}
        else:
            models_to_compare = self.models
        
        if not models_to_compare:
            return {"error": "No models to compare"}
        
        comparison = {
            "models": {},
            "rankings": {},
            "recommendations": []
        }
        
        # Get latest metrics for each model
        for model_id, metrics_list in models_to_compare.items():
            latest = metrics_list[-1]
            comparison["models"][model_id] = latest.to_dict()
        
        # Rank by different metrics
        for metric in ['accuracy', 'f1', 'latency_ms', 'drift_score']:
            sorted_models = sorted(
                comparison["models"].items(),
                key=lambda x: x[1][metric],
                reverse=(metric not in ['latency_ms', 'drift_score'])  # Lower is better for latency/drift
            )
            comparison["rankings"][metric] = [m[0] for m in sorted_models]
        
        # Generate recommendations
        best_accuracy = comparison["rankings"]["accuracy"][0]
        best_latency = comparison["rankings"]["latency_ms"][0]
        lowest_drift = comparison["rankings"]["drift_score"][0]
        
        if best_accuracy == best_latency == lowest_drift:
            comparison["recommendations"].append(
                f"Model '{best_accuracy}' is the best choice across all metrics"
            )
        else:
            comparison["recommendations"].append(
                f"Best accuracy: '{best_accuracy}' ({comparison['models'][best_accuracy]['accuracy']:.4f})"
            )
            comparison["recommendations"].append(
                f"Best latency: '{best_latency}' ({comparison['models'][best_latency]['latency_ms']:.1f}ms)"
            )
            comparison["recommendations"].append(
                f"Lowest drift: '{lowest_drift}' (score: {comparison['models'][lowest_drift]['drift_score']:.4f})"
            )
        
        return comparison
    
    def get_version_history(self, model_id: str) -> Dict[str, Any]:
        """Get version history for a model"""
        
        if model_id not in self.models:
            return {"error": f"Model '{model_id}' not found"}
        
        history = [m.to_dict() for m in self.models[model_id]]
        
        # Calculate trends
        if len(history) > 1:
            accuracy_trend = history[-1]['accuracy'] - history[-2]['accuracy']
            drift_trend = history[-1]['drift_score'] - history[-2]['drift_score']
        else:
            accuracy_trend = 0
            drift_trend = 0
        
        return {
            "model_id": model_id,
            "versions": len(history),
            "history": history,
            "trends": {
                "accuracy": "improving" if accuracy_trend > 0 else "declining" if accuracy_trend < 0 else "stable",
                "drift": "increasing" if drift_trend > 0 else "decreasing" if drift_trend < 0 else "stable"
            }
        }


class ABTestingCoordinator:
    """
    LangFlow Component: A/B Testing Coordinator
    
    Manage A/B tests between model versions.
    """
    
    display_name = "A/B Testing Coordinator"
    description = "Coordinate A/B tests between models"
    
    def __init__(self):
        self.experiments: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))
    
    def create_experiment(
        self,
        experiment_id: str,
        control_model: str,
        treatment_model: str,
        traffic_split: float = 0.5,  # Fraction to treatment
        min_samples: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Create new A/B experiment"""
        
        self.experiments[experiment_id] = {
            "control": control_model,
            "treatment": treatment_model,
            "traffic_split": traffic_split,
            "min_samples": min_samples,
            "confidence_level": confidence_level,
            "status": "running",
            "created_at": datetime.now().isoformat()
        }
        
        return {
            "status": "created",
            "experiment_id": experiment_id,
            "config": self.experiments[experiment_id]
        }
    
    def record_outcome(
        self,
        experiment_id: str,
        variant: str,  # "control" or "treatment"
        outcome: float,  # e.g., conversion rate, accuracy
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Record an outcome for the experiment"""
        
        if experiment_id not in self.experiments:
            return {"error": f"Experiment '{experiment_id}' not found"}
        
        self.results[experiment_id][variant].append({
            "outcome": outcome,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        })
        
        return {
            "status": "recorded",
            "experiment_id": experiment_id,
            "variant": variant,
            "total_samples": {
                "control": len(self.results[experiment_id]["control"]),
                "treatment": len(self.results[experiment_id]["treatment"])
            }
        }
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze experiment results"""
        from scipy import stats
        
        if experiment_id not in self.experiments:
            return {"error": f"Experiment '{experiment_id}' not found"}
        
        exp = self.experiments[experiment_id]
        results = self.results[experiment_id]
        
        control_outcomes = [r["outcome"] for r in results["control"]]
        treatment_outcomes = [r["outcome"] for r in results["treatment"]]
        
        if len(control_outcomes) < 30 or len(treatment_outcomes) < 30:
            return {
                "status": "insufficient_data",
                "control_samples": len(control_outcomes),
                "treatment_samples": len(treatment_outcomes),
                "min_required": exp["min_samples"]
            }
        
        # Calculate statistics
        control_mean = np.mean(control_outcomes)
        treatment_mean = np.mean(treatment_outcomes)
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(control_outcomes, treatment_outcomes)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.var(control_outcomes) + np.var(treatment_outcomes)) / 2
        )
        cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        # Determine winner
        is_significant = p_value < (1 - exp["confidence_level"])
        
        if is_significant:
            winner = "treatment" if treatment_mean > control_mean else "control"
            recommendation = f"Deploy {exp[winner]} - statistically significant improvement"
        else:
            winner = None
            recommendation = "Continue experiment - no significant difference yet"
        
        return {
            "experiment_id": experiment_id,
            "status": "complete" if is_significant else "running",
            "statistics": {
                "control_mean": round(control_mean, 4),
                "treatment_mean": round(treatment_mean, 4),
                "lift": round((treatment_mean - control_mean) / control_mean * 100, 2) if control_mean > 0 else 0,
                "p_value": round(p_value, 4),
                "cohens_d": round(cohens_d, 4),
                "is_significant": is_significant
            },
            "samples": {
                "control": len(control_outcomes),
                "treatment": len(treatment_outcomes)
            },
            "winner": winner,
            "recommendation": recommendation
        }


class AutomatedRetrainingPipeline:
    """
    LangFlow Component: Automated Retraining Pipeline
    
    Orchestrate automated model retraining based on drift/performance.
    """
    
    display_name = "Automated Retraining"
    description = "Orchestrate automated model retraining"
    
    def __init__(self):
        self.triggers: Dict[str, Dict[str, Any]] = {}
        self.pipeline_runs: List[Dict[str, Any]] = []
    
    def configure_trigger(
        self,
        model_id: str,
        drift_threshold: float = 0.2,
        performance_threshold: float = 0.05,
        schedule: Optional[str] = None,  # cron expression
        cooldown_hours: int = 24
    ) -> Dict[str, Any]:
        """Configure retraining triggers"""
        
        self.triggers[model_id] = {
            "drift_threshold": drift_threshold,
            "performance_threshold": performance_threshold,
            "schedule": schedule,
            "cooldown_hours": cooldown_hours,
            "last_trigger": None,
            "enabled": True
        }
        
        return {
            "status": "configured",
            "model_id": model_id,
            "triggers": self.triggers[model_id]
        }
    
    def evaluate_triggers(
        self,
        model_id: str,
        drift_score: float,
        performance_delta: float
    ) -> Dict[str, Any]:
        """Evaluate if retraining should be triggered"""
        
        if model_id not in self.triggers:
            return {"should_retrain": False, "reason": "No triggers configured"}
        
        trigger = self.triggers[model_id]
        
        if not trigger["enabled"]:
            return {"should_retrain": False, "reason": "Triggers disabled"}
        
        # Check cooldown
        if trigger["last_trigger"]:
            last = datetime.fromisoformat(trigger["last_trigger"])
            hours_since = (datetime.now() - last).total_seconds() / 3600
            if hours_since < trigger["cooldown_hours"]:
                return {
                    "should_retrain": False,
                    "reason": f"In cooldown period ({trigger['cooldown_hours'] - hours_since:.1f}h remaining)"
                }
        
        reasons = []
        
        # Check drift
        if drift_score >= trigger["drift_threshold"]:
            reasons.append(f"Drift score ({drift_score:.3f}) exceeds threshold ({trigger['drift_threshold']})")
        
        # Check performance
        if abs(performance_delta) >= trigger["performance_threshold"]:
            reasons.append(f"Performance drop ({abs(performance_delta)*100:.1f}%) exceeds threshold")
        
        should_retrain = len(reasons) > 0
        
        if should_retrain:
            trigger["last_trigger"] = datetime.now().isoformat()
        
        return {
            "should_retrain": should_retrain,
            "reasons": reasons if reasons else ["All metrics within thresholds"],
            "model_id": model_id
        }
    
    def execute_pipeline(
        self,
        model_id: str,
        data_source: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute retraining pipeline"""
        
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        pipeline_run = {
            "run_id": run_id,
            "model_id": model_id,
            "status": "started",
            "stages": [],
            "started_at": datetime.now().isoformat(),
            "config": config
        }
        
        # Define pipeline stages
        stages = [
            ("data_validation", "Validating training data"),
            ("feature_engineering", "Engineering features"),
            ("model_training", "Training model"),
            ("model_evaluation", "Evaluating model"),
            ("model_validation", "Validating against baseline"),
            ("artifact_storage", "Storing model artifacts"),
            ("deployment_ready", "Preparing for deployment")
        ]
        
        for stage_id, stage_desc in stages:
            pipeline_run["stages"].append({
                "stage_id": stage_id,
                "description": stage_desc,
                "status": "pending"
            })
        
        self.pipeline_runs.append(pipeline_run)
        
        return {
            "run_id": run_id,
            "model_id": model_id,
            "status": "started",
            "stages": len(stages),
            "started_at": pipeline_run["started_at"]
        }
    
    def get_pipeline_status(self, run_id: str) -> Dict[str, Any]:
        """Get status of a pipeline run"""
        
        for run in self.pipeline_runs:
            if run["run_id"] == run_id:
                return run
        
        return {"error": f"Pipeline run '{run_id}' not found"}


class FederatedLearningCoordinator:
    """
    LangFlow Component: Federated Learning Coordinator
    
    Coordinate federated learning across distributed nodes.
    """
    
    display_name = "Federated Learning"
    description = "Coordinate federated learning"
    
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.rounds: List[Dict[str, Any]] = []
        self.global_model_version = 0
    
    def register_node(
        self,
        node_id: str,
        node_url: str,
        data_size: int,
        capabilities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Register a federated learning node"""
        
        self.nodes[node_id] = {
            "url": node_url,
            "data_size": data_size,
            "capabilities": capabilities or [],
            "status": "active",
            "last_contribution": None,
            "registered_at": datetime.now().isoformat()
        }
        
        return {
            "status": "registered",
            "node_id": node_id,
            "total_nodes": len(self.nodes)
        }
    
    def initiate_round(
        self,
        min_nodes: int = 2,
        aggregation_strategy: str = "fedavg"
    ) -> Dict[str, Any]:
        """Initiate a new federated learning round"""
        
        active_nodes = [n for n, info in self.nodes.items() if info["status"] == "active"]
        
        if len(active_nodes) < min_nodes:
            return {
                "error": f"Insufficient nodes. Need {min_nodes}, have {len(active_nodes)}"
            }
        
        round_id = len(self.rounds) + 1
        
        round_info = {
            "round_id": round_id,
            "participating_nodes": active_nodes,
            "aggregation_strategy": aggregation_strategy,
            "status": "in_progress",
            "node_updates": {},
            "started_at": datetime.now().isoformat()
        }
        
        self.rounds.append(round_info)
        
        return {
            "round_id": round_id,
            "participating_nodes": len(active_nodes),
            "status": "started"
        }
    
    def submit_update(
        self,
        round_id: int,
        node_id: str,
        model_weights: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Submit model update from a node"""
        
        if round_id > len(self.rounds):
            return {"error": f"Round {round_id} not found"}
        
        round_info = self.rounds[round_id - 1]
        
        if node_id not in round_info["participating_nodes"]:
            return {"error": f"Node {node_id} not participating in round {round_id}"}
        
        round_info["node_updates"][node_id] = {
            "weights": model_weights,
            "metrics": metrics,
            "submitted_at": datetime.now().isoformat()
        }
        
        self.nodes[node_id]["last_contribution"] = datetime.now().isoformat()
        
        # Check if all nodes have submitted
        all_submitted = len(round_info["node_updates"]) == len(round_info["participating_nodes"])
        
        return {
            "status": "submitted",
            "round_id": round_id,
            "node_id": node_id,
            "updates_received": len(round_info["node_updates"]),
            "total_expected": len(round_info["participating_nodes"]),
            "ready_for_aggregation": all_submitted
        }
    
    def aggregate_round(self, round_id: int) -> Dict[str, Any]:
        """Aggregate model updates using FedAvg"""
        
        if round_id > len(self.rounds):
            return {"error": f"Round {round_id} not found"}
        
        round_info = self.rounds[round_id - 1]
        
        if len(round_info["node_updates"]) == 0:
            return {"error": "No updates to aggregate"}
        
        # Calculate weighted average based on data size
        total_data = sum(
            self.nodes[node_id]["data_size"]
            for node_id in round_info["node_updates"]
        )
        
        # Aggregate metrics
        aggregated_metrics = {}
        for node_id, update in round_info["node_updates"].items():
            weight = self.nodes[node_id]["data_size"] / total_data
            for metric, value in update["metrics"].items():
                if metric not in aggregated_metrics:
                    aggregated_metrics[metric] = 0
                aggregated_metrics[metric] += value * weight
        
        self.global_model_version += 1
        round_info["status"] = "completed"
        round_info["completed_at"] = datetime.now().isoformat()
        
        return {
            "round_id": round_id,
            "status": "aggregated",
            "global_model_version": self.global_model_version,
            "nodes_aggregated": len(round_info["node_updates"]),
            "aggregated_metrics": {k: round(v, 4) for k, v in aggregated_metrics.items()}
        }


# LangFlow component registration
LANGFLOW_COMPONENTS = {
    "multi_model_comparison": MultiModelComparisonComponent,
    "ab_testing": ABTestingCoordinator,
    "automated_retraining": AutomatedRetrainingPipeline,
    "federated_learning": FederatedLearningCoordinator
}

