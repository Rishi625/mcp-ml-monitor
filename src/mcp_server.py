"""
MCP Server for ML Model Monitoring

Implements the Model Context Protocol (MCP) for ML monitoring tools.
Provides tools for drift detection, performance monitoring, and health checks.
"""

import json
import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from mcp.server import Server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult
)

from .drift_detector import DriftDetector, DriftReport
from .performance_monitor import PerformanceMonitor, PerformanceReport
from .alert_system import AlertSystem, HealthSummary
from .config import MonitorConfig

logger = logging.getLogger(__name__)


class MLMonitorMCPServer:
    """
    MCP Server for ML Model Monitoring.
    
    Provides tools for:
    - Setting reference data
    - Detecting data drift
    - Monitoring performance
    - Getting model health status
    - Managing alerts
    """
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()
        
        # Initialize components
        self.drift_detector = DriftDetector(
            ks_threshold=self.config.drift.ks_threshold,
            psi_warning=self.config.drift.psi_warning_threshold,
            psi_critical=self.config.drift.psi_critical_threshold,
            wasserstein_threshold=self.config.drift.wasserstein_threshold
        )
        
        self.performance_monitor = PerformanceMonitor(
            baseline_accuracy=self.config.performance.baseline_accuracy,
            baseline_precision=self.config.performance.baseline_precision,
            baseline_recall=self.config.performance.baseline_recall,
            baseline_f1=self.config.performance.baseline_f1,
            warning_threshold=self.config.performance.accuracy_warning_threshold,
            critical_threshold=self.config.performance.accuracy_critical_threshold,
            window_size=self.config.performance.monitoring_window
        )
        
        self.alert_system = AlertSystem()
        
        # Latest reports for health summary
        self.latest_drift_report: Optional[DriftReport] = None
        self.latest_performance_report: Optional[PerformanceReport] = None
        
        # Create MCP server
        self.server = Server("ml-monitor")
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup MCP tool handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="set_reference_data",
                    description="Set reference (training) data for drift detection. Provide data as JSON array of objects.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "description": "Training data as array of objects"
                            }
                        },
                        "required": ["data"]
                    }
                ),
                Tool(
                    name="set_baseline_metrics",
                    description="Set baseline performance metrics from training/validation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "accuracy": {"type": "number", "description": "Baseline accuracy"},
                            "precision": {"type": "number", "description": "Baseline precision"},
                            "recall": {"type": "number", "description": "Baseline recall"},
                            "f1": {"type": "number", "description": "Baseline F1 score"}
                        },
                        "required": ["accuracy", "precision", "recall", "f1"]
                    }
                ),
                Tool(
                    name="detect_drift",
                    description="Detect data drift in production data compared to reference",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "data": {
                                "type": "array",
                                "description": "Production data as array of objects"
                            },
                            "features": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional: specific features to analyze"
                            }
                        },
                        "required": ["data"]
                    }
                ),
                Tool(
                    name="record_predictions",
                    description="Record model predictions for performance monitoring",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "y_true": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "True labels"
                            },
                            "y_pred": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "Predicted labels"
                            }
                        },
                        "required": ["y_true", "y_pred"]
                    }
                ),
                Tool(
                    name="get_performance_report",
                    description="Get current performance metrics and analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="get_health_summary",
                    description="Get comprehensive model health summary with recommendations",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="get_active_alerts",
                    description="Get all active (unresolved) alerts",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="acknowledge_alert",
                    description="Acknowledge an alert by ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "alert_id": {"type": "string", "description": "Alert ID to acknowledge"}
                        },
                        "required": ["alert_id"]
                    }
                ),
                Tool(
                    name="get_retraining_recommendation",
                    description="Get recommendation on whether to retrain the model",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            try:
                result = await self._handle_tool(name, arguments)
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            except Exception as e:
                logger.error(f"Tool error: {e}")
                return [TextContent(type="text", text=json.dumps({
                    "error": str(e),
                    "tool": name
                }))]
    
    async def _handle_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool execution"""
        
        if name == "set_reference_data":
            data = pd.DataFrame(arguments["data"])
            self.drift_detector.set_reference(data)
            return {
                "status": "success",
                "message": f"Reference data set with {len(data)} samples, {len(data.columns)} features",
                "features": list(data.columns)
            }
        
        elif name == "set_baseline_metrics":
            self.performance_monitor.set_baseline(
                accuracy=arguments["accuracy"],
                precision=arguments["precision"],
                recall=arguments["recall"],
                f1=arguments["f1"]
            )
            return {
                "status": "success",
                "baseline": self.performance_monitor.baseline
            }
        
        elif name == "detect_drift":
            data = pd.DataFrame(arguments["data"])
            features = arguments.get("features")
            
            report = self.drift_detector.detect_drift(data, features)
            self.latest_drift_report = report
            
            # Process alerts
            self.alert_system.process_drift_report(report)
            
            return report.to_dict()
        
        elif name == "record_predictions":
            y_true = np.array(arguments["y_true"])
            y_pred = np.array(arguments["y_pred"])
            
            self.performance_monitor.record_predictions(y_true, y_pred)
            
            return {
                "status": "success",
                "predictions_recorded": len(y_true),
                "total_predictions": self.performance_monitor.prediction_count
            }
        
        elif name == "get_performance_report":
            report = self.performance_monitor.generate_report()
            self.latest_performance_report = report
            
            # Process alerts
            self.alert_system.process_performance_report(report)
            
            return report.to_dict()
        
        elif name == "get_health_summary":
            summary = self.alert_system.generate_health_summary(
                self.latest_drift_report,
                self.latest_performance_report
            )
            return summary.to_dict()
        
        elif name == "get_active_alerts":
            alerts = self.alert_system.get_active_alerts()
            return {
                "active_alerts": len(alerts),
                "alerts": [a.to_dict() for a in alerts]
            }
        
        elif name == "acknowledge_alert":
            success = self.alert_system.acknowledge_alert(arguments["alert_id"])
            return {
                "status": "success" if success else "not_found",
                "alert_id": arguments["alert_id"]
            }
        
        elif name == "get_retraining_recommendation":
            recommendation = self.alert_system.generate_retraining_recommendation(
                self.latest_drift_report,
                self.latest_performance_report
            )
            return recommendation.to_dict()
        
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    async def run(self):
        """Run the MCP server"""
        from mcp.server.stdio import stdio_server
        
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


def main():
    """Entry point for MCP server"""
    logging.basicConfig(level=logging.INFO)
    
    server = MLMonitorMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()

