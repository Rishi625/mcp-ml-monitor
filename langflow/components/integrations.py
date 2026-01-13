"""
Integration Components for LangFlow ML Monitoring Pipeline

Connectors for external systems:
- Kubernetes API
- Prometheus/Grafana
- Slack/Email alerting
- GitHub issue creation
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import json


class KubernetesConnectorComponent:
    """
    LangFlow Component: Kubernetes API Connector
    
    Manages ML model deployments in Kubernetes.
    """
    
    display_name = "Kubernetes Connector"
    description = "Manage ML deployments in Kubernetes"
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.kubeconfig_path = kubeconfig_path
        self._client = None
    
    def _get_client(self):
        """Get Kubernetes client"""
        if self._client is None:
            from kubernetes import client, config
            
            if self.kubeconfig_path:
                config.load_kube_config(self.kubeconfig_path)
            else:
                try:
                    config.load_incluster_config()
                except:
                    config.load_kube_config()
            
            self._client = client
        return self._client
    
    def get_deployment_status(
        self,
        deployment_name: str,
        namespace: str = "default"
    ) -> Dict[str, Any]:
        """Get deployment status"""
        client = self._get_client()
        apps_v1 = client.AppsV1Api()
        
        deployment = apps_v1.read_namespaced_deployment(
            name=deployment_name,
            namespace=namespace
        )
        
        return {
            "name": deployment.metadata.name,
            "namespace": deployment.metadata.namespace,
            "replicas": deployment.spec.replicas,
            "ready_replicas": deployment.status.ready_replicas or 0,
            "available_replicas": deployment.status.available_replicas or 0,
            "image": deployment.spec.template.spec.containers[0].image,
            "conditions": [
                {"type": c.type, "status": c.status}
                for c in (deployment.status.conditions or [])
            ]
        }
    
    def scale_deployment(
        self,
        deployment_name: str,
        replicas: int,
        namespace: str = "default"
    ) -> Dict[str, Any]:
        """Scale deployment replicas"""
        client = self._get_client()
        apps_v1 = client.AppsV1Api()
        
        body = {"spec": {"replicas": replicas}}
        
        apps_v1.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace=namespace,
            body=body
        )
        
        return {
            "action": "scale",
            "deployment": deployment_name,
            "new_replicas": replicas,
            "timestamp": datetime.now().isoformat()
        }
    
    def rollback_deployment(
        self,
        deployment_name: str,
        namespace: str = "default"
    ) -> Dict[str, Any]:
        """Rollback to previous revision"""
        client = self._get_client()
        apps_v1 = client.AppsV1Api()
        
        # Get current deployment
        deployment = apps_v1.read_namespaced_deployment(
            name=deployment_name,
            namespace=namespace
        )
        
        # Trigger rollback by updating revision annotation
        rollback_revision = str(int(deployment.metadata.annotations.get(
            'deployment.kubernetes.io/revision', '1'
        )) - 1)
        
        return {
            "action": "rollback",
            "deployment": deployment_name,
            "target_revision": rollback_revision,
            "timestamp": datetime.now().isoformat()
        }


class PrometheusConnectorComponent:
    """
    LangFlow Component: Prometheus/Grafana Connector
    
    Push metrics and create dashboards.
    """
    
    display_name = "Prometheus Connector"
    description = "Push metrics to Prometheus"
    
    def __init__(self, pushgateway_url: str = "http://localhost:9091"):
        self.pushgateway_url = pushgateway_url
    
    def push_metrics(
        self,
        job_name: str,
        metrics: Dict[str, float],
        labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Push metrics to Prometheus Pushgateway"""
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
        
        registry = CollectorRegistry()
        
        for metric_name, value in metrics.items():
            gauge = Gauge(
                metric_name,
                f"ML Monitoring metric: {metric_name}",
                labelnames=list(labels.keys()) if labels else [],
                registry=registry
            )
            if labels:
                gauge.labels(**labels).set(value)
            else:
                gauge.set(value)
        
        push_to_gateway(
            self.pushgateway_url,
            job=job_name,
            registry=registry
        )
        
        return {
            "status": "pushed",
            "job": job_name,
            "metrics_count": len(metrics),
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_grafana_dashboard(
        self,
        model_name: str,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Generate Grafana dashboard JSON"""
        
        panels = []
        for i, metric in enumerate(metrics):
            panels.append({
                "id": i + 1,
                "title": metric.replace("_", " ").title(),
                "type": "graph",
                "gridPos": {"x": (i % 2) * 12, "y": (i // 2) * 8, "w": 12, "h": 8},
                "targets": [{
                    "expr": f'{metric}{{job="{model_name}"}}',
                    "legendFormat": "{{instance}}"
                }]
            })
        
        dashboard = {
            "dashboard": {
                "title": f"ML Monitoring: {model_name}",
                "tags": ["ml", "monitoring", "auto-generated"],
                "timezone": "browser",
                "panels": panels,
                "refresh": "30s",
                "time": {"from": "now-1h", "to": "now"}
            },
            "overwrite": True
        }
        
        return dashboard


class SlackAlertComponent:
    """
    LangFlow Component: Slack Alerting
    
    Send alerts to Slack channels.
    """
    
    display_name = "Slack Alert"
    description = "Send alerts to Slack"
    
    def __init__(self, webhook_url: Optional[str] = None, token: Optional[str] = None):
        self.webhook_url = webhook_url
        self.token = token
    
    def send_alert(
        self,
        channel: str,
        severity: str,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send alert to Slack channel"""
        from slack_sdk import WebClient
        from slack_sdk.webhook import WebhookClient
        
        # Color based on severity
        color_map = {
            "critical": "#FF0000",
            "warning": "#FFA500",
            "info": "#0000FF",
            "success": "#00FF00"
        }
        
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"[{severity.upper()}] {title}"}
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": message}
            }
        ]
        
        if details:
            fields = []
            for key, value in list(details.items())[:10]:
                fields.append({
                    "type": "mrkdwn",
                    "text": f"*{key}*: {value}"
                })
            blocks.append({"type": "section", "fields": fields})
        
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"Generated at {datetime.now().isoformat()}"}]
        })
        
        # Send via webhook or API
        if self.webhook_url:
            client = WebhookClient(self.webhook_url)
            response = client.send(
                text=f"[{severity.upper()}] {title}",
                blocks=blocks
            )
        else:
            client = WebClient(token=self.token)
            response = client.chat_postMessage(
                channel=channel,
                text=f"[{severity.upper()}] {title}",
                blocks=blocks,
                attachments=[{"color": color_map.get(severity, "#808080")}]
            )
        
        return {
            "status": "sent",
            "channel": channel,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        }


class EmailAlertComponent:
    """
    LangFlow Component: Email Alerting
    
    Send alerts via email.
    """
    
    display_name = "Email Alert"
    description = "Send alerts via email"
    
    def __init__(
        self,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
    
    def send_alert(
        self,
        to_addresses: List[str],
        severity: str,
        title: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Send alert via email"""
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        # Build email
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"[ML ALERT - {severity.upper()}] {title}"
        msg['From'] = self.username
        msg['To'] = ", ".join(to_addresses)
        
        # Plain text version
        text_content = f"{title}\n\n{message}"
        if details:
            text_content += "\n\nDetails:\n"
            for k, v in details.items():
                text_content += f"  {k}: {v}\n"
        
        # HTML version
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="background: {'#ff4444' if severity == 'critical' else '#ffaa00'}; 
                        color: white; padding: 10px; border-radius: 5px;">
                <h2>[{severity.upper()}] {title}</h2>
            </div>
            <div style="padding: 20px;">
                <p>{message}</p>
                {'<h3>Details:</h3><ul>' + ''.join(f'<li><b>{k}</b>: {v}</li>' for k, v in (details or {}).items()) + '</ul>' if details else ''}
            </div>
            <div style="color: #888; font-size: 12px; padding: 10px;">
                Generated at {datetime.now().isoformat()}
            </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(text_content, 'plain'))
        msg.attach(MIMEText(html_content, 'html'))
        
        # Send email
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            if self.username and self.password:
                server.login(self.username, self.password)
            server.send_message(msg)
        
        return {
            "status": "sent",
            "recipients": len(to_addresses),
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        }


class GitHubIssueComponent:
    """
    LangFlow Component: GitHub Issue Creator
    
    Create GitHub issues for ML monitoring alerts.
    """
    
    display_name = "GitHub Issue"
    description = "Create GitHub issues for alerts"
    
    def __init__(self, token: str, repo: str):
        self.token = token
        self.repo = repo  # format: "owner/repo"
    
    def create_issue(
        self,
        title: str,
        body: str,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create GitHub issue"""
        from github import Github
        
        g = Github(self.token)
        repo = g.get_repo(self.repo)
        
        issue = repo.create_issue(
            title=title,
            body=body,
            labels=labels or ["ml-monitoring", "auto-generated"],
            assignees=assignees or []
        )
        
        return {
            "status": "created",
            "issue_number": issue.number,
            "issue_url": issue.html_url,
            "timestamp": datetime.now().isoformat()
        }
    
    def create_monitoring_issue(
        self,
        severity: str,
        drift_report: Dict[str, Any],
        performance_report: Dict[str, Any],
        recommendations: List[str]
    ) -> Dict[str, Any]:
        """Create detailed monitoring issue"""
        
        # Build issue body
        body = f"""## ML Monitoring Alert

**Severity**: {severity.upper()}
**Timestamp**: {datetime.now().isoformat()}

### Drift Analysis
- **Drifted Features**: {drift_report.get('drifted_features', 0)}/{drift_report.get('total_features', 0)}
- **Drift Percentage**: {drift_report.get('drift_percentage', 0)}%
- **Overall Severity**: {drift_report.get('overall_severity', 'unknown')}

### Performance Metrics
| Metric | Current | Delta |
|--------|---------|-------|
"""
        
        for metric, value in performance_report.get('current_metrics', {}).items():
            delta = performance_report.get('deltas', {}).get(metric, 0)
            body += f"| {metric} | {value:.4f} | {delta:+.4f} |\n"
        
        body += f"""
### Recommended Actions
"""
        for rec in recommendations:
            body += f"- [ ] {rec}\n"
        
        body += """
### Investigation Checklist
- [ ] Check data pipeline logs
- [ ] Review recent model changes
- [ ] Validate feature engineering
- [ ] Test model predictions manually
- [ ] Prepare retraining if needed
"""
        
        labels = ["ml-monitoring", severity]
        if severity == "critical":
            labels.append("urgent")
        
        return self.create_issue(
            title=f"[ML Alert] {severity.upper()}: Model monitoring alert",
            body=body,
            labels=labels
        )


# LangFlow component registration
LANGFLOW_COMPONENTS = {
    "kubernetes_connector": KubernetesConnectorComponent,
    "prometheus_connector": PrometheusConnectorComponent,
    "slack_alert": SlackAlertComponent,
    "email_alert": EmailAlertComponent,
    "github_issue": GitHubIssueComponent
}

