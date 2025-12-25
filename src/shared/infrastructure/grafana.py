"""
Grafana OTLP Metrics Exporter
==============================

Pushes LLM usage metrics (tokens, latency) to Grafana Cloud via OTLP.

Metrics exported:
- llm_tokens_total: Total tokens used (prompt + completion)
- llm_latency_ms: LLM request latency in milliseconds
- llm_requests_total: Total number of LLM requests
"""

import base64
import json
import time
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import httpx

from config import settings
from shared.infrastructure.logging import get_logger

logger = get_logger(__name__)


class GrafanaOTLPExporter:
    """
    Export LLM metrics to Grafana Cloud via OTLP HTTP endpoint.

    Uses OpenTelemetry Protocol (OTLP) format for metrics.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        api_key: Optional[str] = None,
        instance_id: Optional[str] = None
    ):
        """
        Initialize Grafana OTLP exporter.

        Args:
            host: Grafana OTLP gateway URL (e.g., https://otlp-gateway-prod-ap-south-1.grafana.net)
            api_key: Grafana API key
            instance_id: Instance ID for authentication
        """
        self._host = host or getattr(settings, 'grafana_host', None)
        self._api_key = api_key or getattr(settings, 'grafana_api_key', None)
        self._instance_id = instance_id or getattr(settings, 'grafana_instance_id', None)
        self._enabled = bool(self._host and self._api_key and self._instance_id)

        if self._enabled:
            # Encode authentication
            auth_pair = f"{self._instance_id}:{self._api_key}"
            self._auth_encoded = base64.b64encode(auth_pair.encode()).decode()
            # Don't double-append the path if host already includes it
            if "/otlp/v1/metrics" not in self._host:
                self._url = f"{self._host}/otlp/v1/metrics"
            else:
                self._url = self._host
            logger.info(
                "Grafana OTLP exporter initialized",
                extra={"host": self._host, "instance_id": self._instance_id}
            )
        else:
            logger.warning(
                "Grafana OTLP exporter not configured - metrics will not be exported",
                extra={
                    "host_configured": bool(self._host),
                    "api_key_configured": bool(self._api_key),
                    "instance_id_configured": bool(self._instance_id)
                }
            )

    def is_enabled(self) -> bool:
        """Check if exporter is properly configured."""
        return self._enabled

    async def export_llm_metrics(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        operation: str = "chat_completion",
        attributes: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Export LLM usage metrics to Grafana.

        Args:
            model: LLM model name (e.g., "glm-4.7")
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens generated
            latency_ms: Request latency in milliseconds
            operation: Operation type (chat_completion, classification, rag, etc.)
            attributes: Additional attributes to attach to metrics

        Returns:
            True if export succeeded, False otherwise
        """
        if not self._enabled:
            logger.debug("Grafana exporter not enabled - skipping metrics export")
            return False

        total_tokens = prompt_tokens + completion_tokens
        timestamp_ns = int(time.time() * 1_000_000_000)

        # Build attributes
        metric_attributes = [
            {"key": "model", "value": {"stringValue": model}},
            {"key": "operation", "value": {"stringValue": operation}},
            {"key": "service", "value": {"stringValue": settings.app_name}},
        ]

        if attributes:
            for key, value in attributes.items():
                metric_attributes.append({
                    "key": key,
                    "value": {"stringValue": str(value)}
                })

        # Build OTLP metrics payload
        payload = {
            "resourceMetrics": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": settings.app_name}},
                            {"key": "service.version", "value": {"stringValue": settings.app_version}},
                            {"key": "deployment.environment", "value": {"stringValue": settings.environment}},
                        ]
                    },
                    "scopeMetrics": [
                        {
                            "metrics": [
                                # Total tokens metric - use Gauge instead of Sum for better Grafana compatibility
                                {
                                    "name": "llm_tokens_total",
                                    "unit": "1",
                                    "description": "Total tokens used in LLM requests",
                                    "gauge": {
                                        "dataPoints": [
                                            {
                                                "asInt": total_tokens,
                                                "timeUnixNano": timestamp_ns,
                                                "attributes": metric_attributes
                                            }
                                        ]
                                    }
                                },
                                # Latency metric
                                {
                                    "name": "llm_latency_ms",
                                    "unit": "ms",
                                    "description": "LLM request latency in milliseconds",
                                    "gauge": {
                                        "dataPoints": [
                                            {
                                                "asInt": latency_ms,
                                                "timeUnixNano": timestamp_ns,
                                                "attributes": metric_attributes
                                            }
                                        ]
                                    }
                                },
                                # Prompt tokens metric
                                {
                                    "name": "llm_prompt_tokens",
                                    "unit": "1",
                                    "description": "Number of prompt tokens in LLM requests",
                                    "gauge": {
                                        "dataPoints": [
                                            {
                                                "asInt": prompt_tokens,
                                                "timeUnixNano": timestamp_ns,
                                                "attributes": metric_attributes
                                            }
                                        ]
                                    }
                                },
                                # Completion tokens metric
                                {
                                    "name": "llm_completion_tokens",
                                    "unit": "1",
                                    "description": "Number of completion tokens generated",
                                    "gauge": {
                                        "dataPoints": [
                                            {
                                                "asInt": completion_tokens,
                                                "timeUnixNano": timestamp_ns,
                                                "attributes": metric_attributes
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        # Send to Grafana
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {self._auth_encoded}",
            "X-Grafana-Org-Id": str(self._instance_id)
        }

        try:
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                logger.debug(
                    "Sending metrics to Grafana",
                    extra={
                        "url": self._url,
                        "metrics_count": len(payload["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]),
                        "payload_size": len(str(payload))
                    }
                )
                response = await client.post(
                    self._url,
                    headers=headers,
                    json=payload
                )

                logger.info(
                    "Grafana metrics response",
                    extra={
                        "status_code": response.status_code,
                        "response_text": response.text[:500],
                        "model": model,
                        "operation": operation,
                        "total_tokens": total_tokens
                    }
                )

                if response.status_code == 200 or response.status_code == 202:
                    logger.info(
                        "LLM metrics exported to Grafana successfully",
                        extra={
                            "model": model,
                            "operation": operation,
                            "total_tokens": total_tokens,
                            "latency_ms": latency_ms,
                            "status_code": response.status_code
                        }
                    )
                    return True
                else:
                    logger.warning(
                        "Failed to export metrics to Grafana",
                        extra={
                            "status_code": response.status_code,
                            "response": response.text[:500],
                            "url": self._url
                        }
                    )
                    return False

        except Exception as e:
            logger.error(
                "Error exporting metrics to Grafana",
                extra={"error": str(e)}
            )
            return False

    async def export_request_latency(
        self,
        endpoint: str,
        status_code: int,
        latency_ms: int,
        method: str = "POST",
        attributes: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Export HTTP request latency metrics.

        Args:
            endpoint: API endpoint path
            status_code: HTTP status code
            latency_ms: Request latency in milliseconds
            method: HTTP method
            attributes: Additional attributes

        Returns:
            True if export succeeded
        """
        if not self._enabled:
            return False

        timestamp_ns = int(time.time() * 1_000_000_000)

        metric_attributes = [
            {"key": "endpoint", "value": {"stringValue": endpoint}},
            {"key": "method", "value": {"stringValue": method}},
            {"key": "status_code", "value": {"stringValue": str(status_code)}},
            {"key": "service", "value": {"stringValue": settings.app_name}},
        ]

        if attributes:
            for key, value in attributes.items():
                metric_attributes.append({
                    "key": key,
                    "value": {"stringValue": str(value)}
                })

        payload = {
            "resourceMetrics": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": settings.app_name}},
                            {"key": "service.version", "value": {"stringValue": settings.app_version}},
                        ]
                    },
                    "scopeMetrics": [
                        {
                            "metrics": [
                                {
                                    "name": "http_request_latency_ms",
                                    "unit": "ms",
                                    "gauge": {
                                        "dataPoints": [
                                            {
                                                "asInt": latency_ms,
                                                "timeUnixNano": timestamp_ns,
                                                "attributes": metric_attributes
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {self._auth_encoded}",
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    self._url,
                    headers=headers,
                    json=payload
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(
                "Error exporting request latency to Grafana",
                extra={"error": str(e)}
            )
            return False


# Global exporter instance
_grafana_exporter: Optional[GrafanaOTLPExporter] = None


def get_grafana_exporter() -> Optional[GrafanaOTLPExporter]:
    """Get or create global Grafana exporter instance."""
    global _grafana_exporter
    if _grafana_exporter is None:
        _grafana_exporter = GrafanaOTLPExporter()
    return _grafana_exporter


def init_grafana_exporter(
    host: str,
    api_key: str,
    instance_id: str
) -> GrafanaOTLPExporter:
    """Initialize Grafana exporter with credentials."""
    global _grafana_exporter
    _grafana_exporter = GrafanaOTLPExporter(
        host=host,
        api_key=api_key,
        instance_id=instance_id
    )
    return _grafana_exporter
