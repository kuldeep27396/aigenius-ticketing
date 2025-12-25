"""
SLA External Service Integrations
==================================

External services for SLA monitoring:
- Slack webhook notifications
- YAML config file watcher
- APScheduler for background evaluation
"""

import asyncio
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

import yaml
import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src.shared.infrastructure.logging import get_logger
from src.config import settings
from src.sla.domain.value_objects import SLAConfig

logger = get_logger(__name__)


class ConfigFileHandler(FileSystemEventHandler):
    """Watchdog event handler for SLA config file changes."""

    def __init__(self, config_manager: "SLAConfigManager", config_path: Path):
        self.config_manager = config_manager
        self.config_path = config_path
        super().__init__()

    def on_modified(self, event):
        """Handle file modification event."""
        if event.is_directory:
            return
        if Path(event.src_path).resolve() == self.config_path.resolve():
            logger.info(f"Config file changed: {event.src_path}")
            self.config_manager.reload()


class SLAConfigManager:
    """
    Thread-safe SLA configuration manager with hot-reload support.

    Uses watchdog to monitor file changes and reload configuration
    without restarting the service.
    """

    def __init__(self):
        self._config: Optional[SLAConfig] = None
        self._lock = threading.Lock()
        self._path: Optional[Path] = None
        self._observer = None

    def load(self, path: Path) -> SLAConfig:
        """Initial configuration load."""
        self._path = path
        self._config = self._load_from_file(path)
        return self._config

    def _load_from_file(self, path: Path) -> SLAConfig:
        """Load and parse YAML config file."""
        if not path.exists():
            logger.warning(f"SLA config file not found: {path}, using defaults")
            return SLAConfig()

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        return SLAConfig(**data)

    def reload(self) -> bool:
        """Reload configuration from file."""
        if self._path is None:
            return False

        try:
            new_config = self._load_from_file(self._path)
            with self._lock:
                self._config = new_config
            logger.info("SLA configuration reloaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to reload SLA config: {e}")
            return False

    def start_watching(self) -> None:
        """
        Start watching configuration file for changes.

        Skips watching if:
        - File doesn't exist (production environments often use env vars)
        - Running in a containerized environment where inotify doesn't work
        """
        if self._path is None:
            raise RuntimeError("Config not loaded. Call load() first.")

        # Skip file watching if the config file doesn't exist
        # Production environments typically use environment variables
        if not self._path.exists():
            logger.info(
                f"Config file doesn't exist, skipping file watch: {self._path}. "
                "Using default SLA configuration."
            )
            return

        try:
            self._observer = Observer()
            handler = ConfigFileHandler(self, self._path)
            self._observer.schedule(
                handler,
                str(self._path.parent),
                recursive=False
            )
            self._observer.start()
            logger.info(f"Started watching config file: {self._path}")
        except (OSError, FileNotFoundError) as e:
            # File watching not supported (e.g., in Docker containers)
            logger.warning(
                f"File watching not available, using static config: {e}"
            )
            self._observer = None

    def stop_watching(self) -> None:
        """Stop watching configuration file (safe to call even if not watching)."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

    @property
    def config(self) -> SLAConfig:
        """Get current configuration."""
        if self._config is None:
            raise RuntimeError("SLA configuration not loaded")
        return self._config


class CircuitState:
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Circuit breaker for preventing cascade failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: After N failures, reject all requests for M seconds
    - HALF_OPEN: After timeout, allow one test request
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None

    @property
    def state(self) -> str:
        """Get current circuit state."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time:
                import time
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
        return self._state

    def allow_request(self) -> bool:
        """Check if request should be allowed."""
        state = self.state
        return state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)

    def record_success(self) -> None:
        """Record successful request."""
        self._failure_count = 0
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record failed request."""
        import time
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            logger.warning(
                "Circuit breaker opened",
                extra={
                    "failure_count": self._failure_count,
                    "recovery_timeout": self.recovery_timeout
                }
            )


@dataclass
class SlackMessage:
    """Slack notification message."""
    ticket_id: str
    priority: str
    customer_tier: str
    sla_type: str
    alert_type: str
    remaining_percentage: float
    escalation_level: int
    status: str
    created_at: str
    timestamp: str


class SlackClient:
    """
    Slack webhook client with circuit breaker and retry logic.

    Handles sending structured alerts to Slack with:
    - Circuit breaker to prevent cascade failures
    - Exponential backoff retry
    - Timeout handling
    """

    def __init__(self):
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )
        self._http_client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=settings.slack_timeout_seconds
            )
        return self._http_client

    def _build_message(self, data: SlackMessage) -> Dict[str, Any]:
        """Build Slack Block Kit message."""
        is_breach = data.alert_type == "breach"

        if is_breach:
            emoji = "🚨"
            header_text = "SLA Breach Alert"
            status_emoji = "🔴 BREACHED"
        else:
            emoji = "⚠️"
            header_text = "SLA Warning Alert"
            status_emoji = "🟡 AT RISK"

        ticket_url = f"https://support.example.com/{data.ticket_id}"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {header_text}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Ticket:*\n<{ticket_url}|{data.ticket_id}>"},
                    {"type": "mrkdwn", "text": f"*Priority:*\n{data.priority.title()}"},
                    {"type": "mrkdwn", "text": f"*Customer Tier:*\n{data.customer_tier.title()}"},
                    {"type": "mrkdwn", "text": f"*SLA Type:*\n{data.sla_type.title()}"},
                    {"type": "mrkdwn", "text": f"*Status:*\n{status_emoji}"},
                    {"type": "mrkdwn", "text": f"*Escalation Level:*\n{data.escalation_level}"}
                ]
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Created: {data.created_at} | Remaining: {data.remaining_percentage:.1f}%"
                    }
                ]
            }
        ]

        return {
            "channel": settings.slack_channel,
            "blocks": blocks
        }

    async def send_alert(
        self,
        data: SlackMessage,
        max_retries: int = 3
    ) -> bool:
        """
        Send alert to Slack webhook.

        Returns:
            True if sent successfully, False otherwise
        """
        if not settings.slack_webhook_url:
            logger.debug("Slack webhook URL not configured, skipping notification")
            return False

        if not self._circuit_breaker.allow_request():
            logger.warning(
                "Circuit breaker open, skipping Slack notification",
                extra={"ticket_id": data.ticket_id}
            )
            return False

        message = self._build_message(data)

        for attempt in range(max_retries):
            try:
                client = await self._get_client()
                response = await client.post(
                    settings.slack_webhook_url,
                    json=message
                )

                if response.status_code == 200:
                    self._circuit_breaker.record_success()
                    logger.info(
                        "Slack notification sent",
                        extra={
                            "ticket_id": data.ticket_id,
                            "alert_type": data.alert_type
                        }
                    )
                    return True
                else:
                    logger.warning(
                        "Slack webhook returned non-200",
                        extra={
                            "status_code": response.status_code,
                            "attempt": attempt + 1
                        }
                    )

            except Exception as e:
                logger.error(
                    "Slack notification failed",
                    extra={
                        "error": str(e),
                        "attempt": attempt + 1,
                        "ticket_id": data.ticket_id
                    }
                )

            if attempt < max_retries - 1:
                delay = 2 ** attempt
                await asyncio.sleep(delay)

        self._circuit_breaker.record_failure()
        return False

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


class SLAScheduler:
    """
    Wrapper for APScheduler for background SLA evaluation.

    Manages the lifecycle of the scheduler and jobs.
    """

    def __init__(self, interval_seconds: int = 60):
        self.interval_seconds = interval_seconds
        self._scheduler: Optional[AsyncIOScheduler] = None
        self._running = False

    async def start(self, job_func) -> None:
        """Start the scheduler with the given job function."""
        if self._running:
            logger.warning("SLA scheduler already running")
            return

        self._scheduler = AsyncIOScheduler()

        self._scheduler.add_job(
            job_func,
            "interval",
            seconds=self.interval_seconds,
            id="sla_evaluation",
            name="SLA Evaluation Job",
            misfire_grace_time=60,
            max_instances=1,
            replace_existing=True
        )

        self._scheduler.start()
        self._running = True

        logger.info(
            "SLA scheduler started",
            extra={"interval_seconds": self.interval_seconds}
        )

    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        if not self._running:
            return

        if self._scheduler:
            self._scheduler.shutdown(wait=True)

        self._running = False
        logger.info("SLA scheduler stopped")

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running
