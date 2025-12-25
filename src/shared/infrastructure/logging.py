"""
Structured Logging
==================

JSON-structured logging with correlation ID tracking.

Provides:
- Structured JSON logs (parseable by log aggregators)
- Correlation ID for request tracing
- Contextual loggers for modules
- Performance timing utilities

Usage:
    from shared.infrastructure.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Ticket processed", extra={"ticket_id": "TICKET-001"})
"""

import logging
import sys
import uuid
from datetime import datetime
from typing import Any
from contextlib import contextmanager

from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter with additional fields.

    Adds:
    - timestamp in ISO format
    - correlation_id when available
    - Environment info
    """

    def add_fields(
        self,
        log_record: logging.LogRecord,
        record_dict: dict[str, Any],
        message_dict: dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record_dict, message_dict)

        # Ensure record_dict is a dict (handle compatibility issues)
        if not isinstance(record_dict, dict):
            return

        # Add timestamp
        if "timestamp" not in record_dict or not record_dict.get("timestamp"):
            record_dict["timestamp"] = datetime.utcnow().isoformat()

        # Add correlation_id if present in extra
        if hasattr(log_record, "correlation_id"):
            record_dict["correlation_id"] = log_record.correlation_id
        elif "correlation_id" in message_dict:
            record_dict["correlation_id"] = message_dict["correlation_id"]

        # Add environment
        record_dict["environment"] = getattr(log_record, "environment", "unknown")

        # Sanitize any sensitive data
        for key, value in list(record_dict.items()):
            if isinstance(value, str):
                # Remove common sensitive patterns
                if "password" in key.lower():
                    record_dict[key] = "***REDACTED***"
                elif "token" in key.lower() and "tokens_used" not in key:
                    record_dict[key] = "***REDACTED***"
                elif "api_key" in key.lower():
                    record_dict[key] = "***REDACTED***"


def setup_logging(
    level: str = "INFO",
    environment: str = "development",
) -> None:
    """
    Configure structured JSON logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        environment: Environment name for log context
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Create console handler with JSON formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    formatter = CustomJsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Silence noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        logging.Logger: Configured logger instance
    """
    return logging.getLogger(name)


def get_context_logger(name: str, correlation_id: str | None = None) -> logging.Logger:
    """
    Get a logger with correlation ID for request tracing.

    Args:
        name: Logger name
        correlation_id: Request correlation ID

    Returns:
        logging.Logger: Logger with correlation_id in extra
    """
    logger = get_logger(name)
    if correlation_id:
        logger = logging.LoggerAdapter(logger, {"correlation_id": correlation_id})
    return logger


@contextmanager
def log_latency(logger: logging.Logger, operation: str, **extra_context: Any):
    """
    Context manager for measuring and logging operation latency.

    Usage:
        with log_latency(logger, "database_query", table="tickets"):
            result = await session.execute(query)

    Args:
        logger: Logger instance
        operation: Operation name for logging
        **extra_context: Additional context to include in log
    """
    import time

    start = time.perf_counter()
    try:
        yield
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            f"{operation} completed",
            extra={
                "operation": operation,
                "latency_ms": round(latency_ms, 2),
                **extra_context,
            },
        )
