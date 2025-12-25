"""
Shared API Middleware
======================

Common middleware for all FastAPI applications.
"""

import time
import uuid
from typing import Callable
from datetime import datetime

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from fastapi.responses import JSONResponse

from src.shared.infrastructure.logging import get_logger

logger = get_logger(__name__)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    Adds correlation ID to requests for tracing.

    Correlation IDs are essential for tracing requests through
    distributed systems and linking logs.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get existing correlation ID or generate new one
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))

        # Store in request state for access in endpoints
        request.state.correlation_id = correlation_id

        # Add to response header
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Tracks request metrics for monitoring.

    Records response times and status codes for observability.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.request_count = 0
        self.total_response_time = 0.0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        self.request_count += 1

        response = await call_next(request)

        # Calculate response time
        response_time = time.perf_counter() - start_time
        self.total_response_time += response_time

        # Add metrics as headers (for debugging)
        response.headers["X-Response-Time"] = f"{response_time:.3f}s"
        response.headers["X-Request-Count"] = str(self.request_count)

        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs all requests and responses.

    Provides audit trail and debugging information.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        correlation_id = getattr(request.state, "correlation_id", "unknown")
        start_time = time.perf_counter()

        logger.info(
            "Request started",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "url": str(request.url),
                "client": request.client.host if request.client else None
            }
        )

        try:
            response = await call_next(request)

            response_time = time.perf_counter() - start_time
            logger.info(
                "Request completed",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "url": str(request.url),
                    "status_code": response.status_code,
                    "response_time_ms": int(response_time * 1000)
                }
            )

            return response

        except Exception as e:
            response_time = time.perf_counter() - start_time
            logger.error(
                "Request failed",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "url": str(request.url),
                    "error": str(e),
                    "response_time_ms": int(response_time * 1000)
                }
            )
            raise


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for unhandled exceptions.

    Returns consistent error responses for all exceptions.
    """
    correlation_id = getattr(request.state, "correlation_id", "unknown")

    logger.error(
        "Unhandled exception",
        extra={
            "correlation_id": correlation_id,
            "path": request.url.path,
            "method": request.method,
            "error_type": type(exc).__name__,
            "error_message": str(exc)
        }
    )

    # Don't expose internal details in production
    is_dev = getattr(getattr(request.app.state, "settings", None), "environment", None) == "development"

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "correlation_id": correlation_id,
            "timestamp": datetime.utcnow().isoformat(),
            "debug_info": str(exc) if is_dev else None
        }
    )
