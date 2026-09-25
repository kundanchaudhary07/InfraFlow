"""
Structured telemetry and instrumentation module for DeployGuard AI.

Provides:
- Standardized ISO-8601 UTC JSON log formatting.
- Request correlation via X-Request-ID (propagation and generation).
- Monotonic request latency measurement.
- Non-intrusive Starlette/FastAPI middleware.
- Safe logging that excludes secrets, tokens, authorization headers, and bodies.
"""

from datetime import datetime, timezone
import json
import logging
import re
import sys
import time
from typing import Any, Optional
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.config import settings

# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------

TELEMETRY_LOGGER_NAME = "deployguard.telemetry"

# Safe pattern for incoming client-supplied request IDs:
# Alphanumeric characters, dashes, and underscores only, max length 64.
# Prevents log injection, CRLF injection, and arbitrary payload bloating.
_REQUEST_ID_REGEX = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def get_or_generate_request_id(header_val: Optional[str]) -> str:
    """
    Validate an incoming X-Request-ID header value.

    If valid and safe, preserve it.
    If missing, empty, or failing the safety regex, generate a new UUIDv4.
    """
    if header_val and _REQUEST_ID_REGEX.match(header_val):
        return header_val
    return str(uuid.uuid4())


# -----------------------------------------------------------------------------
# Structured JSON Formatter
# -----------------------------------------------------------------------------


class StructuredJSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.

    If the log message is a dictionary, it is serialized directly as JSON.
    Otherwise, standard logging fields are wrapped into a structured JSON envelope.
    """

    def format(self, record: logging.LogRecord) -> str:
        if isinstance(record.msg, dict):
            payload = record.msg
        else:
            payload = {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "level": record.levelname,
                "service": settings.service_name,
                "version": settings.app_version,
                "message": record.getMessage(),
            }
        return json.dumps(payload)


def get_telemetry_logger() -> logging.Logger:
    """
    Get or configure the dedicated telemetry logger.

    Writes single-line JSON events directly to sys.stdout.
    Sets propagate=False so root handlers do not duplicate or corrupt format.
    """
    logger = logging.getLogger(TELEMETRY_LOGGER_NAME)
    
    # Configure handler only if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredJSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
        logger.propagate = False

    return logger


# Module-level telemetry logger instance
telemetry_logger = get_telemetry_logger()


# -----------------------------------------------------------------------------
# Internal event emission helper
# -----------------------------------------------------------------------------


def _emit_request_log(
    *,
    request: Request,
    request_id: str,
    status_code: int,
    latency_ms: float,
    level: str,
    error_message: Optional[str] = None,
) -> dict[str, Any]:
    """
    Build and log a structured request event.

    Never logs sensitive headers (Authorization, Cookie), tokens, or bodies.
    """
    event: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "level": level,
        "service": settings.service_name,
        "version": settings.app_version,
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": status_code,
        "latency_ms": latency_ms,
        "client_ip": request.client.host if request.client else None,
    }

    if error_message:
        event["error"] = error_message

    numeric_level = getattr(logging, level, logging.INFO)
    telemetry_logger.log(numeric_level, event)
    return event


# -----------------------------------------------------------------------------
# Telemetry Middleware
# -----------------------------------------------------------------------------


class RequestTelemetryMiddleware(BaseHTTPMiddleware):
    """
    Starlette middleware for:
    1. Request correlation (X-Request-ID propagation / generation).
    2. High-resolution monotonic timing (latency_ms).
    3. Structured JSON request logging.
    4. Response header injection (X-Request-ID).
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.perf_counter()

        # 1. Resolve or generate request ID
        raw_header = request.headers.get("X-Request-ID")
        request_id = get_or_generate_request_id(raw_header)

        # Store on request.state for downstream visibility if needed
        request.state.request_id = request_id

        # 2. Process request through pipeline
        try:
            response = await call_next(request)
        except Exception as exc:
            # Monotonic duration measurement on failure
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            _emit_request_log(
                request=request,
                request_id=request_id,
                status_code=500,
                latency_ms=latency_ms,
                level="ERROR",
                error_message=str(exc),
            )
            raise exc

        # 3. Monotonic duration measurement on completion
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # 4. Inject X-Request-ID response header
        response.headers["X-Request-ID"] = request_id

        # 5. Determine log level based on HTTP response code
        if response.status_code >= 500:
            level = "ERROR"
        elif response.status_code >= 400:
            level = "WARNING"
        else:
            level = "INFO"

        # 6. Emit single-line structured JSON log event
        _emit_request_log(
            request=request,
            request_id=request_id,
            status_code=response.status_code,
            latency_ms=latency_ms,
            level=level,
        )

        return response
