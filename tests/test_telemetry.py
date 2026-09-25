"""
Tests for structured telemetry, request correlation, and latency measurement.

Covers:
- X-Request-ID generation when missing
- X-Request-ID propagation when supplied and valid
- X-Request-ID sanitization when malicious/malformed
- Response header injection (X-Request-ID)
- Structured JSON log emission
- Required fields in log entries (timestamp, level, service, version, request_id, method, path, status_code, latency_ms)
- Monotonic latency measurement
"""

import json
import logging
import re
from typing import Any
import uuid

from fastapi.testclient import TestClient
import pytest

from app.config import settings
from app.main import app
from app.telemetry import (
    TELEMETRY_LOGGER_NAME,
    StructuredJSONFormatter,
    get_or_generate_request_id,
    telemetry_logger,
)

client = TestClient(app)


# -----------------------------------------------------------------------------
# Log capture fixture
# -----------------------------------------------------------------------------


class InMemoryLogHandler(logging.Handler):
    """Captures structured log records in-memory for assertion during tests."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        if isinstance(record.msg, dict):
            self.records.append(record.msg)


@pytest.fixture
def captured_logs():
    """Fixture that intercepts structured logs from the telemetry logger."""
    handler = InMemoryLogHandler()
    telemetry_logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        telemetry_logger.removeHandler(handler)


# -----------------------------------------------------------------------------
# Request ID (Correlation) Tests
# -----------------------------------------------------------------------------


def test_request_receives_generated_request_id_when_none_supplied() -> None:
    """When no X-Request-ID is sent, the response must contain a valid UUIDv4."""
    response = client.get("/health")
    assert response.status_code == 200

    request_id = response.headers.get("X-Request-ID")
    assert request_id is not None
    # Validate it is a valid UUID
    parsed_uuid = uuid.UUID(request_id, version=4)
    assert str(parsed_uuid) == request_id


def test_request_preserves_supplied_safe_request_id() -> None:
    """When a valid, safe X-Request-ID is provided, it must be echoed in the response."""
    custom_id = "test-client-trace-12345"
    response = client.get("/health", headers={"X-Request-ID": custom_id})
    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_id


def test_request_sanitizes_unsafe_request_id() -> None:
    """Unsafe request IDs (containing CRLF, control chars, or excess length) must be replaced."""
    # Attempt log-injection payload with CRLF
    malicious_id = "fake-id\r\nSET-COOKIE: malicious=1"
    response = client.get("/health", headers={"X-Request-ID": malicious_id})
    assert response.status_code == 200

    returned_id = response.headers.get("X-Request-ID")
    assert returned_id != malicious_id
    # Must be replaced with a valid UUID
    assert uuid.UUID(returned_id, version=4)


def test_helper_get_or_generate_request_id_edge_cases() -> None:
    """Direct unit tests for request ID validator helper."""
    # None or empty returns UUID
    assert uuid.UUID(get_or_generate_request_id(None), version=4)
    assert uuid.UUID(get_or_generate_request_id(""), version=4)

    # Valid IDs preserved
    assert get_or_generate_request_id("valid-id-123") == "valid-id-123"
    assert get_or_generate_request_id("req_abc_DEF") == "req_abc_DEF"

    # Too long (> 64 characters) rejected
    too_long = "a" * 65
    assert get_or_generate_request_id(too_long) != too_long


# -----------------------------------------------------------------------------
# Structured Logging Tests
# -----------------------------------------------------------------------------


def test_structured_log_emitted_on_successful_request(captured_logs: list[dict[str, Any]]) -> None:
    """Every request must produce a structured log event with required telemetry keys."""
    response = client.get("/health")
    assert response.status_code == 200

    matching_logs = [log for log in captured_logs if log.get("path") == "/health"]
    assert len(matching_logs) >= 1

    entry = matching_logs[-1]
    # Check all required telemetry fields
    assert "timestamp" in entry
    assert entry["level"] == "INFO"
    assert entry["service"] == settings.service_name
    assert entry["version"] == settings.app_version
    assert entry["request_id"] == response.headers["X-Request-ID"]
    assert entry["method"] == "GET"
    assert entry["path"] == "/health"
    assert entry["status_code"] == 200
    assert isinstance(entry["latency_ms"], (int, float))
    assert entry["latency_ms"] >= 0.0


def test_structured_log_emitted_on_404_request(captured_logs: list[dict[str, Any]]) -> None:
    """404 requests must be logged with WARNING level and matching status code."""
    response = client.get("/api/v1/orders/ORD-9999")
    assert response.status_code == 404

    matching_logs = [log for log in captured_logs if log.get("path") == "/api/v1/orders/ORD-9999"]
    assert len(matching_logs) >= 1

    entry = matching_logs[-1]
    assert entry["level"] == "WARNING"
    assert entry["status_code"] == 404
    assert entry["request_id"] == response.headers["X-Request-ID"]
    assert entry["method"] == "GET"
    assert isinstance(entry["latency_ms"], (int, float))


def test_structured_log_contains_correct_correlation_id(captured_logs: list[dict[str, Any]]) -> None:
    """The request_id in the log must match the X-Request-ID in the response header exactly."""
    custom_trace = "trace-order-pipeline-777"
    response = client.get("/api/v1/orders", headers={"X-Request-ID": custom_trace})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == custom_trace

    matching_logs = [log for log in captured_logs if log.get("request_id") == custom_trace]
    assert len(matching_logs) >= 1

    entry = matching_logs[-1]
    assert entry["request_id"] == custom_trace
    assert entry["path"] == "/api/v1/orders"
    assert entry["status_code"] == 200


def test_no_sensitive_headers_or_secrets_logged(captured_logs: list[dict[str, Any]]) -> None:
    """Confirms no Authorization header, cookie, or secrets leak into the telemetry log."""
    sensitive_token = "Bearer secret-jwt-token-never-log"
    client.get(
        "/health",
        headers={
            "Authorization": sensitive_token,
            "Cookie": "session_id=secret-session-id",
        },
    )

    matching_logs = [log for log in captured_logs if log.get("path") == "/health"]
    assert len(matching_logs) >= 1

    for log in matching_logs:
        serialized = json.dumps(log)
        assert sensitive_token not in serialized
        assert "secret-session-id" not in serialized
        assert "Authorization" not in log
        assert "Cookie" not in log


def test_json_formatter_produces_valid_json_string() -> None:
    """The StructuredJSONFormatter must output a valid single-line JSON string."""
    formatter = StructuredJSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg={"key": "value", "count": 42},
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)

    # Must be valid single-line JSON
    assert "\n" not in formatted
    parsed = json.loads(formatted)
    assert parsed == {"key": "value", "count": 42}
