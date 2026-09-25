"""
Tests for Deployment Simulation & Change Ingestion (Phase 4).

Covers:
- Empty change handling
- Single and multiple file change feature extraction
- Accurate additions, deletions, and total churn calculation
- High-risk path classification rules (orders, payment, auth, db, dockerfile)
- Configuration file classification
- Test file classification
- Documentation-only changes
- Unified Git diff parser (single-file, multi-file, additions, deletions)
- Telemetry metrics combination (feature vector generation)
- API endpoint verification (POST /api/v1/analyze-change)
- Validation / error handling for malformed input
- Sample changes verification (Change A, B, C, D)
"""

from fastapi.testclient import TestClient
import pytest

from app.change_analysis import (
    SAMPLE_CHANGES,
    ChangeAnalysisRequest,
    DeploymentChange,
    FileChange,
    RuntimeTelemetryMetrics,
    extract_features,
    is_application_file,
    is_config_file,
    is_doc_file,
    is_high_risk_path,
    is_test_file,
    parse_git_diff,
)
from app.main import app

client = TestClient(app)


# -----------------------------------------------------------------------------
# Unit Tests: Path Classification
# -----------------------------------------------------------------------------


def test_high_risk_path_detection() -> None:
    """Critical business domains must be flagged as high risk."""
    assert is_high_risk_path("app/routes/orders.py") is True
    assert is_high_risk_path("src/payment/gateway.py") is True
    assert is_high_risk_path("app/auth/tokens.py") is True
    assert is_high_risk_path("config/database.yml") is True
    assert is_high_risk_path("Dockerfile") is True
    assert is_high_risk_path("deploy/helm/values.yaml") is True

    # Non-critical files
    assert is_high_risk_path("app/utils/string_helpers.py") is False
    assert is_high_risk_path("README.md") is False


def test_config_file_detection() -> None:
    """Configuration and environment files must be classified correctly."""
    assert is_config_file(".env") is True
    assert is_config_file(".env.example") is True
    assert is_config_file("config/database.yml") is True
    assert is_config_file("app/config.py") is True
    assert is_config_file("pyproject.toml") is True
    assert is_config_file("settings.ini") is True

    # Non-config files
    assert is_config_file("app/routes/orders.py") is False
    assert is_config_file("README.md") is False


def test_test_file_detection() -> None:
    """Test suite files must be recognized regardless of path convention."""
    assert is_test_file("tests/test_orders.py") is True
    assert is_test_file("tests/conftest.py") is True
    assert is_test_file("test_api.py") is True
    assert is_test_file("app/order_test.py") is True

    # Non-test files
    assert is_test_file("app/routes/orders.py") is False
    assert is_test_file("Dockerfile") is False


def test_doc_file_detection() -> None:
    """Documentation files must be identified."""
    assert is_doc_file("README.md") is True
    assert is_doc_file("docs/architecture.md") is True
    assert is_doc_file("LICENSE") is True
    assert is_doc_file("CHANGELOG.rst") is True

    # Docs should never trigger high-risk flags even if path has 'security' or 'order'
    assert is_doc_file("docs/orders_guide.md") is True
    assert is_high_risk_path("docs/orders_guide.md") is False


def test_application_file_detection() -> None:
    """Application source code is distinct from tests, configs, and docs."""
    assert is_application_file("app/routes/orders.py") is True
    assert is_application_file("app/data.py") is True

    # Tests, configs, and docs are not application files
    assert is_application_file("tests/test_orders.py") is False
    assert is_application_file(".env.example") is False
    assert is_application_file("README.md") is False


# -----------------------------------------------------------------------------
# Unit Tests: Git Diff Parser
# -----------------------------------------------------------------------------


def test_parse_git_diff_empty() -> None:
    """Empty or whitespace diff string produces empty list."""
    assert parse_git_diff("") == []
    assert parse_git_diff("   \n  ") == []


def test_parse_git_diff_single_file() -> None:
    """Parses a single-file diff with additions and deletions."""
    sample_diff = """diff --git a/app/routes/orders.py b/app/routes/orders.py
index 1234567..89abcdef 100644
--- a/app/routes/orders.py
+++ b/app/routes/orders.py
@@ -10,4 +10,6 @@
 unchanged line
-old line 1
-old line 2
+new line 1
+new line 2
+new line 3
"""
    changes = parse_git_diff(sample_diff)
    assert len(changes) == 1
    assert changes[0].path == "app/routes/orders.py"
    assert changes[0].lines_added == 3
    assert changes[0].lines_deleted == 2
    assert changes[0].total_lines_changed == 5


def test_parse_git_diff_multiple_files() -> None:
    """Parses a multi-file diff correctly aggregating file metrics."""
    sample_diff = """diff --git a/app/payment.py b/app/payment.py
new file mode 100644
--- /dev/null
+++ b/app/payment.py
@@ -0,0 +1,5 @@
+def process_payment():
+    pass
diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
@@ -1,3 +1,3 @@
-Old title
+New title
"""
    changes = parse_git_diff(sample_diff)
    assert len(changes) == 2

    payment_change = next(c for c in changes if c.path == "app/payment.py")
    assert payment_change.lines_added == 2
    assert payment_change.lines_deleted == 0

    doc_change = next(c for c in changes if c.path == "README.md")
    assert doc_change.lines_added == 1
    assert doc_change.lines_deleted == 1


# -----------------------------------------------------------------------------
# Feature Extraction Engine Tests
# -----------------------------------------------------------------------------


def test_extract_features_empty_change() -> None:
    """An empty change produces a valid feature vector with zeros."""
    change = DeploymentChange(
        deployment_id="DEP-EMPTY",
        commit_sha="000000000000",
        branch="main",
        changed_files=[],
    )
    features = extract_features(change)
    assert features.deployment_id == "DEP-EMPTY"
    assert features.commit_sha == "000000000000"
    assert features.files_changed == 0
    assert features.lines_added == 0
    assert features.lines_deleted == 0
    assert features.total_lines_changed == 0
    assert features.high_risk_files_count == 0
    assert features.high_risk_area_touched is False
    assert features.mean_latency_ms is None


def test_extract_features_with_raw_diff() -> None:
    """extract_features parses raw_diff automatically if changed_files is empty."""
    diff_text = """diff --git a/app/routes/orders.py b/app/routes/orders.py
--- a/app/routes/orders.py
+++ b/app/routes/orders.py
@@ -1,2 +1,3 @@
-old line
+new line 1
+new line 2
"""
    change = DeploymentChange(
        deployment_id="DEP-DIFF-TEST",
        commit_sha="abcdef123456",
        raw_diff=diff_text,
    )
    features = extract_features(change)
    assert features.files_changed == 1
    assert features.lines_added == 2
    assert features.lines_deleted == 1
    assert features.total_lines_changed == 3
    assert features.high_risk_files_count == 1
    assert features.high_risk_area_touched is True


def test_extract_features_documentation_only_change() -> None:
    """Doc-only changes must not flag high-risk area touched."""
    change = SAMPLE_CHANGES["change_c_docs_only"]
    features = extract_features(change)

    assert features.files_changed == 2
    assert features.doc_files_changed == 2
    assert features.high_risk_files_count == 0
    assert features.high_risk_area_touched is False
    assert features.application_files_changed == 0


def test_extract_features_high_risk_payment_db() -> None:
    """High-risk payment and database changes must trigger all risk flags."""
    change = SAMPLE_CHANGES["change_b_payment_db_high_risk"]
    features = extract_features(change)

    assert features.files_changed == 2
    assert features.high_risk_files_count == 2
    assert features.high_risk_area_touched is True
    assert features.config_files_changed == 1
    assert features.application_files_changed == 1
    assert features.lines_added == 179
    assert features.lines_deleted == 45
    assert features.total_lines_changed == 224


def test_extract_features_combined_with_telemetry() -> None:
    """Combining change features with telemetry creates the unified feature vector."""
    change = SAMPLE_CHANGES["change_a_order_feature"]
    telemetry = RuntimeTelemetryMetrics(
        mean_latency_ms=45.2,
        p99_latency_ms=120.8,
        error_rate=0.015,
        request_volume=5400,
    )
    features = extract_features(change, telemetry)

    # Change features
    assert features.files_changed == 2
    assert features.high_risk_area_touched is True
    assert features.test_files_changed == 1
    assert features.application_files_changed == 1

    # Telemetry features
    assert features.mean_latency_ms == 45.2
    assert features.p99_latency_ms == 120.8
    assert features.error_rate == 0.015
    assert features.request_volume == 5400


# -----------------------------------------------------------------------------
# API Route Tests (POST /api/v1/analyze-change)
# -----------------------------------------------------------------------------


def test_api_analyze_change_success() -> None:
    """POST /api/v1/analyze-change returns 200 and the extracted feature vector."""
    payload = {
        "change": {
            "deployment_id": "DEP-TEST-API-001",
            "commit_sha": "1234567890ab",
            "branch": "feature/checkout",
            "changed_files": [
                {"path": "app/routes/orders.py", "lines_added": 15, "lines_deleted": 2},
                {"path": "tests/test_orders.py", "lines_added": 10, "lines_deleted": 0},
            ],
        },
        "telemetry": {
            "mean_latency_ms": 12.5,
            "p99_latency_ms": 35.0,
            "error_rate": 0.002,
            "request_volume": 1200,
        },
    }
    response = client.post("/api/v1/analyze-change", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["deployment_id"] == "DEP-TEST-API-001"
    features = data["features"]
    assert features["files_changed"] == 2
    assert features["lines_added"] == 25
    assert features["lines_deleted"] == 2
    assert features["total_lines_changed"] == 27
    assert features["high_risk_files_count"] == 1
    assert features["high_risk_area_touched"] is True
    assert features["mean_latency_ms"] == 12.5
    assert features["request_volume"] == 1200


def test_api_analyze_change_with_raw_diff() -> None:
    """POST /api/v1/analyze-change parses raw diff in request payload."""
    raw_diff = """diff --git a/config/database.yml b/config/database.yml
--- a/config/database.yml
+++ b/config/database.yml
@@ -1,2 +1,3 @@
-pool_size: 5
+pool_size: 20
+timeout: 30
"""
    payload = {
        "change": {
            "deployment_id": "DEP-DIFF-API-002",
            "commit_sha": "fedcba987654",
            "branch": "fix/db-pool",
            "raw_diff": raw_diff,
        }
    }
    response = client.post("/api/v1/analyze-change", json=payload)
    assert response.status_code == 200

    data = response.json()
    features = data["features"]
    assert features["files_changed"] == 1
    assert features["config_files_changed"] == 1
    assert features["high_risk_area_touched"] is True
    assert features["lines_added"] == 2
    assert features["lines_deleted"] == 1


def test_api_analyze_change_invalid_payload() -> None:
    """Missing mandatory fields produces HTTP 422 Unprocessable Entity."""
    # Missing deployment_id and commit_sha
    payload = {"change": {"changed_files": []}}
    response = client.post("/api/v1/analyze-change", json=payload)
    assert response.status_code == 422


def test_api_list_sample_changes() -> None:
    """GET /api/v1/analyze-change/samples lists all benchmark simulated changes."""
    response = client.get("/api/v1/analyze-change/samples")
    assert response.status_code == 200
    samples = response.json()
    assert "change_a_order_feature" in samples
    assert "change_b_payment_db_high_risk" in samples
    assert "change_c_docs_only" in samples
    assert "change_d_infra_docker" in samples
