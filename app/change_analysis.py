"""
Deployment Simulation and Change Ingestion module for DeployGuard AI.

Provides:
- Data models for code/infrastructure changes and runtime telemetry.
- Unified Git diff parser for extracting file modifications and line diffs.
- Deterministic risk feature extraction (path classification, blast radius).
- Merging of code change features with runtime telemetry into a unified feature vector.
- Sample deployment datasets for simulation and benchmarking.
"""

from pathlib import PurePath
import re
from typing import Any, Optional

from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# High-Risk Path Classification Rules
# -----------------------------------------------------------------------------

# Substrings in file paths that represent critical business or security domains
HIGH_RISK_PATH_KEYWORDS: tuple[str, ...] = (
    "payment",
    "checkout",
    "order",
    "billing",
    "auth",
    "security",
    "token",
    "credential",
    "secret",
    "password",
    "db",
    "database",
    "migration",
    "schema",
    "dockerfile",
    "deploy",
    "pipeline",
    "infra",
)

CONFIG_EXTENSIONS: tuple[str, ...] = (
    ".env",
    ".env.example",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
)

CONFIG_FILENAMES: tuple[str, ...] = (
    "config.py",
    "settings.py",
    "configuration.py",
    "database.yml",
    "database.yaml",
)

DOC_EXTENSIONS: tuple[str, ...] = (
    ".md",
    ".rst",
    ".txt",
    ".pdf",
)


def is_high_risk_path(path: str) -> bool:
    """Return True if the file path intersects with a critical application domain."""
    normalized = path.lower().replace("\\", "/")
    # Doc files are never high risk even if they mention 'order' or 'security' in docs/
    if is_doc_file(path):
        return False
    # Test files touching high risk domains are classified as test files, not high risk core files
    if is_test_file(path):
        return False
    return any(keyword in normalized for keyword in HIGH_RISK_PATH_KEYWORDS)


def is_config_file(path: str) -> bool:
    """Return True if the file is an infrastructure or application configuration file."""
    normalized = path.lower().replace("\\", "/")
    pure_path = PurePath(normalized)
    
    # Check specific filename
    if pure_path.name in CONFIG_FILENAMES or pure_path.name.startswith(".env"):
        return True
    
    # Check extension
    if pure_path.suffix in CONFIG_EXTENSIONS:
        return True
        
    # Check if in config/ or configs/ directory
    if any(part in ("config", "configs", "conf") for part in pure_path.parts):
        return True

    return False


def is_test_file(path: str) -> bool:
    """Return True if the file is part of the test suite."""
    normalized = path.lower().replace("\\", "/")
    pure_path = PurePath(normalized)
    
    if any(part in ("tests", "test", "spec", "specs") for part in pure_path.parts):
        return True
    if pure_path.name.startswith("test_") or pure_path.name.endswith("_test.py"):
        return True
    return False


def is_doc_file(path: str) -> bool:
    """Return True if the file is documentation or metadata only."""
    normalized = path.lower().replace("\\", "/")
    pure_path = PurePath(normalized)
    
    if any(part in ("docs", "doc") for part in pure_path.parts):
        return True
    if pure_path.suffix in DOC_EXTENSIONS:
        return True
    if pure_path.name in ("license", "notice", "authors", "changelog"):
        return True
    return False


def is_application_file(path: str) -> bool:
    """Return True if the file is core application source code."""
    if is_test_file(path) or is_config_file(path) or is_doc_file(path):
        return False
    normalized = path.lower().replace("\\", "/")
    return normalized.endswith(".py") or normalized.startswith("app/") or normalized.startswith("src/")


# -----------------------------------------------------------------------------
# Pydantic Models for Changes & Telemetry
# -----------------------------------------------------------------------------


class FileChange(BaseModel):
    """Represents changes made to a single file in a deployment commit."""

    path: str = Field(..., description="Repository-relative file path")
    lines_added: int = Field(default=0, ge=0, description="Number of added lines")
    lines_deleted: int = Field(default=0, ge=0, description="Number of deleted lines")

    @property
    def total_lines_changed(self) -> int:
        """Total modified lines (additions + deletions)."""
        return self.lines_added + self.lines_deleted


class DeploymentChange(BaseModel):
    """Represents a proposed deployment change submitted for risk analysis."""

    deployment_id: str = Field(..., description="Unique deployment / release candidate ID")
    commit_sha: str = Field(..., description="Git commit hash")
    branch: str = Field(default="main", description="Target deployment branch")
    changed_files: list[FileChange] = Field(
        default_factory=list,
        description="Structured list of files modified in this change",
    )
    raw_diff: Optional[str] = Field(
        None,
        description="Optional raw unified git diff string; parsed automatically if changed_files is empty",
    )


class RuntimeTelemetryMetrics(BaseModel):
    """Runtime telemetry window metrics ingested from service observability."""

    mean_latency_ms: float = Field(..., ge=0.0, description="Mean latency in ms")
    p99_latency_ms: float = Field(..., ge=0.0, description="P99 tail latency in ms")
    error_rate: float = Field(..., ge=0.0, le=1.0, description="HTTP 5xx/4xx error ratio (0.0 to 1.0)")
    request_volume: int = Field(..., ge=0, description="Total requests processed in evaluation window")


class DeploymentRiskFeatures(BaseModel):
    """
    Unified feature vector representing change characteristics and runtime metrics.
    
    This structured object serves as the direct feature input to the DeployGuard
    risk prediction model in subsequent phases.
    """

    # Identifiers
    deployment_id: str
    commit_sha: str

    # Change volume & blast radius features
    files_changed: int = Field(..., ge=0, description="Number of files touched")
    lines_added: int = Field(..., ge=0, description="Total lines added")
    lines_deleted: int = Field(..., ge=0, description="Total lines deleted")
    total_lines_changed: int = Field(..., ge=0, description="Total code churn (added + deleted)")

    # Architectural classification features
    high_risk_files_count: int = Field(..., ge=0, description="Files touching critical business domains")
    config_files_changed: int = Field(..., ge=0, description="Infrastructure or app config files modified")
    application_files_changed: int = Field(..., ge=0, description="Core application logic files modified")
    test_files_changed: int = Field(..., ge=0, description="Test suite files modified")
    doc_files_changed: int = Field(..., ge=0, description="Documentation files modified")
    high_risk_area_touched: bool = Field(..., description="Boolean flag if any high-risk domain was modified")

    # Runtime telemetry features (optional if no baseline telemetry is provided)
    mean_latency_ms: Optional[float] = Field(None, description="Baseline / canary mean latency in ms")
    p99_latency_ms: Optional[float] = Field(None, description="Baseline / canary P99 tail latency in ms")
    error_rate: Optional[float] = Field(None, description="Baseline / canary error rate ratio")
    request_volume: Optional[int] = Field(None, description="Request throughput volume in window")


class ChangeAnalysisRequest(BaseModel):
    """Request payload for the POST /api/v1/analyze-change endpoint."""

    change: DeploymentChange
    telemetry: Optional[RuntimeTelemetryMetrics] = None


class ChangeAnalysisResponse(BaseModel):
    """Response payload containing the extracted risk feature vector."""

    deployment_id: str
    features: DeploymentRiskFeatures


# -----------------------------------------------------------------------------
# Unified Git Diff Parser
# -----------------------------------------------------------------------------


def parse_git_diff(diff_text: str) -> list[FileChange]:
    """
    Parse a standard unified Git diff string into a structured list of FileChange models.

    Supports:
    - Standard `diff --git a/... b/...` headers
    - Single or multi-file diffs
    - New files (`--- /dev/null`) and deleted files (`+++ /dev/null`)
    - Accurate count of added (+) and deleted (-) lines, excluding diff metadata.
    """
    if not diff_text or not diff_text.strip():
        return []

    file_changes: list[FileChange] = []
    current_path: Optional[str] = None
    lines_added = 0
    lines_deleted = 0

    lines = diff_text.splitlines()

    def _flush_current():
        nonlocal current_path, lines_added, lines_deleted
        if current_path is not None:
            file_changes.append(
                FileChange(
                    path=current_path,
                    lines_added=lines_added,
                    lines_deleted=lines_deleted,
                )
            )
            current_path = None
            lines_added = 0
            lines_deleted = 0

    for line in lines:
        # Detect new file diff boundary: diff --git a/path b/path
        if line.startswith("diff --git "):
            _flush_current()
            # Extract b/ path (target path)
            parts = line.split(" ")
            if len(parts) >= 4 and parts[3].startswith("b/"):
                current_path = parts[3][2:]
            elif len(parts) >= 3 and parts[2].startswith("a/"):
                current_path = parts[2][2:]
            continue

        # Detect +++ b/path (fallback / override for new files or renames)
        if line.startswith("+++ "):
            target = line[4:].strip()
            if target != "/dev/null":
                if target.startswith("b/"):
                    target = target[2:]
                current_path = target
            continue

        # Detect --- a/path (used if file was deleted, i.e. +++ /dev/null)
        if line.startswith("--- "):
            target = line[4:].strip()
            if target != "/dev/null" and current_path is None:
                if target.startswith("a/"):
                    target = target[2:]
                current_path = target
            continue

        # Ignore chunk headers and index lines
        if line.startswith("@@") or line.startswith("index ") or line.startswith("new file") or line.startswith("deleted file"):
            continue

        # Count line additions and deletions
        if line.startswith("+"):
            lines_added += 1
        elif line.startswith("-"):
            lines_deleted += 1

    _flush_current()
    return file_changes


# -----------------------------------------------------------------------------
# Feature Extraction Engine
# -----------------------------------------------------------------------------


def extract_features(
    change: DeploymentChange,
    telemetry: Optional[RuntimeTelemetryMetrics] = None,
) -> DeploymentRiskFeatures:
    """
    Extract deterministic risk features from a DeploymentChange and optional telemetry.

    If change.changed_files is empty and change.raw_diff is supplied,
    the raw diff is parsed automatically.
    """
    files = list(change.changed_files)

    # If changed_files was empty but raw_diff was supplied, parse the diff
    if not files and change.raw_diff:
        files = parse_git_diff(change.raw_diff)

    files_changed = len(files)
    lines_added = sum(f.lines_added for f in files)
    lines_deleted = sum(f.lines_deleted for f in files)
    total_lines_changed = lines_added + lines_deleted

    high_risk_files_count = sum(1 for f in files if is_high_risk_path(f.path))
    config_files_changed = sum(1 for f in files if is_config_file(f.path))
    application_files_changed = sum(1 for f in files if is_application_file(f.path))
    test_files_changed = sum(1 for f in files if is_test_file(f.path))
    doc_files_changed = sum(1 for f in files if is_doc_file(f.path))
    high_risk_area_touched = high_risk_files_count > 0

    return DeploymentRiskFeatures(
        deployment_id=change.deployment_id,
        commit_sha=change.commit_sha,
        files_changed=files_changed,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        total_lines_changed=total_lines_changed,
        high_risk_files_count=high_risk_files_count,
        config_files_changed=config_files_changed,
        application_files_changed=application_files_changed,
        test_files_changed=test_files_changed,
        doc_files_changed=doc_files_changed,
        high_risk_area_touched=high_risk_area_touched,
        mean_latency_ms=telemetry.mean_latency_ms if telemetry else None,
        p99_latency_ms=telemetry.p99_latency_ms if telemetry else None,
        error_rate=telemetry.error_rate if telemetry else None,
        request_volume=telemetry.request_volume if telemetry else None,
    )


# -----------------------------------------------------------------------------
# Simulated Sample Changes (for Benchmarking and Demos)
# -----------------------------------------------------------------------------

SAMPLE_CHANGES: dict[str, DeploymentChange] = {
    # Change A: Moderate feature update touching order logic and tests
    "change_a_order_feature": DeploymentChange(
        deployment_id="DEP-SIM-001",
        commit_sha="a1b2c3d4e5f6",
        branch="feature/order-discounts",
        changed_files=[
            FileChange(path="app/routes/orders.py", lines_added=38, lines_deleted=8),
            FileChange(path="tests/test_orders.py", lines_added=24, lines_deleted=2),
        ],
    ),
    # Change B: High-risk payment & database configuration change
    "change_b_payment_db_high_risk": DeploymentChange(
        deployment_id="DEP-SIM-002",
        commit_sha="e4f5a6b7c8d9",
        branch="feature/payment-gateway-v2",
        changed_files=[
            FileChange(path="app/payment.py", lines_added=165, lines_deleted=42),
            FileChange(path="config/database.yml", lines_added=14, lines_deleted=3),
        ],
    ),
    # Change C: Documentation only update (low risk)
    "change_c_docs_only": DeploymentChange(
        deployment_id="DEP-SIM-003",
        commit_sha="f7e8d9c0b1a2",
        branch="docs/update-api-spec",
        changed_files=[
            FileChange(path="README.md", lines_added=52, lines_deleted=10),
            FileChange(path="docs/architecture.md", lines_added=120, lines_deleted=0),
        ],
    ),
    # Change D: Infrastructure / Docker configuration update
    "change_d_infra_docker": DeploymentChange(
        deployment_id="DEP-SIM-004",
        commit_sha="9c8b7a6f5e4d",
        branch="chore/docker-hardening",
        changed_files=[
            FileChange(path="Dockerfile", lines_added=12, lines_deleted=4),
            FileChange(path=".dockerignore", lines_added=6, lines_deleted=1),
        ],
    ),
}
