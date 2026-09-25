"""
Change analysis routes.

Provides:
  POST /api/v1/analyze-change          — Ingest deployment change & extract risk features
  GET  /api/v1/analyze-change/samples  — Retrieve pre-configured sample deployment changes
"""

from fastapi import APIRouter, HTTPException

from app.change_analysis import (
    SAMPLE_CHANGES,
    ChangeAnalysisRequest,
    ChangeAnalysisResponse,
    DeploymentChange,
    extract_features,
)

router = APIRouter(prefix="/api/v1/analyze-change", tags=["Change Analysis"])


@router.post(
    "",
    response_model=ChangeAnalysisResponse,
    summary="Analyze deployment change",
    description=(
        "Ingests a simulated code/configuration change (or unified git diff) "
        "and optional runtime telemetry metrics, returning a deterministic "
        "risk feature vector for pre-deployment evaluation."
    ),
)
def analyze_deployment_change(request: ChangeAnalysisRequest) -> ChangeAnalysisResponse:
    """Extract structured risk features from a proposed deployment change."""
    features = extract_features(request.change, request.telemetry)
    return ChangeAnalysisResponse(
        deployment_id=request.change.deployment_id,
        features=features,
    )


@router.get(
    "/samples",
    response_model=dict[str, DeploymentChange],
    summary="List sample deployment changes",
    description="Returns pre-configured sample changes (Change A, B, C, D) for simulation and benchmarking.",
)
def list_sample_changes() -> dict[str, DeploymentChange]:
    """Return dictionary of sample deployment changes."""
    return SAMPLE_CHANGES
