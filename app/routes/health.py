"""
Health-check route.

A lightweight liveness probe used by load balancers, orchestrators, and
(in later phases) DeployGuard itself to verify the service is reachable.
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Response schema for the health endpoint."""

    status: str


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns HTTP 200 with `{'status': 'healthy'}` when the service is running.",
)
def health_check() -> HealthResponse:
    """Return a simple healthy status."""
    return HealthResponse(status="healthy")
