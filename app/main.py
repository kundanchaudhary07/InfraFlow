"""
Application entry point.

Creates and configures the FastAPI application instance, registers routers,
and exposes the ASGI ``app`` object consumed by Uvicorn at runtime.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.config import settings
from app.ml.model import DEFAULT_MODEL_PATH
from app.routes.change_analysis import router as change_analysis_router
from app.routes.health import router as health_router
from app.routes.orders import router as orders_router
from app.routes.predict import router as predict_router
from app.telemetry import RequestTelemetryMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure baseline XGBoost risk model artifact exists on startup."""
    if not DEFAULT_MODEL_PATH.exists():
        from app.ml.dataset import generate_synthetic_dataset
        from app.ml.model import save_model_artifact, train_xgboost_model

        dataset = generate_synthetic_dataset(num_samples=800, random_seed=42)
        result = train_xgboost_model(dataset, test_size=0.20, random_state=42)
        save_model_artifact(result, DEFAULT_MODEL_PATH)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    description=(
        "Phase 1 demo workload for **DeployGuard AI**. "
        "This Order Service is the target application that DeployGuard will "
        "eventually analyse and protect during pre-deployment risk assessment."
    ),
    contact={
        "name": "DeployGuard AI Project",
    },
    license_info={
        "name": "MIT",
    },
)

# ---------------------------------------------------------------------------
# Middleware registration
# ---------------------------------------------------------------------------

app.add_middleware(RequestTelemetryMiddleware)

# ---------------------------------------------------------------------------
# Router registration
# ---------------------------------------------------------------------------

app.include_router(health_router)
app.include_router(orders_router)
app.include_router(change_analysis_router)
app.include_router(predict_router)
