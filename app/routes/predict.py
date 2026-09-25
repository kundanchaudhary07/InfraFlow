"""
Machine learning risk prediction API routes for DeployGuard AI (Phase 5).

Provides:
  POST /api/v1/predict-risk             — Predict deployment failure probability P(failure)
  GET  /api/v1/predict-risk/model-info  — Retrieve model version, evaluation metrics, and feature importance
"""

from typing import Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.change_analysis import DeploymentRiskFeatures
from app.ml.model import (
    DEFAULT_MODEL_PATH,
    DeployGuardPredictor,
    load_model_artifact,
)

router = APIRouter(prefix="/api/v1/predict-risk", tags=["Risk Prediction"])


class RiskPredictionResponse(BaseModel):
    """Response payload containing failure probability estimate."""

    deployment_id: str = Field(..., description="Deployment identifier")
    failure_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted probability of deployment failure P(failure) in [0.0, 1.0]",
    )
    model_version: str = Field(..., description="Model version identifier")


class ModelInfoResponse(BaseModel):
    """Response payload detailing model metadata, evaluation metrics, and feature importances."""

    model_version: str
    feature_columns: list[str]
    metrics: dict[str, Any]
    feature_importance: dict[str, float]


@router.post(
    "",
    response_model=RiskPredictionResponse,
    summary="Predict deployment failure risk",
    description=(
        "Accepts a Phase 4 DeploymentRiskFeatures vector and estimates "
        "the probability of deployment failure P(failure) between 0.0 and 1.0 "
        "using a trained XGBoost classifier."
    ),
)
def predict_deployment_risk(features: DeploymentRiskFeatures) -> RiskPredictionResponse:
    """Predict deployment failure probability from the extracted feature vector."""
    try:
        predictor = DeployGuardPredictor.get_instance()
        prediction = predictor.predict_failure_probability(features)
        return RiskPredictionResponse(**prediction)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model artifact not found. Please train and persist the XGBoost model "
                "before requesting inference."
            ),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing risk prediction: {str(exc)}",
        )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Retrieve model evaluation metrics and feature importance",
    description=(
        "Returns the model version, held-out test evaluation metrics "
        "(accuracy, precision, recall, F1, ROC-AUC, confusion matrix), "
        "and normalized feature importance rankings."
    ),
)
def get_model_info() -> ModelInfoResponse:
    """Return model artifact metadata and metrics."""
    try:
        artifact = load_model_artifact(DEFAULT_MODEL_PATH)
        return ModelInfoResponse(
            model_version=artifact.get("model_version", "unknown"),
            feature_columns=artifact.get("feature_columns", []),
            metrics=artifact.get("metrics", {}),
            feature_importance=artifact.get("feature_importance", {}),
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model artifact not found. Please train the model first.",
        )
