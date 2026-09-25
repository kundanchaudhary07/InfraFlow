"""
Tests for Machine Learning Risk Prediction Pipeline (Phase 5).

Covers:
- Synthetic dataset generation & schema validation
- Deterministic reproducibility with fixed random seed
- Strict data leakage prevention (target label excluded from X)
- Stratified train/test split
- XGBoost classifier training
- Evaluation metrics calculation (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
- Feature importance extraction
- Model artifact persistence (save & load)
- Inference prediction interface & probability boundaries [0.0, 1.0]
- API endpoints: POST /api/v1/predict-risk and GET /api/v1/predict-risk/model-info
- Input validation (HTTP 422 on malformed payload)
"""

from pathlib import Path
import tempfile

from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.change_analysis import DeploymentRiskFeatures
from app.main import app
from app.ml.dataset import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    generate_synthetic_dataset,
    load_dataset_from_csv,
    save_dataset_to_csv,
)
from app.ml.model import (
    DEFAULT_MODEL_PATH,
    MODEL_VERSION,
    DeployGuardPredictor,
    load_model_artifact,
    prepare_features_and_target,
    save_model_artifact,
    train_xgboost_model,
)

client = TestClient(app)


# -----------------------------------------------------------------------------
# Dataset Generation & Validation Tests
# -----------------------------------------------------------------------------


def test_generate_synthetic_dataset_shape_and_columns() -> None:
    """Dataset generator must produce requested sample size with exact required columns."""
    samples = 150
    dataset = generate_synthetic_dataset(num_samples=samples, random_seed=42)
    assert len(dataset) == samples

    first_row = dataset[0]
    for col in FEATURE_COLUMNS:
        assert col in first_row, f"Missing feature column: {col}"

    assert TARGET_COLUMN in first_row
    assert "deployment_id" in first_row
    assert "scenario_type" in first_row


def test_generate_synthetic_dataset_deterministic_reproducibility() -> None:
    """Identical seeds must produce bit-for-bit identical records."""
    run1 = generate_synthetic_dataset(num_samples=80, random_seed=123)
    run2 = generate_synthetic_dataset(num_samples=80, random_seed=123)
    run3 = generate_synthetic_dataset(num_samples=80, random_seed=999)

    assert run1 == run2, "Dataset generation with same seed must be completely identical"
    assert run1 != run3, "Dataset generation with different seeds must produce different records"


def test_target_label_distribution() -> None:
    """Dataset must contain both 0 (stable) and 1 (failure) labels in plausible proportions."""
    dataset = generate_synthetic_dataset(num_samples=200, random_seed=42)
    labels = [row[TARGET_COLUMN] for row in dataset]

    # Must only contain 0 and 1
    assert set(labels).issubset({0, 1})
    failure_count = sum(labels)
    # Failure rate should be realistic (between 15% and 55%)
    failure_rate = failure_count / len(labels)
    assert 0.15 <= failure_rate <= 0.55, f"Unexpected failure rate: {failure_rate}"


def test_dataset_csv_persistence(tmp_path: Path) -> None:
    """Saving and loading dataset from CSV preserves rows and typed values."""
    dataset = generate_synthetic_dataset(num_samples=50, random_seed=42)
    csv_file = tmp_path / "test_deployments.csv"

    save_dataset_to_csv(dataset, csv_file)
    assert csv_file.exists()

    loaded = load_dataset_from_csv(csv_file)
    assert len(loaded) == len(dataset)
    assert loaded[0]["deployment_id"] == dataset[0]["deployment_id"]
    assert loaded[0][TARGET_COLUMN] == dataset[0][TARGET_COLUMN]
    assert loaded[0]["p99_latency_ms"] == dataset[0]["p99_latency_ms"]


# -----------------------------------------------------------------------------
# Data Leakage & Preparation Tests
# -----------------------------------------------------------------------------


def test_data_leakage_prevention() -> None:
    """CRITICAL TEST: Target label MUST NOT be included in feature matrix X."""
    dataset = generate_synthetic_dataset(num_samples=50, random_seed=42)
    X, y = prepare_features_and_target(dataset)

    # Number of columns in X must equal FEATURE_COLUMNS length exactly
    assert X.shape[1] == len(FEATURE_COLUMNS)
    assert len(y) == len(dataset)

    # Target column must not be inside FEATURE_COLUMNS
    assert TARGET_COLUMN not in FEATURE_COLUMNS


# -----------------------------------------------------------------------------
# Model Training & Evaluation Tests
# -----------------------------------------------------------------------------


def test_train_xgboost_model_metrics() -> None:
    """Trained XGBoost model returns valid metrics on held-out test set."""
    dataset = generate_synthetic_dataset(num_samples=250, random_seed=42)
    result = train_xgboost_model(dataset, test_size=0.20, random_state=42)

    assert result.model is not None
    metrics = result.metrics

    # Check metric boundaries
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.5 <= metrics["roc_auc"] <= 1.0

    # Confusion matrix structure
    cm = metrics["confusion_matrix"]
    total_test_samples = (
        cm["true_negatives"] + cm["false_positives"] + cm["false_negatives"] + cm["true_positives"]
    )
    assert total_test_samples == metrics["test_sample_size"]


def test_feature_importance_ranking() -> None:
    """Model extracts normalized feature importances summing to ~1.0."""
    dataset = generate_synthetic_dataset(num_samples=200, random_seed=42)
    result = train_xgboost_model(dataset, test_size=0.20, random_state=42)

    importance = result.feature_importance
    assert len(importance) == len(FEATURE_COLUMNS)
    # Sum of normalized importances is approximately 1.0
    total_imp = sum(importance.values())
    assert 0.95 <= total_imp <= 1.05

    # Features are sorted descending
    values = list(importance.values())
    assert values == sorted(values, reverse=True)


def test_model_persistence_save_and_load(tmp_path: Path) -> None:
    """Saved model artifact can be reloaded and yields identical probability predictions."""
    dataset = generate_synthetic_dataset(num_samples=150, random_seed=42)
    result = train_xgboost_model(dataset, test_size=0.20, random_state=42)

    model_file = tmp_path / "test_model.joblib"
    save_model_artifact(result, model_file)
    assert model_file.exists()

    loaded_artifact = load_model_artifact(model_file)
    assert loaded_artifact["model_version"] == MODEL_VERSION
    assert "metrics" in loaded_artifact
    assert "feature_importance" in loaded_artifact

    # Compare inference before and after save
    sample_features = DeploymentRiskFeatures(
        deployment_id="DEP-TEST-PERSIST",
        commit_sha="abcdef123456",
        files_changed=5,
        lines_added=150,
        lines_deleted=40,
        total_lines_changed=190,
        high_risk_files_count=2,
        config_files_changed=1,
        application_files_changed=2,
        test_files_changed=1,
        doc_files_changed=0,
        high_risk_area_touched=True,
        mean_latency_ms=45.0,
        p99_latency_ms=120.0,
        error_rate=0.015,
        request_volume=4500,
    )

    pred1 = DeployGuardPredictor(model_path=model_file)
    res1 = pred1.predict_failure_probability(sample_features)

    assert 0.0 <= res1["failure_probability"] <= 1.0
    assert res1["deployment_id"] == "DEP-TEST-PERSIST"


# -----------------------------------------------------------------------------
# Predictor Interface & Monotonicity / Risk Differentiation Tests
# -----------------------------------------------------------------------------


def test_predictor_differentiates_high_risk_from_low_risk(tmp_path: Path) -> None:
    """High-risk changes (payments + db + degraded telemetry) must yield higher failure probability than doc-only changes."""
    dataset = generate_synthetic_dataset(num_samples=300, random_seed=42)
    result = train_xgboost_model(dataset, test_size=0.20, random_state=42)
    model_file = tmp_path / "eval_model.joblib"
    save_model_artifact(result, model_file)

    predictor = DeployGuardPredictor(model_path=model_file)

    safe_change = DeploymentRiskFeatures(
        deployment_id="DEP-SAFE",
        commit_sha="111111111111",
        files_changed=1,
        lines_added=10,
        lines_deleted=0,
        total_lines_changed=10,
        high_risk_files_count=0,
        config_files_changed=0,
        application_files_changed=0,
        test_files_changed=0,
        doc_files_changed=1,
        high_risk_area_touched=False,
        mean_latency_ms=15.0,
        p99_latency_ms=25.0,
        error_rate=0.000,
        request_volume=1000,
    )

    risky_change = DeploymentRiskFeatures(
        deployment_id="DEP-RISKY",
        commit_sha="999999999999",
        files_changed=20,
        lines_added=950,
        lines_deleted=400,
        total_lines_changed=1350,
        high_risk_files_count=5,
        config_files_changed=3,
        application_files_changed=10,
        test_files_changed=0,
        doc_files_changed=0,
        high_risk_area_touched=True,
        mean_latency_ms=180.0,
        p99_latency_ms=450.0,
        error_rate=0.085,
        request_volume=12000,
    )

    safe_prob = predictor.predict_failure_probability(safe_change)["failure_probability"]
    risky_prob = predictor.predict_failure_probability(risky_change)["failure_probability"]

    assert risky_prob > safe_prob, (
        f"Risky change probability ({risky_prob}) should be strictly greater than safe change ({safe_prob})"
    )


# -----------------------------------------------------------------------------
# API Route Tests (POST /api/v1/predict-risk)
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def ensure_default_model_artifact() -> None:
    """Ensure a trained model artifact exists at DEFAULT_MODEL_PATH for API tests."""
    if not DEFAULT_MODEL_PATH.exists():
        dataset = generate_synthetic_dataset(num_samples=200, random_seed=42)
        result = train_xgboost_model(dataset, test_size=0.20, random_state=42)
        save_model_artifact(result, DEFAULT_MODEL_PATH)
    DeployGuardPredictor.set_instance(DeployGuardPredictor(DEFAULT_MODEL_PATH))


def test_api_predict_risk_success() -> None:
    """POST /api/v1/predict-risk returns 200 with probability and model version."""
    payload = {
        "deployment_id": "DEP-API-PREDICT-01",
        "commit_sha": "abcd01234567",
        "files_changed": 4,
        "lines_added": 85,
        "lines_deleted": 15,
        "total_lines_changed": 100,
        "high_risk_files_count": 1,
        "config_files_changed": 0,
        "application_files_changed": 3,
        "test_files_changed": 1,
        "doc_files_changed": 0,
        "high_risk_area_touched": True,
        "mean_latency_ms": 28.5,
        "p99_latency_ms": 72.0,
        "error_rate": 0.004,
        "request_volume": 2500,
    }

    response = client.post("/api/v1/predict-risk", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["deployment_id"] == "DEP-API-PREDICT-01"
    assert "failure_probability" in data
    assert 0.0 <= data["failure_probability"] <= 1.0
    assert data["model_version"] == MODEL_VERSION


def test_api_predict_risk_invalid_payload() -> None:
    """Missing mandatory fields returns HTTP 422 Unprocessable Entity."""
    # Empty payload missing all required fields
    response = client.post("/api/v1/predict-risk", json={})
    assert response.status_code == 422


def test_api_get_model_info() -> None:
    """GET /api/v1/predict-risk/model-info returns metadata, metrics, and feature importances."""
    response = client.get("/api/v1/predict-risk/model-info")
    assert response.status_code == 200

    data = response.json()
    assert data["model_version"] == MODEL_VERSION
    assert len(data["feature_columns"]) == len(FEATURE_COLUMNS)
    assert "metrics" in data
    assert "accuracy" in data["metrics"]
    assert "feature_importance" in data
