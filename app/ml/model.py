"""
Machine Learning Risk Prediction Model for DeployGuard AI (Phase 5).

Provides:
- Data preparation: strictly separating X (features) and y (target).
- Train/test splitting with stratification and fixed random_state.
- Model training using XGBoost Classifier (binary:logistic).
- Comprehensive evaluation metrics: Precision, Recall, F1, ROC-AUC, Accuracy, Confusion Matrix.
- Feature importance extraction for DevOps/SRE interpretability.
- Safe serialization & persistence using joblib.
- Inference interface accepting Phase 4 DeploymentRiskFeatures objects.

DATA LEAKAGE PREVENTION:
The deployment outcome column ('deployment_failed') is explicitly separated from
the training feature matrix X and never passed during feature transformation,
training, or inference.
"""

from pathlib import Path
from typing import Any, Optional
import warnings

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb

from app.change_analysis import DeploymentRiskFeatures
from app.ml.dataset import FEATURE_COLUMNS, TARGET_COLUMN

MODEL_VERSION: str = "xgboost-baseline-v1"
DEFAULT_MODEL_PATH: Path = Path("models") / "deployguard_xgboost.joblib"


# -----------------------------------------------------------------------------
# Data Preparation
# -----------------------------------------------------------------------------


def prepare_features_and_target(
    dataset: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix X and target array y from dataset records.

    CRITICAL SECURITY & DATA LEAKAGE RULE:
    The target column ('deployment_failed') is strictly excluded from X.
    Only explicit FEATURE_COLUMNS are used to construct X.
    """
    X_list: list[list[float]] = []
    y_list: list[int] = []

    for row in dataset:
        feature_row: list[float] = [float(row[col]) for col in FEATURE_COLUMNS]
        X_list.append(feature_row)
        y_list.append(int(row[TARGET_COLUMN]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


# -----------------------------------------------------------------------------
# Model Training & Evaluation
# -----------------------------------------------------------------------------


class TrainedModelResult:
    """Holds a trained XGBoost model along with test evaluation metrics and importance."""

    def __init__(
        self,
        model: xgb.XGBClassifier,
        metrics: dict[str, Any],
        feature_importance: dict[str, float],
        model_version: str = MODEL_VERSION,
    ) -> None:
        self.model = model
        self.metrics = metrics
        self.feature_importance = feature_importance
        self.model_version = model_version


def train_xgboost_model(
    dataset: list[dict[str, Any]],
    test_size: float = 0.20,
    random_state: int = 42,
) -> TrainedModelResult:
    """
    Train an XGBoost classifier on the synthetic deployment dataset.

    Args:
        dataset: List of synthetic deployment records.
        test_size: Ratio of data reserved for held-out evaluation (default: 0.20).
        random_state: Fixed seed for reproducibility.

    Returns:
        TrainedModelResult containing model, evaluation metrics, and feature importance.
    """
    X, y = prepare_features_and_target(dataset)

    # Stratified split guarantees balanced failure ratio across train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Baseline XGBoost configuration for tabular failure classification
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=random_state,
        objective="binary:logistic",
    )

    model.fit(X_train, y_train)

    # Predictions on held-out test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Compute evaluation metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "test_sample_size": len(y_test),
        "train_sample_size": len(y_train),
    }

    # Extract feature importance (normalized gain/weight)
    importances = model.feature_importances_
    feature_importance: dict[str, float] = {}
    for col, imp in zip(FEATURE_COLUMNS, importances):
        feature_importance[col] = round(float(imp), 4)

    # Sort feature importance descending
    sorted_importance = dict(
        sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)
    )

    return TrainedModelResult(
        model=model,
        metrics=metrics,
        feature_importance=sorted_importance,
        model_version=MODEL_VERSION,
    )


# -----------------------------------------------------------------------------
# Persistence: Save and Load
# -----------------------------------------------------------------------------


def save_model_artifact(
    result: TrainedModelResult,
    filepath: Path | str = DEFAULT_MODEL_PATH,
) -> Path:
    """Save the trained model artifact and its metadata to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": result.model,
        "feature_columns": FEATURE_COLUMNS,
        "model_version": result.model_version,
        "metrics": result.metrics,
        "feature_importance": result.feature_importance,
    }
    joblib.dump(artifact, path)
    return path


def load_model_artifact(
    filepath: Path | str = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    """Load the model artifact dictionary from disk."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at {path}. "
            f"Run model training first or verify artifact location."
        )
    return joblib.load(path)


# -----------------------------------------------------------------------------
# Inference Interface
# -----------------------------------------------------------------------------


class DeployGuardPredictor:
    """
    Singleton predictor that loads the persisted model artifact and runs inference.
    """

    _instance: Optional["DeployGuardPredictor"] = None
    _artifact: Optional[dict[str, Any]] = None

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._load_model()

    def _load_model(self) -> None:
        """Load artifact into memory."""
        self._artifact = load_model_artifact(self.model_path)
        self.model: xgb.XGBClassifier = self._artifact["model"]
        self.feature_columns: list[str] = self._artifact["feature_columns"]
        self.model_version: str = self._artifact.get("model_version", MODEL_VERSION)

    @classmethod
    def get_instance(
        cls, model_path: Path | str = DEFAULT_MODEL_PATH
    ) -> "DeployGuardPredictor":
        """Get or initialize singleton instance."""
        if cls._instance is None:
            cls._instance = cls(model_path=model_path)
        return cls._instance

    @classmethod
    def set_instance(cls, instance: Optional["DeployGuardPredictor"]) -> None:
        """Set or reset singleton instance (useful for testing)."""
        cls._instance = instance

    def predict_failure_probability(
        self, features: DeploymentRiskFeatures
    ) -> dict[str, Any]:
        """
        Run inference on a DeploymentRiskFeatures instance.

        Returns:
            Dictionary with deployment_id, failure_probability (0.0 - 1.0), and model_version.
        """
        # Sensible defaults for telemetry when not provided in static analysis
        mean_lat = features.mean_latency_ms if features.mean_latency_ms is not None else 25.0
        p99_lat = features.p99_latency_ms if features.p99_latency_ms is not None else 50.0
        err_rate = features.error_rate if features.error_rate is not None else 0.005
        req_vol = features.request_volume if features.request_volume is not None else 1000

        # Construct feature vector in exact order of FEATURE_COLUMNS
        row = [
            float(features.files_changed),
            float(features.lines_added),
            float(features.lines_deleted),
            float(features.total_lines_changed),
            float(features.high_risk_files_count),
            float(features.config_files_changed),
            float(features.application_files_changed),
            float(features.test_files_changed),
            float(features.doc_files_changed),
            1.0 if features.high_risk_area_touched else 0.0,
            float(mean_lat),
            float(p99_lat),
            float(err_rate),
            float(req_vol),
        ]

        X_input = np.array([row], dtype=np.float32)
        probs = self.model.predict_proba(X_input)
        failure_prob = float(probs[0, 1])

        # Clamp between 0.0 and 1.0
        failure_prob = max(0.0, min(1.0, failure_prob))

        return {
            "deployment_id": features.deployment_id,
            "failure_probability": round(failure_prob, 4),
            "model_version": self.model_version,
        }
