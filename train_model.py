"""
CLI training script for DeployGuard AI XGBoost risk prediction model.

Usage:
    python train_model.py
"""

from pathlib import Path
import sys

from app.ml.dataset import (
    generate_synthetic_dataset,
    save_dataset_to_csv,
)
from app.ml.model import (
    DEFAULT_MODEL_PATH,
    save_model_artifact,
    train_xgboost_model,
)


def main() -> None:
    print("=" * 60)
    print(" DeployGuard AI — Model Training Pipeline (Phase 5)")
    print("=" * 60)

    data_dir = Path("data")
    data_path = data_dir / "synthetic_deployments.csv"

    print("\n1. Generating synthetic deployment benchmark dataset...")
    dataset = generate_synthetic_dataset(num_samples=800, random_seed=42)
    saved_csv = save_dataset_to_csv(dataset, data_path)
    print(f"   Saved {len(dataset)} deployment records to: {saved_csv}")

    print("\n2. Training XGBoost risk prediction model...")
    result = train_xgboost_model(dataset, test_size=0.20, random_state=42)
    saved_model = save_model_artifact(result, DEFAULT_MODEL_PATH)
    print(f"   Model saved to: {saved_model}")

    print("\n3. Held-out Test Set Evaluation Metrics:")
    print("-" * 50)
    metrics = result.metrics
    print(f"   Accuracy  : {metrics['accuracy'] * 100:.2f}%")
    print(f"   Precision : {metrics['precision']:.4f}")
    print(f"   Recall    : {metrics['recall']:.4f}")
    print(f"   F1-Score  : {metrics['f1_score']:.4f}")
    print(f"   ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f"   Test Size : {metrics['test_sample_size']} samples")

    cm = metrics["confusion_matrix"]
    print("\n   Confusion Matrix:")
    print(f"   [True Negatives  (Safe correctly predicted)]: {cm['true_negatives']}")
    print(f"   [False Positives (Safe flagged as risky)  ]: {cm['false_positives']}")
    print(f"   [False Negatives (Risky missed as safe)   ]: {cm['false_negatives']}")
    print(f"   [True Positives  (Risky caught correctly) ]: {cm['true_positives']}")

    print("\n4. Feature Importance Rankings (Normalized Gain):")
    print("-" * 50)
    for rank, (feature, imp) in enumerate(result.feature_importance.items(), start=1):
        bar = "=" * int(imp * 40)
        print(f"   {rank:2d}. {feature:<28} : {imp:.4f} {bar}")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
