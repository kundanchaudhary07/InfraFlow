"""
Synthetic deployment dataset generation and management module.

DeployGuard AI — Phase 5: Machine Learning Risk Prediction.

IMPORTANT NOTICE:
This dataset is a deterministic, controlled SYNTHETIC BENCHMARK designed to
demonstrate the pre-deployment risk estimation pipeline. It is NOT real-world
production deployment telemetry. Real enterprise deployment data will replace
this benchmark in future production deployments.

Responsibilities:
- Generate realistic, synthetic deployment records based on Phase 4 feature concepts.
- Provide clean separation between feature columns (X) and target label (y).
- Enforce strict reproducibility via deterministic random seed.
- Support dataset serialization to CSV/JSON for inspection and model training.
"""

import csv
import json
import math
from pathlib import Path
import random
from typing import Any, Optional


# -----------------------------------------------------------------------------
# Feature and Target Column Definitions
# -----------------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = [
    "files_changed",
    "lines_added",
    "lines_deleted",
    "total_lines_changed",
    "high_risk_files_count",
    "config_files_changed",
    "application_files_changed",
    "test_files_changed",
    "doc_files_changed",
    "high_risk_area_touched",
    "mean_latency_ms",
    "p99_latency_ms",
    "error_rate",
    "request_volume",
]

TARGET_COLUMN: str = "deployment_failed"

METADATA_COLUMNS: list[str] = [
    "deployment_id",
    "scenario_type",
]


# -----------------------------------------------------------------------------
# Synthetic Data Generator
# -----------------------------------------------------------------------------


def generate_synthetic_dataset(
    num_samples: int = 800,
    random_seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Generate a deterministic synthetic dataset of deployment change scenarios.

    Each row represents a proposed deployment with extracted Phase 4 features
    and an outcome label:
      0 = Stable deployment (no post-release incident)
      1 = Deployment failure (incident, rollback, or outage)

    The label is derived from realistic SRE risk factors (code churn, critical
    path exposure, configuration drift, latency degradation, error rates)
    combined via a logistic log-odds model with stochastic noise to reflect
    real-world imperfect observability.

    Args:
        num_samples: Number of deployment records to generate (default: 800).
        random_seed: Fixed seed for deterministic reproducibility.

    Returns:
        List of dictionaries containing metadata, features, and the target label.
    """
    rng = random.Random(random_seed)
    records: list[dict[str, Any]] = []

    for i in range(1, num_samples + 1):
        dep_id = f"DEP-SYNTH-{i:05d}"
        
        # Select deployment archetype
        # 35% Routine feature, 15% Docs/formatting, 25% High-risk domain, 15% Large churn, 10% Degraded telemetry
        scenario_roll = rng.random()
        
        if scenario_roll < 0.15:
            # Archetype 1: Documentation / Minor Metadata (low risk)
            scenario_type = "documentation_only"
            files_changed = rng.randint(1, 3)
            doc_files = files_changed
            test_files = 0
            config_files = 0
            app_files = 0
            high_risk_files = 0
            lines_added = rng.randint(5, 80)
            lines_deleted = rng.randint(0, 20)
            # Healthy baseline telemetry
            mean_latency = round(rng.uniform(12.0, 35.0), 2)
            p99_latency = round(mean_latency + rng.uniform(15.0, 45.0), 2)
            error_rate = round(rng.uniform(0.000, 0.004), 4)
            request_vol = rng.randint(500, 4000)

        elif scenario_roll < 0.50:
            # Archetype 2: Routine Application Feature / Bugfix (moderate-low risk)
            scenario_type = "routine_feature"
            files_changed = rng.randint(2, 6)
            test_files = rng.randint(1, max(1, files_changed // 2))
            config_files = 1 if rng.random() < 0.20 else 0
            doc_files = 1 if rng.random() < 0.30 else 0
            app_files = max(1, files_changed - test_files - config_files - doc_files)
            high_risk_files = 1 if rng.random() < 0.15 else 0
            lines_added = rng.randint(20, 150)
            lines_deleted = rng.randint(5, 50)
            # Normal telemetry
            mean_latency = round(rng.uniform(18.0, 48.0), 2)
            p99_latency = round(mean_latency + rng.uniform(25.0, 75.0), 2)
            error_rate = round(rng.uniform(0.001, 0.012), 4)
            request_vol = rng.randint(1000, 8000)

        elif scenario_roll < 0.75:
            # Archetype 3: High-Risk Core Logic (Payment, Auth, DB Schema)
            scenario_type = "high_risk_core"
            files_changed = rng.randint(3, 10)
            high_risk_files = rng.randint(1, min(4, files_changed))
            config_files = rng.randint(0, 2)
            test_files = rng.randint(0, 3)
            doc_files = 0
            app_files = max(1, files_changed - test_files - config_files - doc_files)
            lines_added = rng.randint(80, 450)
            lines_deleted = rng.randint(20, 160)
            # Slightly elevated telemetry or normal
            mean_latency = round(rng.uniform(25.0, 75.0), 2)
            p99_latency = round(mean_latency + rng.uniform(40.0, 160.0), 2)
            error_rate = round(rng.uniform(0.005, 0.035), 4)
            request_vol = rng.randint(2000, 12000)

        elif scenario_roll < 0.90:
            # Archetype 4: Large Monolithic Refactor / Infrastructure Drift (high risk)
            scenario_type = "monolithic_refactor"
            files_changed = rng.randint(12, 35)
            high_risk_files = rng.randint(2, 6)
            config_files = rng.randint(2, 5)
            test_files = rng.randint(1, 4)
            doc_files = rng.randint(0, 2)
            app_files = max(2, files_changed - test_files - config_files - doc_files)
            lines_added = rng.randint(400, 2200)
            lines_deleted = rng.randint(150, 900)
            # Stressed telemetry
            mean_latency = round(rng.uniform(35.0, 110.0), 2)
            p99_latency = round(mean_latency + rng.uniform(80.0, 280.0), 2)
            error_rate = round(rng.uniform(0.010, 0.055), 4)
            request_vol = rng.randint(3000, 15000)

        else:
            # Archetype 5: Deployment Under Degraded Runtime Telemetry (very high risk)
            scenario_type = "degraded_telemetry"
            files_changed = rng.randint(2, 8)
            high_risk_files = rng.randint(1, 3)
            config_files = rng.randint(0, 2)
            test_files = rng.randint(0, 2)
            doc_files = 0
            app_files = max(1, files_changed - test_files - config_files - doc_files)
            lines_added = rng.randint(50, 300)
            lines_deleted = rng.randint(10, 100)
            # Severely degraded canary/staging telemetry
            mean_latency = round(rng.uniform(80.0, 260.0), 2)
            p99_latency = round(mean_latency + rng.uniform(150.0, 600.0), 2)
            error_rate = round(rng.uniform(0.040, 0.160), 4)
            request_vol = rng.randint(4000, 20000)

        total_lines_changed = lines_added + lines_deleted
        high_risk_touched = 1 if high_risk_files > 0 else 0

        # ---------------------------------------------------------------------
        # Logistic Risk Probability Model (Latent Ground Truth with Noise)
        # ---------------------------------------------------------------------
        # SRE principles:
        # - Code churn (+), file count (+), high-risk domains (+) increase failure risk
        # - Config changes (+) are a primary source of outages
        # - Test changes (-) mitigate risk
        # - Doc-only changes (-) lower risk
        # - Elevated tail latency (+) and error rate (+) strongly indicate instability
        # - Stochastic noise mimics real-world unmeasured factors (network blips, human error)
        # ---------------------------------------------------------------------
        noise = rng.gauss(0.0, 0.45)
        
        logit = (
            -3.40                                  # Intercept: baseline stable (~3% failure rate for small safe changes)
            + 0.065 * min(files_changed, 30)       # Blast radius penalty
            + 0.0018 * min(total_lines_changed, 2000) # Churn penalty
            + 1.25 * high_risk_touched             # Critical domain exposure
            + 0.45 * high_risk_files               # Intensity in critical domain
            + 0.70 * config_files                  # Configuration drift penalty
            - 0.45 * min(test_files, 4)            # Test presence bonus
            - 0.60 * doc_files                     # Documentation dampener
            + 0.012 * max(0.0, p99_latency - 60.0) # Tail latency degradation penalty
            + 28.0 * error_rate                    # Direct error rate penalty
            + noise                                # Real-world observability noise
        )

        prob_failure = 1.0 / (1.0 + math.exp(-logit))

        # Binary label assignment based on simulated failure threshold (0.50)
        # with probabilistic jitter for boundary cases
        outcome = 1 if prob_failure >= 0.50 else 0

        records.append({
            "deployment_id": dep_id,
            "scenario_type": scenario_type,
            "files_changed": files_changed,
            "lines_added": lines_added,
            "lines_deleted": lines_deleted,
            "total_lines_changed": total_lines_changed,
            "high_risk_files_count": high_risk_files,
            "config_files_changed": config_files,
            "application_files_changed": app_files,
            "test_files_changed": test_files,
            "doc_files_changed": doc_files,
            "high_risk_area_touched": high_risk_touched,
            "mean_latency_ms": mean_latency,
            "p99_latency_ms": p99_latency,
            "error_rate": error_rate,
            "request_volume": request_vol,
            TARGET_COLUMN: outcome,
        })

    return records


# -----------------------------------------------------------------------------
# Persistence Helpers
# -----------------------------------------------------------------------------


def save_dataset_to_csv(records: list[dict[str, Any]], filepath: Path | str) -> Path:
    """Save records to a CSV file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if not records:
        return path

    fieldnames = list(records[0].keys())
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return path


def load_dataset_from_csv(filepath: Path | str) -> list[dict[str, Any]]:
    """Load dataset records from a CSV file with typed conversion."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at {path}")

    records: list[dict[str, Any]] = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, Any] = {
                "deployment_id": row["deployment_id"],
                "scenario_type": row["scenario_type"],
                "files_changed": int(row["files_changed"]),
                "lines_added": int(row["lines_added"]),
                "lines_deleted": int(row["lines_deleted"]),
                "total_lines_changed": int(row["total_lines_changed"]),
                "high_risk_files_count": int(row["high_risk_files_count"]),
                "config_files_changed": int(row["config_files_changed"]),
                "application_files_changed": int(row["application_files_changed"]),
                "test_files_changed": int(row["test_files_changed"]),
                "doc_files_changed": int(row["doc_files_changed"]),
                "high_risk_area_touched": int(row["high_risk_area_touched"]),
                "mean_latency_ms": float(row["mean_latency_ms"]),
                "p99_latency_ms": float(row["p99_latency_ms"]),
                "error_rate": float(row["error_rate"]),
                "request_volume": int(row["request_volume"]),
                TARGET_COLUMN: int(row[TARGET_COLUMN]),
            }
            records.append(parsed)

    return records
