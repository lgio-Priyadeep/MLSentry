"""Synthetic dataset and statistical drift injection utilities for integration tests and simulation."""
from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone
from typing import Any


def generate_reference_statistics() -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate valid numerical and categorical baseline reference statistics."""
    num_stats = {
        "mean": 50.0,
        "std": 10.0,
        "min": 0.0,
        "max": 100.0,
        "p25": 43.0,
        "p50": 50.0,
        "p75": 57.0,
        "p95": 66.0,
        "histogram_bin_edges": [0.0, 25.0, 50.0, 75.0, 100.0],
        "histogram_counts": [100, 400, 400, 100],
    }
    cat_stats = {
        "frequency_map": {
            "tier_1": 0.50,
            "tier_2": 0.35,
            "tier_3": 0.15,
        }
    }
    return num_stats, cat_stats


def generate_prediction_batch(
    model_id: str,
    n_samples: int = 40,
    drift_type: str = "none",
) -> list[dict[str, Any]]:
    """Generate a batch of prediction payloads with optional statistical drift injection."""
    batch = []
    now = datetime.now(timezone.utc).isoformat()

    for _ in range(n_samples):
        if drift_type == "none":
            income = random.gauss(50.0, 10.0)
            tier = random.choices(["tier_1", "tier_2", "tier_3"], weights=[0.50, 0.35, 0.15])[0]
            extra_features = {}
        elif drift_type == "numerical_drift":
            # Significant mean shift to induce PSI >= 0.25 (CRITICAL)
            income = random.gauss(85.0, 5.0)
            tier = random.choices(["tier_1", "tier_2", "tier_3"], weights=[0.50, 0.35, 0.15])[0]
            extra_features = {}
        elif drift_type == "categorical_drift":
            # Invert category frequencies to induce Chi2 p <= 0.01 (CRITICAL)
            income = random.gauss(50.0, 10.0)
            tier = random.choices(["tier_1", "tier_2", "tier_3"], weights=[0.05, 0.15, 0.80])[0]
            extra_features = {}
        elif drift_type == "schema_extra_feature":
            income = random.gauss(50.0, 10.0)
            tier = "tier_1"
            extra_features = {"unexpected_score": 99.5}
        else:
            income = 50.0
            tier = "tier_1"
            extra_features = {}

        features = {
            "annual_income": round(max(0.0, min(100.0, income)), 2),
            "tier": tier,
            **extra_features,
        }

        payload = {
            "model_id": model_id,
            "features_json": features,
            "prediction_label": "positive" if income > 50 else "negative",
            "confidence": round(random.uniform(0.6, 0.99), 3),
            "timestamp": now,
        }
        batch.append(payload)

    return batch


def generate_ground_truth_batch(
    pred_ids: list[str],
    degrade_performance: bool = False,
) -> list[dict[str, Any]]:
    """Generate ground truth label payloads for matched predictions."""
    labels = []
    now = datetime.now(timezone.utc).isoformat()

    for pred_id in pred_ids:
        # If performance degradation is injected, introduce label noise
        if degrade_performance:
            label_val = "negative" if random.random() < 0.85 else "positive"
        else:
            label_val = "positive"

        labels.append({
            "pred_id": pred_id,
            "label": label_val,
            "received_at": now,
        })

    return labels
