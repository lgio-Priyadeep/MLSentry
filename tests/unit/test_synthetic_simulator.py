"""Unit tests for synthetic dataset and statistical drift generation utilities."""
from __future__ import annotations

import uuid
import pytest

from mlsentry.synthetic.drift_simulator import (
    generate_ground_truth_batch,
    generate_prediction_batch,
    generate_reference_statistics,
)


def test_generate_reference_statistics():
    """Assert reference statistics generator produces valid distributions."""
    num_stats, cat_stats = generate_reference_statistics()
    assert num_stats["mean"] == 50.0
    assert len(num_stats["histogram_bin_edges"]) == 5
    assert len(num_stats["histogram_counts"]) == 4

    freq_sum = sum(cat_stats["frequency_map"].values())
    assert abs(freq_sum - 1.0) <= 0.001


def test_generate_prediction_batch():
    """Assert prediction batches generate requested sample sizes and schema structure."""
    model_id = str(uuid.uuid4())
    batch = generate_prediction_batch(model_id, n_samples=35, drift_type="none")
    assert len(batch) == 35
    for item in batch:
        assert item["model_id"] == model_id
        assert "annual_income" in item["features_json"]
        assert "tier" in item["features_json"]
        assert item["prediction_label"] in ["positive", "negative"]
        assert 0.0 <= item["confidence"] <= 1.0
        assert "timestamp" in item


def test_generate_ground_truth_batch():
    """Assert ground truth batches match prediction identifiers."""
    pred_ids = [str(uuid.uuid4()) for _ in range(10)]
    gt_batch = generate_ground_truth_batch(pred_ids, degrade_performance=True)
    assert len(gt_batch) == 10
    for idx, item in enumerate(gt_batch):
        assert item["pred_id"] == pred_ids[idx]
        assert "label" in item
