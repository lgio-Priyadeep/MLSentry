"""Unit tests for the Core Drift Detection Engine (PSI + Chi-square).

Validates:
  - PSI numerical drift calculations across identical, shifted, and critical distributions
  - Chi-square categorical drift calculations across identical, shifted, and critical maps
  - Threshold boundaries: PSI (0.100, 0.250), Chi-square p-value (0.050, 0.010)
  - Edge cases: zero-count bins, empty distributions, unseen categories, mismatched shapes
  - Full path coverage for detect_numerical_drift and detect_categorical_drift
"""
import numpy as np
import pytest

from mlsentry.core.constants import (
    AlertSeverity,
    DriftMethod,
    FeatureKind,
)
from mlsentry.core.drift.statistical import (
    DriftResult,
    calculate_chi_square,
    calculate_psi,
    classify_chi2_severity,
    classify_psi_severity,
    compute_histogram_counts,
    detect_categorical_drift,
    detect_numerical_drift,
)


class TestHistogramBinning:
    """Tests for compute_histogram_counts."""

    def test_basic_binning(self) -> None:
        values = [1.0, 2.5, 3.1, 8.9, 12.0]
        bin_edges = [0.0, 5.0, 10.0, 15.0]
        counts = compute_histogram_counts(values, bin_edges)
        assert list(counts) == [3, 1, 1]

    def test_empty_values_returns_zero_counts(self) -> None:
        values: list[float] = []
        bin_edges = [0.0, 5.0, 10.0]
        counts = compute_histogram_counts(values, bin_edges)
        assert list(counts) == [0, 0]

    def test_invalid_bin_edges_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least 2 elements"):
            compute_histogram_counts([1.0, 2.0], [5.0])


class TestPopulationStabilityIndex:
    """Tests for calculate_psi and PSI severity classification."""

    def test_identical_distributions_yields_zero_psi(self) -> None:
        ref_counts = [100, 200, 300, 200, 100]
        cur_counts = [100, 200, 300, 200, 100]
        psi = calculate_psi(ref_counts, cur_counts)
        assert psi == 0.0
        assert classify_psi_severity(psi) == AlertSeverity.INFO

    def test_scaled_identical_proportions_yields_zero_psi(self) -> None:
        ref_counts = [100, 200, 300, 200, 100]
        cur_counts = [50, 100, 150, 100, 50]  # Exact same proportions
        psi = calculate_psi(ref_counts, cur_counts)
        assert psi == 0.0
        assert classify_psi_severity(psi) == AlertSeverity.INFO

    def test_moderate_drift_warning(self) -> None:
        # Shift distribution moderately
        ref_counts = [100, 200, 300, 200, 100]
        cur_counts = [60, 140, 320, 260, 120]
        psi = calculate_psi(ref_counts, cur_counts)
        assert 0.01 <= psi < 0.25

    def test_severe_drift_critical(self) -> None:
        # Major distribution shift (skewed to high bins)
        ref_counts = [300, 300, 200, 100, 100]
        cur_counts = [20, 30, 100, 350, 500]
        psi = calculate_psi(ref_counts, cur_counts)
        assert psi >= 0.25
        assert classify_psi_severity(psi) == AlertSeverity.CRITICAL

    def test_psi_boundary_conditions(self) -> None:
        assert classify_psi_severity(0.099999) == AlertSeverity.INFO
        assert classify_psi_severity(0.100000) == AlertSeverity.WARNING
        assert classify_psi_severity(0.249999) == AlertSeverity.WARNING
        assert classify_psi_severity(0.250000) == AlertSeverity.CRITICAL
        assert classify_psi_severity(0.500000) == AlertSeverity.CRITICAL

    def test_zero_count_bins_handled_with_smoothing(self) -> None:
        ref_counts = [100, 200, 0, 200, 100]
        cur_counts = [100, 0, 300, 200, 100]
        psi = calculate_psi(ref_counts, cur_counts)
        assert isinstance(psi, float)
        assert psi > 0.0
        assert not np.isnan(psi)
        assert not np.isinf(psi)

    def test_mismatched_shapes_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            calculate_psi([10, 20, 30], [10, 20])

    def test_empty_distribution_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="empty distributions"):
            calculate_psi([], [])

    def test_all_zeros_distribution_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="positive sums"):
            calculate_psi([0, 0, 0], [10, 20, 30])
        with pytest.raises(ValueError, match="positive sums"):
            calculate_psi([10, 20, 30], [0, 0, 0])


class TestChiSquareCategoricalDrift:
    """Tests for calculate_chi_square and Chi-square severity classification."""

    def test_identical_proportions_yields_zero_statistic(self) -> None:
        freq_map = {"A": 0.5, "B": 0.3, "C": 0.2}
        counts = {"A": 50, "B": 30, "C": 20}
        stat, p_val = calculate_chi_square(freq_map, counts)
        assert stat == 0.0
        assert p_val == 1.0
        assert classify_chi2_severity(p_val) == AlertSeverity.INFO

    def test_moderate_shift_warning(self) -> None:
        freq_map = {"A": 0.5, "B": 0.3, "C": 0.2}
        counts = {"A": 40, "B": 42, "C": 18}  # Total 100
        stat, p_val = calculate_chi_square(freq_map, counts)
        assert stat > 0.0
        assert classify_chi2_severity(p_val) in (AlertSeverity.INFO, AlertSeverity.WARNING)

    def test_severe_shift_critical(self) -> None:
        freq_map = {"A": 0.5, "B": 0.3, "C": 0.2}
        counts = {"A": 10, "B": 80, "C": 10}  # Major reversal
        stat, p_val = calculate_chi_square(freq_map, counts)
        assert stat > 0.0
        assert p_val <= 0.01
        assert classify_chi2_severity(p_val) == AlertSeverity.CRITICAL

    def test_chi2_boundary_conditions(self) -> None:
        assert classify_chi2_severity(0.050001) == AlertSeverity.INFO
        assert classify_chi2_severity(0.050000) == AlertSeverity.WARNING
        assert classify_chi2_severity(0.010001) == AlertSeverity.WARNING
        assert classify_chi2_severity(0.010000) == AlertSeverity.CRITICAL
        assert classify_chi2_severity(0.000100) == AlertSeverity.CRITICAL

    def test_unseen_category_in_current_counts(self) -> None:
        freq_map = {"A": 0.6, "B": 0.4}
        counts = {"A": 30, "B": 20, "UNSEEN": 50}  # 50% new category
        stat, p_val = calculate_chi_square(freq_map, counts)
        assert stat > 0.0
        assert p_val <= 0.01
        assert classify_chi2_severity(p_val) == AlertSeverity.CRITICAL

    def test_empty_reference_map_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            calculate_chi_square({}, {"A": 10})

    def test_zero_observed_samples_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one sample"):
            calculate_chi_square({"A": 0.5, "B": 0.5}, {})
        with pytest.raises(ValueError, match="at least one sample"):
            calculate_chi_square({"A": 0.5, "B": 0.5}, {"A": 0, "B": 0})


class TestDriftDetectorsEndToEnd:
    """Tests for detect_numerical_drift and detect_categorical_drift wrappers."""

    def test_detect_numerical_drift_healthy(self) -> None:
        np.random.seed(42)
        ref_samples = np.random.normal(50.0, 10.0, 1000)
        bin_edges = np.linspace(10.0, 90.0, 11)  # 10 bins
        ref_counts = compute_histogram_counts(ref_samples, bin_edges)

        cur_samples = np.random.normal(50.0, 10.0, 200)
        result = detect_numerical_drift(
            feature_name="income",
            current_values=cur_samples,
            bin_edges=bin_edges,
            reference_counts=ref_counts,
        )

        assert isinstance(result, DriftResult)
        assert result.feature_name == "income"
        assert result.feature_type == FeatureKind.NUMERICAL
        assert result.method == DriftMethod.PSI
        assert result.severity == AlertSeverity.INFO
        assert result.has_drift is False
        assert result.psi is not None
        assert result.p_value is None
        assert result.sample_count == 200

    def test_detect_numerical_drift_drifted(self) -> None:
        bin_edges = [0.0, 25.0, 50.0, 75.0, 100.0]
        ref_counts = [250, 250, 250, 250]

        # Shifted values heavily into highest bin
        cur_samples = [85.0] * 80 + [90.0] * 20
        result = detect_numerical_drift(
            feature_name="age",
            current_values=cur_samples,
            bin_edges=bin_edges,
            reference_counts=ref_counts,
        )

        assert result.feature_name == "age"
        assert result.severity == AlertSeverity.CRITICAL
        assert result.has_drift is True
        assert result.psi is not None
        assert result.psi >= 0.25

    def test_detect_categorical_drift_from_list(self) -> None:
        freq_map = {"urban": 0.6, "suburban": 0.3, "rural": 0.1}
        cur_samples = ["urban"] * 60 + ["suburban"] * 30 + ["rural"] * 10
        result = detect_categorical_drift(
            feature_name="location",
            current_values=cur_samples,
            frequency_map=freq_map,
        )

        assert isinstance(result, DriftResult)
        assert result.feature_name == "location"
        assert result.feature_type == FeatureKind.CATEGORICAL
        assert result.method == DriftMethod.CHI_SQUARE
        assert result.score == 0.0
        assert result.p_value == 1.0
        assert result.severity == AlertSeverity.INFO
        assert result.has_drift is False
        assert result.sample_count == 100

    def test_detect_categorical_drift_from_dict_drifted(self) -> None:
        freq_map = {"urban": 0.6, "suburban": 0.3, "rural": 0.1}
        cur_counts = {"urban": 10, "suburban": 20, "rural": 70}
        result = detect_categorical_drift(
            feature_name="location",
            current_values=cur_counts,
            frequency_map=freq_map,
        )

        assert result.feature_name == "location"
        assert result.severity == AlertSeverity.CRITICAL
        assert result.has_drift is True
        assert result.p_value is not None
        assert result.p_value <= 0.01
        assert result.sample_count == 100
