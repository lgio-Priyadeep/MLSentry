"""Statistical drift detection engine for numerical and categorical features.

Implements the locked statistical methods:
  - Numerical features: Population Stability Index (PSI) using reference
    histogram bin edges and counts. KS-test, JS-divergence, and Wasserstein
    distance are explicitly banned.
  - Categorical features: Chi-square goodness-of-fit test against baseline
    frequency maps.

Thresholds (from core/constants.py):
  - PSI:
      < 0.10: INFO (no significant drift)
      0.10 <= PSI < 0.25: WARNING (moderate drift)
      >= 0.25: CRITICAL (significant distribution shift)
  - Chi-square:
      p_value > 0.05: INFO (no significant drift)
      0.01 < p_value <= 0.05: WARNING (moderate drift)
      p_value <= 0.01: CRITICAL (significant distribution shift)

This is a pure domain logic module:
  - No database access or ORM imports
  - No HTTP or API dependencies
  - Vectorized array operations using numpy and scipy
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import chisquare

from mlsentry.core.constants import (
    CHI2_P_VALUE_CRITICAL,
    CHI2_P_VALUE_WARNING,
    PSI_THRESHOLD_CRITICAL,
    PSI_THRESHOLD_WARNING,
    AlertSeverity,
    DriftMethod,
    FeatureKind,
)


@dataclass(frozen=True)
class DriftResult:
    """Result of statistical drift computation for a single feature.

    Attributes:
        feature_name: Registered name of the evaluated feature.
        feature_type: NUMERICAL or CATEGORICAL.
        method: PSI for numerical, CHI_SQUARE for categorical.
        score: Computed test score (PSI value or Chi-square statistic).
        p_value: Significance p-value for Chi-square, None for PSI.
        psi: PSI value for numerical features, None for categorical.
        severity: INFO, WARNING, or CRITICAL.
        has_drift: True if severity is WARNING or CRITICAL.
        sample_count: Number of prediction samples in the evaluation window.
    """

    feature_name: str
    feature_type: FeatureKind
    method: DriftMethod
    score: float
    p_value: float | None
    psi: float | None
    severity: AlertSeverity
    has_drift: bool
    sample_count: int


def compute_histogram_counts(
    values: Sequence[float] | np.ndarray,
    bin_edges: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Compute count distribution of numerical values across predefined bin edges.

    Args:
        values: 1D sequence of numerical feature samples.
        bin_edges: Monotonically increasing bin boundary floats.

    Returns:
        np.ndarray: 1D array of sample counts per bin (length = len(bin_edges) - 1).
    """
    arr = np.asarray(values, dtype=np.float64)
    edges = np.asarray(bin_edges, dtype=np.float64)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("bin_edges must be a 1D sequence with at least 2 elements.")
    counts, _ = np.histogram(arr, bins=edges)
    return counts


def calculate_psi(
    reference_counts: Sequence[int | float] | np.ndarray,
    current_counts: Sequence[int | float] | np.ndarray,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between two binned distributions.

    Formula:
        PSI = sum((Actual_i - Expected_i) * ln(Actual_i / Expected_i))

    Epsilon smoothing is applied to prevent division by zero or log(0)
    when a bin contains zero counts in either distribution.

    Args:
        reference_counts: Expected/baseline sample counts per histogram bin.
        current_counts: Actual/current window sample counts per histogram bin.
        epsilon: Small positive constant for zero-bin smoothing.

    Returns:
        float: Non-negative Population Stability Index score rounded to 6 decimals.
    """
    ref = np.asarray(reference_counts, dtype=np.float64)
    cur = np.asarray(current_counts, dtype=np.float64)

    if ref.shape != cur.shape:
        raise ValueError(
            f"Reference shape {ref.shape} and current shape {cur.shape} must match."
        )
    if ref.size == 0:
        raise ValueError("Cannot calculate PSI on empty distributions.")

    ref_sum = np.sum(ref)
    cur_sum = np.sum(cur)

    if ref_sum <= 0 or cur_sum <= 0:
        raise ValueError("Count distributions must have strictly positive sums.")

    ref_prop = ref / ref_sum
    cur_prop = cur / cur_sum

    # Epsilon smoothing for zero bins
    ref_prop = np.where(ref_prop == 0.0, epsilon, ref_prop)
    cur_prop = np.where(cur_prop == 0.0, epsilon, cur_prop)

    # Re-normalize after smoothing
    ref_prop = ref_prop / np.sum(ref_prop)
    cur_prop = cur_prop / np.sum(cur_prop)

    psi_value = np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop))
    return round(float(max(0.0, psi_value)), 6)


def calculate_chi_square(
    reference_frequency_map: dict[str, float],
    current_counts: dict[str, int],
    epsilon: float = 1e-4,
) -> tuple[float, float]:
    """Calculate Chi-square goodness-of-fit test statistic and p-value.

    Compares current categorical counts against the baseline frequency map.

    Args:
        reference_frequency_map: Registered category proportions summing to 1.0 (±0.001).
        current_counts: Observed counts per category in current evaluation window.
        epsilon: Small positive float to prevent expected frequencies of zero.

    Returns:
        tuple[float, float]: (chi2_statistic rounded to 6 decimals, p_value rounded to 8 decimals).
    """
    if not reference_frequency_map:
        raise ValueError("Reference frequency map must not be empty.")

    total_observed = sum(current_counts.values())
    if total_observed <= 0:
        raise ValueError("Current window must have at least one sample.")

    # Union of all registered categories and observed categories in window
    categories = sorted(set(reference_frequency_map.keys()) | set(current_counts.keys()))

    f_obs: list[float] = []
    f_exp: list[float] = []

    for cat in categories:
        obs = float(current_counts.get(cat, 0))
        ref_p = float(reference_frequency_map.get(cat, 0.0))
        exp = total_observed * ref_p
        f_obs.append(obs)
        f_exp.append(max(exp, epsilon))

    f_obs_arr = np.array(f_obs, dtype=np.float64)
    f_exp_arr = np.array(f_exp, dtype=np.float64)

    # Rescale expected frequencies so sum(f_exp) == sum(f_obs)
    f_exp_arr = f_exp_arr * (np.sum(f_obs_arr) / np.sum(f_exp_arr))

    # Exact match shortcut
    if np.allclose(f_obs_arr, f_exp_arr):
        return 0.0, 1.0

    stat, p_val = chisquare(f_obs=f_obs_arr, f_exp=f_exp_arr)
    stat_clean = round(float(max(0.0, stat)), 6)
    p_val_clean = round(float(np.clip(p_val, 0.0, 1.0)), 8)
    return stat_clean, p_val_clean


def classify_psi_severity(psi_score: float) -> AlertSeverity:
    """Classify numerical PSI score into AlertSeverity level.

    Args:
        psi_score: Computed non-negative PSI score.

    Returns:
        AlertSeverity: INFO, WARNING, or CRITICAL.
    """
    if psi_score >= PSI_THRESHOLD_CRITICAL:
        return AlertSeverity.CRITICAL
    if psi_score >= PSI_THRESHOLD_WARNING:
        return AlertSeverity.WARNING
    return AlertSeverity.INFO


def classify_chi2_severity(p_value: float) -> AlertSeverity:
    """Classify categorical Chi-square p-value into AlertSeverity level.

    Args:
        p_value: Computed Chi-square test p-value in [0.0, 1.0].

    Returns:
        AlertSeverity: INFO, WARNING, or CRITICAL.
    """
    if p_value <= CHI2_P_VALUE_CRITICAL:
        return AlertSeverity.CRITICAL
    if p_value <= CHI2_P_VALUE_WARNING:
        return AlertSeverity.WARNING
    return AlertSeverity.INFO


def detect_numerical_drift(
    feature_name: str,
    current_values: Sequence[float] | np.ndarray,
    bin_edges: Sequence[float] | np.ndarray,
    reference_counts: Sequence[int | float] | np.ndarray,
) -> DriftResult:
    """Perform complete numerical drift evaluation for a single feature.

    Args:
        feature_name: Name of the numerical feature.
        current_values: Sequence of feature values in current window.
        bin_edges: Monotonically increasing histogram bin edges from reference_stats.
        reference_counts: Baseline sample counts per bin from reference_stats.

    Returns:
        DriftResult: Aggregated test metrics, severity, and drift status.
    """
    cur_counts = compute_histogram_counts(current_values, bin_edges)
    psi_score = calculate_psi(reference_counts, cur_counts)
    severity = classify_psi_severity(psi_score)
    sample_count = len(current_values)

    return DriftResult(
        feature_name=feature_name,
        feature_type=FeatureKind.NUMERICAL,
        method=DriftMethod.PSI,
        score=psi_score,
        p_value=None,
        psi=psi_score,
        severity=severity,
        has_drift=(severity in (AlertSeverity.WARNING, AlertSeverity.CRITICAL)),
        sample_count=sample_count,
    )


def detect_categorical_drift(
    feature_name: str,
    current_values: Sequence[str] | dict[str, int],
    frequency_map: dict[str, float],
) -> DriftResult:
    """Perform complete categorical drift evaluation for a single feature.

    Args:
        feature_name: Name of the categorical feature.
        current_values: Sequence of string category samples or pre-aggregated count dict.
        frequency_map: Baseline category frequency distribution from reference_stats.

    Returns:
        DriftResult: Aggregated test metrics, severity, and drift status.
    """
    if isinstance(current_values, dict):
        cur_counts = current_values
        sample_count = sum(cur_counts.values())
    else:
        cur_counts = dict(Counter(current_values))
        sample_count = len(current_values)

    stat, p_val = calculate_chi_square(frequency_map, cur_counts)
    severity = classify_chi2_severity(p_val)

    return DriftResult(
        feature_name=feature_name,
        feature_type=FeatureKind.CATEGORICAL,
        method=DriftMethod.CHI_SQUARE,
        score=stat,
        p_value=p_val,
        psi=None,
        severity=severity,
        has_drift=(severity in (AlertSeverity.WARNING, AlertSeverity.CRITICAL)),
        sample_count=sample_count,
    )
