"""Schema validation engine for prediction feature payloads.

Implements all 5 schema violation types defined in the project spec:

Hard violations (HTTP 422, no DB write):
  #2 — Wrong dtype: feature value does not match registered dtype.
  #5 — Oversized/nested JSON: features_json > 64KB or > 200 keys or
       contains nested objects.

Soft violations (HTTP 200, prediction logged with schema_valid flag):
  #1 — Missing required feature: required feature absent from payload.
       schema_valid=false, SCHEMA_VIOLATION alert.
  #3 — Out-of-bounds value: numerical value outside registered min/max.
       schema_valid=false, SCHEMA_VIOLATION alert.
  #4 — Extra feature: feature not in registered schema present in payload.
       schema_valid=true, SCHEMA_WARNING alert.

This module is a pure domain logic module:
  - No HTTP/API imports
  - No database access
  - No side effects (alerts are emitted by the caller)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mlsentry.core.constants import (
    FEATURES_JSON_MAX_BYTES,
    FEATURES_JSON_MAX_KEYS,
    FeatureDtype,
)


class ViolationType(str, Enum):
    """Classification of schema violation outcomes."""

    HARD_WRONG_DTYPE = "HARD_WRONG_DTYPE"
    HARD_TOO_LARGE = "HARD_TOO_LARGE"
    HARD_NESTED = "HARD_NESTED"
    SOFT_MISSING_REQUIRED = "SOFT_MISSING_REQUIRED"
    SOFT_OUT_OF_BOUNDS = "SOFT_OUT_OF_BOUNDS"
    SOFT_EXTRA_FEATURE = "SOFT_EXTRA_FEATURE"


@dataclass(frozen=True)
class SchemaViolation:
    """A single schema violation detected during validation."""

    violation_type: ViolationType
    feature_name: str
    message: str


@dataclass
class ValidationResult:
    """Aggregated result of validating a features_json payload.

    Attributes:
        is_hard_violation: True if any hard violation was detected.
            The caller must reject the request with HTTP 422.
        schema_valid: True if prediction should be included in
            monitoring aggregations. False if any soft violation
            of type MISSING_REQUIRED or OUT_OF_BOUNDS was detected.
            Extra features do NOT set schema_valid to False.
        hard_violations: List of hard violations (dtype, size, nesting).
        soft_violations: List of soft violations (missing, bounds, extra).
    """

    is_hard_violation: bool = False
    schema_valid: bool = True
    hard_violations: list[SchemaViolation] = field(default_factory=list)
    soft_violations: list[SchemaViolation] = field(default_factory=list)

    @property
    def has_violations(self) -> bool:
        """Return True if any violations (hard or soft) were detected."""
        return bool(self.hard_violations) or bool(self.soft_violations)

    @property
    def missing_features(self) -> list[str]:
        """Return names of missing required features."""
        return [
            v.feature_name
            for v in self.soft_violations
            if v.violation_type == ViolationType.SOFT_MISSING_REQUIRED
        ]

    @property
    def extra_features(self) -> list[str]:
        """Return names of extra features not in schema."""
        return [
            v.feature_name
            for v in self.soft_violations
            if v.violation_type == ViolationType.SOFT_EXTRA_FEATURE
        ]


@dataclass(frozen=True)
class FeatureSchema:
    """A single feature's validation rules from model_schemas.

    Attributes:
        feature_name: Registered feature identifier.
        dtype: Expected data type (float, int, bool, string, category).
        required: Whether absence triggers a SCHEMA_VIOLATION.
        min_value: Minimum allowed value (numerical dtypes only).
        max_value: Maximum allowed value (numerical dtypes only).
        allowed_values: Valid category values (category dtype only).
    """

    feature_name: str
    dtype: FeatureDtype
    required: bool = True
    min_value: float | None = None
    max_value: float | None = None
    allowed_values: list[str] | None = None


# Type mapping: FeatureDtype -> set of acceptable Python types.
_DTYPE_TYPE_MAP: dict[FeatureDtype, tuple[type, ...]] = {
    FeatureDtype.FLOAT: (float, int),
    FeatureDtype.INT: (int,),
    FeatureDtype.BOOL: (bool,),
    FeatureDtype.STRING: (str,),
    FeatureDtype.CATEGORY: (str,),
}


def validate_payload_envelope(
    features_json: dict[str, Any],
) -> ValidationResult:
    """Validate features_json envelope constraints (size, nesting).

    Checks Hard Violation #5:
      - Total keys > FEATURES_JSON_MAX_KEYS (200)
      - Serialized size > FEATURES_JSON_MAX_BYTES (64KB)
      - Any value is a dict or list (nested objects)

    Args:
        features_json: The raw features dictionary from the request.

    Returns:
        ValidationResult with hard violations if any envelope
        constraint is breached. Empty result if envelope is valid.
    """
    result = ValidationResult()

    # Check key count limit.
    if len(features_json) > FEATURES_JSON_MAX_KEYS:
        result.is_hard_violation = True
        result.hard_violations.append(
            SchemaViolation(
                violation_type=ViolationType.HARD_TOO_LARGE,
                feature_name="__envelope__",
                message=(
                    f"features_json contains {len(features_json)} keys, "
                    f"exceeding maximum of {FEATURES_JSON_MAX_KEYS}"
                ),
            )
        )
        return result

    # Check serialized byte size.
    serialized_size = len(json.dumps(features_json).encode("utf-8"))
    if serialized_size > FEATURES_JSON_MAX_BYTES:
        result.is_hard_violation = True
        result.hard_violations.append(
            SchemaViolation(
                violation_type=ViolationType.HARD_TOO_LARGE,
                feature_name="__envelope__",
                message=(
                    f"features_json serialized size is {serialized_size} bytes, "
                    f"exceeding maximum of {FEATURES_JSON_MAX_BYTES} bytes"
                ),
            )
        )
        return result

    # Check for nested objects (dicts or lists as values).
    for key, value in features_json.items():
        if isinstance(value, (dict, list)):
            result.is_hard_violation = True
            result.hard_violations.append(
                SchemaViolation(
                    violation_type=ViolationType.HARD_NESTED,
                    feature_name=key,
                    message=(
                        f"Feature '{key}' contains a nested "
                        f"{type(value).__name__}; only scalar values allowed"
                    ),
                )
            )
            return result

    return result


def validate_features(
    features_json: dict[str, Any],
    schema: list[FeatureSchema],
) -> ValidationResult:
    """Validate features_json against a registered model schema.

    Performs all 5 violation checks in order:
      1. Envelope validation (Hard #5: size, keys, nesting)
      2. Dtype validation (Hard #2: wrong type)
      3. Missing required features (Soft #1)
      4. Out-of-bounds values (Soft #3)
      5. Extra features (Soft #4)

    Hard violations short-circuit: if any hard violation is found,
    soft violations are not checked and the result is returned
    immediately.

    Args:
        features_json: The raw features dictionary from the request.
        schema: List of FeatureSchema rules from model_schemas.

    Returns:
        ValidationResult with all detected violations.
    """
    # Step 1: Envelope validation.
    envelope_result = validate_payload_envelope(features_json)
    if envelope_result.is_hard_violation:
        return envelope_result

    result = ValidationResult()
    schema_map = {fs.feature_name: fs for fs in schema}
    registered_names = set(schema_map.keys())
    provided_names = set(features_json.keys())

    # Step 2: Dtype validation (Hard #2).
    # Only check features that exist in both payload and schema.
    for name in provided_names & registered_names:
        fs = schema_map[name]
        value = features_json[name]

        # None/null values are allowed — they indicate missing data
        # but are not a dtype mismatch.
        if value is None:
            continue

        expected_types = _DTYPE_TYPE_MAP[fs.dtype]

        # Special case: bool is a subclass of int in Python.
        # If dtype is INT or FLOAT, reject actual booleans.
        if fs.dtype in (FeatureDtype.INT, FeatureDtype.FLOAT) and isinstance(value, bool):
            result.is_hard_violation = True
            result.hard_violations.append(
                SchemaViolation(
                    violation_type=ViolationType.HARD_WRONG_DTYPE,
                    feature_name=name,
                    message=(
                        f"Feature '{name}' expected type '{fs.dtype.value}', "
                        f"got 'bool'"
                    ),
                )
            )
            return result

        if not isinstance(value, expected_types):
            result.is_hard_violation = True
            result.hard_violations.append(
                SchemaViolation(
                    violation_type=ViolationType.HARD_WRONG_DTYPE,
                    feature_name=name,
                    message=(
                        f"Feature '{name}' expected type "
                        f"'{fs.dtype.value}', got "
                        f"'{type(value).__name__}'"
                    ),
                )
            )
            return result

    # If any hard violation found, stop here.
    if result.is_hard_violation:
        return result

    # Step 3: Missing required features (Soft #1).
    for name, fs in schema_map.items():
        if fs.required and name not in provided_names:
            result.schema_valid = False
            result.soft_violations.append(
                SchemaViolation(
                    violation_type=ViolationType.SOFT_MISSING_REQUIRED,
                    feature_name=name,
                    message=f"Required feature '{name}' is missing",
                )
            )

    # Step 4: Out-of-bounds values (Soft #3).
    for name in provided_names & registered_names:
        fs = schema_map[name]
        value = features_json[name]

        # Only check numerical dtypes with defined bounds.
        if fs.dtype not in (FeatureDtype.FLOAT, FeatureDtype.INT):
            continue
        if value is None:
            continue

        if fs.min_value is not None and value < fs.min_value:
            result.schema_valid = False
            result.soft_violations.append(
                SchemaViolation(
                    violation_type=ViolationType.SOFT_OUT_OF_BOUNDS,
                    feature_name=name,
                    message=(
                        f"Feature '{name}' value {value} is below "
                        f"minimum {fs.min_value}"
                    ),
                )
            )
        elif fs.max_value is not None and value > fs.max_value:
            result.schema_valid = False
            result.soft_violations.append(
                SchemaViolation(
                    violation_type=ViolationType.SOFT_OUT_OF_BOUNDS,
                    feature_name=name,
                    message=(
                        f"Feature '{name}' value {value} is above "
                        f"maximum {fs.max_value}"
                    ),
                )
            )

    # Step 5: Extra features (Soft #4).
    extra_names = provided_names - registered_names
    for name in sorted(extra_names):
        # Extra features do NOT set schema_valid to False.
        result.soft_violations.append(
            SchemaViolation(
                violation_type=ViolationType.SOFT_EXTRA_FEATURE,
                feature_name=name,
                message=(
                    f"Feature '{name}' is not in the registered schema"
                ),
            )
        )

    return result
