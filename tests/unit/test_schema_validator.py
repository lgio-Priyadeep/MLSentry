"""Unit tests for mlsentry.core.schema.validator.

Covers all 5 violation types plus happy path:
  - Hard #2: Wrong dtype
  - Hard #5: Too large (keys), too large (bytes), nested objects
  - Soft #1: Missing required feature
  - Soft #3: Out-of-bounds value
  - Soft #4: Extra feature
  - Happy path: valid payload, no violations
"""
import json

import pytest

from mlsentry.core.constants import (
    FEATURES_JSON_MAX_BYTES,
    FEATURES_JSON_MAX_KEYS,
    FeatureDtype,
)
from mlsentry.core.schema.validator import (
    FeatureSchema,
    ValidationResult,
    ViolationType,
    validate_features,
    validate_payload_envelope,
)


# ─── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture()
def sample_schema() -> list[FeatureSchema]:
    """Return a typical model schema with mixed feature types."""
    return [
        FeatureSchema(
            feature_name="age",
            dtype=FeatureDtype.INT,
            required=True,
            min_value=0,
            max_value=150,
        ),
        FeatureSchema(
            feature_name="income",
            dtype=FeatureDtype.FLOAT,
            required=True,
            min_value=0.0,
            max_value=10_000_000.0,
        ),
        FeatureSchema(
            feature_name="is_active",
            dtype=FeatureDtype.BOOL,
            required=True,
        ),
        FeatureSchema(
            feature_name="name",
            dtype=FeatureDtype.STRING,
            required=False,
        ),
        FeatureSchema(
            feature_name="gender",
            dtype=FeatureDtype.CATEGORY,
            required=False,
            allowed_values=["M", "F", "Other"],
        ),
    ]


@pytest.fixture()
def valid_features() -> dict:
    """Return a valid features_json payload matching sample_schema."""
    return {
        "age": 32,
        "income": 52000.0,
        "is_active": True,
        "name": "test_user",
        "gender": "M",
    }


# ─── Happy Path ───────────────────────────────────────────────────


class TestHappyPath:
    """Valid payloads produce no violations."""

    def test_valid_payload_no_violations(
        self, sample_schema: list[FeatureSchema], valid_features: dict,
    ) -> None:
        result = validate_features(valid_features, sample_schema)
        assert not result.is_hard_violation
        assert result.schema_valid is True
        assert not result.has_violations
        assert result.hard_violations == []
        assert result.soft_violations == []

    def test_valid_payload_with_optional_missing(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {"age": 25, "income": 30000.0, "is_active": False}
        result = validate_features(features, sample_schema)
        assert not result.is_hard_violation
        assert result.schema_valid is True
        assert result.missing_features == []

    def test_float_accepts_int_value(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {
            "age": 25,
            "income": 50000,
            "is_active": True,
        }
        result = validate_features(features, sample_schema)
        assert not result.is_hard_violation
        assert result.schema_valid is True

    def test_null_value_passes_dtype_check(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {
            "age": 25,
            "income": None,
            "is_active": True,
        }
        result = validate_features(features, sample_schema)
        assert not result.is_hard_violation


# ─── Hard Violation #2: Wrong Dtype ───────────────────────────────


class TestHardWrongDtype:
    """Wrong dtype causes hard violation — HTTP 422, no DB write."""

    def test_string_for_int_feature(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {
            "age": "thirty-two",
            "income": 52000.0,
            "is_active": True,
        }
        result = validate_features(features, sample_schema)
        assert result.is_hard_violation is True
        assert len(result.hard_violations) == 1
        assert result.hard_violations[0].violation_type == (
            ViolationType.HARD_WRONG_DTYPE
        )
        assert result.hard_violations[0].feature_name == "age"

    def test_string_for_float_feature(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {
            "age": 25,
            "income": "fifty-thousand",
            "is_active": True,
        }
        result = validate_features(features, sample_schema)
        assert result.is_hard_violation is True
        assert result.hard_violations[0].feature_name == "income"

    def test_int_for_bool_feature(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {
            "age": 25,
            "income": 50000.0,
            "is_active": 1,
        }
        result = validate_features(features, sample_schema)
        assert result.is_hard_violation is True
        assert result.hard_violations[0].feature_name == "is_active"

    def test_bool_for_int_feature_rejected(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {
            "age": True,
            "income": 50000.0,
            "is_active": True,
        }
        result = validate_features(features, sample_schema)
        assert result.is_hard_violation is True
        assert result.hard_violations[0].feature_name == "age"
        assert result.hard_violations[0].violation_type == (
            ViolationType.HARD_WRONG_DTYPE
        )

    def test_bool_for_float_feature_rejected(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {
            "age": 25,
            "income": True,
            "is_active": True,
        }
        result = validate_features(features, sample_schema)
        assert result.is_hard_violation is True
        assert result.hard_violations[0].feature_name == "income"
        assert result.hard_violations[0].violation_type == (
            ViolationType.HARD_WRONG_DTYPE
        )

    def test_int_for_string_feature(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {
            "age": 25,
            "income": 50000.0,
            "is_active": True,
            "name": 12345,
        }
        result = validate_features(features, sample_schema)
        assert result.is_hard_violation is True
        assert result.hard_violations[0].feature_name == "name"


# ─── Hard Violation #5: Too Large / Nested ────────────────────────


class TestHardEnvelopeViolations:
    """Envelope violations (size, keys, nesting) cause hard rejection."""

    def test_too_many_keys(self) -> None:
        features = {f"feature_{i}": 1.0 for i in range(201)}
        result = validate_payload_envelope(features)
        assert result.is_hard_violation is True
        assert result.hard_violations[0].violation_type == (
            ViolationType.HARD_TOO_LARGE
        )

    def test_exactly_max_keys_passes(self) -> None:
        features = {f"feature_{i}": 1.0 for i in range(200)}
        result = validate_payload_envelope(features)
        assert not result.is_hard_violation

    def test_nested_dict_value(self) -> None:
        features = {"age": 25, "nested": {"a": 1}}
        result = validate_payload_envelope(features)
        assert result.is_hard_violation is True
        assert result.hard_violations[0].violation_type == (
            ViolationType.HARD_NESTED
        )
        assert result.hard_violations[0].feature_name == "nested"

    def test_nested_list_value(self) -> None:
        features = {"age": 25, "tags": [1, 2, 3]}
        result = validate_payload_envelope(features)
        assert result.is_hard_violation is True
        assert result.hard_violations[0].violation_type == (
            ViolationType.HARD_NESTED
        )

    def test_envelope_checked_before_schema(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {f"f_{i}": {"nested": True} for i in range(201)}
        result = validate_features(features, sample_schema)
        assert result.is_hard_violation is True
        assert result.hard_violations[0].violation_type == (
            ViolationType.HARD_TOO_LARGE
        )

    def test_exactly_max_bytes_passes(self) -> None:
        # Build a payload just under 65536 bytes.
        # json.dumps({"k": "v"}) overhead is ~7 bytes per key entry.
        # We fill to exactly FEATURES_JSON_MAX_BYTES.
        base = {"a": "x"}
        base_size = len(json.dumps(base).encode("utf-8"))
        # Pad a single value to reach exactly 65536 bytes.
        padding_needed = FEATURES_JSON_MAX_BYTES - base_size + len("x")
        padded = {"a": "x" * padding_needed}
        # Verify we hit exactly 65536
        actual_size = len(json.dumps(padded).encode("utf-8"))
        # Adjust if off by a few bytes (JSON overhead)
        diff = FEATURES_JSON_MAX_BYTES - actual_size
        padded = {"a": "x" * (padding_needed + diff)}
        assert len(json.dumps(padded).encode("utf-8")) == FEATURES_JSON_MAX_BYTES
        result = validate_payload_envelope(padded)
        assert not result.is_hard_violation

    def test_over_max_bytes_rejected(self) -> None:
        # Build a payload of 65537 bytes (one over limit).
        base = {"a": "x"}
        base_size = len(json.dumps(base).encode("utf-8"))
        target = FEATURES_JSON_MAX_BYTES + 1
        padding_needed = target - base_size + len("x")
        padded = {"a": "x" * padding_needed}
        actual_size = len(json.dumps(padded).encode("utf-8"))
        diff = target - actual_size
        padded = {"a": "x" * (padding_needed + diff)}
        assert len(json.dumps(padded).encode("utf-8")) == FEATURES_JSON_MAX_BYTES + 1
        result = validate_payload_envelope(padded)
        assert result.is_hard_violation is True
        assert result.hard_violations[0].violation_type == (
            ViolationType.HARD_TOO_LARGE
        )


# ─── Soft Violation #1: Missing Required Feature ─────────────────


class TestSoftMissingRequired:
    """Missing required features set schema_valid=false."""

    def test_single_missing_required(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {"income": 50000.0, "is_active": True}
        result = validate_features(features, sample_schema)
        assert not result.is_hard_violation
        assert result.schema_valid is False
        assert "age" in result.missing_features

    def test_multiple_missing_required(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {"is_active": True}
        result = validate_features(features, sample_schema)
        assert result.schema_valid is False
        missing = result.missing_features
        assert "age" in missing
        assert "income" in missing
        assert len(missing) == 2

    def test_optional_missing_no_violation(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {"age": 25, "income": 50000.0, "is_active": True}
        result = validate_features(features, sample_schema)
        assert result.schema_valid is True
        assert result.missing_features == []


# ─── Soft Violation #3: Out of Bounds ─────────────────────────────


class TestSoftOutOfBounds:
    """Out-of-bounds values set schema_valid=false."""

    def test_below_min_value(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {
            "age": -1,
            "income": 50000.0,
            "is_active": True,
        }
        result = validate_features(features, sample_schema)
        assert not result.is_hard_violation
        assert result.schema_valid is False
        assert any(
            v.violation_type == ViolationType.SOFT_OUT_OF_BOUNDS
            and v.feature_name == "age"
            for v in result.soft_violations
        )

    def test_above_max_value(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {
            "age": 200,
            "income": 50000.0,
            "is_active": True,
        }
        result = validate_features(features, sample_schema)
        assert result.schema_valid is False
        assert any(
            v.violation_type == ViolationType.SOFT_OUT_OF_BOUNDS
            and v.feature_name == "age"
            for v in result.soft_violations
        )

    def test_at_min_boundary_passes(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {
            "age": 0,
            "income": 0.0,
            "is_active": True,
        }
        result = validate_features(features, sample_schema)
        assert result.schema_valid is True

    def test_at_max_boundary_passes(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {
            "age": 150,
            "income": 10_000_000.0,
            "is_active": True,
        }
        result = validate_features(features, sample_schema)
        assert result.schema_valid is True

    def test_null_value_skips_bounds_check(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {
            "age": None,
            "income": 50000.0,
            "is_active": True,
        }
        result = validate_features(features, sample_schema)
        assert result.schema_valid is True


# ─── Soft Violation #4: Extra Feature ─────────────────────────────


class TestSoftExtraFeature:
    """Extra features log warning but do NOT set schema_valid=false."""

    def test_single_extra_feature(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {
            "age": 25,
            "income": 50000.0,
            "is_active": True,
            "unknown_field": 42,
        }
        result = validate_features(features, sample_schema)
        assert not result.is_hard_violation
        assert result.schema_valid is True
        assert "unknown_field" in result.extra_features

    def test_multiple_extra_features(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {
            "age": 25,
            "income": 50000.0,
            "is_active": True,
            "extra_a": 1,
            "extra_b": 2,
        }
        result = validate_features(features, sample_schema)
        assert result.schema_valid is True
        assert len(result.extra_features) == 2

    def test_extra_feature_does_not_mark_invalid(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {
            "age": 25,
            "income": 50000.0,
            "is_active": True,
            "surprise": "value",
        }
        result = validate_features(features, sample_schema)
        assert result.schema_valid is True
        assert result.has_violations is True


# ─── Mixed Violations ─────────────────────────────────────────────


class TestMixedViolations:
    """Combined soft violations accumulate correctly."""

    def test_missing_and_extra_combined(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {
            "income": 50000.0,
            "is_active": True,
            "unknown_x": 99,
        }
        result = validate_features(features, sample_schema)
        assert not result.is_hard_violation
        assert result.schema_valid is False
        assert "age" in result.missing_features
        assert "unknown_x" in result.extra_features

    def test_missing_and_out_of_bounds_combined(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {
            "income": -100.0,
            "is_active": True,
        }
        result = validate_features(features, sample_schema)
        assert result.schema_valid is False
        assert "age" in result.missing_features
        assert any(
            v.violation_type == ViolationType.SOFT_OUT_OF_BOUNDS
            and v.feature_name == "income"
            for v in result.soft_violations
        )

    def test_hard_violation_short_circuits_soft_checks(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        features = {
            "age": "not-a-number",
            "income": -999.0,
        }
        result = validate_features(features, sample_schema)
        assert result.is_hard_violation is True
        assert result.soft_violations == []


# ─── Edge Cases ───────────────────────────────────────────────────


class TestEdgeCases:
    """Boundary and corner case validations."""

    def test_empty_features_with_required_schema(
        self, sample_schema: list[FeatureSchema],
    ) -> None:
        result = validate_features({}, sample_schema)
        assert not result.is_hard_violation
        assert result.schema_valid is False
        assert len(result.missing_features) == 3

    def test_empty_features_empty_schema(self) -> None:
        result = validate_features({}, [])
        assert not result.is_hard_violation
        assert result.schema_valid is True
        assert not result.has_violations

    def test_validation_result_defaults(self) -> None:
        result = ValidationResult()
        assert result.is_hard_violation is False
        assert result.schema_valid is True
        assert result.hard_violations == []
        assert result.soft_violations == []
        assert not result.has_violations
