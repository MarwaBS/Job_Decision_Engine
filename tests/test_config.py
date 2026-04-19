"""Tests for `src.config`.

Config is the locked source of truth for weights and thresholds. These tests
verify that the values ON DISK match architecture §6 exactly, and that the
objects are runtime-immutable.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config import (
    ENGINE_VERSION,
    MIN_PARSE_CONFIDENCE,
    NEAR_THRESHOLD_DISTANCE,
    THRESHOLDS,
    THRESHOLDS_VERSION,
    WEIGHTS,
    WEIGHTS_VERSION,
)


class TestWeights:
    def test_weights_match_architecture_section_6(self):
        """Architecture §6 table — each value is defensible per-row."""
        assert WEIGHTS.skills == 0.30
        assert WEIGHTS.experience == 0.20
        assert WEIGHTS.semantic == 0.15
        assert WEIGHTS.llm == 0.25
        assert WEIGHTS.role == 0.10

    def test_weights_sum_to_one(self):
        total = (
            WEIGHTS.skills
            + WEIGHTS.experience
            + WEIGHTS.semantic
            + WEIGHTS.llm
            + WEIGHTS.role
        )
        assert total == pytest.approx(1.0)

    def test_llm_weight_not_dominant(self):
        """Architecture Phase 4: LLM cannot single-handedly flip a decision.

        Operationalised here as: LLM weight must be strictly less than the
        sum of the two largest deterministic signals (skills + experience).
        Catches any future retuning that would let the LLM dominate.
        """
        deterministic_top_two = WEIGHTS.skills + WEIGHTS.experience
        assert WEIGHTS.llm < deterministic_top_two

    def test_weights_are_frozen(self):
        with pytest.raises(ValidationError):
            WEIGHTS.skills = 0.99  # type: ignore[misc]


class TestThresholds:
    def test_thresholds_match_architecture_section_6(self):
        assert THRESHOLDS.priority == 80.0
        assert THRESHOLDS.apply_ == 65.0
        assert THRESHOLDS.review == 50.0

    def test_thresholds_monotonic(self):
        assert THRESHOLDS.review < THRESHOLDS.apply_ < THRESHOLDS.priority

    def test_thresholds_version_string_present(self):
        assert THRESHOLDS.version == THRESHOLDS_VERSION
        assert THRESHOLDS_VERSION == "v1.0"

    def test_thresholds_frozen(self):
        with pytest.raises(ValidationError):
            THRESHOLDS.priority = 99.0  # type: ignore[misc]


class TestVersions:
    def test_engine_version_semver_shape(self):
        parts = ENGINE_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_weights_version_matches_thresholds_version(self):
        """Locked convention: v1 weights and v1 thresholds ship together.

        If one retunes, the other is re-audited. This test catches the case
        where a contributor bumps one version without revisiting the other.
        """
        assert WEIGHTS_VERSION == THRESHOLDS_VERSION


class TestHardFilterCutoffs:
    def test_min_parse_confidence_in_range(self):
        assert 0.0 < MIN_PARSE_CONFIDENCE < 1.0

    def test_near_threshold_distance_in_range(self):
        assert 0.0 < NEAR_THRESHOLD_DISTANCE < 100.0
