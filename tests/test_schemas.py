"""Tests for the data contracts (`src.schemas`).

These tests enforce the rules from architecture §5 and §6 at the Pydantic
layer. If any of these fail, the contracts have drifted from the
architecture — fix the code, not the test.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import (
    DecisionSensitivity,
    DecisionTrace,
    FailureMode,
    Signals,
    Thresholds,
    Verdict,
    Weights,
)

# ── Signals ──────────────────────────────────────────────────────────────────


class TestSignals:
    def test_valid_signals_construct(self):
        s = Signals(
            skills_match=0.8,
            experience_match=1.0,
            semantic_similarity=0.7,
            llm_confidence=0.85,
            role_level_fit=1.0,
        )
        assert s.skills_match == 0.8
        assert s.dealbreaker_hit is False
        assert s.parse_confidence == 1.0

    @pytest.mark.parametrize("field", [
        "skills_match", "experience_match", "semantic_similarity", "llm_confidence"
    ])
    def test_signal_out_of_range_rejected(self, field):
        """Architecture §6: all continuous signals are bounded to [0, 1]."""
        payload = {
            "skills_match": 0.5, "experience_match": 0.5,
            "semantic_similarity": 0.5, "llm_confidence": 0.5,
            "role_level_fit": 0.5,
        }
        payload[field] = 1.5
        with pytest.raises(ValidationError):
            Signals(**payload)

    def test_role_level_fit_must_be_discrete(self):
        """Architecture §6: role_level_fit ∈ {0, 0.5, 1}."""
        payload = {
            "skills_match": 0.5, "experience_match": 0.5,
            "semantic_similarity": 0.5, "llm_confidence": 0.5,
            "role_level_fit": 0.75,
        }
        with pytest.raises(ValidationError):
            Signals(**payload)

    def test_signals_frozen(self):
        """Signals are immutable — crossing a module boundary shouldn't allow mutation."""
        s = Signals(
            skills_match=0.5, experience_match=0.5,
            semantic_similarity=0.5, llm_confidence=0.5,
            role_level_fit=0.5,
        )
        with pytest.raises(ValidationError):
            s.skills_match = 0.9  # type: ignore[misc]


# ── Weights ──────────────────────────────────────────────────────────────────


class TestWeights:
    def test_valid_weights_sum_to_one(self):
        w = Weights(skills=0.30, experience=0.20, semantic=0.15, llm=0.25, role=0.10)
        assert w.skills + w.experience + w.semantic + w.llm + w.role == pytest.approx(1.0)

    def test_weights_sum_not_one_rejected(self):
        """Architecture §6: weights MUST sum to 1.0. Any drift is an error."""
        with pytest.raises(ValidationError, match="sum to 1.0"):
            Weights(skills=0.30, experience=0.20, semantic=0.15, llm=0.25, role=0.20)

    def test_weights_negative_rejected(self):
        with pytest.raises(ValidationError):
            Weights(skills=-0.10, experience=0.30, semantic=0.25, llm=0.30, role=0.25)

    def test_weights_frozen(self):
        w = Weights(skills=0.30, experience=0.20, semantic=0.15, llm=0.25, role=0.10)
        with pytest.raises(ValidationError):
            w.skills = 0.99  # type: ignore[misc]


# ── Thresholds ───────────────────────────────────────────────────────────────


class TestThresholds:
    def test_valid_thresholds_construct(self):
        t = Thresholds(priority=80.0, **{"apply": 65.0}, review=50.0)
        assert t.priority == 80.0
        assert t.apply_ == 65.0
        assert t.review == 50.0
        assert t.version == "v1.0"

    def test_non_monotonic_thresholds_rejected(self):
        """Architecture §6: review < apply < priority. Any other ordering is an error."""
        with pytest.raises(ValidationError, match="review < apply < priority"):
            Thresholds(priority=60.0, **{"apply": 70.0}, review=50.0)

    def test_equal_thresholds_rejected(self):
        with pytest.raises(ValidationError):
            Thresholds(priority=65.0, **{"apply": 65.0}, review=50.0)


# ── DecisionTrace + DecisionSensitivity ──────────────────────────────────────


class TestDecisionTrace:
    def test_trace_construct(self):
        trace = DecisionTrace(
            dominant_signal="skills_match",
            failure_mode_detected=None,
            decision_sensitivity=DecisionSensitivity(
                if_llm_removed_score=70.0,
                if_skills_boosted_plus_10pct=85.0,
                if_experience_removed_score=75.0,
            ),
            nearest_threshold_distance=2.5,
            near_threshold_flag=True,
        )
        assert trace.failure_mode_detected is None
        assert trace.near_threshold_flag is True

    def test_trace_failure_mode_accepts_enum(self):
        trace = DecisionTrace(
            dominant_signal="dealbreaker",
            failure_mode_detected=FailureMode.DEALBREAKER_HIT,
            decision_sensitivity=DecisionSensitivity(
                if_llm_removed_score=0.0,
                if_skills_boosted_plus_10pct=0.0,
                if_experience_removed_score=0.0,
            ),
            nearest_threshold_distance=50.0,
            near_threshold_flag=False,
        )
        assert trace.failure_mode_detected == FailureMode.DEALBREAKER_HIT

    def test_trace_invalid_dominant_signal_rejected(self):
        with pytest.raises(ValidationError):
            DecisionTrace(
                dominant_signal="totally_made_up",  # type: ignore[arg-type]
                failure_mode_detected=None,
                decision_sensitivity=DecisionSensitivity(
                    if_llm_removed_score=0.0,
                    if_skills_boosted_plus_10pct=0.0,
                    if_experience_removed_score=0.0,
                ),
                nearest_threshold_distance=0.0,
                near_threshold_flag=False,
            )


# ── Verdict enum ─────────────────────────────────────────────────────────────


def test_verdict_values_locked():
    """Architecture §6 defines exactly 4 verdicts. Adding one is an ADR change."""
    assert {v.value for v in Verdict} == {"PRIORITY", "APPLY", "REVIEW", "SKIP"}
