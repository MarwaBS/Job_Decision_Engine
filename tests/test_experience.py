"""Tests for the experience-match signal."""

from __future__ import annotations

import pytest

from src.schemas import CandidateProfile, ParsedJob, Seniority
from src.signals.experience import compute_experience_match


def _profile(years: float) -> CandidateProfile:
    return CandidateProfile(
        profile_version="v1.0",
        name="Test",
        summary="test",
        years_experience=years,
        seniority=Seniority.SENIOR,
    )


def _job(years: float | None) -> ParsedJob:
    return ParsedJob(title="Test", years_required=years)


class TestExperienceMatch:
    def test_no_years_required_returns_one(self):
        """JD without explicit years → don't penalise."""
        assert compute_experience_match(_job(None), _profile(3.0)) == 1.0

    def test_zero_years_required_returns_one(self):
        """0-year requirement met by anyone."""
        assert compute_experience_match(_job(0.0), _profile(0.0)) == 1.0

    def test_exact_match_returns_one(self):
        assert compute_experience_match(_job(5.0), _profile(5.0)) == 1.0

    def test_more_than_required_returns_one(self):
        assert compute_experience_match(_job(5.0), _profile(10.0)) == 1.0

    def test_zero_years_candidate_scores_zero(self):
        assert compute_experience_match(_job(5.0), _profile(0.0)) == 0.0

    def test_partial_match_linear(self):
        """3 years candidate, 5 years required → 0.6."""
        assert compute_experience_match(_job(5.0), _profile(3.0)) == pytest.approx(0.6)

    @pytest.mark.parametrize(
        "have,required,expected",
        [
            (0.0, 5.0, 0.0),
            (1.0, 5.0, 0.2),
            (2.5, 5.0, 0.5),
            (4.9, 5.0, 0.98),
            (5.0, 5.0, 1.0),
            (100.0, 5.0, 1.0),  # capped
        ],
    )
    def test_parametrized(self, have, required, expected):
        got = compute_experience_match(_job(required), _profile(have))
        assert got == pytest.approx(expected, abs=1e-9)

    def test_signal_is_bounded_zero_to_one(self):
        """experience_match ∈ [0, 1]."""
        for have in (0, 0.5, 2, 5, 10, 100):
            for req in (1, 5, 10, 20):
                got = compute_experience_match(_job(req), _profile(have))
                assert 0.0 <= got <= 1.0

    def test_pure_function_deterministic(self):
        a = compute_experience_match(_job(5.0), _profile(3.0))
        b = compute_experience_match(_job(5.0), _profile(3.0))
        assert a == b

    def test_significant_overqualification_still_scores_one(self):
        """Over-qualification is NOT penalised in the deterministic signal — it is
        left to the LLM's "risks" output (see module docstring). A 4x-experienced
        candidate still scores 1.0, not a reduced value."""
        assert compute_experience_match(_job(5.0), _profile(20.0)) == 1.0
