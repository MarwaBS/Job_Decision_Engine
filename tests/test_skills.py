"""Tests for the skills signal.

Covers: taxonomy invariants, extraction correctness on realistic text,
match-score math, edge cases (empty JD, empty profile).
"""

from __future__ import annotations

import pytest

from src.schemas import CandidateProfile, ParsedJob, Seniority
from src.signals.skills import (
    SKILLS_TAXONOMY,
    compute_skills_match,
    extract_skills,
)


# ── Taxonomy invariants ──────────────────────────────────────────────────────


class TestTaxonomyInvariants:
    def test_taxonomy_no_duplicate_canonicals(self):
        """Every canonical skill name must appear in exactly one bucket."""
        seen: set[str] = set()
        for bucket in SKILLS_TAXONOMY.values():
            for canonical in bucket:
                assert canonical not in seen, (
                    f"duplicate canonical {canonical!r} across buckets"
                )
                seen.add(canonical)

    def test_taxonomy_every_skill_has_alias_list(self):
        for bucket in SKILLS_TAXONOMY.values():
            for canonical, aliases in bucket.items():
                assert isinstance(aliases, list), f"{canonical}: aliases not a list"
                assert len(aliases) >= 1, f"{canonical}: no aliases"

    def test_taxonomy_buckets_are_not_empty(self):
        for bucket_name, bucket in SKILLS_TAXONOMY.items():
            assert len(bucket) >= 1, f"bucket {bucket_name} is empty"


# ── Extraction ───────────────────────────────────────────────────────────────


class TestExtraction:
    def test_extracts_python_and_pytorch(self):
        s = extract_skills("We use Python and PyTorch for modelling.")
        assert "python" in s.tech
        assert "pytorch" in s.tools

    def test_case_insensitive(self):
        s = extract_skills("PYTHON, Pytorch, AWS")
        assert "python" in s.tech
        assert "pytorch" in s.tools
        assert "aws" in s.tools

    def test_word_boundary_avoids_false_positive(self):
        """The 'R' language alias uses strict word boundary — must not match
        inside other words like 'architecture' or 'production'."""
        s = extract_skills("Architecture and production engineering.")
        assert "r" not in s.tech

    def test_empty_text_produces_empty_sets(self):
        s = extract_skills("")
        assert s.tech == ()
        assert s.tools == ()
        assert s.domain == ()

    def test_extraction_is_deterministic(self):
        text = "Python, PyTorch, AWS, MLOps"
        a = extract_skills(text)
        b = extract_skills(text)
        assert a == b

    def test_extraction_sorted_output(self):
        """Lists must be sorted — guarantees byte-stable content hashes."""
        s = extract_skills("TypeScript, Python, Java, Go")
        assert list(s.tech) == sorted(s.tech)


# ── compute_skills_match ─────────────────────────────────────────────────────


def _profile(skills_tech=(), skills_tools=(), skills_domain=()) -> CandidateProfile:
    return CandidateProfile(
        profile_version="v1.0",
        name="Test",
        summary="test",
        years_experience=5.0,
        seniority=Seniority.SENIOR,
        skills_tech=list(skills_tech),
        skills_tools=list(skills_tools),
        skills_domain=list(skills_domain),
    )


class TestSkillsMatch:
    def test_perfect_match_scores_one(self):
        job = ParsedJob(
            title="ML Eng",
            required_skills=["python", "pytorch"],
            preferred_skills=["mlops"],
        )
        profile = _profile(
            skills_tech=["python"],
            skills_tools=["pytorch"],
            skills_domain=["mlops"],
        )
        assert compute_skills_match(job, profile) == pytest.approx(1.0)

    def test_no_overlap_scores_zero(self):
        job = ParsedJob(
            title="ML Eng", required_skills=["python"], preferred_skills=["pytorch"],
        )
        profile = _profile(skills_tech=["scala"])
        assert compute_skills_match(job, profile) == 0.0

    def test_empty_job_skills_returns_zero(self):
        """Architecture §6 hard filter: parser's parse_confidence catches this."""
        job = ParsedJob(title="ML Eng", required_skills=[], preferred_skills=[])
        profile = _profile(skills_tech=["python"])
        assert compute_skills_match(job, profile) == 0.0

    def test_partial_match_exact_math(self):
        """Required: 3 skills, candidate has 2. Preferred: 2 skills, candidate has 1.
        numerator = 2 + 0.5*1 = 2.5
        denominator = 3 + 0.5*2 = 4.0
        score = 0.625
        """
        job = ParsedJob(
            title="ML Eng",
            required_skills=["python", "pytorch", "aws"],
            preferred_skills=["mlops", "kubernetes"],
        )
        profile = _profile(
            skills_tech=["python"],
            skills_tools=["pytorch", "mlops"],  # mlops here is wrong bucket but
            skills_domain=["mlops"],            # _candidate_skill_set flattens
        )
        # Profile skill set (lower-cased, deduped): {python, pytorch, mlops}
        # Required matches: {python, pytorch} = 2
        # Preferred matches: {mlops} = 1
        # score = (2 + 0.5*1) / (3 + 0.5*2) = 2.5 / 4.0 = 0.625
        assert compute_skills_match(job, profile) == pytest.approx(0.625)

    def test_match_case_insensitive(self):
        """Profile free-form entries should match regardless of case."""
        job = ParsedJob(title="ML Eng", required_skills=["python"])
        profile = _profile(skills_tech=["Python"])
        assert compute_skills_match(job, profile) == pytest.approx(1.0)

    def test_match_is_deterministic(self):
        job = ParsedJob(title="ML Eng", required_skills=["python", "pytorch"])
        profile = _profile(skills_tech=["python"], skills_tools=["pytorch"])
        assert compute_skills_match(job, profile) == compute_skills_match(job, profile)
