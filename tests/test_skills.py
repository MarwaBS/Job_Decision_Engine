"""Tests for the skills signal.

Covers: taxonomy invariants, extraction correctness on realistic text,
match-score math, edge cases (empty JD, empty profile).
"""

from __future__ import annotations

import pytest

from src.schemas import CandidateProfile, ParsedJob, Seniority
from src.signals.skills import (
    SKILLS_TAXONOMY,
    _normalise,
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
        """The 'R' language alias must not match inside other words like
        'architecture' or 'production'."""
        s = extract_skills("Architecture and production engineering.")
        assert "r" not in s.tech

    @pytest.mark.parametrize(
        ("text", "phantom"),
        [
            # Every alias is boundary-anchored at COMPILE time, not just the
            # hand-flagged ones. These are the substrings that previously
            # produced phantom skills on essentially every real JD:
            ("Requirements: strong communication skills", "typescript"),  # ...ts
            ("our platform team ships weekly", "tensorflow"),  # ...tf...
            ("a happy team environment", "python"),  # ...py
            ("MongoDB experience preferred", "go"),  # ...go...
            ("JavaScript developer", "java"),  # prefix of a longer word
            ("scalable systems", "scala"),  # prefix of a longer word
            ("the despatch desk", "data engineering"),  # 'de' inside words
            ("global logistics", "go"),  # ...g(lo)...
            ("convolutional layers", "computer vision"),  # 'cv' not present
            ("dragging deadlines", "rag"),  # ...rag...
        ],
    )
    def test_no_substring_phantom_skills(self, text: str, phantom: str):
        """Adversarial regression guard for the boundary-anchoring bug.

        Aliases compiled WITHOUT word boundaries turned ordinary JD prose
        into phantom skills ("RequiremenTS" → typescript), silently
        corrupting skills_match (the highest-weighted signal) and
        parse_confidence. Real prose must extract nothing it doesn't say.
        """
        assert phantom not in extract_skills(text).all

    @pytest.mark.parametrize(
        ("text", "phantom"),
        [
            # Ambiguous short aliases are ALSO ordinary English tokens —
            # word boundaries alone can't disambiguate them. They require
            # list context (delimiter-adjacent), so flowing prose must
            # never credit them:
            ("Please send your CV to jobs@acme.com", "computer vision"),
            ("attach your CV.", "computer vision"),
            ("R&D team of 40 engineers", "r"),
            ("own the go-to-market strategy", "go"),
            ("we'll go over next steps together", "go"),
            ("TF-IDF features for ranking", "tensorflow"),
            ("previously at DE Shaw", "data engineering"),
        ],
    )
    def test_ambiguous_tokens_require_list_context(self, text: str, phantom: str):
        assert phantom not in extract_skills(text).all

    @pytest.mark.parametrize(
        ("text", "expected_subset"),
        [
            # ...while list-style citations — the dominant JD pattern —
            # must still extract them:
            ("Languages: Python, Go, R", {"go", "python", "r"}),
            ("TS/JS stack", {"javascript", "typescript"}),
            ("- Go\n- R\n- Python", {"go", "python", "r"}),
            ("Tools (R, dbt)", {"dbt", "r"}),
            ("Skills: Python, R.", {"python", "r"}),
            ("Golang microservices", {"go"}),  # unambiguous alias unaffected
        ],
    )
    def test_ambiguous_tokens_match_in_list_context(
        self, text: str, expected_subset: set[str]
    ):
        assert expected_subset <= set(extract_skills(text).all)

    @pytest.mark.parametrize(
        ("text", "residual_phantom"),
        [
            # KNOWN, DOCUMENTED residuals of list-context gating (see the
            # _STRONG_LEAD/_STRONG_TRAIL comment in skills.py): delimiter
            # adjacency can't see the far side of the delimiter. These pins
            # make the accepted limitation visible and verifiable — if a
            # future change FIXES one, this test fails and the residual
            # documentation must be updated to match.
            ("ready to go, and we ship", "go"),
            ("Active TS/SCI clearance required", "typescript"),
            ("Send your CV/cover letter to us", "computer vision"),
            ("Microsoft(R) Office proficiency", "r"),
        ],
    )
    def test_documented_residual_phantoms_are_pinned(
        self, text: str, residual_phantom: str
    ):
        """These phantoms are an accepted, documented tradeoff — NOT a bug
        regression. Every one of them also matched under plain word-boundary
        matching, so list-context gating strictly tightened precision."""
        assert residual_phantom in extract_skills(text).all

    def test_profile_side_resolves_ambiguous_alias_without_list_gate(self):
        """The profile-side asymmetry is intentional (see _build_alias_lookup):
        ambiguous tokens are gated to list-context only on the EXTRACTION side
        (JD prose). On the profile side, _normalise matches the WHOLE entry
        exactly, so a discrete list item literally equal to "cv"/"r" resolves to
        its canonical — that entry IS list context — while prose embedding the
        token does not (the whole string is the key, not a substring)."""
        # Discrete ambiguous entries resolve to canonicals.
        assert _normalise(["cv"]) == {"computer vision"}
        assert _normalise(["r"]) == {"r"}
        assert _normalise(["de"]) == {"data engineering"}
        # Prose in a single entry does NOT spuriously resolve the embedded token.
        assert _normalise(["experience with go"]) == {"experience with go"}
        assert "computer vision" not in _normalise(["my cv is attached"])

    def test_boundary_anchoring_still_matches_real_mentions(self):
        """The anchors must not cost recall on legitimate mentions —
        including the awkward symbol-suffixed aliases (c++, c#) where a
        naive \\b would fail."""
        s = extract_skills(
            "Stack: C++, C#, TS/JS, R, dbt, node.js, PySpark on AWS and GCP."
        )
        assert set(s.all) >= {
            "c++",
            "c#",
            "typescript",
            "javascript",
            "r",
            "dbt",
            "node",
            "spark",
            "aws",
            "gcp",
        }

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
            title="ML Eng",
            required_skills=["python"],
            preferred_skills=["pytorch"],
        )
        profile = _profile(skills_tech=["scala"])
        assert compute_skills_match(job, profile) == 0.0

    def test_empty_job_skills_returns_zero(self):
        """Empty job skills score zero; the parser's parse_confidence hard
        filter catches the unparseable case upstream."""
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
            skills_domain=["mlops"],  # _candidate_skill_set flattens
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

    def test_profile_aliases_resolve_to_canonicals(self):
        """Free-form profile entries are normalised through the taxonomy.

        Extraction emits CANONICAL names ("scikit-learn"); profiles are
        written by humans ("sklearn", "k8s", "torch"). Without alias
        resolution those silently never match — the profile side must go
        through the same vocabulary as the JD side.
        """
        job = ParsedJob(
            title="ML Eng",
            required_skills=["scikit-learn", "kubernetes", "pytorch"],
        )
        profile = _profile(skills_tools=["sklearn", "k8s", "torch"])
        assert compute_skills_match(job, profile) == pytest.approx(1.0)

    def test_match_is_deterministic(self):
        job = ParsedJob(title="ML Eng", required_skills=["python", "pytorch"])
        profile = _profile(skills_tech=["python"], skills_tools=["pytorch"])
        assert compute_skills_match(job, profile) == compute_skills_match(job, profile)
