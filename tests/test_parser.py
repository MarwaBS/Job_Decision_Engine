"""Tests for the job-description parser."""

from __future__ import annotations

import pytest

from src.ingestion.parser import parse_job
from src.schemas import Seniority

# ── Realistic fixture: the happy path ────────────────────────────────────────

STRUCTURED_JD = """Title: Senior ML Engineer
Company: Acme Corp
Location: New York (Remote OK)

We're looking for a Senior ML Engineer with 5+ years of experience.

Required:
- Python, PyTorch, AWS
- Experience with MLOps
- Kubernetes and Docker

Nice to have:
- LangChain
- LLM applications

Salary: $150k-$220k
"""


class TestStructuredHappyPath:
    def test_title_extracted(self):
        j = parse_job(STRUCTURED_JD)
        assert j.parsed.title == "Senior ML Engineer"

    def test_company_extracted(self):
        assert parse_job(STRUCTURED_JD).parsed.company == "Acme Corp"

    def test_location_extracted(self):
        j = parse_job(STRUCTURED_JD)
        assert j.parsed.location is not None
        assert "New York" in j.parsed.location

    def test_remote_flag_true(self):
        assert parse_job(STRUCTURED_JD).parsed.remote is True

    def test_seniority_detected(self):
        assert parse_job(STRUCTURED_JD).parsed.seniority == Seniority.SENIOR

    def test_years_required_extracted(self):
        assert parse_job(STRUCTURED_JD).parsed.years_required == 5.0

    def test_required_skills_found(self):
        req = parse_job(STRUCTURED_JD).parsed.required_skills
        for skill in ("python", "pytorch", "aws", "mlops", "kubernetes", "docker"):
            assert skill in req, f"{skill!r} missing from required {req}"

    def test_preferred_skills_found(self):
        pref = parse_job(STRUCTURED_JD).parsed.preferred_skills
        for skill in ("langchain", "llm"):
            assert skill in pref, f"{skill!r} missing from preferred {pref}"

    def test_preferred_and_required_disjoint(self):
        """A skill that appeared before the "nice to have" heading should not
        also appear in preferred — required wins."""
        j = parse_job(STRUCTURED_JD)
        overlap = set(j.parsed.required_skills) & set(j.parsed.preferred_skills)
        assert overlap == set(), f"overlap found: {overlap}"

    def test_salary_range_extracted(self):
        assert parse_job(STRUCTURED_JD).parsed.salary_range_usd == (150000, 220000)

    def test_parse_confidence_high(self):
        """Fully-structured JD should score ≥ 0.9 on the confidence heuristic."""
        assert parse_job(STRUCTURED_JD).parse_confidence >= 0.9

    def test_no_warnings_on_clean_input(self):
        assert parse_job(STRUCTURED_JD).parse_warnings == []


# ── Content hash + determinism ───────────────────────────────────────────────


class TestContentHash:
    def test_hash_is_sha256(self):
        h = parse_job(STRUCTURED_JD).content_hash
        assert h.startswith("sha256:")
        assert len(h) == len("sha256:") + 64

    def test_same_input_same_hash(self):
        h1 = parse_job(STRUCTURED_JD).content_hash
        h2 = parse_job(STRUCTURED_JD).content_hash
        assert h1 == h2

    def test_different_input_different_hash(self):
        h1 = parse_job(STRUCTURED_JD).content_hash
        h2 = parse_job(STRUCTURED_JD + "\n\nextra content").content_hash
        assert h1 != h2

    def test_trailing_whitespace_does_not_change_hash(self):
        """Normalisation strips trailing whitespace per line — cosmetic
        differences must not change the hash, or dedupe breaks."""
        h1 = parse_job(STRUCTURED_JD).content_hash
        # Add trailing spaces on every line
        munged = "\n".join(line + "   " for line in STRUCTURED_JD.splitlines())
        h2 = parse_job(munged).content_hash
        assert h1 == h2


# ── Low-structure inputs trigger low confidence ──────────────────────────────


class TestLowStructureInputs:
    def test_empty_string_returns_untitled_and_zero_confidence(self):
        j = parse_job("")
        assert j.parsed.title == "Untitled Role"
        assert j.parse_confidence == 0.0
        assert "empty_input" in j.parse_warnings

    def test_whitespace_only_returns_zero_confidence(self):
        j = parse_job("   \n\n   \n")
        assert j.parse_confidence == 0.0

    def test_prose_only_is_low_confidence(self):
        """A paragraph with no skills and no structure should score below the
        MIN_PARSE_CONFIDENCE threshold (0.5)."""
        j = parse_job(
            "We are a startup looking for someone great to help us grow."
            " No specifics."
        )
        assert j.parse_confidence < 0.5

    def test_partial_structure_produces_partial_confidence(self):
        """Skills-only JD (no heading, no seniority, no workplace cue)
        produces a confidence strictly between fully-empty and fully-structured.

        Expected buckets that fire on "We use Python and PyTorch":
          - ≥1 skill matched: 0.20
        Nothing else. Total: 0.20. Below MIN_PARSE_CONFIDENCE (0.5) → routes
        to REVIEW at scoring time, which is the correct behaviour: the
        user should confirm before an application decision is made on this
        little evidence.
        """
        jd = "Machine Learning Engineer\nWe use Python and PyTorch."
        j = parse_job(jd)
        assert 0.15 <= j.parse_confidence < 0.5

    def test_adding_structure_raises_confidence(self):
        """Monotonicity check — adding structure cues must not lower confidence."""
        bare = "We use Python and PyTorch."
        enriched = (
            "Title: ML Engineer\n"
            "Company: Acme\n"
            "Location: Remote\n"
            "Senior role, 5+ years of experience.\n"
            "We use Python, PyTorch, and AWS."
        )
        bare_conf = parse_job(bare).parse_confidence
        enriched_conf = parse_job(enriched).parse_confidence
        assert enriched_conf > bare_conf
        assert enriched_conf >= 0.7


# ── Seniority detection ──────────────────────────────────────────────────────


class TestSeniority:
    @pytest.mark.parametrize("title,expected", [
        ("Staff ML Engineer", Seniority.STAFF),
        ("Principal Data Scientist", Seniority.PRINCIPAL),
        ("Senior Software Engineer", Seniority.SENIOR),
        ("Sr. Backend Dev", Seniority.SENIOR),
        ("Junior Developer", Seniority.JUNIOR),
        ("Mid-level Engineer", Seniority.MID),
        ("Senior Staff Software Engineer", Seniority.STAFF),  # staff wins
    ])
    def test_seniority_keyword_match(self, title, expected):
        assert parse_job(title).parsed.seniority == expected

    def test_no_keyword_no_seniority(self):
        assert parse_job("Software Engineer").parsed.seniority is None


# ── Years-required regex variants ────────────────────────────────────────────


class TestYearsRequired:
    @pytest.mark.parametrize("text,expected", [
        ("5+ years of experience", 5.0),
        ("3 years of experience required", 3.0),
        ("10+ years", 10.0),
        ("5-7 years", 5.0),
        ("5 to 7 years", 5.0),
    ])
    def test_years_patterns(self, text, expected):
        assert parse_job(text).parsed.years_required == expected

    def test_no_years_returns_none(self):
        assert parse_job("Python developer.").parsed.years_required is None


# ── Remote / hybrid / on-site ────────────────────────────────────────────────


class TestWorkplace:
    def test_remote_keyword_detects_remote(self):
        assert parse_job("Fully remote role.").parsed.remote is True

    def test_hybrid_detects_remote(self):
        assert parse_job("Hybrid work arrangement.").parsed.remote is True

    def test_on_site_only_detects_non_remote(self):
        assert parse_job("On-site role in NYC.").parsed.remote is False

    def test_no_cue_defaults_to_false(self):
        assert parse_job("Software engineer position.").parsed.remote is False


# ── Salary parsing edge cases ────────────────────────────────────────────────


class TestSalary:
    def test_salary_range_with_k(self):
        assert parse_job("Salary: $150k-$220k").parsed.salary_range_usd == (150000, 220000)

    def test_salary_range_with_dash(self):
        assert parse_job("Salary $100 to $150").parsed.salary_range_usd == (100000, 150000)

    def test_no_salary_returns_none(self):
        assert parse_job("Great role.").parsed.salary_range_usd is None

    def test_inverted_salary_range_returns_none_with_warning(self):
        """$200k-$100k is malformed — don't invent an answer, warn."""
        j = parse_job("Salary $200k-$100k")
        assert j.parsed.salary_range_usd is None
        assert "salary_range_inverted" in j.parse_warnings


# ── Parser purity ────────────────────────────────────────────────────────────


class TestPurity:
    def test_parser_does_not_import_network_modules(self):
        """Architecture §3: parser is pure. The url_scraper is the I/O
        module, not the parser."""
        from pathlib import Path

        src = (
            Path(__file__).parent.parent / "src" / "ingestion" / "parser.py"
        ).read_text(encoding="utf-8")
        forbidden = ["import requests", "from requests", "import httpx", "urllib.request"]
        for needle in forbidden:
            assert needle not in src, f"parser.py imports network module {needle!r}"


# ── Long/degenerate title handling ───────────────────────────────────────────


class TestTitleDegenerate:
    def test_overlong_first_line_truncated(self):
        """A 200-char first line gets truncated to 140 + ellipsis."""
        huge = "A" * 200
        j = parse_job(huge)
        assert len(j.parsed.title) <= 141  # 140 + "…"
        assert j.parsed.title.endswith("…")
