"""Pins the deterministic outputs of `scripts/demo_example.py`.

The README's Example B quotes concrete numbers; this suite pins every
hermetically-computable one (parser + pure signals) so README drift fails
CI. `semantic_sim` and the final `apply_score` depend on the pinned
sentence-transformer model and are reproduced by running the script
itself — not asserted here, to keep the suite model-free and fast.
"""

from __future__ import annotations

import pytest

from scripts.demo_example import EXAMPLE_B_JD, EXAMPLE_PROFILE
from src.ingestion.parser import parse_job
from src.signals.experience import compute_experience_match
from src.signals.skills import compute_skills_match


class TestExampleBDeterministicSignals:
    def test_parse_confidence_is_full(self):
        """All 8 structural cues present → parse_confidence = 1.0."""
        assert parse_job(EXAMPLE_B_JD).parse_confidence == pytest.approx(1.0)

    def test_required_skills_extraction(self):
        parsed = parse_job(EXAMPLE_B_JD).parsed
        assert parsed.required_skills == [
            "aws",
            "docker",
            "fastapi",
            "llm",
            "mlops",
            "python",
            "pytorch",
            "sql",
        ]

    def test_preferred_skills_extraction(self):
        parsed = parse_job(EXAMPLE_B_JD).parsed
        assert parsed.preferred_skills == ["kubernetes", "langchain", "rag"]

    def test_skills_match_value(self):
        """8 of 8 required + 1 of 3 preferred → (8 + 0.5) / (8 + 1.5)."""
        parsed = parse_job(EXAMPLE_B_JD).parsed
        assert compute_skills_match(parsed, EXAMPLE_PROFILE) == pytest.approx(8.5 / 9.5)

    def test_experience_and_workplace(self):
        job = parse_job(EXAMPLE_B_JD)
        assert job.parsed.years_required == 5.0
        assert compute_experience_match(job.parsed, EXAMPLE_PROFILE) == 1.0
        assert job.parsed.remote is True  # "(Remote)" in the location line
