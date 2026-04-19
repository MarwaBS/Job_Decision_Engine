"""Tests for the LLM reasoning layer.

All tests use `MockReasoner` / `FailingReasoner` — no OpenAI calls, no
network, hermetic. The `OpenAIReasoner` is exercised by a separate
integration smoke test (needs a real API key) that is NOT part of this
suite.

Focus:
- Protocol conformance
- ReasoningOutput schema enforcement (the retry-once-or-fail contract
  depends on strict validation)
- Prompt versioning file loads correctly
- Mock/Failing reasoners behave as documented
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.llm.reasoning import (
    FailingReasoner,
    LLMReasoner,
    LLMReasoningFailed,
    MockReasoner,
    _validate_reasoning_json,
    load_prompt,
    render_user_message,
)
from src.schemas import CandidateProfile, ParsedJob, ReasoningOutput, Seniority, Signals


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _job() -> ParsedJob:
    return ParsedJob(
        title="Senior ML Engineer",
        seniority=Seniority.SENIOR,
        years_required=5.0,
        required_skills=["python", "pytorch"],
        preferred_skills=["aws"],
    )


def _profile() -> CandidateProfile:
    return CandidateProfile(
        profile_version="v1.0",
        name="Marwa",
        summary="ML engineer with 5 years of Python + PyTorch",
        years_experience=5.0,
        seniority=Seniority.SENIOR,
        skills_tech=["python"],
        skills_tools=["pytorch", "aws"],
    )


def _signals() -> Signals:
    return Signals(
        skills_match=0.8, experience_match=1.0,
        semantic_similarity=0.7, llm_confidence=0.75, role_level_fit=1.0,
    )


# ── Protocol conformance ─────────────────────────────────────────────────────


class TestProtocolConformance:
    def test_mock_reasoner_satisfies_protocol(self):
        assert isinstance(MockReasoner(), LLMReasoner)

    def test_failing_reasoner_satisfies_protocol(self):
        assert isinstance(FailingReasoner(), LLMReasoner)


# ── MockReasoner behaviour ───────────────────────────────────────────────────


class TestMockReasoner:
    def test_returns_valid_reasoning_output(self):
        r = MockReasoner().reason(job=_job(), profile=_profile(), signals=_signals())
        assert isinstance(r, ReasoningOutput)

    def test_default_confidence_is_sane(self):
        r = MockReasoner().reason(job=_job(), profile=_profile(), signals=_signals())
        assert 0.0 <= r.llm_confidence <= 1.0

    def test_custom_confidence_honored(self):
        r = MockReasoner(llm_confidence=0.42).reason(
            job=_job(), profile=_profile(), signals=_signals()
        )
        assert r.llm_confidence == pytest.approx(0.42)

    def test_mock_is_deterministic(self):
        r1 = MockReasoner().reason(job=_job(), profile=_profile(), signals=_signals())
        r2 = MockReasoner().reason(job=_job(), profile=_profile(), signals=_signals())
        assert r1 == r2


# ── FailingReasoner ──────────────────────────────────────────────────────────


class TestFailingReasoner:
    def test_always_raises(self):
        with pytest.raises(LLMReasoningFailed):
            FailingReasoner().reason(
                job=_job(), profile=_profile(), signals=_signals()
            )


# ── ReasoningOutput schema contract ──────────────────────────────────────────


class TestReasoningOutputSchema:
    """Architecture §7 bounds. If these break, the LLM retry-or-fail
    contract can't tell good output from bad."""

    VALID_PAYLOAD: dict = {
        "strengths": ["a", "b", "c"],
        "gaps": ["g1", "g2"],
        "risks": ["r1"],
        "llm_confidence": 0.7,
        "recommended_talking_points": ["t1", "t2", "t3"],
    }

    def test_valid_payload_validates(self):
        ReasoningOutput.model_validate(self.VALID_PAYLOAD)

    def test_too_few_strengths_rejected(self):
        payload = dict(self.VALID_PAYLOAD, strengths=["only one", "two"])
        with pytest.raises(ValidationError):
            ReasoningOutput.model_validate(payload)

    def test_too_many_strengths_rejected(self):
        payload = dict(self.VALID_PAYLOAD, strengths=["a", "b", "c", "d", "e", "f"])
        with pytest.raises(ValidationError):
            ReasoningOutput.model_validate(payload)

    def test_bullet_over_120_chars_rejected(self):
        long_bullet = "x" * 121
        payload = dict(self.VALID_PAYLOAD, strengths=[long_bullet, "b", "c"])
        with pytest.raises(ValidationError, match="exceeds 120"):
            ReasoningOutput.model_validate(payload)

    def test_empty_bullet_rejected(self):
        payload = dict(self.VALID_PAYLOAD, strengths=["", "b", "c"])
        with pytest.raises(ValidationError, match="empty or whitespace"):
            ReasoningOutput.model_validate(payload)

    def test_confidence_out_of_range_rejected(self):
        payload = dict(self.VALID_PAYLOAD, llm_confidence=1.5)
        with pytest.raises(ValidationError):
            ReasoningOutput.model_validate(payload)


# ── _validate_reasoning_json (parses + validates) ────────────────────────────


class TestValidateReasoningJson:
    def test_valid_json_returns_reasoning_output(self):
        payload = {
            "strengths": ["a", "b", "c"],
            "gaps": ["g1", "g2"],
            "risks": ["r1"],
            "llm_confidence": 0.6,
            "recommended_talking_points": ["t1", "t2", "t3"],
        }
        r = _validate_reasoning_json(json.dumps(payload))
        assert r.llm_confidence == pytest.approx(0.6)

    def test_malformed_json_raises_decode_error(self):
        """Malformed JSON raises JSONDecodeError (caught by reasoner retry)."""
        with pytest.raises(json.JSONDecodeError):
            _validate_reasoning_json("{not valid json")

    def test_valid_json_wrong_schema_raises_validation_error(self):
        """Structurally valid JSON but missing required fields → ValidationError."""
        with pytest.raises(ValidationError):
            _validate_reasoning_json('{"strengths": ["only one"]}')


# ── Prompt versioning ────────────────────────────────────────────────────────


class TestPromptVersioning:
    def test_load_default_prompt(self):
        prompt = load_prompt("reasoning_v1")
        assert prompt["version"] == "v1"
        assert "system" in prompt
        assert "user" in prompt

    def test_load_unknown_version_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_prompt("reasoning_v999")

    def test_render_user_message_substitutes(self):
        prompt = load_prompt("reasoning_v1")
        rendered = render_user_message(
            prompt, job=_job(), profile=_profile(), signals=_signals()
        )
        assert "Senior ML Engineer" in rendered
        assert "Marwa" in rendered
        assert "python" in rendered.lower()


# ── LLM-layer module purity: no I/O at import time ───────────────────────────


class TestLLMModulePurity:
    def test_importing_reasoning_does_not_require_openai(self):
        """Lazy import: the module must load on a machine without openai."""
        from pathlib import Path

        src = (
            Path(__file__).parent.parent / "src" / "llm" / "reasoning.py"
        ).read_text(encoding="utf-8")
        # openai must only appear inside a function body (lazy import), not at
        # module top level. Quick heuristic: no "import openai" at col 0.
        for line in src.splitlines():
            stripped = line.rstrip()
            if stripped == "import openai" or stripped.startswith("import openai "):
                pytest.fail(
                    "src/llm/reasoning.py has top-level `import openai` — "
                    "tests should not require openai installed"
                )
            if stripped == "from openai import OpenAI":
                pytest.fail(
                    "src/llm/reasoning.py has top-level OpenAI import"
                )
