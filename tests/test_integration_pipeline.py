"""End-to-end integration test: parser → signals → scorer.

This is the proof that Step 2 and Step 3 compose correctly. It exercises
the full data flow a Step 4 orchestrator will later wire into an API
endpoint:

    raw JD text
        → parse_job                          (ingestion)
        → compute_skills_match               (signals)
        → compute_experience_match           (signals)
        → compute_semantic_similarity        (signals, mock provider)
        → Signals                            (schemas)
        → score                              (engine)
        → DecisionResult with verdict + trace

The LLM signal is stubbed with a fixed value because Step 4 is where the
LLM layer lands. That stub does NOT violate the integrity claim — it's an
explicit Step-4 seam, documented as such.
"""

from __future__ import annotations

import pytest

from src.engine.scorer import score
from src.ingestion.parser import parse_job
from src.schemas import (
    CandidateProfile,
    Seniority,
    Signals,
    Verdict,
)
from src.signals.experience import compute_experience_match
from src.signals.semantic import MockEmbeddingProvider, compute_semantic_similarity
from src.signals.skills import compute_skills_match


# ── Shared fixtures ──────────────────────────────────────────────────────────


STRONG_JD = """Title: Senior ML Engineer
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


WEAK_JD_MISMATCH = """Title: Senior Rust Systems Engineer
Company: LowLevel Inc
Location: Austin, TX

Required:
- 10+ years of Rust and C++
- Kernel-level programming
- Lock-free data structures

On-site only.
"""


def _marwa_profile() -> CandidateProfile:
    """A stable candidate profile used across the integration tests.

    Representative of the real user of the engine — ML engineer with Python
    / PyTorch / AWS + MLOps background, 5.5 years of experience, senior.
    """
    return CandidateProfile(
        profile_version="v1.0",
        name="Marwa",
        summary=(
            "Senior ML engineer with 5+ years of experience building "
            "end-to-end ML systems in Python. Comfortable with PyTorch, "
            "XGBoost, FastAPI, AWS, Docker, and MLOps."
        ),
        years_experience=5.5,
        seniority=Seniority.SENIOR,
        skills_tech=["python", "sql"],
        skills_tools=["pytorch", "xgboost", "aws", "docker", "mlops", "fastapi"],
        skills_domain=["mlops", "llm"],
    )


def _build_signals(
    job_raw: str,
    profile: CandidateProfile,
    *,
    llm_confidence: float,
    role_level_fit: float,
) -> Signals:
    """Compose the full Signals object from the raw JD.

    `llm_confidence` and `role_level_fit` are parameters — the LLM layer
    lands in Step 4 and the role-level matcher is part of the Step 4
    orchestrator. Until then, this integration test passes them as
    explicit inputs so the test is hermetic.
    """
    job = parse_job(job_raw)
    skills = compute_skills_match(job.parsed, profile)
    exp = compute_experience_match(job.parsed, profile)
    sem = compute_semantic_similarity(
        job.parsed, profile, provider=MockEmbeddingProvider()
    )
    return Signals(
        skills_match=skills,
        experience_match=exp,
        semantic_similarity=sem,
        llm_confidence=llm_confidence,
        role_level_fit=role_level_fit,
        parse_confidence=job.parse_confidence,
    )


# ── Happy path: strong candidate, matching JD ────────────────────────────────


class TestStrongMatchEndToEnd:
    def test_pipeline_produces_apply_or_priority_verdict(self):
        """A strong-match JD + matching profile must not SKIP."""
        signals = _build_signals(
            STRONG_JD, _marwa_profile(),
            llm_confidence=0.85,  # simulated Step-4 LLM output
            role_level_fit=1.0,   # senior role, senior candidate
        )
        result = score(signals)
        assert result.verdict in {Verdict.APPLY, Verdict.PRIORITY}, (
            f"expected APPLY or PRIORITY, got {result.verdict} "
            f"(score={result.apply_score:.2f}, signals={signals.model_dump()})"
        )

    def test_skills_signal_is_high_for_strong_match(self):
        signals = _build_signals(
            STRONG_JD, _marwa_profile(),
            llm_confidence=0.5, role_level_fit=1.0,
        )
        # Profile has python, pytorch, aws, mlops, docker — JD required has
        # python, pytorch, aws, mlops, kubernetes, docker → 5/6 match.
        # Preferred has langchain, llm → profile has llm (1/2).
        # numerator: 5 + 0.5 = 5.5. denominator: 6 + 1 = 7. = 0.785...
        assert signals.skills_match > 0.6

    def test_parse_confidence_passes_hard_filter(self):
        """Strong structured JD must not trigger the low-parse-confidence
        REVIEW filter."""
        signals = _build_signals(
            STRONG_JD, _marwa_profile(),
            llm_confidence=0.5, role_level_fit=1.0,
        )
        assert signals.parse_confidence >= 0.5


# ── Negative path: bad-fit JD ────────────────────────────────────────────────


class TestWeakMatchEndToEnd:
    def test_pipeline_does_not_force_apply_on_mismatch(self):
        """A 10-year Rust JD vs a 5-year Python profile should not produce
        a strong APPLY verdict."""
        signals = _build_signals(
            WEAK_JD_MISMATCH, _marwa_profile(),
            llm_confidence=0.2,
            role_level_fit=0.5,
        )
        result = score(signals)
        assert result.verdict in {Verdict.SKIP, Verdict.REVIEW}, (
            f"mismatch JD should not APPLY; got {result.verdict} "
            f"(score={result.apply_score:.2f})"
        )

    def test_experience_signal_low_when_underqualified(self):
        """5.5 years of experience vs 10 required → signal < 1.0."""
        signals = _build_signals(
            WEAK_JD_MISMATCH, _marwa_profile(),
            llm_confidence=0.5, role_level_fit=1.0,
        )
        assert signals.experience_match < 1.0
        assert signals.experience_match == pytest.approx(5.5 / 10.0, abs=1e-9)


# ── Reproducibility: Step 2 + Step 3 together ────────────────────────────────


class TestPipelineReproducibility:
    def test_same_input_produces_same_result(self):
        """Architecture Phase 10: consistent decisions on consistent inputs."""
        profile = _marwa_profile()
        r1 = score(_build_signals(STRONG_JD, profile, llm_confidence=0.7, role_level_fit=1.0))
        r2 = score(_build_signals(STRONG_JD, profile, llm_confidence=0.7, role_level_fit=1.0))
        assert r1.apply_score == r2.apply_score
        assert r1.verdict == r2.verdict
        assert r1.decision_trace.model_dump() == r2.decision_trace.model_dump()

    def test_decision_carries_weights_and_version_stamps(self):
        """Step 3 output → Step 2 scorer → DecisionResult stamped with
        weights + thresholds_version + engine_version.

        This is the Phase-10 reproducibility contract in action: a decision
        read back from Mongo (Step 4) can be re-scored and compared against
        current config. If the stamps don't match, the decision is from a
        different engine version and the comparison is invalid.
        """
        from src.config import ENGINE_VERSION, THRESHOLDS_VERSION, WEIGHTS

        signals = _build_signals(
            STRONG_JD, _marwa_profile(),
            llm_confidence=0.7, role_level_fit=1.0,
        )
        result = score(signals)
        assert result.weights == WEIGHTS
        assert result.thresholds_version == THRESHOLDS_VERSION
        assert result.engine_version == ENGINE_VERSION
