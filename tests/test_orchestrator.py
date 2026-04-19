"""End-to-end orchestrator tests.

The orchestrator is the seam between the deterministic core and the
production dependencies (store, LLM, embeddings). These tests wire it
with in-memory / mock implementations so the full flow is exercised
without any network or external service.

Key invariants tested:

1. Happy path: returns a DecisionResult with populated reasoning.
2. LLM-failure fallback: reasoning=None, llm_confidence=0.0, decision ships.
3. LLM cannot override the verdict — even a `llm_confidence=1.0` MockReasoner
   cannot flip a SKIP to APPLY on hard-filter input (dealbreaker).
4. Persistence: one job + one decision per call (jobs deduped by hash).
5. Role-level fit: discrete {0, 0.5, 1.0}.
"""

from __future__ import annotations

import pytest

from src.db import InMemoryStore
from src.engine.orchestrator import compute_role_level_fit, evaluate_job
from src.llm.reasoning import FailingReasoner, MockReasoner
from src.schemas import (
    CandidateProfile,
    ParsedJob,
    Seniority,
    Verdict,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


STRONG_JD = """Title: Senior ML Engineer
Company: Acme
Location: Remote

5+ years of Python, PyTorch, AWS experience required. MLOps work.

Nice to have: LangChain, LLM applications.

Salary: $150k-$220k
"""


def _marwa() -> CandidateProfile:
    return CandidateProfile(
        profile_version="v1.0",
        name="Marwa",
        summary="Senior ML engineer with 5+ years of Python, PyTorch, AWS",
        years_experience=5.5,
        seniority=Seniority.SENIOR,
        skills_tech=["python", "sql"],
        skills_tools=["pytorch", "aws", "docker", "mlops"],
        skills_domain=["mlops", "llm"],
    )


# ── Happy path ───────────────────────────────────────────────────────────────


class TestHappyPath:
    def test_strong_match_produces_apply_or_priority(self):
        store = InMemoryStore()
        d = evaluate_job(
            STRONG_JD, _marwa(),
            store=store, reasoner=MockReasoner(llm_confidence=0.8),
        )
        assert d.verdict in {Verdict.APPLY, Verdict.PRIORITY}

    def test_reasoning_populated_on_success(self):
        store = InMemoryStore()
        d = evaluate_job(
            STRONG_JD, _marwa(),
            store=store, reasoner=MockReasoner(llm_confidence=0.8),
        )
        assert d.reasoning is not None
        assert "strengths" in d.reasoning
        assert "llm_confidence" in d.reasoning

    def test_llm_confidence_reflected_in_signals(self):
        store = InMemoryStore()
        d = evaluate_job(
            STRONG_JD, _marwa(),
            store=store, reasoner=MockReasoner(llm_confidence=0.7),
        )
        assert d.signals.llm_confidence == pytest.approx(0.7)

    def test_persists_one_job_and_one_decision(self):
        store = InMemoryStore()
        evaluate_job(STRONG_JD, _marwa(), store=store, reasoner=MockReasoner())
        assert store.count("jobs") == 1
        assert store.count("decisions") == 1


# ── LLM failure fallback (the architecture §7 contract) ──────────────────────


class TestLLMFailureFallback:
    def test_reasoning_is_none_when_llm_fails(self):
        store = InMemoryStore()
        d = evaluate_job(
            STRONG_JD, _marwa(),
            store=store, reasoner=FailingReasoner(),
        )
        assert d.reasoning is None

    def test_llm_confidence_is_zero_when_llm_fails(self):
        store = InMemoryStore()
        d = evaluate_job(
            STRONG_JD, _marwa(),
            store=store, reasoner=FailingReasoner(),
        )
        assert d.signals.llm_confidence == 0.0

    def test_decision_still_persisted_when_llm_fails(self):
        """Architecture §7: the decision ships even without the LLM."""
        store = InMemoryStore()
        evaluate_job(
            STRONG_JD, _marwa(),
            store=store, reasoner=FailingReasoner(),
        )
        assert store.count("decisions") == 1

    def test_score_with_and_without_llm_differs_by_at_most_25(self):
        """LLM weight is capped at 0.25. Removing the LLM contribution
        should drop the score by at most 100 * 0.25 = 25 points."""
        store = InMemoryStore()
        d_with = evaluate_job(
            STRONG_JD, _marwa(),
            store=store, reasoner=MockReasoner(llm_confidence=1.0),
        )
        d_without = evaluate_job(
            STRONG_JD, _marwa(),
            store=store, reasoner=FailingReasoner(),
        )
        assert d_with.apply_score - d_without.apply_score <= 25.0 + 1e-9


# ── LLM cannot override the verdict ──────────────────────────────────────────


class TestLLMCannotOverrideVerdict:
    def test_dealbreaker_forces_skip_regardless_of_llm(self):
        """Even with an overconfident LLM, a dealbreaker must SKIP.

        Architecture §6: hard filters fire BEFORE the weighted sum. The
        LLM's `llm_confidence` is never evaluated in the hard-filter path.
        """
        profile = _marwa().model_copy(
            update={"dealbreakers": ["on_site_only"]}
        )
        on_site_jd = """Title: Senior ML Engineer
On-site only in NYC. 5+ years Python, PyTorch.
"""
        store = InMemoryStore()
        # Even an LLM returning confidence=1.0 cannot flip the dealbreaker SKIP.
        d = evaluate_job(
            on_site_jd, profile,
            store=store, reasoner=MockReasoner(llm_confidence=1.0),
        )
        assert d.verdict == Verdict.SKIP
        assert d.signals.dealbreaker_hit is True

    def test_low_parse_confidence_forces_review(self):
        """Empty JD → parse_confidence=0 → REVIEW. LLM can't rescue it."""
        store = InMemoryStore()
        d = evaluate_job(
            "",  # empty → parse_confidence=0
            _marwa(),
            store=store, reasoner=MockReasoner(llm_confidence=1.0),
        )
        assert d.verdict == Verdict.REVIEW


# ── compute_role_level_fit ───────────────────────────────────────────────────


class TestRoleLevelFit:
    @pytest.mark.parametrize("job_sen,cand_sen,expected", [
        (Seniority.SENIOR, Seniority.SENIOR, 1.0),
        (Seniority.STAFF, Seniority.SENIOR, 0.5),
        (Seniority.SENIOR, Seniority.STAFF, 0.5),
        (Seniority.MID, Seniority.SENIOR, 0.5),
        (Seniority.JUNIOR, Seniority.SENIOR, 0.0),
        (Seniority.PRINCIPAL, Seniority.MID, 0.0),
        (Seniority.SENIOR, Seniority.PRINCIPAL, 0.0),
    ])
    def test_discrete_level_fit(self, job_sen, cand_sen, expected):
        job = ParsedJob(title="X", seniority=job_sen)
        profile = _marwa().model_copy(update={"seniority": cand_sen})
        assert compute_role_level_fit(job, profile) == expected

    def test_no_job_seniority_returns_one(self):
        job = ParsedJob(title="X", seniority=None)
        assert compute_role_level_fit(job, _marwa()) == 1.0


# ── Reproducibility stamps ───────────────────────────────────────────────────


class TestReproducibilityStamps:
    def test_decision_carries_version_stamps(self):
        """The decision written to Mongo carries the exact config it was
        scored against (architecture §5.3 + Phase 10 contract)."""
        from src.config import ENGINE_VERSION, THRESHOLDS_VERSION, WEIGHTS

        store = InMemoryStore()
        d = evaluate_job(STRONG_JD, _marwa(), store=store, reasoner=MockReasoner())
        assert d.engine_version == ENGINE_VERSION
        assert d.thresholds_version == THRESHOLDS_VERSION
        assert d.weights == WEIGHTS
