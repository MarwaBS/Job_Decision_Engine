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
from src.signals.semantic import MockEmbeddingProvider


def _mock_embeddings() -> MockEmbeddingProvider:
    """Fresh deterministic mock embedder for one test call.

    `compute_semantic_similarity` requires an explicit provider — there is
    no default. Tests construct their own here; the production Streamlit
    app constructs `SentenceTransformerProvider()` at boot.
    """
    return MockEmbeddingProvider()


# ── Fixtures ─────────────────────────────────────────────────────────────────


STRONG_JD = """Title: Senior ML Engineer
Company: Acme
Location: Remote

5+ years of Python, PyTorch, AWS experience required. MLOps work.

Nice to have: LangChain, LLM applications.

Salary: $150k-$220k
"""


def _alex_rivera() -> CandidateProfile:
    return CandidateProfile(
        profile_version="v1.0",
        name="Alex Rivera",
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
            STRONG_JD,
            _alex_rivera(),
            store=store,
            reasoner=MockReasoner(llm_confidence=0.8),
            embedding_provider=_mock_embeddings(),
        )
        assert d.verdict in {Verdict.APPLY, Verdict.PRIORITY}

    def test_reasoning_populated_on_success(self):
        store = InMemoryStore()
        d = evaluate_job(
            STRONG_JD,
            _alex_rivera(),
            store=store,
            reasoner=MockReasoner(llm_confidence=0.8),
            embedding_provider=_mock_embeddings(),
        )
        assert d.reasoning is not None
        assert "strengths" in d.reasoning
        assert "llm_confidence" in d.reasoning

    def test_llm_confidence_reflected_in_signals(self):
        store = InMemoryStore()
        d = evaluate_job(
            STRONG_JD,
            _alex_rivera(),
            store=store,
            reasoner=MockReasoner(llm_confidence=0.7),
            embedding_provider=_mock_embeddings(),
        )
        assert d.signals.llm_confidence == pytest.approx(0.7)

    def test_persists_one_job_and_one_decision(self):
        store = InMemoryStore()
        evaluate_job(
            STRONG_JD,
            _alex_rivera(),
            store=store,
            reasoner=MockReasoner(),
            embedding_provider=_mock_embeddings(),
        )
        assert store.count("jobs") == 1
        assert store.count("decisions") == 1


# ── LLM failure fallback (the architecture §7 contract) ──────────────────────


class TestLLMFailureFallback:
    def test_reasoning_is_none_when_llm_fails(self):
        store = InMemoryStore()
        d = evaluate_job(
            STRONG_JD,
            _alex_rivera(),
            store=store,
            reasoner=FailingReasoner(),
            embedding_provider=_mock_embeddings(),
        )
        assert d.reasoning is None

    def test_llm_confidence_is_zero_when_llm_fails(self):
        store = InMemoryStore()
        d = evaluate_job(
            STRONG_JD,
            _alex_rivera(),
            store=store,
            reasoner=FailingReasoner(),
            embedding_provider=_mock_embeddings(),
        )
        assert d.signals.llm_confidence == 0.0

    def test_decision_still_persisted_when_llm_fails(self):
        """Architecture §7: the decision ships even without the LLM."""
        store = InMemoryStore()
        evaluate_job(
            STRONG_JD,
            _alex_rivera(),
            store=store,
            reasoner=FailingReasoner(),
            embedding_provider=_mock_embeddings(),
        )
        assert store.count("decisions") == 1

    def test_score_with_and_without_llm_differs_by_at_most_25(self):
        """LLM weight is capped at 0.25. Removing the LLM contribution
        should drop the score by at most 100 * 0.25 = 25 points."""
        store = InMemoryStore()
        d_with = evaluate_job(
            STRONG_JD,
            _alex_rivera(),
            store=store,
            reasoner=MockReasoner(llm_confidence=1.0),
            embedding_provider=_mock_embeddings(),
        )
        d_without = evaluate_job(
            STRONG_JD,
            _alex_rivera(),
            store=store,
            reasoner=FailingReasoner(),
            embedding_provider=_mock_embeddings(),
        )
        assert d_with.apply_score - d_without.apply_score <= 25.0 + 1e-9


# ── LLM cannot override the verdict ──────────────────────────────────────────


class TestLLMCannotOverrideVerdict:
    def test_dealbreaker_forces_skip_regardless_of_llm(self):
        """Even with an overconfident LLM, a dealbreaker must SKIP.

        Architecture §6: hard filters fire BEFORE the weighted sum. The
        LLM's `llm_confidence` is never evaluated in the hard-filter path.
        """
        profile = _alex_rivera().model_copy(update={"dealbreakers": ["on_site_only"]})
        on_site_jd = """Title: Senior ML Engineer
On-site only in NYC. 5+ years Python, PyTorch.
"""
        store = InMemoryStore()
        # Even an LLM returning confidence=1.0 cannot flip the dealbreaker SKIP.
        d = evaluate_job(
            on_site_jd,
            profile,
            store=store,
            reasoner=MockReasoner(llm_confidence=1.0),
            embedding_provider=_mock_embeddings(),
        )
        assert d.verdict == Verdict.SKIP
        assert d.signals.dealbreaker_hit is True

    def test_low_parse_confidence_forces_parse_failure(self):
        """Empty JD → parse_confidence=0 → PARSE_FAILURE. LLM can't rescue it.

        BUG-004: this path used to return REVIEW with apply_score=0.0, which
        users misread as "0% match". It now returns the PARSE_FAILURE input-
        quality verdict with apply_score=None.
        """
        store = InMemoryStore()
        d = evaluate_job(
            "",  # empty → parse_confidence=0
            _alex_rivera(),
            store=store,
            reasoner=MockReasoner(llm_confidence=1.0),
            embedding_provider=_mock_embeddings(),
        )
        assert d.verdict == Verdict.PARSE_FAILURE
        assert d.apply_score is None


# ── Dealbreaker semantics (missing data must not fire hard filters) ──────────


class TestDealbreakerSemantics:
    WORKPLACE_SILENT_JD = """Title: Senior ML Engineer
Company: Initech
Location: New York, NY
5+ years of experience with Python, PyTorch, AWS.
"""

    def test_on_site_only_does_not_fire_when_jd_is_silent_on_workplace(self):
        """Regression guard: `remote` defaulting to False made this
        dealbreaker hard-SKIP every JD that simply never mentioned
        workplace. Absence of evidence is not evidence of on-site — the
        dealbreaker fires only on an explicit on-site mention
        (job.remote is False), mirroring the "don't penalise missing
        data" rule of the experience and role-level signals."""
        profile = _alex_rivera().model_copy(update={"dealbreakers": ["on_site_only"]})
        d = evaluate_job(
            self.WORKPLACE_SILENT_JD,
            profile,
            store=InMemoryStore(),
            reasoner=MockReasoner(),
            embedding_provider=_mock_embeddings(),
        )
        assert d.signals.dealbreaker_hit is False
        assert d.verdict != Verdict.SKIP

    def test_no_pytorch_fires_when_jd_requires_pytorch(self):
        profile = _alex_rivera().model_copy(update={"dealbreakers": ["no_pytorch"]})
        d = evaluate_job(
            self.WORKPLACE_SILENT_JD,  # requires pytorch
            profile,
            store=InMemoryStore(),
            reasoner=MockReasoner(),
            embedding_provider=_mock_embeddings(),
        )
        assert d.signals.dealbreaker_hit is True
        assert d.verdict == Verdict.SKIP

    def test_no_pytorch_does_not_fire_without_pytorch(self):
        profile = _alex_rivera().model_copy(update={"dealbreakers": ["no_pytorch"]})
        jd = """Title: Senior Data Engineer
Company: Initech
Location: Remote
5+ years of experience with Python, SQL, Airflow.
"""
        d = evaluate_job(
            jd,
            profile,
            store=InMemoryStore(),
            reasoner=MockReasoner(),
            embedding_provider=_mock_embeddings(),
        )
        assert d.signals.dealbreaker_hit is False


# ── compute_role_level_fit ───────────────────────────────────────────────────


class TestRoleLevelFit:
    @pytest.mark.parametrize(
        "job_sen,cand_sen,expected",
        [
            (Seniority.SENIOR, Seniority.SENIOR, 1.0),
            (Seniority.STAFF, Seniority.SENIOR, 0.5),
            (Seniority.SENIOR, Seniority.STAFF, 0.5),
            (Seniority.MID, Seniority.SENIOR, 0.5),
            (Seniority.JUNIOR, Seniority.SENIOR, 0.0),
            (Seniority.PRINCIPAL, Seniority.MID, 0.0),
            (Seniority.SENIOR, Seniority.PRINCIPAL, 0.0),
        ],
    )
    def test_discrete_level_fit(self, job_sen, cand_sen, expected):
        job = ParsedJob(title="X", seniority=job_sen)
        profile = _alex_rivera().model_copy(update={"seniority": cand_sen})
        assert compute_role_level_fit(job, profile) == expected

    def test_no_job_seniority_returns_one(self):
        job = ParsedJob(title="X", seniority=None)
        assert compute_role_level_fit(job, _alex_rivera()) == 1.0


# ── Reproducibility stamps ───────────────────────────────────────────────────


class TestReproducibilityStamps:
    def test_decision_carries_version_stamps(self):
        """The decision written to Mongo carries the exact config it was
        scored against (architecture §5.3 + Phase 10 contract)."""
        from src.config import ENGINE_VERSION, THRESHOLDS_VERSION, WEIGHTS

        store = InMemoryStore()
        d = evaluate_job(
            STRONG_JD,
            _alex_rivera(),
            store=store,
            reasoner=MockReasoner(),
            embedding_provider=_mock_embeddings(),
        )
        assert d.engine_version == ENGINE_VERSION
        assert d.thresholds_version == THRESHOLDS_VERSION
        assert d.weights == WEIGHTS


# ── Silent-mock-fallback guardrail ───────────────────────────────────────────


class TestEmbeddingProviderRequired:
    """Regression: HIGH #3 from Phase 2 audit (2026-05-28).

    Before this fix, both `compute_semantic_similarity` and
    `evaluate_job` defaulted `embedding_provider=None` and silently
    substituted `MockEmbeddingProvider()` — so a caller that forgot to
    inject the real `SentenceTransformerProvider` would get hash-based
    mock embeddings in production and score wrong without any warning.

    These tests lock in the new contract: both functions require
    explicit injection. Removing the keyword should fail loudly with
    TypeError, never silently fall back to mock.
    """

    def test_evaluate_job_raises_when_embedding_provider_omitted(self):
        store = InMemoryStore()
        with pytest.raises(TypeError):
            evaluate_job(  # type: ignore[call-arg]
                STRONG_JD,
                _alex_rivera(),
                store=store,
                reasoner=MockReasoner(),
            )

    def test_orchestrator_source_has_no_silent_mock_fallback(self):
        """Source-level guard: no `or MockEmbeddingProvider` expression
        should re-introduce the silent fallback in a future refactor."""
        from pathlib import Path

        src = (
            Path(__file__).parent.parent / "src" / "engine" / "orchestrator.py"
        ).read_text(encoding="utf-8")
        assert "or MockEmbeddingProvider" not in src, (
            "src/engine/orchestrator.py re-introduced a silent "
            "MockEmbeddingProvider fallback. The embedding provider MUST "
            "be injected by the caller; there is no production-safe default."
        )
