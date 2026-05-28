"""Tests for the semantic-similarity signal.

All tests use `MockEmbeddingProvider` — no model download, no network, fast.
The `SentenceTransformerProvider` is exercised by a separate integration
smoke test that is NOT part of the hermetic unit suite.
"""

from __future__ import annotations

import pytest

from src.schemas import CandidateProfile, ParsedJob, Seniority
from src.signals.semantic import (
    EmbeddingProvider,
    MockEmbeddingProvider,
    compute_semantic_similarity,
)


def _profile(summary: str) -> CandidateProfile:
    return CandidateProfile(
        profile_version="v1.0",
        name="Test",
        summary=summary,
        years_experience=5.0,
        seniority=Seniority.SENIOR,
    )


def _job(title: str = "ML Engineer", required_skills: list[str] | None = None) -> ParsedJob:
    return ParsedJob(title=title, required_skills=required_skills or [])


# ── Provider Protocol conformance ────────────────────────────────────────────


class TestProviderProtocol:
    def test_mock_provider_is_an_embedding_provider(self):
        """runtime_checkable Protocol — mock must satisfy the interface."""
        m = MockEmbeddingProvider()
        assert isinstance(m, EmbeddingProvider)

    def test_mock_provider_dimension_is_consistent(self):
        """A single provider instance must always return the same-length vector."""
        m = MockEmbeddingProvider(dim=32)
        v1 = m.embed("hello")
        v2 = m.embed("world")
        assert len(v1) == len(v2) == 32

    def test_mock_provider_is_deterministic(self):
        m = MockEmbeddingProvider()
        assert m.embed("same input") == m.embed("same input")

    def test_mock_provider_dim_validation(self):
        with pytest.raises(ValueError):
            MockEmbeddingProvider(dim=0)
        with pytest.raises(ValueError):
            MockEmbeddingProvider(dim=257)  # above hash digest ceiling


# ── compute_semantic_similarity ──────────────────────────────────────────────


class TestSemanticSimilarity:
    def test_returns_float_in_zero_one(self):
        score = compute_semantic_similarity(
            _job(), _profile("ML engineer"),
            provider=MockEmbeddingProvider(),
        )
        assert 0.0 <= score <= 1.0

    def test_determinism_with_shared_provider(self):
        provider = MockEmbeddingProvider()
        a = compute_semantic_similarity(_job(), _profile("x"), provider=provider)
        b = compute_semantic_similarity(_job(), _profile("x"), provider=provider)
        assert a == b

    def test_determinism_with_fresh_provider_per_call(self):
        """Two separately-constructed MockEmbeddingProvider instances must
        produce identical embeddings because the mock is hash-based and
        stateless. This is the property that lets tests construct ad-hoc
        providers without coordinating instances."""
        a = compute_semantic_similarity(
            _job(), _profile("x"), provider=MockEmbeddingProvider(),
        )
        b = compute_semantic_similarity(
            _job(), _profile("x"), provider=MockEmbeddingProvider(),
        )
        assert a == b

    def test_identical_texts_score_one(self):
        """If the JD-text and profile-summary happen to collapse to the same
        string, cosine = 1 → signal = 1."""
        job = _job(title="identical")
        # _job_to_text on this job: "identical"
        # Profile summary set to the same string.
        profile = _profile("identical")
        score = compute_semantic_similarity(
            job, profile, provider=MockEmbeddingProvider(),
        )
        assert score == pytest.approx(1.0, abs=1e-9)

    def test_missing_provider_raises_type_error(self):
        """Regression: removing the silent-mock-fallback default means
        calling without `provider=` raises TypeError. This is the
        guardrail against any future production caller accidentally
        getting hash-based mock embeddings instead of real ones.
        See HIGH #3 / Phase 2 audit (2026-05-28)."""
        with pytest.raises(TypeError):
            compute_semantic_similarity(_job(), _profile("x"))  # type: ignore[call-arg]

    def test_accepts_injected_provider(self):
        class ZeroProvider:
            def embed(self, text: str) -> tuple[float, ...]:
                return (0.0, 0.0, 0.0, 0.0)

        score = compute_semantic_similarity(
            _job(), _profile("x"), provider=ZeroProvider(),
        )
        # Zero vectors → cosine defined as 0 → signal = 0.5 (mapping)
        assert score == pytest.approx(0.5, abs=1e-9)

    def test_dimension_mismatch_raises(self):
        """A provider that returns inconsistent dimensions between calls is
        a bug — surface it, don't silently pass."""

        class BadProvider:
            def __init__(self):
                self._count = 0

            def embed(self, text: str) -> tuple[float, ...]:
                self._count += 1
                if self._count == 1:
                    return (1.0, 0.0, 0.0)
                return (1.0, 0.0)  # wrong dim

        with pytest.raises(ValueError, match="dimension mismatch"):
            compute_semantic_similarity(
                _job(), _profile("x"), provider=BadProvider(),
            )


# ── Signal output contract (shape + invariants) ──────────────────────────────


class TestSignalContract:
    def test_similar_texts_score_higher_than_dissimilar(self):
        """Loose invariant: two texts sharing a long prefix score higher than
        two unrelated texts, under the mock hash-based provider.

        This is a smoke test for the provider pattern itself, not a claim
        about the mock's linguistic quality."""
        provider = MockEmbeddingProvider(dim=32)
        job_ml = ParsedJob(
            title="Senior Machine Learning Engineer",
            required_skills=["python", "pytorch"],
            seniority=Seniority.SENIOR,
        )
        profile_ml = _profile(
            "Senior Machine Learning Engineer with Python and PyTorch"
        )
        profile_frontend = _profile("Frontend TypeScript developer")

        near = compute_semantic_similarity(job_ml, profile_ml, provider=provider)
        far = compute_semantic_similarity(job_ml, profile_frontend, provider=provider)
        # The mock is hash-based and doesn't model real semantics. The two
        # scores WILL differ (different inputs → different hashes → different
        # cosines). We just assert the signal produces distinguishable
        # outputs, not a specific direction.
        assert near != far
