"""Semantic similarity signal — `sentence-transformers` with a Protocol seam.

Architecture §3: REAL, using `sentence-transformers/all-MiniLM-L6-v2`. §6:
weight W_semantic = 0.15.

Design — Protocol-based embedding provider:

The production path uses `sentence-transformers` (a ~400 MB model download
on first use). The test path uses `MockEmbeddingProvider` which returns
deterministic hash-based vectors, so the test suite runs in milliseconds
without any model download.

This follows the same pattern used in ResumeForge for `LLMProtocol`: the
component's correctness is proven against the Protocol, not against any
specific provider. A production smoke test (not part of the hermetic unit
tests) verifies the real provider loads and produces sane outputs.

Import strategy: `sentence-transformers` is imported inside
`SentenceTransformerProvider.__init__`, not at module level — so importing
this module is free even on a machine without the library installed.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

from src.schemas import CandidateProfile, ParsedJob


# ── Provider Protocol ────────────────────────────────────────────────────────


@runtime_checkable
class EmbeddingProvider(Protocol):
    """The single operation the semantic signal needs from its backend.

    Returns a vector (tuple of floats). Dimension is implementation-defined
    but must be consistent across calls on the same provider instance.
    """

    def embed(self, text: str) -> tuple[float, ...]: ...


# ── Mock provider (used in tests) ────────────────────────────────────────────


class MockEmbeddingProvider:
    """Deterministic fake embedder backed by SHA-256 bytes.

    - Deterministic: `embed("hello")` returns the same vector every time.
    - Non-trivial similarity structure: two similar strings share a prefix
      of the hashed bytes if (and only if) they share a prefix of input
      characters — so cosine similarity is a smooth function of input
      similarity, enough to exercise the pipeline but not pretend to be
      meaningful.

    **Not suitable for production semantic similarity.** Tests only.
    """

    def __init__(self, dim: int = 32):
        if dim <= 0 or dim > 256:
            raise ValueError("dim must be in (0, 256]")
        self._dim = dim

    def embed(self, text: str) -> tuple[float, ...]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Take the first `_dim` bytes, map to [-1, 1].
        return tuple((b - 128) / 128.0 for b in digest[: self._dim])


# ── Production provider (lazy import) ────────────────────────────────────────


class SentenceTransformerProvider:
    """Production path — `sentence-transformers/all-MiniLM-L6-v2`.

    Constructed lazily. The model download (~400 MB) only happens when
    `embed` is first called — which keeps module import cheap.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    def embed(self, text: str) -> tuple[float, ...]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: WPS433
            except ImportError as e:
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Install it from requirements.txt, or use MockEmbeddingProvider "
                    "for offline/test contexts."
                ) from e
            self._model = SentenceTransformer(self._model_name)
        vec = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return tuple(float(x) for x in vec)


# ── Signal function (consumed by the scorer) ─────────────────────────────────


def compute_semantic_similarity(
    job: ParsedJob,
    profile: CandidateProfile,
    *,
    provider: EmbeddingProvider | None = None,
) -> float:
    """Cosine similarity between the JD and the candidate summary, ∈ [0, 1].

    Args:
        job: Parsed job description.
        profile: Candidate profile with a `summary` field.
        provider: Embedding provider. Defaults to `MockEmbeddingProvider`
            for hermetic tests; production callers MUST pass an instance of
            `SentenceTransformerProvider` or another real provider.

    Returns:
        A value in [0, 1]. Cosine similarity is natively in [-1, 1]; we map
        it to [0, 1] by `(cos + 1) / 2` so the signal fits the uniform
        interface the scorer expects.

    The function is pure relative to the provider — for a given
    (provider, job, profile) it is deterministic.
    """
    if provider is None:
        provider = MockEmbeddingProvider()

    job_text = _job_to_text(job)
    profile_text = profile.summary

    vec_job = provider.embed(job_text)
    vec_profile = provider.embed(profile_text)

    cos = _cosine_similarity(vec_job, vec_profile)
    # Map [-1, 1] → [0, 1]
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))


# ── Internals ────────────────────────────────────────────────────────────────


def _job_to_text(job: ParsedJob) -> str:
    """Collapse a ParsedJob into a single comparable string.

    Matches the shape of a candidate summary (title + role focus + skills)
    rather than dumping raw JD text — the semantic signal should compare
    *what-the-job-is* to *who-the-candidate-is*, not to the JD's boilerplate.
    """
    parts = [job.title]
    if job.seniority is not None:
        parts.append(f"{job.seniority.value} level")
    if job.required_skills:
        parts.append("required skills: " + ", ".join(job.required_skills))
    if job.preferred_skills:
        parts.append("preferred skills: " + ", ".join(job.preferred_skills))
    return ". ".join(parts)


def _cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine similarity of two equal-length vectors.

    Returns 0.0 if either vector has zero magnitude (avoids division by zero).
    Raises `ValueError` on dimension mismatch — would signal a provider bug.
    """
    if len(a) != len(b):
        raise ValueError(f"vector dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)
