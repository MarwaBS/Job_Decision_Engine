"""Orchestrator — glues parser + signals + LLM + scorer + persistence.

Architecture §4. Single public entrypoint:

    evaluate_job(raw_text, profile, *, store, reasoner, provider) -> DecisionResult

The orchestrator is the one function the UI (Step 5) and any future API
will call. It holds the dependency wiring and the order-of-operations, but
does NOT itself make decisions — the scorer does.

Order of operations:

    1. parse_job(raw_text)                          [pure, deterministic]
    2. compute_skills_match(job, profile)           [pure]
    3. compute_experience_match(job, profile)       [pure]
    4. compute_semantic_similarity(job, profile)    [provider-backed]
    5. compute_role_level_fit(job, profile)         [pure, here]
    6. reasoner.reason(job, profile, signals_draft) [may raise]
          ├─ success: get llm_confidence + reasoning
          └─ failure: llm_confidence = 0.0, reasoning = None
    7. assemble full Signals (with llm_confidence slotted)
    8. score(signals)                               [deterministic]
    9. attach reasoning dict to DecisionResult
    10. persist: upsert_job → insert_decision       [store]
    11. return decision

Step 6 happens BEFORE step 8 because the scorer needs `llm_confidence` as
an input. The LLM's own confidence contributes at most 25% to the weighted
sum (architecture §6 cap); it does not set the verdict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import ENGINE_VERSION
from src.db import Store
from src.engine.scorer import score
from src.ingestion.parser import parse_job
from src.llm.reasoning import LLMReasoner, LLMReasoningFailed
from src.logging.persistence import persist_decision
from src.schemas import (
    CandidateProfile,
    DecisionResult,
    ParsedJob,
    Seniority,
    Signals,
)
from src.signals.experience import compute_experience_match
from src.signals.semantic import (
    EmbeddingProvider,
    MockEmbeddingProvider,
    compute_semantic_similarity,
)
from src.signals.skills import compute_skills_match

if TYPE_CHECKING:
    from src.schemas import ReasoningOutput


# ── Role-level fit (kept here; pure function, not its own module) ────────────


_SENIORITY_RANK: dict[Seniority, int] = {
    Seniority.JUNIOR: 1,
    Seniority.MID: 2,
    Seniority.SENIOR: 3,
    Seniority.STAFF: 4,
    Seniority.PRINCIPAL: 5,
}


def compute_role_level_fit(job: ParsedJob, profile: CandidateProfile) -> float:
    """Role-level match ∈ {0.0, 0.5, 1.0}. Architecture §6.

    - 1.0: exact match (Senior candidate applying to Senior JD) or the JD
      is unlabelled (don't penalise — same rationale as experience match).
    - 0.5: one level apart (Senior → Staff; Mid → Senior).
    - 0.0: two or more levels apart.

    Why a discrete scale: the deterministic levels are coarse; a
    continuous score would imply a precision we don't have.
    """
    job_level = job.seniority
    if job_level is None:
        return 1.0
    job_rank = _SENIORITY_RANK[job_level]
    cand_rank = _SENIORITY_RANK[profile.seniority]
    diff = abs(job_rank - cand_rank)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.5
    return 0.0


# ── Orchestrator ─────────────────────────────────────────────────────────────


def evaluate_job(
    raw_text: str,
    profile: CandidateProfile,
    *,
    store: Store,
    reasoner: LLMReasoner,
    embedding_provider: EmbeddingProvider | None = None,
) -> DecisionResult:
    """End-to-end: raw JD text → scored, explained, persisted DecisionResult.

    Args:
        raw_text: Job description as pasted by the user.
        profile: The candidate profile (loaded by the UI layer from
            `store.get_active_profile()`).
        store: The persistence interface. Called exactly twice:
            `upsert_job` + `insert_decision`.
        reasoner: LLM reasoner. A `FailingReasoner` is valid — the
            orchestrator will catch and substitute the null-reasoning path.
        embedding_provider: Override for the semantic-similarity provider.
            Defaults to `MockEmbeddingProvider` — in production the caller
            SHOULD pass `SentenceTransformerProvider()`.

    Returns:
        A persisted `DecisionResult` with `reasoning` populated when the
        LLM succeeded, or `reasoning=None` + `llm_confidence=0.0` when it
        did not. The score + verdict are deterministic either way.
    """
    # ── Step 1: ingestion ──────────────────────────────────────────────────
    job = parse_job(raw_text)

    # ── Steps 2-5: deterministic signals ───────────────────────────────────
    skills_s = compute_skills_match(job.parsed, profile)
    exp_s = compute_experience_match(job.parsed, profile)
    sem_s = compute_semantic_similarity(
        job.parsed, profile,
        provider=embedding_provider or MockEmbeddingProvider(),
    )
    role_s = compute_role_level_fit(job.parsed, profile)

    # Hard-filter flags: a dealbreaker hit on the profile is detected here.
    # v1 uses a literal string-match against the profile's `dealbreakers`
    # list. Example: dealbreaker "requires_10_yr_exp" fires if
    # job.years_required >= 10.
    dealbreaker_hit = _check_dealbreakers(job.parsed, profile)

    # ── Step 6: LLM reasoning (bounded signal) ─────────────────────────────
    # Build a DRAFT Signals with llm_confidence=0 so we can pass the
    # deterministic context to the LLM. The LLM sees the four
    # deterministic signals, not its own (would create a feedback loop).
    draft_signals = Signals(
        skills_match=skills_s,
        experience_match=exp_s,
        semantic_similarity=sem_s,
        llm_confidence=0.0,  # placeholder; replaced below
        role_level_fit=role_s,
        dealbreaker_hit=dealbreaker_hit,
        parse_confidence=job.parse_confidence,
    )

    reasoning_obj: ReasoningOutput | None
    try:
        reasoning_obj = reasoner.reason(
            job=job.parsed, profile=profile, signals=draft_signals
        )
        llm_confidence = reasoning_obj.llm_confidence
    except LLMReasoningFailed:
        # Architecture §7 contract: scoring continues without the LLM
        # signal. Decision still ships.
        reasoning_obj = None
        llm_confidence = 0.0

    # ── Step 7: full Signals ───────────────────────────────────────────────
    signals = Signals(
        skills_match=skills_s,
        experience_match=exp_s,
        semantic_similarity=sem_s,
        llm_confidence=llm_confidence,
        role_level_fit=role_s,
        dealbreaker_hit=dealbreaker_hit,
        parse_confidence=job.parse_confidence,
    )

    # ── Step 8: deterministic scoring ──────────────────────────────────────
    decision = score(signals)

    # ── Step 9: attach reasoning ───────────────────────────────────────────
    if reasoning_obj is not None:
        decision = decision.model_copy(
            update={"reasoning": reasoning_obj.model_dump(mode="json")}
        )
    # else: decision.reasoning stays None (the schemas default)

    # Sanity: engine_version sanity check so a future upgrade doesn't
    # silently persist stale decisions.
    assert decision.engine_version == ENGINE_VERSION

    # ── Step 10: persist ───────────────────────────────────────────────────
    persist_decision(store, job, decision)

    return decision


# ── Dealbreaker detection ────────────────────────────────────────────────────


def _check_dealbreakers(
    job: ParsedJob, profile: CandidateProfile
) -> bool:
    """Return True if the JD violates any of the profile's dealbreakers.

    v1 dealbreaker vocabulary (extensible):
        "requires_10_yr_exp"  — fires when job.years_required >= 10
        "on_site_only"        — fires when job.remote is False
        "no_pytorch"          — placeholder; v1 doesn't match against the
                                 profile's own stack (that would flip the
                                 semantics). Kept in vocabulary for v2.

    Anything the function doesn't recognise is ignored (forward-compatible
    with future vocabulary expansion).
    """
    for item in profile.dealbreakers:
        key = item.strip().lower()
        if key == "requires_10_yr_exp":
            if job.years_required is not None and job.years_required >= 10:
                return True
        elif key == "on_site_only":
            if not job.remote:
                return True
    return False


__all__ = ["evaluate_job", "compute_role_level_fit"]
