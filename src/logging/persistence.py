"""Persistence layer — pure I/O on top of the `Store` Protocol.

Step 4 rules (authorization message, §4.2):
    "Pure I/O layer only. NO logic transformation here."

This module composes the primitive `Store` operations into the specific
write patterns the orchestrator needs:

    persist_decision(store, job, decision) -> decision_id
    persist_outcome_submitted(store, decision_id) -> outcome_id
    advance_outcome(store, decision_id, stage) -> None
    close_outcome(store, decision_id, final_stage) -> None
    persist_feedback(store, decision_id, feedback_type, ...) -> feedback_id

Every function here takes a `Store` and returns a primitive id or None. No
decisions are computed, no signals are transformed, no LLM is called. The
boundary is strict: if a function needs to *decide* something, it belongs
somewhere else.

Version tagging:

Every `decisions` write inherits the `weights`, `thresholds_version`, and
`engine_version` stamped on the `DecisionResult` by the scorer (Step 2).
This layer does not set them — it merely forwards. That's the Phase-10
reproducibility contract: a decision read back later can be re-scored
against its original config and compared.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from src.db import Store
from src.schemas import (
    DecisionResult,
    FeedbackLog,
    Job,
    Outcome,
    OutcomeStage,
    Verdict,
)

# ── Decision writes ──────────────────────────────────────────────────────────


def persist_decision(
    store: Store,
    job: Job,
    decision: DecisionResult,
) -> str:
    """Persist a decision alongside its source job.

    The job is upserted by `content_hash` (dedupes repeat submissions).
    The decision is strictly appended — multiple decisions against the same
    job (e.g., after a profile bump) each get their own doc.

    Returns the decision's id.
    """
    store.upsert_job(job)
    return store.insert_decision(decision)


# ── Outcome writes (state machine) ───────────────────────────────────────────


_TERMINAL_STAGES = {"OFFER", "REJECTED", "GHOSTED", "WITHDRAWN"}


def persist_outcome_submitted(store: Store, decision_id: str) -> str:
    """Create the outcome row for a newly-submitted application.

    One outcome doc per decision. The initial doc always has exactly one
    stage: SUBMITTED. Subsequent stages arrive via `advance_outcome`.
    """
    now = _utc_now()
    outcome = Outcome(
        decision_id=decision_id,
        submitted_at=now,
        stages=[OutcomeStage(stage="SUBMITTED", at=now)],
        final_stage=None,
    )
    return store.insert_outcome(outcome)


def advance_outcome(
    store: Store,
    decision_id: str,
    stage: Literal["CALLBACK", "INTERVIEW", "OFFER", "REJECTED", "GHOSTED", "WITHDRAWN"],
) -> None:
    """Append a new stage to an existing outcome and close it if terminal.

    The terminal stages (OFFER / REJECTED / GHOSTED / WITHDRAWN) automatically
    set `final_stage` on the document. Non-terminal stages (CALLBACK,
    INTERVIEW) only push.
    """
    now = _utc_now()
    store.push_outcome_stage(decision_id, OutcomeStage(stage=stage, at=now))
    if stage in _TERMINAL_STAGES:
        store.set_outcome_final_stage(decision_id, stage)


def close_outcome(
    store: Store,
    decision_id: str,
    final_stage: Literal["OFFER", "REJECTED", "GHOSTED", "WITHDRAWN"],
) -> None:
    """Explicitly close an outcome without pushing a new stage.

    Useful when the user knows the final verdict without a stage-by-stage
    history (e.g., retrospectively marking a ghosted application).
    """
    if final_stage not in _TERMINAL_STAGES:
        raise ValueError(
            f"final_stage must be terminal ({_TERMINAL_STAGES}); got {final_stage!r}"
        )
    store.set_outcome_final_stage(decision_id, final_stage)


# ── Feedback writes ──────────────────────────────────────────────────────────


def persist_feedback(
    store: Store,
    *,
    decision_id: str,
    feedback_type: Literal[
        "score_too_low", "score_too_high", "verdict_wrong", "reasoning_off"
    ],
    reason: str,
    expected_verdict: Verdict | None = None,
    actual_verdict: Verdict | None = None,
) -> str:
    """Append a user-authored correction note.

    v1: logged only — never consumed by the scorer. The feedback loop
    activates in v2 when N≥50 accumulate. See architecture §5.5.
    """
    feedback = FeedbackLog(
        decision_id=decision_id,
        feedback_type=feedback_type,
        reason=reason,
        expected_verdict=expected_verdict,
        actual_verdict=actual_verdict,
    )
    return store.insert_feedback(feedback)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "persist_decision",
    "persist_outcome_submitted",
    "advance_outcome",
    "close_outcome",
    "persist_feedback",
]
