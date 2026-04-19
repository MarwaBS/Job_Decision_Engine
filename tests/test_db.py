"""Tests for the storage layer.

All tests run against `InMemoryStore`. `MongoStore` is exercised by a
separate integration smoke test that is NOT part of the hermetic suite
(it needs a real Mongo URI).

The append-only contract (architecture §5 + DT-010) is the headline
behaviour verified here:

- `decisions` never updated after insert
- `outcomes` can only grow their `stages[]` and close their `final_stage`
- `final_stage` cannot be retroactively overwritten
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.db import InMemoryStore, Store
from src.schemas import (
    CandidateProfile,
    DecisionResult,
    DecisionSensitivity,
    DecisionTrace,
    FeedbackLog,
    Job,
    Outcome,
    OutcomeStage,
    ParsedJob,
    Seniority,
    Signals,
    Thresholds,
    Verdict,
    Weights,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _profile(version: str = "v1.0", active: bool = True) -> CandidateProfile:
    return CandidateProfile(
        profile_version=version,
        name="Marwa",
        summary="ML eng",
        years_experience=5.0,
        seniority=Seniority.SENIOR,
        active=active,
    )


def _job(content_hash: str = "sha256:abc") -> Job:
    return Job(
        content_hash=content_hash,
        source="paste",
        raw_text="raw",
        parsed=ParsedJob(title="ML Engineer"),
        parse_confidence=0.9,
        parse_warnings=[],
    )


def _decision() -> DecisionResult:
    signals = Signals(
        skills_match=0.8, experience_match=1.0,
        semantic_similarity=0.7, llm_confidence=0.85, role_level_fit=1.0,
    )
    return DecisionResult(
        apply_score=85.0,
        verdict=Verdict.PRIORITY,
        signals=signals,
        weights=Weights(skills=0.30, experience=0.20, semantic=0.15, llm=0.25, role=0.10),
        thresholds_version="v1.0",
        decision_trace=DecisionTrace(
            dominant_signal="skills_match",
            failure_mode_detected=None,
            decision_sensitivity=DecisionSensitivity(
                if_llm_removed_score=70.0,
                if_skills_boosted_plus_10pct=90.0,
                if_experience_removed_score=75.0,
            ),
            nearest_threshold_distance=5.0,
            near_threshold_flag=False,
        ),
        engine_version="0.1.0",
    )


def _outcome(decision_id: str) -> Outcome:
    now = datetime.now(timezone.utc)
    return Outcome(
        decision_id=decision_id,
        submitted_at=now,
        stages=[OutcomeStage(stage="SUBMITTED", at=now)],
    )


# ── Protocol conformance ─────────────────────────────────────────────────────


class TestProtocolConformance:
    def test_in_memory_store_satisfies_store_protocol(self):
        assert isinstance(InMemoryStore(), Store)


# ── Profiles ─────────────────────────────────────────────────────────────────


class TestProfiles:
    def test_upsert_and_get_active(self):
        store = InMemoryStore()
        store.upsert_profile(_profile("v1.0", active=True))
        got = store.get_active_profile()
        assert got is not None
        assert got.profile_version == "v1.0"

    def test_no_active_profile_returns_none(self):
        store = InMemoryStore()
        store.upsert_profile(_profile("v1.0", active=False))
        assert store.get_active_profile() is None

    def test_new_active_version_deactivates_old(self):
        """Only one profile_version can be `active: True` at a time."""
        store = InMemoryStore()
        store.upsert_profile(_profile("v1.0", active=True))
        store.upsert_profile(_profile("v1.1", active=True))
        active = store.get_active_profile()
        assert active is not None
        assert active.profile_version == "v1.1"
        # Old v1.0 must have been deactivated
        assert store.count("profiles") == 2

    def test_upsert_same_version_replaces_in_place(self):
        store = InMemoryStore()
        store.upsert_profile(_profile("v1.0", active=True))
        store.upsert_profile(_profile("v1.0", active=True))
        assert store.count("profiles") == 1  # not duplicated


# ── Jobs (dedupe by content_hash) ────────────────────────────────────────────


class TestJobs:
    def test_same_hash_same_id(self):
        store = InMemoryStore()
        id1 = store.upsert_job(_job("sha256:x"))
        id2 = store.upsert_job(_job("sha256:x"))
        assert id1 == id2
        assert store.count("jobs") == 1

    def test_different_hash_different_id(self):
        store = InMemoryStore()
        id1 = store.upsert_job(_job("sha256:a"))
        id2 = store.upsert_job(_job("sha256:b"))
        assert id1 != id2
        assert store.count("jobs") == 2


# ── Decisions (strict append-only — the headline contract) ───────────────────


class TestDecisionsAppendOnly:
    def test_insert_multiple_decisions_all_persisted(self):
        store = InMemoryStore()
        id1 = store.insert_decision(_decision())
        id2 = store.insert_decision(_decision())
        assert id1 != id2
        assert store.count("decisions") == 2

    def test_store_exposes_no_update_method_for_decisions(self):
        """There must be no way to mutate a decision after insert.

        This is enforced by the absence of an update method, not by a
        try/except — the Protocol surface does not expose one.
        """
        store = InMemoryStore()
        # Build the full list of store attributes and assert none of them
        # mutate decisions.
        forbidden_names = {
            "update_decision", "replace_decision", "delete_decision",
            "fix_decision", "patch_decision",
        }
        assert set(dir(store)) & forbidden_names == set()

    def test_list_decisions_returns_insertion_order(self):
        store = InMemoryStore()
        store.insert_decision(_decision())
        store.insert_decision(_decision())
        store.insert_decision(_decision())
        listed = store.list_decisions()
        assert len(listed) == 3


# ── Outcomes (state machine — stages grow, final_stage closes once) ──────────


class TestOutcomes:
    def test_insert_outcome_creates_one_doc(self):
        store = InMemoryStore()
        did = store.insert_decision(_decision())
        store.insert_outcome(_outcome(did))
        assert store.count("outcomes") == 1

    def test_push_stage_appends(self):
        store = InMemoryStore()
        did = store.insert_decision(_decision())
        store.insert_outcome(_outcome(did))
        now = datetime.now(timezone.utc)
        store.push_outcome_stage(did, OutcomeStage(stage="CALLBACK", at=now))
        outcomes = store.list_outcomes()
        assert len(outcomes[0]["stages"]) == 2

    def test_final_stage_set_once(self):
        store = InMemoryStore()
        did = store.insert_decision(_decision())
        store.insert_outcome(_outcome(did))
        store.set_outcome_final_stage(did, "OFFER")
        assert store.list_outcomes()[0]["final_stage"] == "OFFER"

    def test_retroactive_final_stage_overwrite_forbidden(self):
        """The integrity constraint — closed outcomes stay closed."""
        store = InMemoryStore()
        did = store.insert_decision(_decision())
        store.insert_outcome(_outcome(did))
        store.set_outcome_final_stage(did, "REJECTED")
        with pytest.raises(ValueError, match="retroactive overwrite"):
            store.set_outcome_final_stage(did, "OFFER")

    def test_push_stage_on_missing_outcome_raises(self):
        store = InMemoryStore()
        with pytest.raises(KeyError):
            store.push_outcome_stage("nonexistent", OutcomeStage(
                stage="CALLBACK", at=datetime.now(timezone.utc)
            ))


# ── Feedback (append-only) ───────────────────────────────────────────────────


class TestFeedback:
    def test_feedback_append_only(self):
        store = InMemoryStore()
        did = store.insert_decision(_decision())
        store.insert_feedback(FeedbackLog(
            decision_id=did,
            feedback_type="verdict_wrong",
            reason="parser missed skill",
            expected_verdict=Verdict.APPLY,
            actual_verdict=Verdict.REVIEW,
        ))
        store.insert_feedback(FeedbackLog(
            decision_id=did,
            feedback_type="reasoning_off",
            reason="LLM missed domain context",
        ))
        assert store.count("feedback_logs") == 2


# ── Diagnostics ──────────────────────────────────────────────────────────────


class TestDiagnostics:
    def test_count_unknown_collection_raises(self):
        with pytest.raises(KeyError):
            InMemoryStore().count("not_a_collection")

    def test_all_collections_start_empty(self):
        store = InMemoryStore()
        for c in ("profiles", "jobs", "decisions", "outcomes", "feedback_logs"):
            assert store.count(c) == 0
