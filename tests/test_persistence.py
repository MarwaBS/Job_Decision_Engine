"""Tests for the persistence layer (thin I/O wrappers)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.db import InMemoryStore
from src.logging.persistence import (
    advance_outcome,
    close_outcome,
    persist_decision,
    persist_feedback,
    persist_outcome_submitted,
)
from src.schemas import (
    DecisionResult,
    DecisionSensitivity,
    DecisionTrace,
    Job,
    ParsedJob,
    Seniority,
    Signals,
    Thresholds,
    Verdict,
    Weights,
)


def _job() -> Job:
    return Job(
        content_hash="sha256:test",
        source="paste",
        raw_text="raw",
        parsed=ParsedJob(title="ML Engineer"),
        parse_confidence=0.9,
        parse_warnings=[],
    )


def _decision() -> DecisionResult:
    sig = Signals(
        skills_match=0.8, experience_match=1.0,
        semantic_similarity=0.7, llm_confidence=0.85, role_level_fit=1.0,
    )
    return DecisionResult(
        apply_score=85.0,
        verdict=Verdict.PRIORITY,
        signals=sig,
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


# ── persist_decision ─────────────────────────────────────────────────────────


class TestPersistDecision:
    def test_upserts_job_and_inserts_decision(self):
        store = InMemoryStore()
        did = persist_decision(store, _job(), _decision())
        assert store.count("jobs") == 1
        assert store.count("decisions") == 1
        assert did  # not empty

    def test_duplicate_job_deduped(self):
        """Same JD submitted twice → one job doc, two decision docs."""
        store = InMemoryStore()
        persist_decision(store, _job(), _decision())
        persist_decision(store, _job(), _decision())
        assert store.count("jobs") == 1
        assert store.count("decisions") == 2


# ── Outcome state machine ────────────────────────────────────────────────────


class TestOutcomeLifecycle:
    def test_submit_creates_initial_outcome(self):
        store = InMemoryStore()
        did = persist_decision(store, _job(), _decision())
        oid = persist_outcome_submitted(store, did)
        assert oid
        outcome = store.list_outcomes()[0]
        assert outcome["final_stage"] is None
        assert len(outcome["stages"]) == 1
        assert outcome["stages"][0]["stage"] == "SUBMITTED"

    def test_advance_non_terminal_stage(self):
        store = InMemoryStore()
        did = persist_decision(store, _job(), _decision())
        persist_outcome_submitted(store, did)
        advance_outcome(store, did, "CALLBACK")
        outcome = store.list_outcomes()[0]
        assert len(outcome["stages"]) == 2
        assert outcome["final_stage"] is None

    def test_advance_terminal_stage_closes_outcome(self):
        """OFFER / REJECTED / GHOSTED / WITHDRAWN automatically close."""
        store = InMemoryStore()
        did = persist_decision(store, _job(), _decision())
        persist_outcome_submitted(store, did)
        advance_outcome(store, did, "OFFER")
        outcome = store.list_outcomes()[0]
        assert outcome["final_stage"] == "OFFER"

    def test_full_cycle(self):
        """Real-world path: SUBMITTED → CALLBACK → INTERVIEW → OFFER."""
        store = InMemoryStore()
        did = persist_decision(store, _job(), _decision())
        persist_outcome_submitted(store, did)
        advance_outcome(store, did, "CALLBACK")
        advance_outcome(store, did, "INTERVIEW")
        advance_outcome(store, did, "OFFER")
        outcome = store.list_outcomes()[0]
        assert len(outcome["stages"]) == 4
        assert [s["stage"] for s in outcome["stages"]] == [
            "SUBMITTED", "CALLBACK", "INTERVIEW", "OFFER",
        ]
        assert outcome["final_stage"] == "OFFER"

    def test_cannot_reopen_closed_outcome(self):
        store = InMemoryStore()
        did = persist_decision(store, _job(), _decision())
        persist_outcome_submitted(store, did)
        advance_outcome(store, did, "REJECTED")
        with pytest.raises(ValueError, match="retroactive overwrite"):
            advance_outcome(store, did, "OFFER")

    def test_close_outcome_rejects_non_terminal(self):
        store = InMemoryStore()
        did = persist_decision(store, _job(), _decision())
        persist_outcome_submitted(store, did)
        with pytest.raises(ValueError, match="must be terminal"):
            close_outcome(store, did, "CALLBACK")  # type: ignore[arg-type]


# ── Feedback ─────────────────────────────────────────────────────────────────


class TestFeedback:
    def test_persist_feedback_appends(self):
        store = InMemoryStore()
        did = persist_decision(store, _job(), _decision())
        persist_feedback(
            store,
            decision_id=did,
            feedback_type="verdict_wrong",
            reason="parser missed skill",
            expected_verdict=Verdict.APPLY,
            actual_verdict=Verdict.REVIEW,
        )
        assert store.count("feedback_logs") == 1


# ── Pure-I/O invariant (no business logic in persistence) ────────────────────


class TestPureIOInvariant:
    def test_persistence_module_does_not_import_scorer(self):
        """Architecture / Step 4 rule: "Pure I/O layer only. NO logic."

        The persistence module must not import or call the scorer.
        """
        from pathlib import Path

        src = (
            Path(__file__).parent.parent / "src" / "logging" / "persistence.py"
        ).read_text(encoding="utf-8")
        forbidden = [
            "from src.engine.scorer",
            "import src.engine.scorer",
            "from src.llm",
            "from src.ingestion",
            "from src.signals",
        ]
        for needle in forbidden:
            assert needle not in src, (
                f"persistence.py imports {needle!r} — violates pure-I/O rule"
            )
