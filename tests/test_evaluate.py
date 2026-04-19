"""Tests for the evaluation script.

The headline test: **`evaluate()` MUST return a STUB when N < 50**.

This is the integrity claim of the whole project — simulating feedback
data would be worse than showing no evaluation. If this test fails, an
actual deployment could ship fake metrics.

The REAL-metric path is tested with seeded fake data SOLELY to verify the
metric shape (not the semantic correctness of the metrics themselves —
that's a portfolio conversation, not a test).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from scripts.evaluate import MIN_OUTCOMES_FOR_EVALUATION, EvaluationResult, evaluate
from src.db import InMemoryStore
from src.schemas import (
    DecisionResult,
    DecisionSensitivity,
    DecisionTrace,
    Outcome,
    OutcomeStage,
    Signals,
    Thresholds,
    Verdict,
    Weights,
)


def _decision(verdict: Verdict = Verdict.APPLY) -> DecisionResult:
    sig = Signals(
        skills_match=0.7, experience_match=1.0,
        semantic_similarity=0.6, llm_confidence=0.6, role_level_fit=1.0,
    )
    return DecisionResult(
        apply_score=70.0,
        verdict=verdict,
        signals=sig,
        weights=Weights(skills=0.30, experience=0.20, semantic=0.15, llm=0.25, role=0.10),
        thresholds_version="v1.0",
        decision_trace=DecisionTrace(
            dominant_signal="skills_match",
            failure_mode_detected=None,
            decision_sensitivity=DecisionSensitivity(
                if_llm_removed_score=55.0,
                if_skills_boosted_plus_10pct=75.0,
                if_experience_removed_score=60.0,
            ),
            nearest_threshold_distance=5.0,
            near_threshold_flag=False,
        ),
        engine_version="0.1.0",
    )


def _outcome_with_stages(
    decision_id: str, stages: list[str], final: str | None = None,
    ttr_days: int | None = None,
) -> Outcome:
    now = datetime.now(timezone.utc)
    return Outcome(
        decision_id=decision_id,
        submitted_at=now,
        stages=[OutcomeStage(stage=s, at=now) for s in stages],  # type: ignore[arg-type]
        final_stage=final,  # type: ignore[arg-type]
        time_to_first_response_days=ttr_days,
    )


# ── Stub path (the one that matters for integrity) ───────────────────────────


class TestStubPathIntegrity:
    def test_empty_store_returns_stub(self):
        result = evaluate(InMemoryStore())
        assert result.is_stub is True
        assert result.n_outcomes == 0
        assert result.metrics == {}

    def test_stub_message_says_insufficient_data(self):
        result = evaluate(InMemoryStore())
        assert "INSUFFICIENT DATA" in result.message
        assert "50" in result.message  # cites the threshold

    def test_n_just_below_threshold_returns_stub(self):
        """N=49: still STUB. No projection, no "approximate" metrics."""
        store = InMemoryStore()
        for _ in range(MIN_OUTCOMES_FOR_EVALUATION - 1):
            did = store.insert_decision(_decision())
            store.insert_outcome(_outcome_with_stages(did, ["SUBMITTED"]))
        result = evaluate(store)
        assert result.is_stub is True
        assert result.n_outcomes == MIN_OUTCOMES_FOR_EVALUATION - 1
        assert result.metrics == {}

    def test_stub_result_metrics_field_is_empty_dict_not_none(self):
        """A consistent shape prevents downstream code from crashing on
        `None`. The metrics field is always a dict — empty in STUB case."""
        result = evaluate(InMemoryStore())
        assert isinstance(result.metrics, dict)

    def test_threshold_is_locked_at_50(self):
        """Architectural guarantee: the threshold is 50 and not
        accidentally tweakable. The test PINS the constant."""
        assert MIN_OUTCOMES_FOR_EVALUATION == 50


# ── Metric-shape path (requires N >= 50 seeded) ──────────────────────────────


class TestMetricShape:
    def _seed_n_outcomes(self, n: int, interview_rate: float) -> InMemoryStore:
        store = InMemoryStore()
        n_interviews = int(n * interview_rate)
        for i in range(n):
            did = store.insert_decision(_decision())
            stages = ["SUBMITTED"]
            final: str | None = None
            if i < n_interviews:
                stages.extend(["CALLBACK", "INTERVIEW"])
            else:
                final = "REJECTED"
            store.insert_outcome(_outcome_with_stages(did, stages, final=final, ttr_days=3))
        return store

    def test_exactly_50_outcomes_triggers_metric_path(self):
        store = self._seed_n_outcomes(50, interview_rate=0.2)
        result = evaluate(store)
        assert result.is_stub is False
        assert result.n_outcomes == 50
        assert "precision_apply" in result.metrics

    def test_metric_keys_match_architecture_section_8(self):
        store = self._seed_n_outcomes(50, interview_rate=0.2)
        result = evaluate(store)
        expected = {"precision_apply", "interview_rate", "false_positive_rate"}
        assert expected.issubset(result.metrics.keys())

    def test_precision_apply_is_in_zero_one(self):
        store = self._seed_n_outcomes(50, interview_rate=0.3)
        result = evaluate(store)
        assert 0.0 <= result.metrics["precision_apply"] <= 1.0


# ── EvaluationResult.as_dict shape ───────────────────────────────────────────


class TestEvaluationResultDict:
    def test_as_dict_has_all_fields(self):
        result = evaluate(InMemoryStore())
        d = result.as_dict()
        assert set(d.keys()) == {"n_outcomes", "is_stub", "message", "metrics"}
