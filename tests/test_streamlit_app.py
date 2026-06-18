"""Behavioral tests for the Streamlit render layer.

The render layer (`render_decision`, including the PARSE_FAILURE "N/A" branch) was
previously exercised only by "it imports". These run the real Streamlit script via
``streamlit.testing.v1.AppTest`` and assert on the rendered elements — using a
hermetic ``MockEmbeddingProvider`` + ``MockReasoner`` so no model/network is needed.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

STRONG_JD = """Title: Senior ML Engineer
Company: Acme
Location: Remote

5+ years of Python, PyTorch, AWS experience required. MLOps work.
"""


def _render_decision_script(jd_text: str) -> None:
    """Streamlit script body (run by AppTest): build a REAL DecisionResult with a
    hermetic embedder and render it through the production render path."""
    from src.db import InMemoryStore
    from src.engine.orchestrator import evaluate_job
    from src.llm.reasoning import MockReasoner
    from src.schemas import CandidateProfile, Seniority
    from src.signals.semantic import MockEmbeddingProvider
    from streamlit_app.app import render_decision

    profile = CandidateProfile(
        profile_version="v1.0",
        name="Tester",
        summary="ML engineer",
        years_experience=6.0,
        seniority=Seniority.SENIOR,
    )
    decision = evaluate_job(
        jd_text,
        profile,
        store=InMemoryStore(),
        reasoner=MockReasoner(),
        embedding_provider=MockEmbeddingProvider(),
    )
    render_decision(decision)


def test_render_decision_scored_jd_renders_score_and_verdict() -> None:
    at = AppTest.from_function(
        _render_decision_script, kwargs={"jd_text": STRONG_JD}
    ).run()
    assert not at.exception, at.exception
    values = [m.value for m in at.metric]
    # A scored JD shows a "<score> / 100" apply-score metric and a verdict, never
    # the parse-failure placeholder.
    assert any("/ 100" in v for v in values), values
    assert not any("N/A" in v for v in values), values
    verdicts = {"PRIORITY", "APPLY", "REVIEW", "SKIP"}
    assert any(v in verdicts for v in values), values


def test_render_decision_parse_failure_shows_na_not_zero() -> None:
    """The PARSE_FAILURE branch must render "N/A — parse failure", not "0.0/100"
    (BUG-004: an undefined score must not read as a 0% match)."""
    at = AppTest.from_function(
        _render_decision_script, kwargs={"jd_text": ""}
    ).run()
    assert not at.exception, at.exception
    values = [m.value for m in at.metric]
    assert any("N/A — parse failure" in v for v in values), values
    assert not any("0.0 / 100" in v for v in values), values
    assert any(v == "PARSE_FAILURE" for v in values), values
