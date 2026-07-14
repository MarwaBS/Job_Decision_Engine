"""Behavioral tests for the Streamlit render layer.

The render layer (`render_decision`, including the PARSE_FAILURE "N/A" branch) was
previously exercised only by "it imports". These run the real Streamlit script via
``streamlit.testing.v1.AppTest`` and assert on the rendered elements — using a
hermetic ``MockEmbeddingProvider`` + ``MockReasoner`` so no model/network is needed.
"""

from __future__ import annotations

import ast
from pathlib import Path

from streamlit.testing.v1 import AppTest

_APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app" / "app.py"

# AppTest's default script timeout is 3 s — measured cold runs of these tests
# (first import of streamlit + pydantic model build) have exceeded it on a
# developer laptop, failing the suite spuriously. A generous explicit timeout
# keeps the tests deterministic; warm runs still finish in well under 1 s.
_APPTEST_TIMEOUT_S = 30.0

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
        _render_decision_script,
        kwargs={"jd_text": STRONG_JD},
        default_timeout=_APPTEST_TIMEOUT_S,
    ).run()
    assert not at.exception, at.exception
    values = [m.value for m in at.metric]
    # A scored JD shows a "<score> / 100" apply-score metric and a verdict, never
    # the parse-failure placeholder.
    assert any("/ 100" in v for v in values), values
    assert not any("N/A" in v for v in values), values
    verdicts = {"PRIORITY", "APPLY", "REVIEW", "SKIP"}
    assert any(v in verdicts for v in values), values


def _render_header_script(mode_name: str, reasoner_state: str) -> None:
    """Streamlit script: render the header banner for a given mode with either a
    DEAD reasoner (FailingReasoner) or a LIVE one (MockReasoner)."""
    from src.db import InMemoryStore
    from src.llm.reasoning import FailingReasoner, MockReasoner
    from streamlit_app.app import RuntimeMode, render_header

    mode = RuntimeMode(
        name=mode_name,
        label="OpenAI + in-memory store",
        banner_kind="info",
        store_kind="InMemoryStore — session-only",
        reasoner_kind="OpenAIReasoner (gpt-4o) — reasoning panel populated",
        embedding_kind="SentenceTransformer (all-MiniLM-L6-v2)",
    )
    reasoner = FailingReasoner() if reasoner_state == "dead" else MockReasoner()
    render_header(mode, InMemoryStore(), reasoner)


def test_banner_reports_llm_degraded_when_key_is_dead() -> None:
    """Regression for the live-demo banner lie: an OpenAI mode whose key is dead
    (reasoner degraded to FailingReasoner) must NOT keep claiming "reasoning
    panel populated". render_header reconciles the banner against the reasoner
    ACTUALLY built and surfaces the degradation."""
    at = AppTest.from_function(
        _render_header_script,
        kwargs={"mode_name": "openai_only", "reasoner_state": "dead"},
        default_timeout=_APPTEST_TIMEOUT_S,
    ).run()
    assert not at.exception, at.exception
    warnings = [w.value for w in at.warning]
    assert any("LLM degraded" in w for w in warnings), warnings
    all_banner_text = " ".join(
        [b.value for b in at.info] + [b.value for b in at.success] + warnings
    )
    # The self-contradiction that shipped must be gone: no "populated" claim.
    assert "reasoning panel populated" not in all_banner_text, all_banner_text
    assert "DEGRADED" in all_banner_text, all_banner_text


def test_banner_reports_live_llm_when_reasoner_works() -> None:
    """The honest positive: an OpenAI mode with a working reasoner shows the
    populated-panel banner and NO degradation warning."""
    at = AppTest.from_function(
        _render_header_script,
        kwargs={"mode_name": "openai_only", "reasoner_state": "live"},
        default_timeout=_APPTEST_TIMEOUT_S,
    ).run()
    assert not at.exception, at.exception
    assert not any("LLM degraded" in w.value for w in at.warning)
    banner_text = " ".join(b.value for b in at.info)
    assert "reasoning panel populated" in banner_text, banner_text


def test_render_decision_parse_failure_shows_na_not_zero() -> None:
    """The PARSE_FAILURE branch must render "N/A — parse failure", not "0.0/100"
    (BUG-004: an undefined score must not read as a 0% match)."""
    at = AppTest.from_function(
        _render_decision_script,
        kwargs={"jd_text": ""},
        default_timeout=_APPTEST_TIMEOUT_S,
    ).run()
    assert not at.exception, at.exception
    values = [m.value for m in at.metric]
    assert any("N/A — parse failure" in v for v in values), values
    assert not any("0.0 / 100" in v for v in values), values
    assert any(v == "PARSE_FAILURE" for v in values), values


def test_no_stray_module_level_strings_leak_to_ui() -> None:
    """Streamlit "magic" renders any top-level bare expression — including a
    stray triple-quoted string — into the live UI. A PEP-257 "attribute
    docstring" placed after a module-level assignment is exactly such a bare
    expression, so it leaks developer commentary onto the page above the title.

    (This is real: the `DEMO_PROFILE` attribute docstring once rendered "Kept
    inline … to avoid expanding the module layout for a presentation-layer
    artefact" as the first thing a visitor saw on the public Space.)

    Guard against any recurrence: the ONLY legitimate module-level string
    expression is the module docstring (the first statement). Anything else
    must be a real ``#`` comment, not a bare string, so magic can't render it.
    """
    module = ast.parse(_APP_PATH.read_text(encoding="utf-8"))
    stray = [
        node.value.lineno
        for i, node in enumerate(module.body)
        if i != 0  # index 0 is the module docstring — Streamlit skips it
        and isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    assert not stray, (
        f"Bare module-level string(s) at line(s) {stray} in {_APP_PATH.name} "
        "will be rendered into the UI by Streamlit magic — make them comments."
    )
