"""Smoke tests for the Streamlit presentation layer.

The UI is a DUMB SHELL: it must not diverge from the tested pipeline.
These tests prove that:

1. The app module imports cleanly (no syntax / import-time errors).
2. `detect_mode()` produces the expected 4 modes given all env-var combos.
3. The demo profile is a valid `CandidateProfile`.
4. `resolve_profile()` prefers a Mongo-seeded profile when one exists,
   and falls back to the demo only when none is active.
5. The UI does not contain any code that recomputes a score — enforced
   by a grep-test against the forbidden symbols.

Streamlit's `st.*` calls inside `render_decision` / `render_header` /
`main` are NOT exercised here — those require a running Streamlit
context. A separate manual smoke (`streamlit run`) covers that path.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from src.db import InMemoryStore
from src.schemas import CandidateProfile, Seniority

# ── Module imports cleanly ───────────────────────────────────────────────────


def test_streamlit_app_imports():
    mod = importlib.import_module("streamlit_app.app")
    assert hasattr(mod, "main")
    assert hasattr(mod, "detect_mode")
    assert hasattr(mod, "resolve_profile")
    assert hasattr(mod, "render_decision")


# ── detect_mode() exhaustive over env ────────────────────────────────────────


class TestDetectMode:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("MONGODB_URI", raising=False)

    def test_no_env_is_demo(self):
        from streamlit_app.app import detect_mode

        m = detect_mode()
        assert m.name == "demo"
        assert m.banner_kind == "warning"

    def test_openai_only(self, monkeypatch):
        from streamlit_app.app import detect_mode

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        m = detect_mode()
        assert m.name == "openai_only"

    def test_mongo_only(self, monkeypatch):
        from streamlit_app.app import detect_mode

        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        m = detect_mode()
        assert m.name == "mongo_only"

    def test_both_is_production(self, monkeypatch):
        from streamlit_app.app import detect_mode

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("MONGODB_URI", "mongodb://test")
        m = detect_mode()
        assert m.name == "production"
        assert m.banner_kind == "success"

    def test_mode_fields_are_user_visible_strings(self):
        from streamlit_app.app import detect_mode

        m = detect_mode()
        # Every visible field must be non-empty and human-readable.
        for field in (m.label, m.store_kind, m.reasoner_kind, m.embedding_kind):
            assert isinstance(field, str)
            assert len(field) >= 5


# ── Demo profile validity ────────────────────────────────────────────────────


class TestDemoProfile:
    def test_demo_profile_is_valid_candidate_profile(self):
        from streamlit_app.app import DEMO_PROFILE

        assert isinstance(DEMO_PROFILE, CandidateProfile)
        assert DEMO_PROFILE.active is True
        # A sane baseline — the demo must have some skills or the UI
        # renders an empty card.
        assert len(DEMO_PROFILE.skills_tech) + len(DEMO_PROFILE.skills_tools) > 0


# ── resolve_profile precedence ──────────────────────────────────────────────


class TestResolveProfile:
    def test_empty_store_returns_demo(self):
        from streamlit_app.app import DEMO_PROFILE, resolve_profile

        profile, degraded = resolve_profile(InMemoryStore())
        assert profile is DEMO_PROFILE
        # A clean "no active profile" lookup is NOT a degradation — no
        # warning banner should fire for the ordinary demo path.
        assert degraded is False

    def test_mongo_resident_profile_wins(self):
        from streamlit_app.app import DEMO_PROFILE, resolve_profile

        store = InMemoryStore()
        real = CandidateProfile(
            profile_version="real-v1.0",
            name="Real Candidate",
            summary="real",
            years_experience=7.0,
            seniority=Seniority.STAFF,
            active=True,
        )
        store.upsert_profile(real)
        got, degraded = resolve_profile(store)
        assert got.name == "Real Candidate"
        assert got is not DEMO_PROFILE
        assert degraded is False

    def test_raising_store_falls_back_to_demo_and_flags_degraded(self):
        """A store that RAISES (Mongo outage mid-session) must surface
        `degraded=True` so the UI renders a visible warning instead of
        silently scoring against the demo profile."""
        from streamlit_app.app import DEMO_PROFILE, resolve_profile

        class _ExplodingStore(InMemoryStore):
            def get_active_profile(self):
                raise ConnectionError("mongo unreachable")

        profile, degraded = resolve_profile(_ExplodingStore())
        assert profile is DEMO_PROFILE
        assert degraded is True


# ── No-score-recomputation invariant (UI is a dumb shell) ────────────────────


class TestUIDoesNotRecomputeScores:
    """UI must not recompute scores, re-interpret signals, or introduce
    logic branches that diverge from tests."""

    def _app_source(self) -> str:
        return (Path(__file__).parent.parent / "streamlit_app" / "app.py").read_text(
            encoding="utf-8"
        )

    def test_app_does_not_import_scorer_directly(self):
        """The UI must go through the orchestrator, not call score() itself.

        Otherwise a UI tweak could diverge from the scorer's contract
        without failing any test.
        """
        src = self._app_source()
        forbidden = [
            "from src.engine.scorer import score",
            "from src.engine.scorer import _weighted_contributions",
            "from src.engine.scorer import _score_to_verdict",
            "from src.engine.scorer import _dominant_signal",
        ]
        for needle in forbidden:
            assert needle not in src, (
                f"streamlit_app/app.py imports {needle!r} — the UI must "
                "not call the scorer directly; it must go through "
                "orchestrator.evaluate_job()."
            )

    def test_app_does_not_redefine_weights_or_thresholds(self):
        """No UI-local override of the locked config."""
        src = self._app_source()
        forbidden_literals = [
            "WEIGHTS = Weights(",
            "THRESHOLDS = Thresholds(",
            "apply_score =",  # no computing an apply_score anywhere in UI
        ]
        for needle in forbidden_literals:
            assert needle not in src, (
                f"streamlit_app/app.py contains {needle!r} — the UI must "
                "not redefine or recompute locked config."
            )

    def test_app_does_not_call_llm_more_than_via_orchestrator(self):
        """No second LLM call from the UI. The orchestrator calls the
        reasoner exactly once per evaluate_job."""
        src = self._app_source()
        # The UI may REFERENCE reasoner classes for provider selection,
        # but must not call `.reason(` anywhere.
        assert ".reason(" not in src, (
            "streamlit_app/app.py calls .reason() directly — the UI must "
            "only call evaluate_job() which calls the reasoner exactly once."
        )

    def test_app_never_uses_mock_embedding_provider(self):
        """Scoring determinism invariant.

        `MockEmbeddingProvider` produces hash-based embeddings whose
        cosine similarity differs from the real sentence-transformer's.
        Using it in the UI path would change `semantic_similarity` and
        therefore `apply_score` — creating two different score
        behaviours depending on deployment environment.

        The only correct provider in the UI is `SentenceTransformerProvider`.
        The Docker image pre-downloads the model at build time.
        Tests use the mock directly (passed as an argument), never
        through the UI.
        """
        src = self._app_source()
        assert "MockEmbeddingProvider" not in src, (
            "streamlit_app/app.py references MockEmbeddingProvider. "
            "The UI must never fall back to the mock — it changes "
            "semantic_similarity and therefore the score. "
            "If sentence-transformers is unavailable at runtime, the "
            "app should fail loudly, not silently score differently."
        )


# ── README.md required sections ──────────────────────────────────────────────


class TestReadmeContract:
    """The README is a system contract document with a fixed set of
    required content sections."""

    def _readme(self) -> str:
        return (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")

    def test_readme_has_hf_frontmatter(self):
        readme = self._readme()
        assert readme.startswith("---\n"), "README must open with YAML frontmatter"
        assert "sdk: docker" in readme
        assert "app_port: 7860" in readme

    @pytest.mark.parametrize(
        "required_heading",
        [
            "## 1. The problem",
            "## 2. How I use it",
            "## 3. System design",
            "## 4. The decision formula",
            "## 5. LLM boundary",
            "## 6. Evaluation and honest scope",
            "## 7. Proof of correctness",
        ],
    )
    def test_readme_has_required_section(self, required_heading):
        assert required_heading in self._readme(), (
            f"README missing required section: {required_heading!r}"
        )

    def test_readme_cites_the_exact_formula_values(self):
        """The weights + thresholds quoted in the README must match
        src/config.py exactly. Any drift is a system contract violation."""
        readme = self._readme()
        for literal in (
            "0.30 × skills_match",
            "0.20 × experience_match",
            "0.15 × semantic_similarity",
            "0.25 × llm_confidence",
            "0.10 × role_level_fit",
            "score ≥ 80",
            "65 ≤ s < 80",
            "50 ≤ s < 65",
            "score < 50",
        ):
            assert literal in readme, f"README missing {literal!r}"

    def test_readme_cites_the_n_50_threshold(self):
        assert "50" in self._readme()
        assert "INSUFFICIENT DATA" in self._readme()


# ── Claim honesty: determinism scope + measured figures ─────────────────────


class TestClaimHonesty:
    """The repo's brand is claim-honesty: every determinism / size / speed
    claim in the README and the UI must be scoped to what is actually
    verified. These grep-tests pin the corrected wording so an overclaim
    cannot silently return.
    """

    def _readme(self) -> str:
        return (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")

    def _app_source(self) -> str:
        return (Path(__file__).parent.parent / "streamlit_app" / "app.py").read_text(
            encoding="utf-8"
        )

    def _semantic_source(self) -> str:
        return (
            Path(__file__).parent.parent / "src" / "signals" / "semantic.py"
        ).read_text(encoding="utf-8")

    def test_ui_banner_scopes_determinism_to_the_deterministic_signals(self):
        """`apply_score` is NOT identical across modes when OPENAI_API_KEY is
        set: the orchestrator feeds live `llm_confidence` into `score()` at
        weight 0.25 (the repo's own
        test_orchestrator.py::test_score_with_and_without_llm_differs_by_at_most_25
        allows a 25-point delta). The banner must scope the claim to the
        deterministic signals and disclose the LLM-signal delta."""
        src = self._app_source()
        forbidden = [
            "Scoring is identical across all modes",
            "IDENTICAL across all four modes",
            "scoring is deterministic across all",
        ]
        for needle in forbidden:
            assert needle not in src, (
                f"streamlit_app/app.py claims {needle!r} — false when "
                "OPENAI_API_KEY is set (live llm_confidence, weight 0.25, "
                "shifts apply_score by up to 25 points). Scope the claim "
                "to the deterministic signals instead."
            )
        # The corrected, truthful banner wording must be present.
        assert "The deterministic core is identical in every mode" in src
        assert "up to 25 points" in src

    def test_readme_determinism_claims_are_scoped(self):
        """'Same input → same output, every time' and '1e-9 across local,
        CI, and HuggingFace Spaces' were falsifiable: the LLM signal is live
        (stochastic) whenever a key is present, and no HF-side 1e-9
        verification exists (tests/ are excluded from the Space image by
        .dockerignore). The README must scope both claims."""
        readme = self._readme()
        assert "Same input → same output, every time." not in readme
        assert "local, CI, and HuggingFace Spaces" not in readme
        assert "job-offer scoring engine" not in readme  # it scores JDs
        # Corrected scoping must be present.
        assert "LLM-absent path" in readme
        assert "verified to 1e-9 in local and CI test runs" in readme

    def test_readme_grep_test_count_matches_reality(self):
        """README §3 cites the number of UI grep-tests; keep it equal to the
        actual number of test methods in TestUIDoesNotRecomputeScores."""
        n = len([m for m in dir(TestUIDoesNotRecomputeScores) if m.startswith("test_")])
        assert n == 4, "update README §3 and this test if the count changes"
        readme = self._readme()
        assert "Three grep-tests" not in readme
        assert "Four grep-tests" in readme

    def test_model_size_claims_are_honest(self):
        """all-MiniLM-L6-v2 is ~90 MB of weights (~175 MB on disk with both
        cached snapshots), not '~400 MB'. Both the README first-visit note
        and the semantic module docstring must state the measured figure."""
        assert "~400 MB" not in self._readme()
        assert "~400 MB" not in self._semantic_source()
        assert "~90 MB" in self._readme()

    def test_suite_runtime_claim_is_honest(self):
        """The hermetic suite measures ~5-9 s on a developer laptop,
        not '~1-2 seconds'. (Deliberately count-free: a hardcoded test
        count here would rot the moment the suite grows.)"""
        readme = self._readme()
        assert "~1–2 seconds" not in readme
        assert "~5–10 seconds" in readme
