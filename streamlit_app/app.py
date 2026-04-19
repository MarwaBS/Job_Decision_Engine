"""Streamlit UI for the Job Decision Engine.

Role: PRESENTATION LAYER ONLY. Per Step 5 hard rules:

- Inputs (JD text) → `orchestrator.evaluate_job()` → render the returned
  `DecisionResult`. No recompute, no re-interpret.
- Provider selection is **explicit at boot**. No hidden switches. The
  current mode (Production vs Demo) is surfaced in the UI header so a
  viewer always knows whether real OpenAI / real Mongo is being used.
- Caching via `@st.cache_resource` is for *connection reuse only*
  (sentence-transformers model, OpenAI client, Mongo client). It never
  caches decisions or signal values.

Startup modes (INFORMATIONAL ONLY — scoring is deterministic across all):

    OPENAI_API_KEY present   + MONGODB_URI present        -> "Production"
    OPENAI_API_KEY present   + MONGODB_URI absent         -> "OpenAI + in-memory store"
    OPENAI_API_KEY absent    + MONGODB_URI present        -> "Mongo-backed demo (no LLM)"
    both absent                                           -> "Demo mode (in-memory, no LLM)"

Deterministic-score invariant (enforced by always using the real
embedding provider):

    For a given (job, profile) input, the `apply_score` and `verdict`
    are IDENTICAL across all four modes. The only things that change:

    1. Persistence layer — decisions saved to Atlas vs session-only.
       Same scoring either way.
    2. LLM reasoning panel — rich text (OpenAI present) vs null
       reasoning (OpenAI absent, llm_confidence=0.0 per architecture
       §7). That zero IS the scored value in both deployments; the
       LLM is never a hidden free parameter.

The mock embedding provider from `src.signals.semantic` exists for
HERMETIC TESTS ONLY and is never instantiated by the UI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import streamlit as st

# IMPORTANT: set_page_config MUST be the first Streamlit command invoked
# by this module. It lives at module top — before any decorators, before
# any cached-resource registration — because Streamlit re-executes the
# script on every interaction and the rule is "first st.* call wins".
# Moving this inside a function that runs after cache_resource decorators
# caused StreamlitSetPageConfigMustBeFirstCommandError on HF Space.
st.set_page_config(
    page_title="Job Decision Engine",
    page_icon="⚖️",
    layout="wide",
)

from src.config import ENGINE_VERSION, THRESHOLDS, WEIGHTS
from src.db import InMemoryStore, MongoStore, Store
from src.engine.orchestrator import evaluate_job
from src.llm.reasoning import (
    FailingReasoner,
    LLMReasoner,
    MockReasoner,
    OpenAIReasoner,
)
from src.schemas import CandidateProfile, DecisionResult, Seniority, Verdict
from src.signals.semantic import (
    EmbeddingProvider,
    SentenceTransformerProvider,
)


# ── Mode / provider selection (explicit, surfaced in UI) ────────────────────


@dataclass(frozen=True)
class RuntimeMode:
    name: Literal["production", "openai_only", "mongo_only", "demo"]
    label: str
    banner_kind: Literal["success", "info", "warning"]
    store_kind: str
    reasoner_kind: str
    embedding_kind: str


def detect_mode() -> RuntimeMode:
    """Explicit boot-time detection. No fallbacks hidden later."""
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_mongo = bool(os.getenv("MONGODB_URI"))

    # Embedding provider is ALWAYS the real SentenceTransformer — mock
    # embeddings produce different semantic_similarity values and would
    # change scoring across modes. Docker image pre-downloads the model
    # at build time so this is always available in the HF Space.
    embedding_kind = "SentenceTransformer (all-MiniLM-L6-v2)"

    if has_openai and has_mongo:
        return RuntimeMode(
            name="production",
            label="Production",
            banner_kind="success",
            store_kind="MongoStore (Atlas) — decisions persisted across sessions",
            reasoner_kind="OpenAIReasoner (gpt-4o) — reasoning panel populated",
            embedding_kind=embedding_kind,
        )
    if has_openai:
        return RuntimeMode(
            name="openai_only",
            label="OpenAI + in-memory store",
            banner_kind="info",
            store_kind="InMemoryStore — session-only; restart loses decisions",
            reasoner_kind="OpenAIReasoner (gpt-4o) — reasoning panel populated",
            embedding_kind=embedding_kind,
        )
    if has_mongo:
        return RuntimeMode(
            name="mongo_only",
            label="Mongo-backed demo (no LLM)",
            banner_kind="warning",
            store_kind="MongoStore (Atlas) — decisions persisted across sessions",
            reasoner_kind="LLM disabled — reasoning=None, llm_confidence=0.0 per architecture §7",
            embedding_kind=embedding_kind,
        )
    return RuntimeMode(
        name="demo",
        label="Demo mode",
        banner_kind="warning",
        store_kind="InMemoryStore — session-only; restart loses decisions",
        reasoner_kind="LLM disabled — reasoning=None, llm_confidence=0.0 per architecture §7",
        embedding_kind=embedding_kind,
    )


# ── Cached providers (reuse connections, never cache decisions) ─────────────


@st.cache_resource
def _build_store(mode_name: str) -> Store:
    if mode_name in ("production", "mongo_only"):
        try:
            return MongoStore()
        except RuntimeError:
            return InMemoryStore()
    return InMemoryStore()


@st.cache_resource
def _build_reasoner(mode_name: str) -> LLMReasoner:
    if mode_name in ("production", "openai_only"):
        try:
            return OpenAIReasoner()
        except RuntimeError:
            return FailingReasoner()
    return FailingReasoner()


@st.cache_resource
def _build_embedding_provider() -> EmbeddingProvider:
    """Always return the real SentenceTransformer.

    NEVER falls back to a hash-based mock provider. Mock embeddings
    differ from the real model's output, so using them in the UI would
    change `semantic_similarity` and therefore `apply_score` — violating
    the deterministic-score invariant across deployment modes. The
    Docker image pre-downloads the model at build time, so in the HF
    Space runtime this is always available.

    For hermetic tests, individual tests construct a mock provider
    directly and pass it into signal / orchestrator calls — this UI
    helper is not used in tests.
    """
    return SentenceTransformerProvider()


# ── Profile resolution (Mongo first, demo fallback) ──────────────────────────


DEMO_PROFILE: CandidateProfile = CandidateProfile(
    profile_version="demo-1.0",
    name="Demo Candidate (Marwa Ben Salem)",
    summary=(
        "Senior ML engineer with 5+ years of end-to-end experience building "
        "production ML systems in Python — data engineering, model training, "
        "SHAP explainability, FastAPI serving, Docker, HuggingFace. "
        "Comfortable across tabular ML, LLM pipelines, and MLOps."
    ),
    years_experience=5.5,
    seniority=Seniority.SENIOR,
    skills_tech=["python", "sql"],
    skills_tools=["pytorch", "xgboost", "lightgbm", "aws", "docker", "mlops", "fastapi"],
    skills_domain=["mlops", "llm", "rag", "data engineering"],
    target_roles=["Senior ML Engineer", "Staff ML Engineer", "AI Engineer"],
    target_locations=["Remote", "EU", "US"],
    must_haves=[],
    nice_to_haves=[],
    dealbreakers=[],
    active=True,
)
"""The bundled demo profile for the public HF Space.

In a real multi-tenant deployment this would be replaced by a per-user
profile seeded via `scripts/seed_profile.py`. Kept inline (not in a
separate module) to avoid expanding the architecture §4 module layout
for a presentation-layer artefact.
"""


def resolve_profile(store: Store) -> CandidateProfile:
    """Prefer a Mongo-resident active profile; fall back to the demo."""
    try:
        from_store = store.get_active_profile()
    except Exception:
        from_store = None
    return from_store or DEMO_PROFILE


# ── UI renderers (pure — they read a DecisionResult and write Streamlit) ────


def render_header(mode: RuntimeMode) -> None:
    # set_page_config lives at module top (first-command rule).
    st.title("Job Decision Engine")
    st.caption(
        "Deterministic scoring + bounded-signal LLM reasoning. "
        f"engine {ENGINE_VERSION} · thresholds {THRESHOLDS.version}"
    )

    banner = {
        "success": st.success,
        "info": st.info,
        "warning": st.warning,
    }[mode.banner_kind]
    banner(
        f"**Mode: {mode.label}**\n\n"
        f"- Store: {mode.store_kind}\n"
        f"- Reasoner: {mode.reasoner_kind}\n"
        f"- Embeddings: {mode.embedding_kind}\n\n"
        f"**Scoring is identical across all modes.** Only the persistence "
        f"layer (where decisions are saved) and the LLM reasoning panel "
        f"change. The 5-signal weighted formula is deterministic."
    )


def render_decision(decision: DecisionResult) -> None:
    """Render a DecisionResult. No calculations — fields are read straight."""
    trace = decision.decision_trace
    signals = decision.signals

    # ── Verdict card ─────────────────────────────────────────────────────
    col_score, col_verdict, col_engine = st.columns([1, 1, 1])

    col_score.metric(
        label="Apply score",
        value=f"{decision.apply_score:.1f} / 100",
        delta=(
            f"nearest boundary: {trace.nearest_threshold_distance:.1f} pts"
            + (" (near!)" if trace.near_threshold_flag else "")
        ),
    )
    col_verdict.metric(
        label="Verdict",
        value=decision.verdict.value,
    )
    col_engine.metric(
        label="Engine / thresholds",
        value=f"{decision.engine_version} / {decision.thresholds_version}",
    )

    # ── Signals breakdown ────────────────────────────────────────────────
    st.subheader("Signals (direct from DecisionResult.signals)")
    st.table(
        {
            "Signal": [
                "skills_match", "experience_match", "semantic_similarity",
                "llm_confidence", "role_level_fit", "parse_confidence",
            ],
            "Value": [
                f"{signals.skills_match:.3f}",
                f"{signals.experience_match:.3f}",
                f"{signals.semantic_similarity:.3f}",
                f"{signals.llm_confidence:.3f}",
                f"{signals.role_level_fit:.3f}",
                f"{signals.parse_confidence:.3f}",
            ],
            "Weight (locked)": [
                f"{decision.weights.skills:.2f}",
                f"{decision.weights.experience:.2f}",
                f"{decision.weights.semantic:.2f}",
                f"{decision.weights.llm:.2f}",
                f"{decision.weights.role:.2f}",
                "— (hard filter)",
            ],
        }
    )

    # ── Decision trace (auditing panel) ──────────────────────────────────
    st.subheader("Decision trace")
    st.markdown(
        f"- **Dominant signal:** `{trace.dominant_signal}`\n"
        f"- **Failure mode detected:** "
        f"`{trace.failure_mode_detected.value if trace.failure_mode_detected else 'none'}`\n"
        f"- **Near threshold flag:** `{trace.near_threshold_flag}`"
    )
    st.markdown("**Counterfactual sensitivity replays:**")
    sens = trace.decision_sensitivity
    st.table(
        {
            "Counterfactual": [
                "LLM removed (confidence → 0)",
                "Skills boosted (+10%)",
                "Experience removed",
            ],
            "Re-scored value": [
                f"{sens.if_llm_removed_score:.1f}",
                f"{sens.if_skills_boosted_plus_10pct:.1f}",
                f"{sens.if_experience_removed_score:.1f}",
            ],
        }
    )

    # ── LLM reasoning (if present) ───────────────────────────────────────
    st.subheader("LLM reasoning")
    if decision.reasoning is None:
        st.info(
            "No reasoning attached. The LLM was unavailable or its output "
            "failed schema validation — per architecture §7, the decision "
            "ships anyway with llm_confidence = 0.0."
        )
    else:
        r = decision.reasoning
        cols = st.columns(3)
        cols[0].markdown("**Strengths**")
        for s in r.get("strengths", []):
            cols[0].markdown(f"- {s}")
        cols[1].markdown("**Gaps**")
        for s in r.get("gaps", []):
            cols[1].markdown(f"- {s}")
        cols[2].markdown("**Risks**")
        for s in r.get("risks", []):
            cols[2].markdown(f"- {s}")

        if r.get("recommended_talking_points"):
            st.markdown("**Recommended talking points (for cover letter / outreach)**")
            for tp in r["recommended_talking_points"]:
                st.markdown(f"- {tp}")


def render_footer() -> None:
    st.divider()
    st.caption(
        "This UI is a thin rendering layer. The scorer, signals, decision "
        "trace, LLM boundary, and append-only persistence are covered by "
        "261 hermetic unit tests. See the public README for the system "
        "contract."
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    mode = detect_mode()
    render_header(mode)

    store = _build_store(mode.name)
    reasoner = _build_reasoner(mode.name)
    embedding_provider = _build_embedding_provider()
    profile = resolve_profile(store)

    with st.sidebar:
        st.markdown("### Candidate profile")
        st.markdown(
            f"- **Name:** {profile.name}\n"
            f"- **Version:** `{profile.profile_version}`\n"
            f"- **Seniority:** `{profile.seniority.value}`\n"
            f"- **Experience:** {profile.years_experience} years"
        )
        st.markdown("**Tech:** " + (", ".join(profile.skills_tech) or "—"))
        st.markdown("**Tools:** " + (", ".join(profile.skills_tools) or "—"))
        st.markdown("**Domain:** " + (", ".join(profile.skills_domain) or "—"))

    st.subheader("Paste a job description")
    raw_text = st.text_area(
        label="Job description",
        placeholder="Title: Senior ML Engineer\nCompany: ...\n...",
        height=280,
        label_visibility="collapsed",
    )
    evaluate_clicked = st.button("Evaluate", type="primary", disabled=not raw_text.strip())

    if evaluate_clicked:
        with st.spinner("Running decision engine…"):
            decision = evaluate_job(
                raw_text, profile,
                store=store,
                reasoner=reasoner,
                embedding_provider=embedding_provider,
            )
        render_decision(decision)

    render_footer()


if __name__ == "__main__":  # pragma: no cover
    main()
