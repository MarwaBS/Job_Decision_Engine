"""Reproduce README "Example B" end-to-end — no API key, no database.

Every number quoted in the README's Example B block is the output of this
script, so a stranger can verify the README instead of taking it on faith:

    python -m scripts.demo_example

What runs:
- the REAL parser on the bundled structured JD below,
- the REAL deterministic signals (skills, experience, role-level fit),
- the REAL pinned sentence-transformer for semantic similarity
  (requires `sentence-transformers` from requirements.txt; first run
  downloads the pinned model revision),
- the LLM-absent path (`llm_confidence = 0.0`) — the same path the public
  demo runs without an OpenAI key, and the reason this script is fully
  deterministic: same input → same score, every run, every machine.

The hermetic test suite pins the parser + deterministic-signal outputs of
this exact JD (`tests/test_demo_example.py`), so README drift fails CI.
"""

from __future__ import annotations

from src.db import InMemoryStore
from src.engine.orchestrator import evaluate_job
from src.llm.reasoning import FailingReasoner
from src.schemas import CandidateProfile, Seniority
from src.signals.semantic import SentenceTransformerProvider

#: The structured JD behind README Example B — verbatim.
EXAMPLE_B_JD = """\
Title: Senior ML Engineer
Company: Acme AI
Location: Berlin, Germany (Remote)

We build LLM-powered document intelligence for enterprise customers.

Requirements:
- 5+ years of experience building production ML systems
- Python, PyTorch, SQL
- FastAPI, Docker, AWS
- MLOps practices: CI/CD for models, monitoring, reproducible training

Nice to have:
- Kubernetes, LangChain
- RAG pipelines
"""

#: The profile the README narrative scores against — mirrors the bundled
#: demo profile in `streamlit_app/app.py` (Alex Rivera).
EXAMPLE_PROFILE = CandidateProfile(
    profile_version="demo-1.0",
    name="Demo Candidate (Alex Rivera)",
    summary=(
        "Senior ML engineer with 5+ years of end-to-end experience building "
        "production ML systems in Python — data engineering, model training, "
        "SHAP explainability, FastAPI serving, Docker, HuggingFace. "
        "Comfortable across tabular ML, LLM pipelines, and MLOps."
    ),
    years_experience=6.0,
    seniority=Seniority.SENIOR,
    skills_tech=["python", "sql"],
    skills_tools=[
        "pytorch",
        "xgboost",
        "lightgbm",
        "aws",
        "docker",
        "mlops",
        "fastapi",
    ],
    skills_domain=["mlops", "llm", "rag", "data engineering"],
)


def main() -> int:
    decision = evaluate_job(
        EXAMPLE_B_JD,
        EXAMPLE_PROFILE,
        store=InMemoryStore(),
        reasoner=FailingReasoner(),  # LLM-absent path: llm_confidence = 0.0
        embedding_provider=SentenceTransformerProvider(),
    )
    s = decision.signals
    print("README Example B — reproduced from source (LLM-absent mode)")
    print(f"engine_version:    {decision.engine_version}")
    print(f"parse_confidence:  {s.parse_confidence:.2f}")
    print(f"skills_match:      {s.skills_match:.3f}")
    print(f"experience_match:  {s.experience_match:.3f}")
    print(f"semantic_sim:      {s.semantic_similarity:.3f}")
    print(f"llm_confidence:    {s.llm_confidence:.3f}   (no API key — LLM-absent path)")
    print(f"role_level_fit:    {s.role_level_fit:.3f}")
    print(f"apply_score:       {decision.apply_score:.1f}")
    print(f"verdict:           {decision.verdict.value}")
    print(f"dominant_signal:   {decision.decision_trace.dominant_signal}")
    print(f"near_threshold:    {decision.decision_trace.near_threshold_flag}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
