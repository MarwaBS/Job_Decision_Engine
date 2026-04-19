---
title: Job Decision Engine
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Job Decision Engine

A deterministic decision system for job applications. Given a job
description and a candidate profile, it produces a score in `[0, 100]`, a
verdict (`PRIORITY` / `APPLY` / `REVIEW` / `SKIP`), a machine-readable
decision trace (dominant signal, 3 counterfactual sensitivity replays,
distance to the nearest verdict boundary), and — when an LLM is
available — structured reasoning (strengths / gaps / risks / talking
points). Decisions are persisted append-only to MongoDB. Evaluation of
the system's own accuracy is **intentionally inert** until at least 50
real application outcomes have been logged.

## 1. What the system does

This is not a resume tool. It is a **decision system**. The core is a
pure, deterministic scoring function over five signals with locked
weights and fixed verdict thresholds. An LLM reasoning layer
**explains** the decision after the fact; it does not make the decision.
If the LLM is unavailable, the system still ships a verdict with
`reasoning=None` and `llm_confidence=0.0`. Every decision persisted to
MongoDB is stamped with the exact weights, thresholds version, and
engine version it was scored against — a stored decision can be
re-scored later and compared against the current configuration to prove
that the recorded result is still reproducible.

## 2. Architecture

```
┌────────────────────────┐          ┌──────────────────────┐
│       Streamlit UI     │────────▶ │   Orchestrator       │
│   (presentation only)  │          │ evaluate_job(...)    │
└────────────────────────┘          └─────────┬────────────┘
                                              │
      ┌────────────┬──────────────────┬──────────┬──────────┐
      ▼            ▼                  ▼          ▼          ▼
 Ingestion     Signals             LLM        Scorer   Persistence
 (parser)   (skills, exp,       reasoning   (pure,    (MongoStore or
            semantic, role)     (bounded    determ-     InMemoryStore;
                                 signal     inistic)    strict
                                 only)                  append-only)
```

Each box maps to a directory under `src/`. The engine layer (scorer) is
pure — no I/O, no network, no model calls. All I/O lives behind the
`Store` and `LLMReasoner` Protocol seams in `src/db.py` and
`src/llm/reasoning.py`. The orchestrator in `src/engine/orchestrator.py`
wires the layers together into a single `evaluate_job()` entrypoint.

## 3. Decision formula

Hard filters (applied BEFORE the weighted sum):

- `dealbreaker_hit == True`          → verdict `SKIP` (score = 0)
- `parse_confidence < 0.5`            → verdict `REVIEW` (score = 0)

Otherwise:

```
apply_score = 100 × ( 0.30 × skills_match
                    + 0.20 × experience_match
                    + 0.15 × semantic_similarity
                    + 0.25 × llm_confidence
                    + 0.10 × role_level_fit )
```

All signals are in `[0, 1]` except `role_level_fit` which is discrete
`{0, 0.5, 1}`. The weights sum to `1.0` exactly (enforced by a
model-level Pydantic validator). The weights are **priors, not learned
parameters** — they represent design intent, not statistical
optimisation.

Verdict thresholds:

| Score range    | Verdict    |
|----------------|------------|
| `score ≥ 80`   | `PRIORITY` |
| `65 ≤ s < 80`  | `APPLY`    |
| `50 ≤ s < 65`  | `REVIEW`   |
| `score < 50`   | `SKIP`     |

Weights and thresholds are stored on every decision document, so a
decision from today stays reproducible even if v2 re-tunes them.

## 4. LLM boundary (strict)

The LLM contributes to the system in exactly two ways:

1. As a **bounded numeric signal** (`llm_confidence` in `[0, 1]`, weighted
   at `0.25`). Its contribution is capped so that it cannot single-handedly
   flip a decision — this is enforced by a test
   (`test_config.py::test_llm_weight_not_dominant`).
2. As an **explanatory layer** (strengths, gaps, risks, talking points).
   These are stored on the decision for display; no downstream
   calculation reads them.

The LLM does NOT:

- define the decision boundary
- control verdict thresholds
- override scoring logic
- re-compute any signal

If the LLM call fails (network error, schema violation, retry
exhausted), the decision still ships — with `reasoning=None` and
`llm_confidence=0.0`. The integration test
`test_orchestrator.py::test_dealbreaker_forces_skip_regardless_of_llm`
proves that an LLM returning `llm_confidence=1.0` cannot override a
dealbreaker SKIP verdict.

## 5. Evaluation rule

`scripts/evaluate.py` computes real metrics (precision of APPLY,
interview rate, false-positive rate, precision of PRIORITY) **only when
at least 50 real application outcomes are logged**. Below that
threshold, the script returns:

```
INSUFFICIENT DATA (N=X, need >= 50). Evaluation framework ready;
metrics will populate when real outcome data reaches the threshold.
No simulated metrics will ever be shown.
```

This is deliberate. Simulating feedback data to produce a "working"
evaluation would make the whole system unfalsifiable. The evaluation
framework is ready; the metrics arrive when real outcomes do. The
threshold constant (`MIN_OUTCOMES_FOR_EVALUATION = 50`) is locked at
the module level of `scripts/evaluate.py` and pinned by a dedicated
test, specifically to make "temporary tweaks at N=47" harder to ship.
