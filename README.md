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

A deterministic decision system I built to triage real job descriptions
during my own active job search. The verdict comes from an explicit
weighted formula; the LLM only **explains** the decision — it never
decides it. Every decision is persisted with a full audit trace so I
can re-derive any past verdict from the stored data.

> **First-time visitor note.** The HF Space cold-start downloads the
> `sentence-transformers/all-MiniLM-L6-v2` model (~400 MB) and builds the
> Docker image. Allow 5–10 minutes on the very first visit. Subsequent
> boots are near-instant.

---

## 1. The problem

I'm in active job search for senior ML / AI engineering roles. Manual
triage of 50+ job descriptions per week is noisy, biased, and exhausting.
By Friday I'm rejecting things I should have applied to and applying to
things I should have skipped — because my own filter drifts as I get tired.

The system I needed:

- **Deterministic** — same JD + same profile → same verdict, every time, no matter who runs it or where
- **Auditable** — every decision logged with the exact signal values + weights it was scored under, so any past verdict can be re-derived
- **Honest about its limits** — flags structurally-bad JDs for manual review instead of pretending to score them; refuses to fake performance metrics until real outcomes accumulate

What I did NOT need: another LLM-driven "AI tool" that gives different
answers on different runs. The decision should be reproducible; the
*explanation* can be enriched by the LLM.

---

## 2. How I use it

Two real examples from my own usage. Both JDs are reproducible test
fixtures (in the repo); the screenshots / scores below are exactly what
the system returns for those inputs.

> *Snapshot note:* the example outputs and the "4.3 years" / "5.5 years"
> figures below were captured against my profile as it stood when these
> runs were recorded. My current `profile.yaml` has `years_experience: 6`;
> re-running the examples today would shift the experience-derived numbers
> accordingly. The verdicts (REVIEW for Example A, PRIORITY for Example B)
> are unchanged.

### Example A — Adobe Senior ML Engineer (prose JD)

I pasted Adobe's "Platform & AI Engineer" posting as-is. The JD is a
well-written prose paragraph but has no labeled headers (no `Title:`,
`Company:`, `Location:` lines).

```
parse_confidence:  0.45    ← BELOW the MIN_PARSE_CONFIDENCE = 0.5 hard filter
verdict:           PARSE_FAILURE
failure_mode:      low_parse_confidence
score:             None    (undefined — the JD couldn't be parsed, so no
                            weighted sum is computed; "N/A" in the UI)
```

**What the system was telling me:** "I extracted some skills and a
years-of-experience number, but I missed enough structural cues that I'm
not confident I parsed this correctly. Don't trust my numeric score —
read it yourself." I read it manually. The role required 7+ years of
AWS data engineering plus deep agentic-AI experience — a stretch given
my 4.3 years. I skipped it.

**The system was right to flag it.** A different design would have
soft-weighted the low parse confidence and shipped a confident-looking
score from garbage parser output. The hard filter is the integrity gate.

### Example B — Acme AI Senior ML Engineer (structured JD, fully reproducible)

Same engine, this time on a JD with explicit `Title:` / `Company:` /
`Location:` headers + a "Requirements" section + a "Nice to have"
section. **Every number below is reproducible from a clone of this
repo** — the exact JD and profile are committed in
`scripts/demo_example.py`, and the run is deterministic (pinned model
revision, LLM-absent path, no API key needed):

```
$ python -m scripts.demo_example

parse_confidence:  1.00    (8 of 8 structural cues found)
skills_match:      0.895   (8/8 required matched + 1/3 nice-to-have)
experience_match:  1.000   (5+ years required, demo profile has 6)
semantic_sim:      0.879   (all-MiniLM-L6-v2 @ pinned revision)
llm_confidence:    0.000   (LLM-absent path — no API key)
role_level_fit:    1.000   (senior JD, senior profile)

apply_score:       70.0    ← APPLY band (65 ≤ s < 80)
verdict:           APPLY
dominant_signal:   skills_match
near_threshold:    False
```

The deterministic signal values are also pinned by
`tests/test_demo_example.py`, so if this README block ever drifts from
what the engine computes, CI fails.

With OpenAI enabled, the LLM adds its bounded signal (≤ 25 points at
`llm_confidence = 1.0`); a calibrated confidence around 0.75 lifts this
JD into PRIORITY. The reasoning panel from a live GPT-4o session on this
JD (illustrative — LLM output is stochastic and not reproducible by
design; it is captured per-decision for replay):

- Strengths: "6 years in ML engineering, meets experience requirement" · "Proficient in Python, PyTorch, FastAPI, and Docker" · "Strong MLOps background with AWS experience" · "Experience with LLM pipelines aligns with preferred skills"
- Gaps: "No explicit mention of Kubernetes experience" · "Lacks direct mention of LangChain experience"
- Risks: "Potential gap in Kubernetes could affect deployment tasks"
- Talking points (5, copy-pasteable for cover letter)

### How I read the verdicts in practice

| Verdict | What I do |
|---|---|
| **PRIORITY** (≥80) | Apply same day. Use the talking points; address gaps in cover letter. |
| **APPLY** (65–80) | Apply within the week. Read the trace first — if `near_threshold_flag = True` (within 3 points of REVIEW), check the LLM gaps before drafting. |
| **REVIEW** (50–65) | Read the JD manually. The system flags this when it can't be confident. About 30% of my real-world JDs land here. |
| **SKIP** (<50) | Trust the system. Move on. |
| **PARSE_FAILURE** (orthogonal — not a score tier) | The JD couldn't be parsed reliably (`parse_confidence < 0.5`). `apply_score` is `None` (not 0) — score is undefined, not "0% match". Re-paste a cleaner copy of the JD or read it manually. |

Note: PRIORITY/APPLY/REVIEW/SKIP are **fit-signal** verdicts derived from
`apply_score`. PARSE_FAILURE is an **input-quality** verdict on a separate
axis — a JD whose text is unparseable is not "a 0% match", it's
unscorable. The two distinctions lead to different next steps, so they
are reported as different verdicts.

The audit log is in MongoDB Atlas. Every decision has its signal vector,
weights, thresholds version, engine version, and (when LLM ran) the raw
reasoning blob — so I can re-derive any past verdict and verify it was
correct given what was known at the time.

---

## 3. System design

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
**pure** — no I/O, no network, no model calls. All I/O lives behind the
`Store` and `LLMReasoner` Protocol seams in `src/db.py` and
`src/llm/reasoning.py`. The orchestrator in `src/engine/orchestrator.py`
wires the layers into a single `evaluate_job()` entrypoint.

The Streamlit UI is a thin renderer over `evaluate_job()` — it does not
recompute scores, re-interpret signals, or call the LLM independently.
Three grep-tests enforce this so a UI tweak can't silently drift the
contract.

---

## 4. The decision formula

Hard filters (applied BEFORE the weighted sum, in this order):

1. `parse_confidence < 0.5`  → verdict `PARSE_FAILURE` (score = `None`, undefined)
2. `dealbreaker_hit == True` → verdict `SKIP` (score = 0)

Input quality is checked first on purpose: a dealbreaker inferred from a
JD the parser couldn't reliably read is itself unreliable, so garbage
input yields PARSE_FAILURE — never a confident-looking SKIP.

Otherwise:

```
apply_score = 100 × ( 0.30 × skills_match
                    + 0.20 × experience_match
                    + 0.15 × semantic_similarity
                    + 0.25 × llm_confidence
                    + 0.10 × role_level_fit )
```

All signals are in `[0, 1]` except `role_level_fit` which is discrete
`{0, 0.5, 1}`. The weights sum to `1.0` exactly (enforced by a Pydantic
validator). The weights are **priors, not learned parameters** —
defensible per row, intentionally not fit to data. They will be retuned
only when at least 50 real outcomes accumulate.

Verdict thresholds (fit-signal verdicts, when a numeric `apply_score` exists):

| Score range    | Verdict    |
|----------------|------------|
| `score ≥ 80`   | `PRIORITY` |
| `65 ≤ s < 80`  | `APPLY`    |
| `50 ≤ s < 65`  | `REVIEW`   |
| `score < 50`   | `SKIP`     |

`PARSE_FAILURE` sits on an orthogonal axis: when the JD cannot be parsed,
`apply_score` is `None` and no threshold mapping applies.

Both weights and thresholds version are stored on every persisted
decision document, so a decision from today stays reproducible even if
v2 retunes them.

---

## 5. LLM boundary

The LLM contributes in exactly two ways:

1. As a **bounded numeric signal** (`llm_confidence ∈ [0, 1]`, weighted
   at `0.25`). Capped so it cannot single-handedly flip a decision —
   enforced by `test_config.py::test_llm_weight_not_dominant`.
2. As an **explanatory layer** (strengths, gaps, risks, talking points).
   These are stored on the decision for display; no downstream
   calculation reads them.

The LLM does **not** define the decision boundary, control thresholds,
override scoring logic, or recompute any signal.

If the LLM call fails (network error, rate limit, timeout, schema
violation, retry exhausted), the decision still ships — with
`reasoning=None` and `llm_confidence=0.0`. Both failure classes are
tested: `test_llm_reasoning.py::TestTransportFailures` proves a dead
network cannot crash an evaluation (each API request is also bounded by
an explicit 30-second timeout), and
`test_orchestrator.py::test_dealbreaker_forces_skip_regardless_of_llm`
proves that an LLM returning `llm_confidence=1.0` cannot override a
dealbreaker SKIP verdict.

---

## 6. Evaluation and honest scope

`scripts/evaluate.py` computes precision-of-APPLY, interview rate,
false-positive rate, and precision-of-PRIORITY — **only when at least
50 real outcomes are logged**. Below that threshold, the script returns:

```
INSUFFICIENT DATA (N=X, need >= 50). Evaluation framework ready;
metrics will populate when real outcome data reaches the threshold.
No simulated metrics will ever be shown.
```

The `MIN_OUTCOMES_FOR_EVALUATION = 50` constant is module-locked at
`scripts/evaluate.py` and pinned by a dedicated test, specifically to
make "temporary tweaks at N=47" harder to ship. Simulating feedback data
to populate the metrics would make the system unfalsifiable; that
self-imposed constraint is the integrity claim of the project.

### What this system is NOT

- Not a hiring tool. It does not score candidates. It scores job
  descriptions for one specific candidate (me).
- Not a recommendation engine. It does not search for jobs; you paste
  the JD in.
- Not multi-tenant. Single-user by design — a v2 trigger when
  multi-user actually matters.
- Not a substitute for reading a JD. The REVIEW verdict explicitly
  routes ambiguous cases to human-in-the-loop.

---

## 7. Proof of correctness

The above is only credible if the system actually behaves the way the
spec says. Three layers of evidence in the repo:

- **A 300+ test hermetic suite** covering schemas, the deterministic
  scorer, the parser (including adversarial extraction cases — prose
  like "Requirements" must never produce phantom skills), every signal,
  persistence (with append-only contract), the LLM Protocol seam (with
  retry-or-fallback AND transport-failure wrapping), the orchestrator
  end-to-end, and the README contract itself. Runtime: ~1–2 seconds on
  a developer laptop. No network, no model downloads.
- **Determinism by construction, enforced in tests.** The scorer is a
  pure function (no I/O imports — grep-tested), the embedding model is
  pinned to an exact revision in both the runtime and the Docker
  pre-warm (a test fails if the two pins drift), and same-input →
  same-output is asserted at the extraction, signal, and scorer layers.
  `python -m scripts.demo_example` reproduces the README's Example B
  numbers from source on any machine. (LLM output is stochastic but
  bounded at 25% weight and captured per-decision for replay.)
- **CI-enforced architectural invariants** in `.github/workflows/ci.yml`:
  four-job gate (privacy audit · hermetic test suite · ruff + mypy lint
  and type gates · auto-deploy to HF Space). Privacy audit fails if any
  internal artefact leaks into git. Tests, lint, and types all gate the
  deploy. Branch protection on `main` enforces the whole pipeline.

The README itself is contract-tested — formula values quoted here must
match `src/config.py` exactly, and Example B's deterministic signal
values are pinned by `tests/test_demo_example.py` — drift fails CI.
