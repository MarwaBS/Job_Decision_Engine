"""Immutable configuration for the Job Decision Engine.

This module is the single source of truth for:

- Scoring weights
- Verdict thresholds
- Engine version
- Near-threshold distance cutoff

Values here are **priors**, not learned parameters. Changing any of them is a
deliberate change-control action:

1. Bump the matching version constant (WEIGHTS_VERSION or THRESHOLDS_VERSION)
   so existing decisions stay reproducible against the values they used.
2. Record the rationale for the new numbers alongside the change.

Immutability is enforced at runtime: the exported objects are `frozen`
Pydantic models, so any attempt to reassign a field raises `ValidationError`.
Reassigning the module-level binding itself is beyond Python's control; the
rule (not the technology) is what protects us from drift.
"""

from __future__ import annotations

from src.schemas import Thresholds, Weights

# ── Engine identity ──────────────────────────────────────────────────────────

# 0.2.0: skill extraction gained boundary-anchored alias matching, the
# `on_site_only` dealbreaker stopped firing on workplace-silent JDs, and the
# parse-confidence hard filter now precedes the dealbreaker filter. Same JD +
# profile can score differently than under 0.1.0 — hence the bump (decisions
# persist the engine_version they were scored with).
ENGINE_VERSION: str = "0.2.0"
WEIGHTS_VERSION: str = "v1.0"
THRESHOLDS_VERSION: str = "v1.0"


# ── Scoring weights (priors — not learned) ───────────────────────────────────
#
# These numbers came from design intent, not from fitting. They will be retuned
# only when the evaluation framework has N≥50 real outcomes.

WEIGHTS: Weights = Weights(
    skills=0.30,
    experience=0.20,
    semantic=0.15,
    llm=0.25,
    role=0.10,
)


# ── Verdict thresholds ───────────────────────────────────────────────────────

THRESHOLDS: Thresholds = Thresholds(
    priority=80.0,
    **{"apply": 65.0},  # `apply` is a reserved-ish keyword; use alias
    review=50.0,
    version=THRESHOLDS_VERSION,
)


# ── Near-threshold distance ──────────────────────────────────────────────────

NEAR_THRESHOLD_DISTANCE: float = 3.0
"""Absolute score distance within which `near_threshold_flag` fires.

At `NEAR_THRESHOLD_DISTANCE = 3`, a score of 78 or 82 flags as near-PRIORITY.
A flagged decision surfaces to the UI as a warning; it does NOT change the
verdict itself.
"""


# ── Hard-filter thresholds ───────────────────────────────────────────────────

MIN_PARSE_CONFIDENCE: float = 0.5
"""If `Signals.parse_confidence < MIN_PARSE_CONFIDENCE`, the scorer
short-circuits to the PARSE_FAILURE verdict with `apply_score=None` —
the score is undefined when the JD could not be parsed (BUG-004). This is
one of the scorer's hard filters, applied before the weighted sum."""
