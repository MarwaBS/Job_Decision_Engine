"""Pydantic data contracts for the Job Decision Engine.

Every type that crosses a module boundary is defined here. These Pydantic
models ARE the authoritative data contracts for the engine — they are the
source of truth, and every other module is held to the shapes defined below.

Design rules enforced here:
- Weights sum to 1.0.
- Signals are all bounded to [0, 1] except `role_level_fit` which is {0, 0.5, 1}.
- Every decision stores the weights and thresholds_version it was scored with,
  so old decisions remain reproducible after v2 retuning.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ── Enums ────────────────────────────────────────────────────────────────────


class Verdict(StrEnum):
    """Final decision output.

    Two orthogonal axes:

    - Fit-signal verdicts (threshold-derived from apply_score):
        PRIORITY, APPLY, REVIEW, SKIP.
    - Input-quality verdict (orthogonal — no apply_score is computed):
        PARSE_FAILURE. Returned when the JD could not be parsed reliably
        enough to score (parse_confidence < MIN_PARSE_CONFIDENCE). The
        `apply_score` on a PARSE_FAILURE result is `None`, not 0.0 — the
        score is undefined, not "0% match" (BUG-004).
    """

    PRIORITY = "PRIORITY"
    APPLY = "APPLY"
    REVIEW = "REVIEW"
    SKIP = "SKIP"
    PARSE_FAILURE = "PARSE_FAILURE"


class FailureMode(StrEnum):
    """Coded labels for structural issues flagged during scoring.

    None (the absence of this label on a decision) means no structural issue
    was detected. The labels themselves are fixed — adding a new one is a
    deliberate change to the scoring contract, not an ad-hoc addition.
    """

    DEALBREAKER_HIT = "dealbreaker_hit"
    LOW_PARSE_CONFIDENCE = "low_parse_confidence"
    SKILLS_PARSER_MISSED_MATCH = "skills_parser_missed_match"
    LLM_SCHEMA_VIOLATION = "llm_schema_violation"


class Seniority(StrEnum):
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"
    PRINCIPAL = "principal"


# ── Core I/O types ───────────────────────────────────────────────────────────


class Signals(BaseModel):
    """Signal vector computed from (Job, Profile). Input to the scorer.

    The five scoring signals, all in [0, 1] except role_level_fit.
    Additional fields (dealbreaker_hit, parse_confidence) gate the hard filters
    applied *before* the weighted sum.
    """

    model_config = ConfigDict(frozen=True)

    skills_match: float = Field(ge=0.0, le=1.0)
    experience_match: float = Field(ge=0.0, le=1.0)
    semantic_similarity: float = Field(ge=0.0, le=1.0)
    llm_confidence: float = Field(ge=0.0, le=1.0)
    role_level_fit: float = Field(ge=0.0, le=1.0)

    # Hard filter inputs
    dealbreaker_hit: bool = False
    parse_confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("role_level_fit")
    @classmethod
    def _check_role_level_discrete(cls, v: float) -> float:
        """role_level_fit is constrained to the discrete set {0, 0.5, 1}."""
        allowed = {0.0, 0.5, 1.0}
        if v not in allowed:
            raise ValueError(f"role_level_fit must be one of {allowed}, got {v}")
        return v


class Weights(BaseModel):
    """Scoring weights.

    Weights are priors, not learned parameters. They represent design intent,
    not statistical optimization. Any change is a deliberate retuning, paired
    with a version bump so old decisions stay reproducible.
    """

    model_config = ConfigDict(frozen=True)

    skills: float = Field(ge=0.0, le=1.0)
    experience: float = Field(ge=0.0, le=1.0)
    semantic: float = Field(ge=0.0, le=1.0)
    llm: float = Field(ge=0.0, le=1.0)
    role: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_sum_to_one(self) -> Weights:
        total = self.skills + self.experience + self.semantic + self.llm + self.role
        # Allow a tiny float tolerance but refuse meaningful drift.
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"weights must sum to 1.0. got {total!r}")
        return self


class Thresholds(BaseModel):
    """Score → verdict cutoffs.

    Versioned so old decisions remain reproducible after retuning.
    """

    model_config = ConfigDict(frozen=True)

    priority: float = Field(ge=0.0, le=100.0)  # score >= priority → PRIORITY
    apply_: float = Field(alias="apply", ge=0.0, le=100.0)
    review: float = Field(ge=0.0, le=100.0)
    version: str = "v1.0"

    @model_validator(mode="after")
    def _check_monotonic(self) -> Thresholds:
        if not (self.review < self.apply_ < self.priority):
            raise ValueError(
                "thresholds must satisfy review < apply < priority "
                f"(got review={self.review}, apply={self.apply_}, priority={self.priority})"
            )
        return self


# ── Decision trace (added during review) ─────────────────────────────────────


class DecisionSensitivity(BaseModel):
    """Counterfactual replays of the scoring function.

    Each field is the score that would have been produced under a single
    counterfactual perturbation, with all other signals held constant.
    """

    model_config = ConfigDict(frozen=True)

    if_llm_removed_score: float = Field(ge=0.0, le=100.0)
    if_skills_boosted_plus_10pct: float = Field(ge=0.0, le=100.0)
    if_experience_removed_score: float = Field(ge=0.0, le=100.0)


#: The vocabulary of `DecisionTrace.dominant_signal`. A named alias (not an
#: inline Literal) so the scorer can type its helpers against the same
#: vocabulary instead of suppressing mypy at the construction site.
DominantSignal = Literal[
    "skills_match",
    "experience_match",
    "semantic_similarity",
    "llm_confidence",
    "role_level_fit",
    "dealbreaker",
    "low_parse_confidence",
]


class DecisionTrace(BaseModel):
    """Explains WHY a given score produced a given verdict.

    The differentiator. Answers "what if one signal was different?" with
    numbers, not prose.
    """

    model_config = ConfigDict(frozen=True)

    dominant_signal: DominantSignal
    failure_mode_detected: FailureMode | None
    decision_sensitivity: DecisionSensitivity
    nearest_threshold_distance: float = Field(ge=0.0, le=100.0)
    near_threshold_flag: bool


# ── Output of the scorer ─────────────────────────────────────────────────────


class DecisionResult(BaseModel):
    """Full output of a single evaluate_job() call.

    Stored as a `decisions` document. Includes the weights
    and thresholds_version so old decisions remain reproducible.
    """

    model_config = ConfigDict(frozen=True)

    # `None` only on the PARSE_FAILURE path: the JD could not be parsed
    # reliably enough to score, so the score is semantically undefined.
    # All fit-signal verdicts (PRIORITY/APPLY/REVIEW/SKIP) carry a float.
    apply_score: float | None = Field(default=None, ge=0.0, le=100.0)
    verdict: Verdict
    signals: Signals
    weights: Weights
    thresholds_version: str
    decision_trace: DecisionTrace
    engine_version: str

    # Reasoning is produced by the LLM layer (Step 4). Absent in Step 2.
    # A Field with default=None keeps Step 2 green while documenting the hook.
    reasoning: dict | None = None


# ── Persistence envelopes ────────────────────────────────────────────────────


#: Recognized dealbreaker keys. The orchestrator's hard filter silently ignores
#: keys it doesn't recognize, so an unvalidated typo (e.g. "requires_10yr_exp")
#: would disable a dealbreaker with no signal. CandidateProfile rejects unknown
#: keys at construction instead. Semantics live in
#: ``src.engine.orchestrator._check_dealbreakers`` — every key listed here is
#: enforced there; the vocabulary contains no no-op entries.
KNOWN_DEALBREAKERS: frozenset[str] = frozenset(
    {"requires_10_yr_exp", "on_site_only", "no_pytorch"}
)


class CandidateProfile(BaseModel):
    """Candidate profile.

    Versioned: changing the profile bumps `profile_version` and persists a new
    document so past decisions stay explainable against their exact profile.
    """

    model_config = ConfigDict(frozen=True)

    profile_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    active: bool = True
    name: str
    summary: str
    years_experience: float = Field(ge=0.0)
    seniority: Seniority
    skills_tech: list[str] = Field(default_factory=list)
    skills_tools: list[str] = Field(default_factory=list)
    skills_domain: list[str] = Field(default_factory=list)
    target_roles: list[str] = Field(default_factory=list)
    target_locations: list[str] = Field(default_factory=list)
    must_haves: list[str] = Field(default_factory=list)
    nice_to_haves: list[str] = Field(default_factory=list)
    dealbreakers: list[str] = Field(default_factory=list)

    @field_validator("dealbreakers")
    @classmethod
    def _validate_dealbreakers(cls, value: list[str]) -> list[str]:
        """Reject unrecognized dealbreaker keys so a typo fails fast here
        rather than silently disabling a hard filter at scoring time."""
        unknown = sorted(
            {d for d in value if d.strip().lower() not in KNOWN_DEALBREAKERS}
        )
        if unknown:
            raise ValueError(
                f"unknown dealbreaker key(s): {unknown}; "
                f"valid keys are {sorted(KNOWN_DEALBREAKERS)}"
            )
        return value


class ParsedJob(BaseModel):
    """Output of the ingestion layer.

    `parse_confidence < 0.5` triggers the REVIEW hard filter in the scorer.
    """

    model_config = ConfigDict(frozen=True)

    title: str
    company: str | None = None
    location: str | None = None
    # Tri-state: True = remote/hybrid mentioned, False = on-site explicitly
    # mentioned, None = the JD is silent on workplace. The `on_site_only`
    # dealbreaker fires only on an explicit False — never on None.
    remote: bool | None = None
    seniority: Seniority | None = None
    years_required: float | None = None
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    salary_range_usd: tuple[int, int] | None = None


class Job(BaseModel):
    """A single ingested job description."""

    model_config = ConfigDict(frozen=True)

    content_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: Literal["paste", "url", "upload"]
    source_url: str | None = None
    raw_text: str
    parsed: ParsedJob
    parse_confidence: float = Field(ge=0.0, le=1.0)
    parse_warnings: list[str] = Field(default_factory=list)


# Outcome + FeedbackLog are defined now so Step 4 persistence doesn't break the
# data contract. They are not referenced in Step 2 code beyond export.


class OutcomeStage(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: Literal[
        "SUBMITTED",
        "CALLBACK",
        "INTERVIEW",
        "OFFER",
        "REJECTED",
        "GHOSTED",
        "WITHDRAWN",
    ]
    at: datetime


class Outcome(BaseModel):
    """Job-search outcome — manually entered by the candidate."""

    model_config = ConfigDict(frozen=True)

    decision_id: str
    submitted_at: datetime
    stages: list[OutcomeStage]
    final_stage: Literal["OFFER", "REJECTED", "GHOSTED", "WITHDRAWN"] | None = None
    time_to_first_response_days: int | None = Field(default=None, ge=0)
    notes: str | None = None


class ReasoningOutput(BaseModel):
    """LLM reasoning output — the strict JSON contract.

    The LLM MUST produce JSON matching this exact shape.
    If it doesn't, the reasoning layer retries once; if that also fails,
    the reasoning is set to `None` on the `DecisionResult` and the scorer
    continues with `llm_confidence = 0.0` — the LLM never causes a decision
    failure.

    Bounds:
        - strengths:  3-5 bullets, each ≤ 120 chars
        - gaps:       2-4 bullets, each ≤ 120 chars
        - risks:      1-3 bullets, each ≤ 120 chars
        - llm_confidence: [0, 1], calibrated
        - recommended_talking_points: 3-5 bullets
    """

    model_config = ConfigDict(frozen=True)

    strengths: list[str] = Field(min_length=3, max_length=5)
    gaps: list[str] = Field(min_length=2, max_length=4)
    risks: list[str] = Field(min_length=1, max_length=3)
    llm_confidence: float = Field(ge=0.0, le=1.0)
    recommended_talking_points: list[str] = Field(min_length=3, max_length=5)

    @field_validator("strengths", "gaps", "risks", "recommended_talking_points")
    @classmethod
    def _max_bullet_length(cls, bullets: list[str]) -> list[str]:
        for i, b in enumerate(bullets):
            if len(b) > 120:
                raise ValueError(f"bullet {i} exceeds 120 chars: {len(b)} chars")
            if not b.strip():
                raise ValueError(f"bullet {i} is empty or whitespace-only")
        return bullets


class FeedbackLog(BaseModel):
    """User-authored corrections. Logged only, not fitted."""

    model_config = ConfigDict(frozen=True)

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    decision_id: str
    feedback_type: Literal[
        "score_too_low", "score_too_high", "verdict_wrong", "reasoning_off"
    ]
    expected_verdict: Verdict | None = None
    actual_verdict: Verdict | None = None
    reason: str
