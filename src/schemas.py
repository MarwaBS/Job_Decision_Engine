"""Pydantic data contracts for the Job Decision Engine.

Every type that crosses a module boundary is defined here. The architecture
(docs/ARCHITECTURE.md §5 and §6) is the authoritative source. If the code below
diverges from the architecture, the architecture wins — fix the code.

Design rules enforced here:
- Weights sum to 1.0 (architecture §6).
- Signals are all bounded to [0, 1] except `role_level_fit` which is {0, 0.5, 1}.
- Every decision stores the weights and thresholds_version it was scored with,
  so old decisions remain reproducible after v2 retuning.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── Enums ────────────────────────────────────────────────────────────────────


class Verdict(str, Enum):
    """Final decision output. Architecture §6 thresholds."""

    PRIORITY = "PRIORITY"
    APPLY = "APPLY"
    REVIEW = "REVIEW"
    SKIP = "SKIP"


class FailureMode(str, Enum):
    """Coded labels for structural issues flagged during scoring.

    None (the absence of this label on a decision) means no structural issue
    was detected. The labels themselves are fixed — adding a new one is an
    architecture change that requires an ADR.
    """

    DEALBREAKER_HIT = "dealbreaker_hit"
    LOW_PARSE_CONFIDENCE = "low_parse_confidence"
    SKILLS_PARSER_MISSED_MATCH = "skills_parser_missed_match"
    LLM_SCHEMA_VIOLATION = "llm_schema_violation"


class Seniority(str, Enum):
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"
    PRINCIPAL = "principal"


# ── Core I/O types ───────────────────────────────────────────────────────────


class Signals(BaseModel):
    """Signal vector computed from (Job, Profile). Input to the scorer.

    Architecture §6 — five signals, all in [0, 1] except role_level_fit.
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
        """role_level_fit is {0, 0.5, 1} per architecture §6."""
        allowed = {0.0, 0.5, 1.0}
        if v not in allowed:
            raise ValueError(f"role_level_fit must be one of {allowed}, got {v}")
        return v


class Weights(BaseModel):
    """Scoring weights. Architecture §6.

    Weights are priors, not learned parameters. They represent design intent,
    not statistical optimization. Any change requires an ADR.
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
            raise ValueError(
                f"weights must sum to 1.0 (architecture §6). got {total!r}"
            )
        return self


class Thresholds(BaseModel):
    """Score → verdict cutoffs. Architecture §6.

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


# ── Decision trace (architecture §5.3, added during review) ──────────────────


class DecisionSensitivity(BaseModel):
    """Counterfactual replays of the scoring function.

    Each field is the score that would have been produced under a single
    counterfactual perturbation, with all other signals held constant.
    """

    model_config = ConfigDict(frozen=True)

    if_llm_removed_score: float = Field(ge=0.0, le=100.0)
    if_skills_boosted_plus_10pct: float = Field(ge=0.0, le=100.0)
    if_experience_removed_score: float = Field(ge=0.0, le=100.0)


class DecisionTrace(BaseModel):
    """Explains WHY a given score produced a given verdict.

    Architecture §5.3 — the differentiator. Answers "what if one signal was
    different?" with numbers, not prose.
    """

    model_config = ConfigDict(frozen=True)

    dominant_signal: Literal[
        "skills_match",
        "experience_match",
        "semantic_similarity",
        "llm_confidence",
        "role_level_fit",
        "dealbreaker",
        "low_parse_confidence",
    ]
    failure_mode_detected: FailureMode | None
    decision_sensitivity: DecisionSensitivity
    nearest_threshold_distance: float = Field(ge=0.0, le=100.0)
    near_threshold_flag: bool


# ── Output of the scorer ─────────────────────────────────────────────────────


class DecisionResult(BaseModel):
    """Full output of a single evaluate_job() call.

    Stored as a `decisions` document (architecture §5.3). Includes the weights
    and thresholds_version so old decisions remain reproducible.
    """

    model_config = ConfigDict(frozen=True)

    apply_score: float = Field(ge=0.0, le=100.0)
    verdict: Verdict
    signals: Signals
    weights: Weights
    thresholds_version: str
    decision_trace: DecisionTrace
    engine_version: str

    # Reasoning is produced by the LLM layer (Step 4). Absent in Step 2.
    # A Field with default=None keeps Step 2 green while documenting the hook.
    reasoning: dict | None = None


# ── Persistence envelopes (architecture §5) ──────────────────────────────────


class CandidateProfile(BaseModel):
    """Candidate profile. Architecture §5.1.

    Versioned: changing the profile bumps `profile_version` and persists a new
    document so past decisions stay explainable against their exact profile.
    """

    model_config = ConfigDict(frozen=True)

    profile_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
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


class ParsedJob(BaseModel):
    """Output of the ingestion layer. Architecture §5.2.

    `parse_confidence < 0.5` triggers the REVIEW hard filter in the scorer.
    """

    model_config = ConfigDict(frozen=True)

    title: str
    company: str | None = None
    location: str | None = None
    remote: bool = False
    seniority: Seniority | None = None
    years_required: float | None = None
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    salary_range_usd: tuple[int, int] | None = None


class Job(BaseModel):
    """A single ingested job description. Architecture §5.2."""

    model_config = ConfigDict(frozen=True)

    content_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
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
        "SUBMITTED", "CALLBACK", "INTERVIEW", "OFFER", "REJECTED", "GHOSTED", "WITHDRAWN"
    ]
    at: datetime


class Outcome(BaseModel):
    """Architecture §5.4 — manually entered by the candidate."""

    model_config = ConfigDict(frozen=True)

    decision_id: str
    submitted_at: datetime
    stages: list[OutcomeStage]
    final_stage: Literal["OFFER", "REJECTED", "GHOSTED", "WITHDRAWN"] | None = None
    time_to_first_response_days: int | None = Field(default=None, ge=0)
    notes: str | None = None


class ReasoningOutput(BaseModel):
    """LLM reasoning output — the strict JSON contract.

    Architecture §7. The LLM MUST produce JSON matching this exact shape.
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
    """Architecture §5.5 — user-authored corrections. Logged only, not fitted."""

    model_config = ConfigDict(frozen=True)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    decision_id: str
    feedback_type: Literal[
        "score_too_low", "score_too_high", "verdict_wrong", "reasoning_off"
    ]
    expected_verdict: Verdict | None = None
    actual_verdict: Verdict | None = None
    reason: str
