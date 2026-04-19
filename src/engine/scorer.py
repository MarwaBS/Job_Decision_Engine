"""Deterministic scoring function for the Job Decision Engine.

Implements architecture §6 exactly. Pure function: same (signals, weights,
thresholds) → same (score, verdict, decision_trace), always.

This is the only module in the system that can claim "provably deterministic".
It has no imports from `db`, `llm`, or any I/O layer, by design. If that ever
changes, the test suite will catch it (`test_scorer.py::test_no_io_imports`).
"""

from __future__ import annotations

from src.config import (
    ENGINE_VERSION,
    MIN_PARSE_CONFIDENCE,
    NEAR_THRESHOLD_DISTANCE,
    THRESHOLDS,
    WEIGHTS,
)
from src.schemas import (
    DecisionResult,
    DecisionSensitivity,
    DecisionTrace,
    FailureMode,
    Signals,
    Thresholds,
    Verdict,
    Weights,
)

# ── Public API ───────────────────────────────────────────────────────────────


def score(
    signals: Signals,
    weights: Weights = WEIGHTS,
    thresholds: Thresholds = THRESHOLDS,
) -> DecisionResult:
    """Score a (Job, Profile) signal vector into a verdict.

    Args:
        signals: Five-signal vector + hard-filter inputs. Architecture §6.
        weights: Scoring weights. Defaults to the locked v1.0 priors.
        thresholds: Score cutoffs. Defaults to the locked v1.0 thresholds.

    Returns:
        DecisionResult with apply_score, verdict, full decision_trace.

    The function is pure — no side effects, no I/O. It is safe to call from a
    test with any Signals instance and get a reproducible result.
    """
    # ── Hard filters (applied BEFORE the weighted sum) ───────────────────────
    # Architecture §6: "Hard filters" subsection.

    if signals.dealbreaker_hit:
        return _short_circuit(
            signals=signals,
            weights=weights,
            thresholds=thresholds,
            apply_score=0.0,
            verdict=Verdict.SKIP,
            dominant_signal="dealbreaker",
            failure_mode=FailureMode.DEALBREAKER_HIT,
        )

    if signals.parse_confidence < MIN_PARSE_CONFIDENCE:
        return _short_circuit(
            signals=signals,
            weights=weights,
            thresholds=thresholds,
            apply_score=0.0,
            verdict=Verdict.REVIEW,
            dominant_signal="low_parse_confidence",
            failure_mode=FailureMode.LOW_PARSE_CONFIDENCE,
        )

    # ── Weighted sum (architecture §6) ───────────────────────────────────────

    weighted = _weighted_contributions(signals, weights)
    apply_score = 100.0 * sum(weighted.values())

    # ── Verdict from thresholds ──────────────────────────────────────────────

    verdict = _score_to_verdict(apply_score, thresholds)

    # ── Decision trace (architecture §5.3) ───────────────────────────────────

    dominant = _dominant_signal(weighted)
    sensitivity = _compute_sensitivity(signals, weights)
    nearest = _nearest_threshold_distance(apply_score, thresholds)
    near_flag = nearest < NEAR_THRESHOLD_DISTANCE

    trace = DecisionTrace(
        dominant_signal=dominant,
        failure_mode_detected=None,
        decision_sensitivity=sensitivity,
        nearest_threshold_distance=nearest,
        near_threshold_flag=near_flag,
    )

    return DecisionResult(
        apply_score=apply_score,
        verdict=verdict,
        signals=signals,
        weights=weights,
        thresholds_version=thresholds.version,
        decision_trace=trace,
        engine_version=ENGINE_VERSION,
    )


# ── Internals (private — never imported by other modules) ────────────────────


def _weighted_contributions(signals: Signals, weights: Weights) -> dict[str, float]:
    """Return {signal_name: weighted_contribution} for the five signals."""
    return {
        "skills_match": weights.skills * signals.skills_match,
        "experience_match": weights.experience * signals.experience_match,
        "semantic_similarity": weights.semantic * signals.semantic_similarity,
        "llm_confidence": weights.llm * signals.llm_confidence,
        "role_level_fit": weights.role * signals.role_level_fit,
    }


def _dominant_signal(weighted: dict[str, float]) -> str:
    """Return the signal name with the highest weighted contribution.

    Ties are broken by the signal's order of definition in the architecture
    (skills > experience > semantic > llm > role) — not alphabetical. This
    matters: determinism requires tie-breaking rules that are specified, not
    inherited from Python's dict ordering.
    """
    order = [
        "skills_match",
        "experience_match",
        "semantic_similarity",
        "llm_confidence",
        "role_level_fit",
    ]
    best_name = order[0]
    best_value = weighted[best_name]
    for name in order[1:]:
        if weighted[name] > best_value:
            best_name = name
            best_value = weighted[name]
    return best_name


def _score_to_verdict(apply_score: float, thresholds: Thresholds) -> Verdict:
    """Map apply_score ∈ [0, 100] to a Verdict.

    Architecture §6:
        score >= priority → PRIORITY
        apply <= score < priority → APPLY
        review <= score < apply   → REVIEW
        score < review            → SKIP
    """
    if apply_score >= thresholds.priority:
        return Verdict.PRIORITY
    if apply_score >= thresholds.apply_:
        return Verdict.APPLY
    if apply_score >= thresholds.review:
        return Verdict.REVIEW
    return Verdict.SKIP


def _compute_sensitivity(signals: Signals, weights: Weights) -> DecisionSensitivity:
    """Three counterfactual replays of the weighted sum.

    Each field re-runs the scorer under a single perturbation:

    - `if_llm_removed_score`      → set llm_confidence to 0 (weight kept)
    - `if_skills_boosted_plus_10pct` → boost skills_match by +0.10, clipped to 1.0
    - `if_experience_removed_score`  → set experience_match to 0 (weight kept)

    Weights are unchanged in all three counterfactuals — only the signal
    value moves. This keeps the replays comparable to the actual score.
    """

    def _score_only(s: Signals) -> float:
        return 100.0 * sum(_weighted_contributions(s, weights).values())

    s_no_llm = signals.model_copy(update={"llm_confidence": 0.0})
    s_skills_plus = signals.model_copy(
        update={"skills_match": min(1.0, signals.skills_match + 0.10)}
    )
    s_no_exp = signals.model_copy(update={"experience_match": 0.0})

    return DecisionSensitivity(
        if_llm_removed_score=_score_only(s_no_llm),
        if_skills_boosted_plus_10pct=_score_only(s_skills_plus),
        if_experience_removed_score=_score_only(s_no_exp),
    )


def _nearest_threshold_distance(apply_score: float, thresholds: Thresholds) -> float:
    """Absolute distance from the score to the nearest verdict boundary.

    Architecture §5.3: a small distance means the decision is borderline and
    should be reviewed. The three boundaries are `priority`, `apply`, `review`.
    """
    boundaries = (thresholds.priority, thresholds.apply_, thresholds.review)
    return min(abs(apply_score - b) for b in boundaries)


def _short_circuit(
    *,
    signals: Signals,
    weights: Weights,
    thresholds: Thresholds,
    apply_score: float,
    verdict: Verdict,
    dominant_signal: str,
    failure_mode: FailureMode,
) -> DecisionResult:
    """Build a DecisionResult for hard-filter paths (dealbreaker / low parse).

    The weighted sum is NOT computed when a hard filter fires — that's the
    whole point of a hard filter. But we still populate the decision_trace
    with zero-sensitivity placeholders so downstream consumers can treat
    every DecisionResult uniformly.
    """
    trace = DecisionTrace(
        dominant_signal=dominant_signal,  # type: ignore[arg-type]
        failure_mode_detected=failure_mode,
        decision_sensitivity=DecisionSensitivity(
            if_llm_removed_score=0.0,
            if_skills_boosted_plus_10pct=0.0,
            if_experience_removed_score=0.0,
        ),
        nearest_threshold_distance=_nearest_threshold_distance(apply_score, thresholds),
        near_threshold_flag=False,
    )
    return DecisionResult(
        apply_score=apply_score,
        verdict=verdict,
        signals=signals,
        weights=weights,
        thresholds_version=thresholds.version,
        decision_trace=trace,
        engine_version=ENGINE_VERSION,
    )
