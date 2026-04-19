"""Evaluation script — STUB-enforced until N≥50 real outcomes.

Step 4 rule #3 (non-negotiable):
    "Until N >= 50 real outcomes:
     - NO fake metrics
     - NO simulated outcomes
     - NO synthetic 'performance reporting'
     It is intentionally inert."

Architecture §8: the framework is built; the metrics are only computed
when enough real data exists. This is the integrity claim of the whole
project — simulating feedback data to show a "working" evaluation would
be worse than no evaluation.

Usage:
    python -m scripts.evaluate         # reads from Mongo, returns STUB if N<50
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any

from src.db import MongoStore, Store


MIN_OUTCOMES_FOR_EVALUATION: int = 50
"""Hard gate. Below this, the script MUST return the STUB message.

Matches architecture §8 and the Step 4 rule. Intentionally locked at the
module level (not in config.py) so it's harder to "temporarily tweak" when
the temptation hits at N=47.
"""


# ── Result type ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EvaluationResult:
    """Output of the evaluation script.

    One of two shapes:
        - STUB: `n_outcomes < 50`, `metrics = {}`, `message` explains why.
        - REAL: `n_outcomes >= 50`, `metrics` populated, `message` is a summary.
    """

    n_outcomes: int
    is_stub: bool
    message: str
    metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_outcomes": self.n_outcomes,
            "is_stub": self.is_stub,
            "message": self.message,
            "metrics": self.metrics,
        }


# ── Public API ───────────────────────────────────────────────────────────────


def evaluate(store: Store) -> EvaluationResult:
    """Run the evaluation over whatever outcomes the store currently holds.

    Always returns the STUB shape if fewer than 50 outcomes exist. Never
    fabricates data. Never "projects" from small samples.
    """
    outcomes = store.list_outcomes(limit=100_000)
    n = len(outcomes)

    if n < MIN_OUTCOMES_FOR_EVALUATION:
        return EvaluationResult(
            n_outcomes=n,
            is_stub=True,
            message=(
                f"INSUFFICIENT DATA (N={n}, need >= {MIN_OUTCOMES_FOR_EVALUATION}). "
                "Evaluation framework ready; metrics will populate when real "
                "outcome data reaches the threshold. No simulated metrics will "
                "ever be shown."
            ),
        )

    # N >= 50. Compute the real metrics documented in architecture §8.
    metrics = _compute_metrics(outcomes, store)
    return EvaluationResult(
        n_outcomes=n,
        is_stub=False,
        message=(
            f"Evaluation over N={n} real outcomes. "
            f"precision_apply={metrics.get('precision_apply', 0):.3f}, "
            f"interview_rate={metrics.get('interview_rate', 0):.3f}, "
            f"false_positive_rate={metrics.get('false_positive_rate', 0):.3f}"
        ),
        metrics=metrics,
    )


# ── Internals ────────────────────────────────────────────────────────────────


def _compute_metrics(
    outcomes: list[dict[str, Any]], store: Store
) -> dict[str, float]:
    """Compute the architecture §8 metric set.

    - precision_apply: (callbacks + interviews + offers) / applications
    - precision_priority: same, filtered to verdict=PRIORITY decisions
    - interview_rate: interviews / applications
    - false_positive_rate: (rejected within 7 days) / applications

    Only called when `n >= MIN_OUTCOMES_FOR_EVALUATION`. The presence of
    this function is the "framework ready" claim; the gate in `evaluate()`
    is the "intentionally inert" claim.
    """
    if not outcomes:
        return {}

    total = len(outcomes)

    def _had_any(stages: list[dict[str, Any]], *names: str) -> bool:
        return any(s.get("stage") in names for s in stages)

    positive = sum(
        1 for o in outcomes
        if _had_any(o.get("stages", []), "CALLBACK", "INTERVIEW", "OFFER")
    )
    interviews = sum(
        1 for o in outcomes
        if _had_any(o.get("stages", []), "INTERVIEW", "OFFER")
    )
    fast_rejections = sum(
        1 for o in outcomes
        if o.get("final_stage") == "REJECTED"
        and (o.get("time_to_first_response_days") or 999) <= 7
    )

    metrics = {
        "precision_apply": positive / total if total else 0.0,
        "interview_rate": interviews / total if total else 0.0,
        "false_positive_rate": fast_rejections / total if total else 0.0,
    }

    # precision_priority requires joining outcomes to their originating
    # decisions. Keep it simple: fetch all decisions, index by id.
    decisions = {d.get("_id"): d for d in store.list_decisions(limit=100_000)}
    priority_outcomes = [
        o for o in outcomes
        if decisions.get(o.get("decision_id"), {}).get("verdict") == "PRIORITY"
    ]
    if priority_outcomes:
        priority_positive = sum(
            1 for o in priority_outcomes
            if _had_any(o.get("stages", []), "CALLBACK", "INTERVIEW", "OFFER")
        )
        metrics["precision_priority"] = priority_positive / len(priority_outcomes)

    return metrics


# ── CLI ──────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    _ = argv
    try:
        store = MongoStore()
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    result = evaluate(store)
    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [
    "MIN_OUTCOMES_FOR_EVALUATION",
    "EvaluationResult",
    "evaluate",
    "main",
]
