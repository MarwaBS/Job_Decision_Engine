"""Evaluation script — STUB-enforced until N≥50 real outcomes.

Integrity rule (non-negotiable): until N >= 50 real outcomes, there are
NO fake metrics, NO simulated outcomes, and NO synthetic "performance
reporting" — the script is intentionally inert.

The framework is built; the metrics are only computed when enough real
data exists. This is the integrity claim of the whole project — simulating
feedback data to show a "working" evaluation would be worse than no
evaluation.

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

Intentionally locked at the module level (not in config.py) so it's harder
to "temporarily tweak" when the temptation hits at N=47.
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

    # N >= 50. Compute the real outcome metrics.
    metrics = _compute_metrics(outcomes, store)

    def _fmt(key: str) -> str:
        # "n/a" when a metric is legitimately absent (e.g. no outcome joins
        # to an APPLY-verdict decision) — never a fabricated 0.000.
        return f"{metrics[key]:.3f}" if key in metrics else "n/a"

    return EvaluationResult(
        n_outcomes=n,
        is_stub=False,
        message=(
            f"Evaluation over N={n} real outcomes. "
            f"precision_apply={_fmt('precision_apply')}, "
            f"interview_rate={_fmt('interview_rate')}, "
            f"false_positive_rate={_fmt('false_positive_rate')}"
        ),
        metrics=metrics,
    )


# ── Internals ────────────────────────────────────────────────────────────────


def _compute_metrics(outcomes: list[dict[str, Any]], store: Store) -> dict[str, float]:
    """Compute the outcome metric set.

    - precision_apply: (callbacks + interviews + offers) / outcomes whose
      originating decision had verdict=APPLY — precision-of-APPLY, per
      README §6. Omitted when no outcome joins to an APPLY decision.
    - precision_priority: same, for verdict=PRIORITY decisions.
    - interview_rate: interviews / all submitted outcomes
    - false_positive_rate: (rejected within 7 days) / all submitted outcomes

    The two precision metrics are verdict-scoped via the decision join;
    interview_rate and false_positive_rate are deliberately over ALL
    submitted outcomes (their names claim no verdict scope).

    Only called when `n >= MIN_OUTCOMES_FOR_EVALUATION`. The presence of
    this function is the "framework ready" claim; the gate in `evaluate()`
    is the "intentionally inert" claim.
    """
    if not outcomes:
        return {}

    total = len(outcomes)

    def _had_any(stages: list[dict[str, Any]], *names: str) -> bool:
        return any(s.get("stage") in names for s in stages)

    def _positive(subset: list[dict[str, Any]]) -> int:
        return sum(
            1
            for o in subset
            if _had_any(o.get("stages", []), "CALLBACK", "INTERVIEW", "OFFER")
        )

    interviews = sum(
        1 for o in outcomes if _had_any(o.get("stages", []), "INTERVIEW", "OFFER")
    )
    fast_rejections = sum(
        1
        for o in outcomes
        if o.get("final_stage") == "REJECTED"
        and (ttr := o.get("time_to_first_response_days")) is not None
        and ttr <= 7
    )

    metrics = {
        "interview_rate": interviews / total if total else 0.0,
        "false_positive_rate": fast_rejections / total if total else 0.0,
    }

    # The precision metrics require joining outcomes to their originating
    # decisions to filter by verdict. Outcomes store `decision_id` as the
    # STRING form of the inserted id (db.py returns `str(inserted_id)`),
    # while Mongo documents carry a raw ObjectId in `_id` — so the index
    # key must be stringified or the join silently never matches on the
    # production path.
    decisions = {str(d.get("_id")): d for d in store.list_decisions(limit=100_000)}

    def _outcomes_with_verdict(verdict: str) -> list[dict[str, Any]]:
        return [
            o
            for o in outcomes
            if decisions.get(str(o.get("decision_id")), {}).get("verdict") == verdict
        ]

    apply_outcomes = _outcomes_with_verdict("APPLY")
    if apply_outcomes:
        metrics["precision_apply"] = _positive(apply_outcomes) / len(apply_outcomes)

    priority_outcomes = _outcomes_with_verdict("PRIORITY")
    if priority_outcomes:
        metrics["precision_priority"] = _positive(priority_outcomes) / len(
            priority_outcomes
        )

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
