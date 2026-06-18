"""Experience-match signal — pure function.

A REAL signal. Weight W_experience = 0.20.

Returns a float in [0, 1] representing how well the candidate's years of
experience match what the job requires.

Design:

- If the job doesn't specify `years_required`, the signal is **1.0**. Rationale:
  the JD didn't make a hard claim we can measure against, so we don't
  penalise. This is intentional — penalising for missing data would
  incentivise the parser to invent values.
- If the candidate has ≥ required years → **1.0**.
- If the candidate has fewer years → linear fall-off with a floor at 0.0.
  A 5-year requirement and a 3-year candidate scores 3/5 = 0.6. A 5-year
  requirement and a 0-year candidate scores 0.0.
- If the candidate has significantly MORE years than required, the signal is
  still **1.0**. Over-qualification is a valid concern, but it is deliberately
  NOT expressed in the deterministic score or the decision trace: it is left to
  the LLM, whose prompt receives both `years_required` and `years_experience`
  and is explicitly instructed to raise over-qualification under "risks". Baking
  a fixed over-qualification penalty into the score would punish strong
  candidates by rule; flagging it as a soft, contextual risk is the right place.

The function is pure: same inputs → same output, no I/O.
"""

from __future__ import annotations

from src.schemas import CandidateProfile, ParsedJob


def compute_experience_match(job: ParsedJob, profile: CandidateProfile) -> float:
    """Return the experience-match signal ∈ [0, 1].

    See module docstring for the exact rules.
    """
    required = job.years_required
    have = profile.years_experience

    if required is None:
        # No structural claim from the JD → don't penalise.
        return 1.0

    if required <= 0:
        # Defensive: a "0 years required" JD is treated as fully met by
        # anyone, including new grads.
        return 1.0

    if have >= required:
        return 1.0

    # Linear fall-off from required → 0.
    ratio = have / required
    return max(0.0, min(1.0, ratio))
