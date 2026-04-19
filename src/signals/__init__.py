"""Signal-extraction layer.

Each module here computes one signal from `(Job, CandidateProfile)` and
returns a float in `[0, 1]`. Signals are combined (not here) by the scoring
engine per architecture §6.
"""
