"""Streamlit UI — presentation layer only.

Per EXECUTION_RULES + Step 5 authorization: this package is a thin
rendering layer over a deterministic decision engine. It does not
recompute scores, re-interpret signals, or introduce logic branches
that diverge from the tested pipeline.
"""
