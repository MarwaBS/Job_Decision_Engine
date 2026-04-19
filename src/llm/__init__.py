"""LLM reasoning layer — produces structured explanations for decisions.

Architecture §7: the LLM is a SIGNAL (bounded `llm_confidence` contribution)
AND an explanatory layer (strengths / gaps / risks / talking points).

It is NOT the decision boundary, threshold controller, or verdict setter.
The scorer always runs; the LLM enriches. If the LLM is unavailable, the
decision still ships with `reasoning = None` and `llm_confidence = 0.0`.
"""
