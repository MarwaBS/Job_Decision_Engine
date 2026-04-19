"""Engine layer — the decision core.

Pure functions only. No I/O, no database, no HTTP, no LLM calls. The engine
is the one module in the system that must be provably deterministic and
testable with fixed inputs — see EXECUTION_RULES §7 and architecture §11
acceptance criteria.
"""
