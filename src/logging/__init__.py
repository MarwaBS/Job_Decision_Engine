"""Persistence / audit-logging layer.

The package name `logging` is the project's own audit-logging package, not
the standard library. Python's stdlib
`logging` is unaffected — absolute imports in Python 3 resolve stdlib
first, and `src.logging` is a distinct import path.
"""
