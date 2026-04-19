"""Persistence / audit-logging layer.

The package name `logging` matches architecture §4. Python's stdlib
`logging` is unaffected — absolute imports in Python 3 resolve stdlib
first, and `src.logging` is a distinct import path.
"""
