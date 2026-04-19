"""Ingestion layer — turn raw job-description text (or URLs) into a `Job`.

Architecture §3: this layer is REAL. URL scraping is explicitly best-effort
(LinkedIn / Indeed block most scrapers) — the primary path is text paste.
"""
