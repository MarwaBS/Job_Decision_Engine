"""Job-description parser.

Architecture §5.2: produces a `Job` with a `parsed: ParsedJob` payload, a
content hash (SHA-256 of the normalised raw text), a `parse_confidence`
score in [0, 1], and a list of `parse_warnings`.

Strategy: regex + heuristics only. No LLM. Intentionally lightweight —
`parse_confidence` surfaces how much structure was actually recovered, and
a confidence below `config.MIN_PARSE_CONFIDENCE` routes the decision to
REVIEW at scoring time.

Extracted fields (each contributes to `parse_confidence` when found):

    title                   (required — falls back to "Untitled Role")
    company                 (optional)
    location                (optional)
    remote                  (boolean, default False)
    seniority               (optional)
    years_required          (optional)
    required_skills         (taxonomy-matched from the text)
    preferred_skills        (taxonomy-matched from the "nice to have" block)
    salary_range_usd        (optional)

The parser does not attempt to distinguish required vs preferred without a
structural cue (a "nice to have" / "preferred" heading). Everything
outside such a block is treated as required.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Literal

from src.schemas import Job, ParsedJob, Seniority
from src.signals.skills import _ALIAS_PATTERNS, SKILLS_TAXONOMY


# ── Regex patterns ───────────────────────────────────────────────────────────

_TITLE_LINE_PATTERN = re.compile(
    r"^(?:title|position|role|job\s*title)\s*:?\s*(.+?)\s*$",
    re.IGNORECASE,
)
_COMPANY_LINE_PATTERN = re.compile(
    r"^(?:company|employer|organization)\s*:?\s*(.+?)\s*$",
    re.IGNORECASE,
)
_LOCATION_LINE_PATTERN = re.compile(
    r"^(?:location|based\s+in)\s*:?\s*(.+?)\s*$",
    re.IGNORECASE,
)
_YEARS_PATTERN = re.compile(
    r"(\d+)\s*\+?\s*(?:to\s*\d+\s*)?(?:years?|yrs?)\s+of\s+experience"
    r"|(\d+)\s*\+\s*(?:years?|yrs?)"
    r"|(\d+)\s*(?:-|to|–)\s*\d+\s*(?:years?|yrs?)",
    re.IGNORECASE,
)
_REMOTE_PATTERN = re.compile(r"\b(?:remote|work\s+from\s+home|wfh)\b", re.IGNORECASE)
_HYBRID_PATTERN = re.compile(r"\bhybrid\b", re.IGNORECASE)
_ONSITE_PATTERN = re.compile(r"\b(?:on[-\s]?site|in[-\s]?office)\b", re.IGNORECASE)
_SALARY_PATTERN = re.compile(
    r"\$\s*(\d{2,3})\s*[kK]?\s*(?:-|to|–)\s*\$?\s*(\d{2,3})\s*[kK]?",
)
_NICE_TO_HAVE_HEADING = re.compile(
    r"(?i)(?:preferred|nice[-\s]?to[-\s]?have|bonus|plus|would\s+be\s+nice)"
    r"\s*(?:qualifications|skills|experience)?\s*:?"
)

_SENIORITY_KEYWORDS: list[tuple[re.Pattern[str], Seniority]] = [
    (re.compile(r"\bprincipal\b", re.IGNORECASE), Seniority.PRINCIPAL),
    (re.compile(r"\bstaff\b", re.IGNORECASE), Seniority.STAFF),
    (re.compile(r"\b(?:senior|sr\.?)\b", re.IGNORECASE), Seniority.SENIOR),
    (re.compile(r"\b(?:junior|jr\.?|entry[-\s]?level)\b", re.IGNORECASE), Seniority.JUNIOR),
    (re.compile(r"\b(?:mid[-\s]?level|mid)\b", re.IGNORECASE), Seniority.MID),
]


# ── Public API ───────────────────────────────────────────────────────────────


def parse_job(
    raw_text: str,
    *,
    source: Literal["paste", "url", "upload"] = "paste",
    source_url: str | None = None,
) -> Job:
    """Parse a raw job-description into a `Job`.

    Args:
        raw_text: The JD body as free text.
        source: Where this JD came from (tag only; doesn't affect parsing).
        source_url: Optional URL if `source == "url"`.

    Returns:
        A `Job` with structured fields and a `parse_confidence` signal.

    The function is pure — no I/O, no logging, no external calls. Runnable
    in a unit test with any string input.
    """
    normalised = _normalise(raw_text)
    if not normalised.strip():
        # Empty input → minimal Job with confidence 0
        return Job(
            content_hash="sha256:" + hashlib.sha256(b"").hexdigest(),
            created_at=datetime.now(timezone.utc),
            source=source,
            source_url=source_url,
            raw_text=raw_text,
            parsed=ParsedJob(title="Untitled Role"),
            parse_confidence=0.0,
            parse_warnings=["empty_input"],
        )

    warnings: list[str] = []

    title, had_title = _extract_title(normalised, warnings)
    company = _extract_company(normalised)
    location = _extract_location(normalised)
    remote = _extract_remote(normalised)
    seniority = _extract_seniority(title, normalised)
    years = _extract_years(normalised)
    salary = _extract_salary(normalised, warnings)
    required, preferred = _extract_skills(normalised)

    parsed = ParsedJob(
        title=title,
        company=company,
        location=location,
        remote=remote,
        seniority=seniority,
        years_required=years,
        required_skills=required,
        preferred_skills=preferred,
        salary_range_usd=salary,
    )

    confidence = _compute_confidence(
        had_title=had_title,
        company=company,
        location=location,
        seniority=seniority,
        years=years,
        required=required,
        remote_detected=_any_workplace_cue(normalised),
    )

    content_hash = "sha256:" + hashlib.sha256(normalised.encode("utf-8")).hexdigest()

    return Job(
        content_hash=content_hash,
        created_at=datetime.now(timezone.utc),
        source=source,
        source_url=source_url,
        raw_text=raw_text,
        parsed=parsed,
        parse_confidence=confidence,
        parse_warnings=warnings,
    )


# ── Field extractors ─────────────────────────────────────────────────────────


def _normalise(text: str) -> str:
    """Collapse multiple blank lines, strip trailing whitespace per line.

    Deterministic — the content hash relies on this being byte-stable.
    """
    lines = [line.rstrip() for line in text.splitlines()]
    # Collapse 3+ consecutive blank lines to 2
    out: list[str] = []
    blank_run = 0
    for line in lines:
        if not line:
            blank_run += 1
            if blank_run <= 2:
                out.append(line)
        else:
            blank_run = 0
            out.append(line)
    return "\n".join(out)


def _extract_title(text: str, warnings: list[str]) -> tuple[str, bool]:
    """Find the role title. Two strategies, in order:

    1. An explicit "Title:" / "Position:" / "Role:" line.
    2. The first non-empty line of the JD.

    Returns (title, had_structural_signal). `had_structural_signal` is True
    only when strategy 1 fired.
    """
    for line in text.splitlines():
        m = _TITLE_LINE_PATTERN.match(line)
        if m:
            return m.group(1).strip(), True

    for line in text.splitlines():
        if line.strip():
            # Guardrail: a line longer than 140 chars probably isn't a title.
            candidate = line.strip()
            if len(candidate) <= 140:
                return candidate, False
            return candidate[:140].rstrip() + "…", False

    warnings.append("no_title_found")
    return "Untitled Role", False


def _extract_company(text: str) -> str | None:
    for line in text.splitlines():
        m = _COMPANY_LINE_PATTERN.match(line)
        if m:
            return m.group(1).strip()
    return None


def _extract_location(text: str) -> str | None:
    for line in text.splitlines():
        m = _LOCATION_LINE_PATTERN.match(line)
        if m:
            return m.group(1).strip()
    return None


def _extract_remote(text: str) -> bool:
    """Return True if the JD mentions remote/WFH and does NOT mention on-site.

    Hybrid is treated as "partially remote" → True. On-site-only → False.
    """
    if _REMOTE_PATTERN.search(text) or _HYBRID_PATTERN.search(text):
        return True
    return False


def _any_workplace_cue(text: str) -> bool:
    """Used by `_compute_confidence` to reward JDs that specify workplace at all."""
    return bool(
        _REMOTE_PATTERN.search(text)
        or _HYBRID_PATTERN.search(text)
        or _ONSITE_PATTERN.search(text)
    )


def _extract_seniority(title: str, body: str) -> Seniority | None:
    """Match seniority keywords in the title first (strongest signal), then body.

    Order of the keyword list is meaningful — principal/staff win over senior
    so a "Senior Staff" title correctly resolves to STAFF.
    """
    for pattern, seniority in _SENIORITY_KEYWORDS:
        if pattern.search(title):
            return seniority
    for pattern, seniority in _SENIORITY_KEYWORDS:
        if pattern.search(body):
            return seniority
    return None


def _extract_years(text: str) -> float | None:
    """Extract the minimum-years-of-experience value.

    Patterns handled:
        "5+ years of experience"
        "5 years of experience"
        "5+ years"
        "5-7 years"
    Returns the smaller number of a range (the floor).
    """
    m = _YEARS_PATTERN.search(text)
    if not m:
        return None
    for group in m.groups():
        if group is not None:
            try:
                return float(group)
            except ValueError:
                continue
    return None


def _extract_salary(
    text: str, warnings: list[str]
) -> tuple[int, int] | None:
    """Extract a salary range in USD. Returns None if not present.

    Only supports $NNN-$NNN shapes with a 'k' or 'K' suffix (or bare numbers
    large enough to be annual salaries). Anything else gets a warning and
    returns None.
    """
    m = _SALARY_PATTERN.search(text)
    if not m:
        return None
    low = int(m.group(1))
    high = int(m.group(2))
    # Normalize "k" notation: "$100-$150" interpreted as 100k-150k if both
    # numbers are small (<1000), otherwise taken as absolute values.
    if low < 1000 and high < 1000:
        low, high = low * 1000, high * 1000
    if low > high:
        warnings.append("salary_range_inverted")
        return None
    return (low, high)


def _extract_skills(text: str) -> tuple[list[str], list[str]]:
    """Partition skills into (required, preferred) based on a heading split.

    Splits the JD at the first "nice to have" / "preferred" heading. Any
    taxonomy match before the split is required; after the split is
    preferred. If there's no split, everything is required.
    """
    split_match = _NICE_TO_HAVE_HEADING.search(text)
    if split_match:
        required_chunk = text[: split_match.start()]
        preferred_chunk = text[split_match.end() :]
    else:
        required_chunk = text
        preferred_chunk = ""

    required = _taxonomy_hits(required_chunk)
    preferred = [s for s in _taxonomy_hits(preferred_chunk) if s not in required]
    return required, preferred


def _taxonomy_hits(chunk: str) -> list[str]:
    """Return canonical taxonomy names found in `chunk`, sorted + deduped."""
    if not chunk.strip():
        return []
    hits: set[str] = set()
    for canonical, pattern in _ALIAS_PATTERNS.items():
        if pattern.search(chunk):
            hits.add(canonical)
    return sorted(hits)


# ── Confidence ───────────────────────────────────────────────────────────────


def _compute_confidence(
    *,
    had_title: bool,
    company: str | None,
    location: str | None,
    seniority: Seniority | None,
    years: float | None,
    required: list[str],
    remote_detected: bool,
) -> float:
    """Heuristic parse confidence in [0, 1].

    Each extracted field adds a small, documented amount. Weights sum to 1.0
    so confidence is interpretable as "fraction of expected structure
    recovered". The cutoff in `config.MIN_PARSE_CONFIDENCE` (0.5) routes
    low-structure JDs to REVIEW.

    Weights:
        structural title heading         0.15
        company                          0.10
        location                         0.10
        seniority                        0.10
        years_required                   0.10
        any workplace cue (remote/hybrid/on-site)  0.10
        ≥1 taxonomy skill matched        0.20
        ≥3 taxonomy skills matched       0.15 (additional)
        Total if everything present:      1.00
    """
    score = 0.0
    if had_title:
        score += 0.15
    if company:
        score += 0.10
    if location:
        score += 0.10
    if seniority is not None:
        score += 0.10
    if years is not None:
        score += 0.10
    if remote_detected:
        score += 0.10
    if len(required) >= 1:
        score += 0.20
    if len(required) >= 3:
        score += 0.15
    return min(1.0, score)


# Re-export for test discoverability
__all__ = ["parse_job", "SKILLS_TAXONOMY"]
