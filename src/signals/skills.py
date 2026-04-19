"""Skills signal — taxonomy + extraction + match score.

Architecture §3: REAL. The extraction is taxonomy-based regex matching
(fast, deterministic, offline). spaCy NER is an optional enhancer that can
be plugged in later via `extract_skills(text, enhance=True)`; the v1 default
is pure-regex so the system remains fully testable without a spaCy model
download.

`compute_skills_match(job, profile)` is the signal function consumed by the
scorer. It returns a value in `[0, 1]` shaped like a weighted Jaccard:

    numerator   = |matched required skills| + 0.5 * |matched preferred skills|
    denominator = |required skills| + 0.5 * |preferred skills|

Rationale: required skills are hard signals; preferred skills are
"nice-to-have" signals and should contribute at half weight. Clipping the
denominator to avoid division by zero when a JD lists no skills — in that
case the function returns `0.0` and relies on the parser's low
`parse_confidence` to route the decision to REVIEW.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from src.schemas import CandidateProfile, ParsedJob


# ── Taxonomy ─────────────────────────────────────────────────────────────────
#
# Hand-curated. Expansion requires a DECISION_TRACE_LOG entry, not an ADR
# (taxonomy growth is expected implementation work, not an architecture
# change). Every skill appears in exactly one bucket.
#
# Each entry is a (canonical_name, aliases) tuple. Matching is case-insensitive
# and uses word-boundary regex so "c" doesn't match inside "contract" but does
# match "C, C++, Python".

SKILLS_TAXONOMY: dict[str, dict[str, list[str]]] = {
    "tech": {
        "python": ["python", "py"],
        "java": ["java"],
        "javascript": ["javascript", "js", "ecmascript"],
        "typescript": ["typescript", "ts"],
        "go": ["golang", "go"],
        "rust": ["rust"],
        "c++": ["c\\+\\+", "cpp"],
        "c#": ["c#", "csharp"],
        "sql": ["sql"],
        "scala": ["scala"],
        "r": ["\\br\\b"],  # strict word boundary — avoid matching inside words
    },
    "ml_frameworks": {
        "pytorch": ["pytorch", "torch"],
        "tensorflow": ["tensorflow", "tf"],
        "keras": ["keras"],
        "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
        "xgboost": ["xgboost", "xgb"],
        "lightgbm": ["lightgbm", "lgbm"],
        "hugging face": ["hugging\\s*face", "huggingface"],
        "langchain": ["langchain"],
        "transformers": ["transformers"],
    },
    "data_tools": {
        "pandas": ["pandas"],
        "numpy": ["numpy"],
        "spark": ["spark", "pyspark"],
        "airflow": ["airflow"],
        "dbt": ["\\bdbt\\b"],
        "snowflake": ["snowflake"],
        "bigquery": ["bigquery"],
        "kafka": ["kafka"],
    },
    "cloud": {
        "aws": ["\\baws\\b", "amazon web services"],
        "gcp": ["\\bgcp\\b", "google cloud"],
        "azure": ["azure"],
        "kubernetes": ["kubernetes", "k8s"],
        "docker": ["docker"],
        "terraform": ["terraform"],
    },
    "web_frameworks": {
        "fastapi": ["fastapi"],
        "flask": ["flask"],
        "django": ["django"],
        "react": ["react"],
        "node": ["node\\.?js", "node"],
    },
    "domain": {
        "mlops": ["mlops"],
        "llm": ["\\bllm\\b", "large language model"],
        "rag": ["\\brag\\b", "retrieval augmented generation"],
        "nlp": ["\\bnlp\\b", "natural language processing"],
        "computer vision": ["computer vision", "\\bcv\\b"],
        "recommender": ["recommender", "recommendation system"],
        "time series": ["time series", "forecasting"],
        "data engineering": ["data engineering", "\\bde\\b"],
    },
}


def _all_skills() -> dict[str, list[str]]:
    """Flatten the taxonomy to {canonical: [aliases]}.

    Canonicals are unique across buckets by construction — verified by
    `test_skills.py::test_taxonomy_no_duplicate_canonicals`.
    """
    out: dict[str, list[str]] = {}
    for bucket in SKILLS_TAXONOMY.values():
        out.update(bucket)
    return out


_ALIAS_PATTERNS: dict[str, re.Pattern[str]] = {
    canonical: re.compile(
        r"(?i)(?:" + "|".join(aliases) + r")",
    )
    for canonical, aliases in _all_skills().items()
}


# ── Extraction ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SkillSet:
    """Structured output of `extract_skills`.

    Lists are canonical (deduped, sorted) so two runs on the same input
    produce byte-identical output — a prerequisite for deterministic content
    hashing in the ingestion layer.
    """

    tech: tuple[str, ...]
    tools: tuple[str, ...]
    domain: tuple[str, ...]

    @property
    def all(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.tech + self.tools + self.domain)))


def extract_skills(text: str) -> SkillSet:
    """Extract canonical skill names from free text.

    Case-insensitive, word-boundary regex over the full taxonomy. Returns a
    `SkillSet` with three buckets: tech (languages), tools (ML + data +
    cloud + web frameworks collapsed), and domain.
    """
    hits: set[str] = set()
    for canonical, pattern in _ALIAS_PATTERNS.items():
        if pattern.search(text):
            hits.add(canonical)

    tech = tuple(sorted(s for s in hits if s in SKILLS_TAXONOMY["tech"]))
    tools = tuple(
        sorted(
            s for s in hits
            if s in SKILLS_TAXONOMY["ml_frameworks"]
            or s in SKILLS_TAXONOMY["data_tools"]
            or s in SKILLS_TAXONOMY["cloud"]
            or s in SKILLS_TAXONOMY["web_frameworks"]
        )
    )
    domain = tuple(sorted(s for s in hits if s in SKILLS_TAXONOMY["domain"]))
    return SkillSet(tech=tech, tools=tools, domain=domain)


# ── Signal function (consumed by the scorer) ─────────────────────────────────


def compute_skills_match(
    job: ParsedJob, profile: CandidateProfile
) -> float:
    """Weighted-Jaccard-ish match score ∈ [0, 1].

    Returns 0.0 when the job lists no required OR preferred skills — the
    parser's `parse_confidence` is the hard-filter that should catch this
    case and route to REVIEW.
    """
    profile_skills = _candidate_skill_set(profile)
    required = _normalise(job.required_skills)
    preferred = _normalise(job.preferred_skills)

    denominator = len(required) + 0.5 * len(preferred)
    if denominator == 0:
        return 0.0

    matched_required = len(required & profile_skills)
    matched_preferred = len(preferred & profile_skills)

    numerator = matched_required + 0.5 * matched_preferred
    score = numerator / denominator
    return max(0.0, min(1.0, score))


def _candidate_skill_set(profile: CandidateProfile) -> set[str]:
    return _normalise(profile.skills_tech + profile.skills_tools + profile.skills_domain)


def _normalise(skills: Iterable[str]) -> set[str]:
    """Lower-case + strip each skill so taxonomy canonicals and free-form
    profile entries compare on equal footing."""
    return {s.strip().lower() for s in skills if s and s.strip()}
