"""LLM reasoning — Protocol seam + OpenAI implementation + Mock for tests.

Architecture §7. The LLM:

- Takes a (job, profile, signals) triple.
- Returns a `ReasoningOutput` (strict JSON, Pydantic-validated).
- Retries ONCE on schema violation with the error appended to the prompt.
- On second failure: raises `LLMReasoningFailed`. The orchestrator catches
  this and attaches `reasoning=None` + `llm_confidence=0.0` to the decision.

The LLM does not compute the score. It does not set the verdict. It does
not modify the decision_trace. It is a SIGNAL (bounded `llm_confidence`
contribution ≤ 0.25 weighted) and an EXPLANATORY LAYER (strengths / gaps /
risks / talking points).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from src.schemas import CandidateProfile, ParsedJob, ReasoningOutput, Signals


# ── Errors ───────────────────────────────────────────────────────────────────


class LLMReasoningFailed(RuntimeError):
    """Raised after the retry-once-then-fail path exhausts.

    Caller (the orchestrator) MUST catch this and substitute
    `reasoning=None` + `llm_confidence=0.0` on the DecisionResult — the
    scorer continues without the LLM signal.
    """


# ── Protocol ─────────────────────────────────────────────────────────────────


@runtime_checkable
class LLMReasoner(Protocol):
    """The single operation the LLM layer exposes.

    A reasoner takes all the deterministic context the LLM needs and
    returns a validated `ReasoningOutput` OR raises `LLMReasoningFailed`.
    No partial returns; no null-sentinel values at this boundary — that
    translation happens in the orchestrator.
    """

    def reason(
        self,
        *,
        job: ParsedJob,
        profile: CandidateProfile,
        signals: Signals,
    ) -> ReasoningOutput: ...


# ── Prompt loading ───────────────────────────────────────────────────────────


_PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(version: str = "reasoning_v1") -> dict[str, Any]:
    """Load a versioned prompt YAML.

    Args:
        version: Filename without extension (e.g. `"reasoning_v1"`).

    Returns:
        Parsed YAML as a dict with keys `version`, `model_default`,
        `temperature`, `system`, `user`.

    Raises:
        FileNotFoundError: unknown version.
    """
    path = _PROMPTS_DIR / f"{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"prompt version {version!r} not found at {path}. "
            "Rollback-safe convention: new prompts create a new file, never "
            "edit an existing version in place."
        )
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_user_message(
    prompt: dict[str, Any],
    *,
    job: ParsedJob,
    profile: CandidateProfile,
    signals: Signals,
) -> str:
    """Render the user-message template with the current triple.

    Uses string `.format()` with named fields. The template itself is held
    in the YAML; this function only substitutes. That keeps the prompt
    version-controlled and the code minimal.
    """
    template: str = prompt["user"]
    return template.format(
        job=_JobView(job),
        profile=_ProfileView(profile),
        signals=_SignalsView(signals),
    )


# ── Mock reasoner (used in tests) ────────────────────────────────────────────


class MockReasoner:
    """A reasoner that always returns a fixed, valid `ReasoningOutput`.

    Useful for:
    - orchestrator tests (wire the whole pipeline without a network)
    - negative tests of the "bounded numeric signal" invariant

    The fixed output is deterministic: same inputs → same outputs.
    """

    def __init__(
        self,
        *,
        llm_confidence: float = 0.75,
        strengths: list[str] | None = None,
        gaps: list[str] | None = None,
        risks: list[str] | None = None,
    ) -> None:
        self._llm_confidence = llm_confidence
        self._strengths = strengths or [
            "Strong signal match on core required skills",
            "Seniority level aligns with the role",
            "Relevant recent experience in the domain",
        ]
        self._gaps = gaps or [
            "Preferred skills partially covered",
            "No direct evidence of cross-team leadership",
        ]
        self._risks = risks or [
            "JD does not specify the team's on-call burden",
        ]

    def reason(
        self,
        *,
        job: ParsedJob,
        profile: CandidateProfile,
        signals: Signals,
    ) -> ReasoningOutput:
        return ReasoningOutput(
            strengths=list(self._strengths),
            gaps=list(self._gaps),
            risks=list(self._risks),
            llm_confidence=self._llm_confidence,
            recommended_talking_points=[
                f"{int(profile.years_experience)}+ years in {job.title.split()[0]} roles",
                "Shipping production systems with measurable impact",
                "Comfortable operating across the full stack when needed",
            ],
        )


class FailingReasoner:
    """Always raises `LLMReasoningFailed`. Tests the fallback path."""

    def reason(self, **_kwargs: Any) -> ReasoningOutput:
        raise LLMReasoningFailed("mock reasoner configured to fail")


# ── OpenAI reasoner (production) ─────────────────────────────────────────────


class OpenAIReasoner:
    """Production reasoner — OpenAI `gpt-4o` with strict JSON response format.

    Retry-once-then-fail policy:

        attempt 1: normal prompt → response → validate
        on ValidationError:
          attempt 2: same prompt + appended error message → response → validate
          on second ValidationError: raise LLMReasoningFailed

    The OpenAI SDK is lazy-imported (`openai` is in requirements.txt but tests
    never need it — they use `MockReasoner`).
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        prompt_version: str = "reasoning_v1",
    ) -> None:
        self._prompt = load_prompt(prompt_version)
        self._model = model or self._prompt.get("model_default", "gpt-4o")
        self._temperature = float(self._prompt.get("temperature", 0.2))
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "OPENAI_API_KEY missing. Set the env var or pass api_key="
                "to OpenAIReasoner()."
            )
        self._client = None  # lazy

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI  # noqa: WPS433
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "openai SDK is not installed. Install from requirements.txt, "
                    "or use MockReasoner in test contexts."
                ) from e
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def reason(
        self,
        *,
        job: ParsedJob,
        profile: CandidateProfile,
        signals: Signals,
    ) -> ReasoningOutput:
        user_msg = render_user_message(
            self._prompt, job=job, profile=profile, signals=signals
        )
        messages = [
            {"role": "system", "content": self._prompt["system"]},
            {"role": "user", "content": user_msg},
        ]

        # Attempt 1 — catch both JSONDecodeError (malformed JSON) and
        # ValidationError (schema drift). Both trigger the single retry.
        raw = self._call_openai(messages)
        try:
            return _validate_reasoning_json(raw)
        except (ValidationError, json.JSONDecodeError) as first_error:
            first_error_msg = str(first_error)

        # Attempt 2 — retry once with the error appended
        messages_retry = messages + [
            {
                "role": "user",
                "content": (
                    "Your previous response failed validation with this "
                    f"error:\n\n{first_error_msg}\n\nReturn ONLY the JSON, fixed. "
                    "No prose, no markdown."
                ),
            }
        ]
        raw_retry = self._call_openai(messages_retry)
        try:
            return _validate_reasoning_json(raw_retry)
        except (ValidationError, json.JSONDecodeError) as second_error:
            raise LLMReasoningFailed(
                f"LLM returned invalid output twice. second error: {second_error}"
            ) from second_error

    def _call_openai(self, messages: list[dict[str, str]]) -> str:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            response_format={"type": "json_object"},
            messages=messages,
        )
        return resp.choices[0].message.content or ""


# ── Internals ────────────────────────────────────────────────────────────────


def _validate_reasoning_json(raw: str) -> ReasoningOutput:
    """Parse JSON string → ReasoningOutput.

    Raises:
        json.JSONDecodeError: raw isn't valid JSON.
        ValidationError: JSON is valid but doesn't match the schema.

    Both are caught by the retry-or-fail path in `OpenAIReasoner.reason`.
    """
    payload = json.loads(raw)
    return ReasoningOutput.model_validate(payload)


# ── View objects (for prompt template rendering) ─────────────────────────────
#
# The YAML template uses `{job.title}` / `{profile.skills_tech}` / etc.
# Pydantic models are not directly compatible with `str.format()` attribute
# lookups when fields are complex types. These thin views expose
# str-friendly representations.


class _JobView:
    def __init__(self, job: ParsedJob) -> None:
        self._job = job

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._job, name)
        if isinstance(value, list):
            return ", ".join(str(x) for x in value) or "(none)"
        if value is None:
            return "(unspecified)"
        return value


class _ProfileView:
    def __init__(self, profile: CandidateProfile) -> None:
        self._profile = profile

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._profile, name)
        if isinstance(value, list):
            return ", ".join(str(x) for x in value) or "(none)"
        return value


class _SignalsView:
    def __init__(self, signals: Signals) -> None:
        self._signals = signals

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._signals, name)
        if isinstance(value, float):
            return f"{value:.3f}"
        return value


__all__ = [
    "LLMReasoner",
    "LLMReasoningFailed",
    "MockReasoner",
    "FailingReasoner",
    "OpenAIReasoner",
    "load_prompt",
    "render_user_message",
]
