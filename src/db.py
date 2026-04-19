"""MongoDB layer — connection abstraction + collection accessors.

Architecture §4, §5. This module is the ONLY place in the project that
talks to pymongo. All higher layers depend on the `Store` Protocol, never
on pymongo directly. That keeps:

- the scorer and orchestrator pure-testable
- schema + persistence changes localised
- tests hermetic (use `InMemoryStore`)

Non-business-logic by design. Per Step 4 rules: "connection abstraction
+ collection accessors, no business logic". A method here either reads
from Mongo or writes to Mongo, nothing more.

Append-only contract (locked per DT-010):

- `decisions` — `insert_decision` only. No update, no replace, no delete.
- `outcomes` — `insert_outcome` OR `push_outcome_stage` / `set_outcome_final_stage`
  for the state-machine path. Hard rule: a document's `stages[]` only grows;
  `final_stage` transitions only from None → terminal.
- `feedback_logs` — append-only.
- `profiles` — `upsert_profile` (per-version doc). Versioned means new
  versions write new docs, they do not overwrite.
- `jobs` — upsert by `content_hash`. Same content → same doc. No mutation
  of the `parsed` payload after creation.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Iterable, Protocol, runtime_checkable

from src.schemas import (
    CandidateProfile,
    DecisionResult,
    FeedbackLog,
    Job,
    Outcome,
    OutcomeStage,
)


# ── Store Protocol ───────────────────────────────────────────────────────────


@runtime_checkable
class Store(Protocol):
    """The append-only storage interface the whole system depends on.

    Every method takes a validated Pydantic model (or a primitive) and
    returns either a string id (created/upserted docs) or a typed result.
    No method here mutates a previously-inserted `decision` document.
    """

    # Profiles (upsert by version — new version = new doc)
    def upsert_profile(self, profile: CandidateProfile) -> str: ...

    def get_active_profile(self) -> CandidateProfile | None: ...

    # Jobs (upsert by content_hash)
    def upsert_job(self, job: Job) -> str: ...

    # Decisions (strict append-only)
    def insert_decision(self, decision: DecisionResult) -> str: ...

    def list_decisions(self, limit: int = 100) -> list[dict[str, Any]]: ...

    # Outcomes (state machine — insert once per decision, then push stages)
    def insert_outcome(self, outcome: Outcome) -> str: ...

    def push_outcome_stage(self, decision_id: str, stage: OutcomeStage) -> None: ...

    def set_outcome_final_stage(self, decision_id: str, final_stage: str) -> None: ...

    def list_outcomes(self, limit: int = 1000) -> list[dict[str, Any]]: ...

    # Feedback logs (append-only)
    def insert_feedback(self, feedback: FeedbackLog) -> str: ...

    # Diagnostics
    def count(self, collection: str) -> int: ...


# ── In-memory store (production-grade for tests, trivial code) ───────────────


class InMemoryStore:
    """Dict-of-lists implementation of the `Store` Protocol.

    Used by the unit test suite so tests remain hermetic: no network, no
    Mongo install, no flaky fixtures. Shape-identical to the real store so
    the same business-layer code works against both.

    Storage layout:
        self._collections[name] -> list[dict]
    """

    _COLLECTIONS = ("profiles", "jobs", "decisions", "outcomes", "feedback_logs")

    def __init__(self) -> None:
        self._collections: dict[str, list[dict[str, Any]]] = {
            name: [] for name in self._COLLECTIONS
        }
        self._next_id = 1

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _mint_id(self) -> str:
        oid = f"mem_{self._next_id:08d}"
        self._next_id += 1
        return oid

    def _append(self, collection: str, doc: dict[str, Any]) -> str:
        oid = self._mint_id()
        stored = {"_id": oid, **doc}
        self._collections[collection].append(stored)
        return oid

    # ── Profiles ─────────────────────────────────────────────────────────────

    def upsert_profile(self, profile: CandidateProfile) -> str:
        doc = profile.model_dump(mode="json")
        # If a doc for the same profile_version exists, replace it in-place
        # (this is "upsert by version" — not a mutation of a DIFFERENT version).
        # If a *new* profile_version is being written, deactivate the old active.
        if doc.get("active"):
            for existing in self._collections["profiles"]:
                if existing.get("active"):
                    existing["active"] = False

        for existing in self._collections["profiles"]:
            if existing.get("profile_version") == doc.get("profile_version"):
                oid = existing["_id"]
                existing.clear()
                existing.update({"_id": oid, **doc})
                return oid

        return self._append("profiles", doc)

    def get_active_profile(self) -> CandidateProfile | None:
        for doc in self._collections["profiles"]:
            if doc.get("active"):
                return CandidateProfile.model_validate({
                    k: v for k, v in doc.items() if k != "_id"
                })
        return None

    # ── Jobs ─────────────────────────────────────────────────────────────────

    def upsert_job(self, job: Job) -> str:
        doc = job.model_dump(mode="json")
        for existing in self._collections["jobs"]:
            if existing.get("content_hash") == doc.get("content_hash"):
                return existing["_id"]
        return self._append("jobs", doc)

    # ── Decisions (strict append-only) ───────────────────────────────────────

    def insert_decision(self, decision: DecisionResult) -> str:
        return self._append("decisions", decision.model_dump(mode="json"))

    def list_decisions(self, limit: int = 100) -> list[dict[str, Any]]:
        return list(self._collections["decisions"][-limit:])

    # ── Outcomes (state machine) ─────────────────────────────────────────────

    def insert_outcome(self, outcome: Outcome) -> str:
        return self._append("outcomes", outcome.model_dump(mode="json"))

    def push_outcome_stage(self, decision_id: str, stage: OutcomeStage) -> None:
        """Append a stage to an existing outcome document.

        `stages[]` is strictly growing — no reorder, no removal. This is
        the only form of "outcome mutation" allowed (see DT-010).
        """
        doc = self._find_outcome_by_decision_id(decision_id)
        doc["stages"].append(stage.model_dump(mode="json"))

    def set_outcome_final_stage(self, decision_id: str, final_stage: str) -> None:
        """Set the terminal `final_stage`. Allowed to set once (None → value).

        Refuses to overwrite a non-None final_stage — that would be a
        retroactive "fix" of a closed outcome, explicitly banned.
        """
        doc = self._find_outcome_by_decision_id(decision_id)
        if doc.get("final_stage") is not None:
            raise ValueError(
                f"final_stage already set for decision_id={decision_id}; "
                "retroactive overwrite is not permitted."
            )
        doc["final_stage"] = final_stage

    def list_outcomes(self, limit: int = 1000) -> list[dict[str, Any]]:
        return list(self._collections["outcomes"][-limit:])

    def _find_outcome_by_decision_id(self, decision_id: str) -> dict[str, Any]:
        for doc in self._collections["outcomes"]:
            if doc.get("decision_id") == decision_id:
                return doc
        raise KeyError(f"no outcome found for decision_id={decision_id}")

    # ── Feedback ─────────────────────────────────────────────────────────────

    def insert_feedback(self, feedback: FeedbackLog) -> str:
        return self._append("feedback_logs", feedback.model_dump(mode="json"))

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def count(self, collection: str) -> int:
        if collection not in self._collections:
            raise KeyError(f"unknown collection {collection!r}")
        return len(self._collections[collection])


# ── Real Mongo store (pymongo-backed) ────────────────────────────────────────


class MongoStore:
    """pymongo-backed `Store`.

    `pymongo` is imported lazily in `__init__` so the rest of the codebase
    (and the test suite) can run without it installed. Production callers
    construct this store once at app startup and pass it everywhere.

    No caching. No smart querying. Same methods as `InMemoryStore`, same
    semantics, same append-only discipline.
    """

    def __init__(
        self,
        uri: str | None = None,
        database: str = "job_decision_engine",
    ) -> None:
        try:
            from pymongo import MongoClient  # noqa: WPS433
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "pymongo is not installed. Install from requirements.txt "
                "or use InMemoryStore for hermetic test contexts."
            ) from e
        uri = uri or os.getenv("MONGODB_URI")
        if not uri:
            raise RuntimeError(
                "MongoDB URI missing. Set MONGODB_URI env var or pass uri=."
            )
        self._client = MongoClient(uri, tz_aware=True)
        self._db = self._client[database]

    # ── Profiles ─────────────────────────────────────────────────────────────

    def upsert_profile(self, profile: CandidateProfile) -> str:
        doc = profile.model_dump(mode="json")
        # Deactivate any currently-active profile of a different version.
        if doc.get("active"):
            self._db.profiles.update_many(
                {"profile_version": {"$ne": doc["profile_version"]}, "active": True},
                {"$set": {"active": False}},
            )
        res = self._db.profiles.update_one(
            {"profile_version": doc["profile_version"]},
            {"$set": doc},
            upsert=True,
        )
        if res.upserted_id is not None:
            return str(res.upserted_id)
        existing = self._db.profiles.find_one(
            {"profile_version": doc["profile_version"]}, {"_id": 1}
        )
        return str(existing["_id"]) if existing else ""

    def get_active_profile(self) -> CandidateProfile | None:
        doc = self._db.profiles.find_one({"active": True})
        if not doc:
            return None
        doc.pop("_id", None)
        return CandidateProfile.model_validate(doc)

    # ── Jobs ─────────────────────────────────────────────────────────────────

    def upsert_job(self, job: Job) -> str:
        doc = job.model_dump(mode="json")
        existing = self._db.jobs.find_one({"content_hash": doc["content_hash"]}, {"_id": 1})
        if existing:
            return str(existing["_id"])
        res = self._db.jobs.insert_one(doc)
        return str(res.inserted_id)

    # ── Decisions (strict append-only) ───────────────────────────────────────

    def insert_decision(self, decision: DecisionResult) -> str:
        res = self._db.decisions.insert_one(decision.model_dump(mode="json"))
        return str(res.inserted_id)

    def list_decisions(self, limit: int = 100) -> list[dict[str, Any]]:
        return list(
            self._db.decisions.find().sort("_id", -1).limit(limit)
        )

    # ── Outcomes (state machine) ─────────────────────────────────────────────

    def insert_outcome(self, outcome: Outcome) -> str:
        res = self._db.outcomes.insert_one(outcome.model_dump(mode="json"))
        return str(res.inserted_id)

    def push_outcome_stage(self, decision_id: str, stage: OutcomeStage) -> None:
        res = self._db.outcomes.update_one(
            {"decision_id": decision_id},
            {"$push": {"stages": stage.model_dump(mode="json")}},
        )
        if res.matched_count == 0:
            raise KeyError(f"no outcome found for decision_id={decision_id}")

    def set_outcome_final_stage(self, decision_id: str, final_stage: str) -> None:
        res = self._db.outcomes.update_one(
            {"decision_id": decision_id, "final_stage": None},
            {"$set": {"final_stage": final_stage}},
        )
        if res.matched_count == 0:
            # Either the outcome doesn't exist or it's already closed.
            existing = self._db.outcomes.find_one({"decision_id": decision_id})
            if existing is None:
                raise KeyError(f"no outcome found for decision_id={decision_id}")
            raise ValueError(
                f"final_stage already set for decision_id={decision_id}; "
                "retroactive overwrite is not permitted."
            )

    def list_outcomes(self, limit: int = 1000) -> list[dict[str, Any]]:
        return list(
            self._db.outcomes.find().sort("_id", -1).limit(limit)
        )

    # ── Feedback ─────────────────────────────────────────────────────────────

    def insert_feedback(self, feedback: FeedbackLog) -> str:
        res = self._db.feedback_logs.insert_one(feedback.model_dump(mode="json"))
        return str(res.inserted_id)

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def count(self, collection: str) -> int:
        valid = {"profiles", "jobs", "decisions", "outcomes", "feedback_logs"}
        if collection not in valid:
            raise KeyError(f"unknown collection {collection!r}")
        return self._db[collection].count_documents({})


def _now_utc() -> datetime:
    """Single UTC-now source — makes tests easier to pin deterministically."""
    return datetime.now(timezone.utc)


__all__ = ["Store", "InMemoryStore", "MongoStore"]
