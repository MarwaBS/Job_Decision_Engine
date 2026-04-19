"""Seed the candidate profile from a local YAML file into MongoDB.

Architecture §5.1 + Step 4 rule #4: YAML is the dev-only input artefact.
MongoDB is the runtime state. The YAML file itself never reaches production
logic — this script is the one-way gate between the two.

Usage:
    python -m scripts.seed_profile              # reads ./profile.yaml
    python -m scripts.seed_profile path/to.yaml # explicit path

The YAML must validate as a `CandidateProfile` (strict Pydantic validation).
If it doesn't, the script fails loudly — better to refuse to persist bad
data than to silently load a malformed profile that then misleads every
subsequent decision.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from src.db import MongoStore, Store
from src.schemas import CandidateProfile


def load_profile_from_yaml(path: Path) -> CandidateProfile:
    """Parse + validate a `profile.yaml` into a `CandidateProfile`.

    Raises:
        FileNotFoundError: path doesn't exist.
        pydantic.ValidationError: YAML doesn't match the schema.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"profile YAML not found at {path}. Create it (gitignored) "
            f"from profile.example.yaml."
        )
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return CandidateProfile.model_validate(payload)


def seed_profile(path: Path, store: Store) -> str:
    """Validate the YAML at `path` and upsert into the given store.

    Returns the profile's store id (primary key).
    """
    profile = load_profile_from_yaml(path)
    return store.upsert_profile(profile)


def _default_path() -> Path:
    return Path(__file__).resolve().parent.parent / "profile.yaml"


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint.

    Exit code 0 on success, non-zero on any failure. Errors print to stderr.
    """
    args = argv if argv is not None else sys.argv[1:]
    path = Path(args[0]) if args else _default_path()

    try:
        store = MongoStore()  # reads MONGODB_URI from env
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    try:
        oid = seed_profile(path, store)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    except Exception as e:  # pragma: no cover
        print(f"ERROR: failed to seed profile: {e}", file=sys.stderr)
        return 4

    print(f"seeded profile {oid} from {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["load_profile_from_yaml", "seed_profile", "main"]
