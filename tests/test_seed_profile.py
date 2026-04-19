"""Tests for `scripts/seed_profile.py`."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.seed_profile import load_profile_from_yaml, seed_profile
from src.db import InMemoryStore


_VALID_YAML = """
profile_version: "v1.0"
active: true
name: "Marwa"
summary: "ML engineer"
years_experience: 5.0
seniority: "senior"
skills_tech: ["python"]
skills_tools: ["pytorch"]
skills_domain: ["mlops"]
target_roles: ["Senior ML Engineer"]
dealbreakers: []
"""


_INVALID_YAML = """
profile_version: "v1.0"
name: "Marwa"
# missing summary, years_experience, seniority
"""


class TestLoadProfileFromYaml:
    def test_valid_yaml_loads(self, tmp_path: Path):
        path = tmp_path / "profile.yaml"
        path.write_text(_VALID_YAML, encoding="utf-8")
        profile = load_profile_from_yaml(path)
        assert profile.profile_version == "v1.0"
        assert profile.name == "Marwa"

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_profile_from_yaml(tmp_path / "does-not-exist.yaml")

    def test_invalid_yaml_raises_validation_error(self, tmp_path: Path):
        path = tmp_path / "profile.yaml"
        path.write_text(_INVALID_YAML, encoding="utf-8")
        with pytest.raises(ValidationError):
            load_profile_from_yaml(path)


class TestSeedProfile:
    def test_seeds_into_store(self, tmp_path: Path):
        path = tmp_path / "profile.yaml"
        path.write_text(_VALID_YAML, encoding="utf-8")

        store = InMemoryStore()
        oid = seed_profile(path, store)
        assert oid
        assert store.count("profiles") == 1
        assert store.get_active_profile() is not None

    def test_reseeding_same_version_does_not_duplicate(self, tmp_path: Path):
        path = tmp_path / "profile.yaml"
        path.write_text(_VALID_YAML, encoding="utf-8")

        store = InMemoryStore()
        seed_profile(path, store)
        seed_profile(path, store)  # same version — in-place
        assert store.count("profiles") == 1
