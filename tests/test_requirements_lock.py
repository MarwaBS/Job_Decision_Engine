"""Contract tests for the transitive dependency lock.

The claim under test (2026-07 re-audit, ROUND 6 item B3): the shipped
image must equal the tested universe. requirements.txt pins the direct
deps; requirements-lock.txt locks every transitive wheel; the Dockerfile
and both CI install steps must consume the lock as a pip constraints
file, and CI must pip-audit the lock so a new advisory turns the build
red.

These are file-contract greps in the same spirit as
tests/test_semantic.py::test_dockerfile_prewarm_revision_matches_pin —
they make it impossible to edit one side of the contract without the
suite noticing.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

_PIN_RE = re.compile(r"^([A-Za-z0-9._-]+)==([A-Za-z0-9.!+_-]+)")


def _pins(path: Path) -> dict[str, str]:
    """Parse `name==version` pins from a requirements-format file.

    Comments, blank lines, and environment markers are ignored; names are
    normalized per PEP 503 so `PyYAML` and `pyyaml` compare equal.
    """
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PIN_RE.match(line)
        if m:
            name = re.sub(r"[-_.]+", "-", m.group(1)).lower()
            pins[name] = m.group(2)
    return pins


class TestLockCoversDirectPins:
    def test_every_direct_pin_appears_identically_in_the_lock(self):
        """requirements.txt and requirements-lock.txt must never drift:
        each direct `name==version` pin must appear with the same version
        in the lock, or the image installs something the suite never
        tested."""
        direct = _pins(_ROOT / "requirements.txt")
        locked = _pins(_ROOT / "requirements-lock.txt")
        assert direct, "requirements.txt parsed to zero pins — parser broken?"
        for name, version in direct.items():
            assert locked.get(name) == version, (
                f"{name}=={version} is pinned in requirements.txt but the "
                f"lock has {name}=={locked.get(name)!r} — regenerate "
                "requirements-lock.txt (see its header) so the direct pins "
                "and the transitive lock agree."
            )

    def test_lock_is_a_superset_of_direct_pins(self):
        """The lock must contain strictly more packages than the direct
        list — if it ever collapses to just the direct pins, the
        transitive universe is unlocked again."""
        direct = _pins(_ROOT / "requirements.txt")
        locked = _pins(_ROOT / "requirements-lock.txt")
        assert len(locked) > len(direct)

    def test_lock_resolves_patched_starlette(self):
        """Regression pin for PYSEC-2026-248 / PYSEC-2026-249: the
        deployed set once carried starlette 1.2.1 because transitives
        floated. The lock must hold starlette at >= 1.3.1 (the fixed
        release)."""
        locked = _pins(_ROOT / "requirements-lock.txt")
        version = locked.get("starlette")
        assert version is not None, "starlette missing from the lock"
        major, minor, patch = (int(p) for p in version.split(".")[:3])
        assert (major, minor, patch) >= (1, 3, 1), (
            f"starlette=={version} in the lock is below the 1.3.1 security "
            "floor (PYSEC-2026-248 / PYSEC-2026-249)."
        )


class TestInstallersConsumeTheLock:
    def test_dockerfile_installs_with_lock_constraint(self):
        """The image build must install with `-c requirements-lock.txt`
        (and copy the lock in first), or the deployed wheels are not the
        tested wheels."""
        dockerfile = (_ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "-r requirements.txt -c requirements-lock.txt" in dockerfile
        assert "requirements-lock.txt ./requirements-lock.txt" in dockerfile

    def test_ci_installs_with_lock_constraint_and_audits_it(self):
        """Both CI install steps must use the lock constraint, and CI must
        run pip-audit against the lock so a new advisory in the shipped
        set fails the build instead of rotting silently."""
        ci = (_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        assert ci.count("-r requirements.txt -c requirements-lock.txt") >= 2, (
            "both the test and lint jobs must install with the lock "
            "constraint so CI tests the exact universe the image ships"
        )
        assert "pip-audit" in ci, "CI must pip-audit the locked dependency set"
        assert "--no-deps" in ci, (
            "pip-audit should run in --no-deps mode against the fully "
            "pinned lock (no re-resolution)"
        )
