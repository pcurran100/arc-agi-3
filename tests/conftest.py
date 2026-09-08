"""Shared fixtures.

The perception tests run entirely from recorded episodes in ``fixtures/`` so
they need no engine, no network and no actions. Tests that genuinely need the
live engine are marked ``@pytest.mark.engine`` and skip when
``environment_files/`` is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arc3.client.replay import read_episode
from arc3.types import Episode

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "fixtures"
ENVIRONMENTS_DIR = REPO_ROOT / "environment_files"

#: Fixture games and what each one is here to exercise.
FIXTURE_GAMES = ("ls20", "sp80", "bp35", "vc33", "sb26")


def load(game: str) -> Episode:
    path = FIXTURE_DIR / f"{game}.jsonl.gz"
    if not path.exists():
        pytest.skip(f"missing fixture {path}; run scripts/record_fixtures.py")
    return read_episode(path)


@pytest.fixture(scope="session")
def episodes() -> dict[str, Episode]:
    return {g: load(g) for g in FIXTURE_GAMES}


@pytest.fixture(scope="session")
def ls20() -> Episode:
    return load("ls20")


@pytest.fixture(scope="session")
def sp80() -> Episode:
    """4x upscaled camera - the case geometry has to get right."""
    return load("sp80")


@pytest.fixture(scope="session")
def sb26() -> Episode:
    """Long animated transitions - up to 42 frames for one action."""
    return load("sb26")


@pytest.fixture(scope="session")
def bp35() -> Episode:
    """UI-rendered, 8x camera underneath - broke global scale induction."""
    return load("bp35")


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("engine") and not ENVIRONMENTS_DIR.is_dir():
        pytest.skip("no environment_files/; run scripts/fetch_environments.py")
