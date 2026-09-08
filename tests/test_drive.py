"""Driving policies.

These run entirely on synthetic observations — no engine, no actions spent —
because a driver is a pure function of the observation it is handed.
"""

from __future__ import annotations

import numpy as np
import pytest

from arc3.drive import (
    ACTION_CYCLE,
    RandomDriver,
    RoundRobinDriver,
    lattice_targets,
    make_driver,
    object_targets,
)
from arc3.types import GameState, Observation, freeze_grid


def obs(available: tuple[int, ...], frame: np.ndarray | None = None) -> Observation:
    if frame is None:
        frame = np.zeros((64, 64), dtype=np.int8)
        frame[10:14, 10:20] = 3
        frame[40:44, 30:36] = 7
    return Observation(
        frames=(freeze_grid(frame),),
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        win_levels=6,
        available_actions=available,
    )


# -- coverage, the whole point of round-robin -------------------------------


def test_round_robin_covers_every_legal_action():
    d = RoundRobinDriver()
    seen = {d.next_action(obs((1, 2, 3, 4))).id for _ in range(12)}
    assert seen == {1, 2, 3, 4}


def test_round_robin_covers_a_sparse_mask():
    """Non-contiguous masks are real: bp35 publishes [3, 4, 6, 7]."""
    d = RoundRobinDriver()
    seen = {d.next_action(obs((3, 4, 6, 7))).id for _ in range(16)}
    assert seen == {3, 4, 6, 7}


def test_round_robin_is_even_not_merely_covering():
    d = RoundRobinDriver()
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for _ in range(40):
        counts[d.next_action(obs((1, 2, 3, 4))).id] += 1
    assert set(counts.values()) == {10}


def test_reset_is_never_emitted_even_if_offered():
    """RESET is never published by the engine, but a driver must not step it
    regardless: it would restart the level and put a discontinuity in the corpus."""
    d = RoundRobinDriver()
    for _ in range(20):
        assert d.next_action(obs((0, 1, 2))).id != 0


# -- the two degenerate game shapes -----------------------------------------


def test_click_only_game_varies_the_target_not_the_action():
    """6 of 25 games publish only [6]; the target is then the entire policy."""
    d = RoundRobinDriver()
    actions = [d.next_action(obs((6,))) for _ in range(8)]
    assert all(a.id == 6 for a in actions)
    assert len({(a.x, a.y) for a in actions}) > 1, "target cursor never advanced"


def test_keyboard_only_game_never_emits_a_click():
    """6 of 25 games never publish 6 at all."""
    d = RoundRobinDriver()
    for _ in range(20):
        a = d.next_action(obs((1, 2, 3, 4, 5)))
        assert a.id != 6 and a.x is None and a.y is None


# -- stopping ---------------------------------------------------------------


def test_driver_stops_on_terminal_state():
    o = Observation(
        frames=(freeze_grid(np.zeros((64, 64), dtype=np.int8)),),
        state=GameState.WIN,
        levels_completed=6,
        win_levels=6,
        available_actions=(1, 2),
    )
    assert RoundRobinDriver().next_action(o) is None
    assert RandomDriver().next_action(o) is None


def test_driver_stops_when_no_actions_are_available():
    assert RoundRobinDriver().next_action(obs(())) is None


# -- determinism (D-006: seed everything, reproduce from the log alone) ------


@pytest.mark.parametrize("name", ["random", "round_robin", "lattice"])
def test_same_seed_gives_the_same_sequence(name):
    def seq():
        d = make_driver(name, seed=7)
        return [str(d.next_action(obs((1, 2, 3, 4, 5, 6, 7)))) for _ in range(20)]

    assert seq() == seq()


def test_round_robin_does_not_depend_on_the_seed():
    """Round-robin is deterministic by construction, not by seeding."""
    a = RoundRobinDriver(seed=0)
    b = RoundRobinDriver(seed=999)
    for _ in range(10):
        assert str(a.next_action(obs((1, 2, 6)))) == str(b.next_action(obs((1, 2, 6))))


# -- click targets ----------------------------------------------------------


def test_lattice_targets_are_in_bounds_and_cover_the_frame():
    t = lattice_targets(8)
    assert len(t) == 64
    assert all(0 <= x < 64 and 0 <= y < 64 for x, y in t)
    assert len(set(t)) == len(t)


def test_object_targets_land_on_objects_not_background():
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[10:14, 10:20] = 3
    frame[40:44, 30:36] = 7
    targets = object_targets(frame)
    assert targets
    hits = sum(1 for x, y in targets if frame[y, x] != 0)
    assert hits >= 2, f"expected targets on both objects, got {hits}"


def test_object_targets_fall_back_to_lattice_on_a_blank_frame():
    """A frame with nothing in it must still yield somewhere to click, or a
    click-only game would deadlock with no action to take."""
    blank = np.zeros((64, 64), dtype=np.int8)
    d = RoundRobinDriver(target_policy="objects")
    a = d.next_action(obs((6,), frame=blank))
    assert a is not None and a.id == 6
    assert 0 <= a.x < 64 and 0 <= a.y < 64


def test_lattice_policy_ignores_the_frame_entirely():
    d = RoundRobinDriver(target_policy="lattice")
    a = d.next_action(obs((6,)))
    assert (a.x, a.y) in lattice_targets(8)


# -- wiring -----------------------------------------------------------------


def test_action_cycle_excludes_reset_and_covers_the_space():
    assert 0 not in ACTION_CYCLE
    assert set(ACTION_CYCLE) == {1, 2, 3, 4, 5, 6, 7}


def test_make_driver_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown driver"):
        make_driver("greedy")
