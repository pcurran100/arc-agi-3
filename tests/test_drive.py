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
    RandomObjectDriver,
    RoundRobinDriver,
    foreground_object_targets,
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


@pytest.mark.parametrize("name", ["random", "random_objects", "round_robin", "lattice"])
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


# -- random_objects: random actions, object-targeted clicks (D-026) ----------


def test_random_objects_clicks_land_inside_objects():
    """The whole point. Under RandomDriver these land anywhere at all.

    On lp85 only 2.3% of the 4,096 pixels do anything (D-021), so a uniformly
    random pixel wastes roughly forty-nine clicks in fifty. This driver draws
    from the object proposal set instead.
    """
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[10:14, 10:20] = 3
    frame[40:44, 30:36] = 7
    d = RandomObjectDriver(seed=0)
    clicks = 0
    for _ in range(60):
        a = d.next_action(obs((6,), frame))
        assert a.is_click
        clicks += 1
        assert frame[a.y, a.x] != 0, f"clicked background at ({a.x},{a.y})"
    assert clicks == 60


def test_random_objects_keeps_the_action_distribution_of_random():
    """Only the click *placement* changes, so the comparison isolates one variable.

    Both drivers draw uniformly from the legal actions with the same RNG calls
    for the action choice, so on a keyboard-only game they must agree exactly.
    """
    a = RandomDriver(seed=3)
    b = RandomObjectDriver(seed=3)
    for _ in range(30):
        assert a.next_action(obs((1, 2, 3, 4))).id == b.next_action(obs((1, 2, 3, 4))).id


def test_random_objects_falls_back_to_a_pixel_on_a_blank_frame():
    """A blank frame yields no object targets; the action must still be spent.

    Skipping would silently change the action distribution, which is exactly
    what this driver is designed not to do.
    """
    blank = np.zeros((64, 64), dtype=np.int8)
    d = RandomObjectDriver(seed=0)
    a = d.next_action(obs((6,), blank))
    assert a.is_click
    assert 0 <= a.x < 64 and 0 <= a.y < 64


def test_random_objects_never_emits_a_click_on_a_keyboard_only_game():
    d = RandomObjectDriver(seed=0)
    for _ in range(40):
        assert not d.next_action(obs((1, 2, 3, 4))).is_click


def test_random_objects_stops_on_terminal_state():
    o = Observation(
        frames=(freeze_grid(np.zeros((64, 64), dtype=np.int8)),),
        state=GameState.GAME_OVER,
        levels_completed=0,
        win_levels=6,
        available_actions=(1, 2, 6),
    )
    assert RandomObjectDriver(seed=0).next_action(o) is None


def test_random_objects_reaches_several_distinct_objects():
    """One target forever would be worse than random pixels, not better.

    The rule language's only relational feature is `clicked` (D-024), which
    needs the *same* object hit repeatedly to be learnable - but a driver that
    only ever hits one object cannot teach anything about the others.
    """
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[5:9, 5:9] = 3
    frame[20:24, 20:24] = 5
    frame[40:44, 40:44] = 7
    d = RandomObjectDriver(seed=0)
    hit = {int(frame[a.y, a.x]) for a in (d.next_action(obs((6,), frame)) for _ in range(60))}
    assert len(hit) >= 3, f"only reached colours {hit}"


def test_object_targets_include_the_background_but_foreground_ones_do_not():
    """A finding, not a preference: the colour carvings partition the whole frame.

    `object_targets` therefore offers the background region's centroid as a
    click candidate, which is usually the least informative pixel on the frame.
    D-021's large proposal counts are partly this.
    """
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[10:14, 10:20] = 3
    frame[40:44, 30:36] = 7
    everything = object_targets(frame)
    foreground = foreground_object_targets(frame)
    assert any(frame[y, x] == 0 for (x, y) in everything), "expected a background target"
    assert foreground, "expected some non-background targets"
    assert all(frame[y, x] != 0 for (x, y) in foreground)
    assert len(foreground) < len(everything)
