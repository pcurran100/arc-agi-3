"""The exploring driver, and the online bookkeeping-band estimate it depends on.

Runs entirely on synthetic observations: a driver is a function of what it has
been shown, so none of this needs the engine or spends an action.

The tests worth reading are the ones encoding measured facts rather than
intentions — that a counter must not be mistaken for the game (vc33 changed its
frame on 99 of 99 actions and its playfield on zero), and that an arm which has
failed repeatedly is still occasionally retried (responsiveness is state-gated
on eight of nineteen games).
"""

from __future__ import annotations

import numpy as np
import pytest

from arc3.explore import BAND_WARMUP, ExploringDriver, OnlineBand, _Arm
from arc3.types import CLICK, Action, GameState, Observation, freeze_grid


def frame(objects=((10, 10, 3),), counter: int = -1) -> np.ndarray:
    g = np.zeros((64, 64), dtype=np.int8)
    for x, y, c in objects:
        g[y : y + 4, x : x + 4] = c
    if counter >= 0:
        g[62:64, :counter] = 11
    return freeze_grid(g)


def obs(f: np.ndarray, available=(1, 2, 6)) -> Observation:
    return Observation(
        frames=(f,),
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        win_levels=6,
        available_actions=available,
    )


# -- the band ----------------------------------------------------------------


def test_band_finds_a_counter_and_ignores_the_playfield():
    """The distinction the whole policy rests on.

    vc33 changed its frame on 99 of 99 actions and its playfield on zero: every
    change was the counter. A policy rewarded for frame change is satisfied by
    a game that never responds.
    """
    b = OnlineBand()
    for k in range(BAND_WARMUP + 6):
        before = frame(((10 + k, 10, 3),), counter=k)
        after = frame(((10 + k, 10, 3),), counter=k + 1)
        b.update(before, after)
    m = b.mask()
    assert m[62].all() and m[63].all()
    assert not m[10].any()


def test_band_says_a_counter_only_transition_did_not_move_the_playfield():
    b = OnlineBand()
    for k in range(BAND_WARMUP + 6):
        b.update(frame(counter=k), frame(counter=k + 1))
    assert not b.moved_playfield(frame(counter=30), frame(counter=31))
    assert b.moved_playfield(frame(((10, 10, 3),), counter=30), frame(((20, 10, 3),), counter=31))


def test_band_is_empty_before_warmup_so_nothing_is_wrongly_excluded():
    """Under-confident early rather than wrong early."""
    b = OnlineBand()
    b.update(frame(counter=1), frame(counter=2))
    assert not b.ready
    assert int(b.mask().sum()) == 0
    assert b.moved_playfield(frame(counter=1), frame(counter=2))


def test_band_refuses_a_band_thick_enough_to_be_playfield():
    b = OnlineBand()
    for _ in range(BAND_WARMUP + 4):
        g = np.zeros((64, 64), dtype=np.int8)
        h = g.copy()
        h[20:40, :] = 5
        b.update(freeze_grid(g), freeze_grid(h))
    assert int(b.mask().sum()) == 0


# -- the posterior -----------------------------------------------------------


def test_a_failing_arm_is_down_weighted_but_never_eliminated():
    """Guardrail 1, and it is load-bearing here.

    Responsiveness is state-dependent: vc33 moves the playfield on 2 of 256
    probes at the level start and 254 of 256 five actions later. An arm written
    off at the start would never be retried when the game comes alive.
    """
    import random

    rng = random.Random(0)
    arm = _Arm()
    for _ in range(50):
        arm.update(False)
        arm.decay(0.97)
    assert arm.mean < 0.35
    assert max(arm.sample(rng) for _ in range(200)) > 0.05


def test_decay_lets_an_arm_recover_when_the_game_changes():
    """A stationary estimate cannot follow a game that becomes responsive."""
    arm = _Arm()
    for _ in range(30):
        arm.update(False)
        arm.decay(0.97)
    low = arm.mean
    for _ in range(10):
        arm.update(True)
        arm.decay(0.97)
    assert arm.mean > low + 0.2


# -- the driver --------------------------------------------------------------


def test_driver_learns_which_action_moves_the_playfield():
    """The core behaviour: prefer what works, without being told which it is."""
    d = ExploringDriver(seed=0)
    pos = 10
    counter = 0
    picks = []
    for _ in range(80):
        a = d.next_action(obs(frame(((pos, 10, 3),), counter=counter), available=(1, 2)))
        picks.append(a.id)
        counter += 1
        if a.id == 1:  # only action 1 does anything real
            pos += 1
    early = picks[:20].count(1)
    late = picks[-20:].count(1)
    assert late > early, f"did not learn: early {early}, late {late} of 20"
    assert late >= 12


def test_driver_is_not_fooled_by_a_counter_that_ticks_on_every_action():
    """If it were, it would rate both actions equally and learn nothing.

    Asserts the property the design actually promises, not an absolute zero.
    Before the band is estimated every change is credited as playfield
    (`BAND_WARMUP` observations of deliberate over-crediting, on the grounds
    that being under-confident early beats being wrong early), so the honest
    test is that crediting **stops** once the band is known.
    """
    d = ExploringDriver(seed=0)
    counter = 0
    for _ in range(BAND_WARMUP + 4):
        d.next_action(obs(frame(counter=counter), available=(1, 2)))
        counter += 1
    assert d.band.ready
    credited_after_warmup = d.n_playfield

    for _ in range(40):
        d.next_action(obs(frame(counter=counter), available=(1, 2)))
        counter += 1
    assert d.n_playfield == credited_after_warmup, (
        "counter ticks were still being credited as playfield activity"
    )
    assert d.playfield_rate < 0.25  # the warmup residue, washing out


def test_driver_clicks_objects_not_background():
    d = ExploringDriver(seed=0)
    f = frame(((10, 10, 3), (40, 40, 7)))
    for _ in range(25):
        a = d.next_action(obs(f, available=(CLICK,)))
        assert a.is_click
        assert f[a.y, a.x] != 0, f"clicked background at ({a.x},{a.y})"


def test_driver_stops_on_terminal_state():
    o = Observation(
        frames=(frame(),),
        state=GameState.GAME_OVER,
        levels_completed=0,
        win_levels=6,
        available_actions=(1, 2, 6),
    )
    assert ExploringDriver(seed=0).next_action(o) is None


def test_driver_is_deterministic_given_a_seed():
    def seq():
        d = ExploringDriver(seed=5)
        return [str(d.next_action(obs(frame(counter=i)))) for i in range(30)]

    assert seq() == seq()


def test_driver_survives_a_missing_frame():
    o = Observation(
        frames=(),
        state=GameState.NOT_FINISHED,
        levels_completed=0,
        win_levels=6,
        available_actions=(1, 2),
    )
    assert ExploringDriver(seed=0).next_action(o) is not None


def test_arms_are_object_signatures_so_experience_survives_movement():
    """An arm keyed on a pixel would be worthless the moment the object moves."""
    d = ExploringDriver(seed=0)
    for x in (10, 20, 30, 40):
        d.next_action(obs(frame(((x, 10, 3),)), available=(CLICK,)))
    keys = [k for k in d._arms if k[0] == "click"]
    assert len(keys) == 1, f"expected one arm for one object across frames, got {keys}"
