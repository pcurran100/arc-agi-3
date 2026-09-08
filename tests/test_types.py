from __future__ import annotations

import numpy as np
import pytest

from arc3.types import (
    Action,
    Episode,
    GameMetadata,
    GameState,
    Observation,
    freeze_grid,
)


def test_grids_are_immutable_and_validated():
    g = freeze_grid(np.zeros((64, 64), dtype=np.int8))
    assert g.dtype == np.int8 and g.shape == (64, 64)
    with pytest.raises(ValueError):
        g[0, 0] = 1
    with pytest.raises(ValueError):
        freeze_grid(np.zeros((32, 32)))
    with pytest.raises(ValueError):
        freeze_grid(np.full((64, 64), 16))


def test_click_requires_coordinates_in_range():
    assert Action.click(0, 63).data == {"x": 0, "y": 63}
    with pytest.raises(ValueError):
        Action(6)
    with pytest.raises(ValueError):
        Action.click(64, 0)
    with pytest.raises(ValueError):
        Action(1, x=3, y=3)  # a simple action must not carry coordinates
    with pytest.raises(ValueError):
        Action(9)


def test_action_roundtrips_through_json():
    for a in (Action.simple(0), Action.simple(7), Action.click(12, 34)):
        assert Action.from_json(a.to_json()) == a


def _obs(frames, **kw):
    return Observation(
        frames=tuple(freeze_grid(f) for f in frames),
        state=kw.pop("state", GameState.NOT_FINISHED),
        levels_completed=kw.pop("levels_completed", 0),
        win_levels=kw.pop("win_levels", 6),
        available_actions=kw.pop("available_actions", (1, 2)),
        **kw,
    )


def test_noop_and_error_are_different_things():
    """An engine failure must never be mistaken for 'the action did nothing'."""
    noop = _obs([], action=Action.simple(1))
    err = _obs([], action=Action.simple(1), error="engine wrapper returned None")
    assert noop.is_noop and not noop.is_error
    assert err.is_error and not err.is_noop
    assert noop.last is None and err.last is None


def test_observation_roundtrips_with_frames():
    o = _obs([np.zeros((64, 64)), np.ones((64, 64))], action=Action.click(1, 2), actions_used=3)
    back = Observation.from_json(o.to_json())
    assert back.action == o.action and back.actions_used == 3
    assert len(back) == 2
    assert np.array_equal(back.last, o.last)


def test_metadata_withholding_is_total():
    """D-012: consumers must cope when tags and baselines are unavailable."""
    m = GameMetadata("ls20", 7, tags=("keyboard",), baseline_actions=(22, 123))
    assert m.baseline_for_level(1) == 123
    assert m.baseline_for_level(5) is None
    w = m.withheld()
    assert w.tags is None and w.baseline_actions is None
    assert w.baseline_for_level(0) is None
    assert w.game_id == m.game_id and w.win_levels == m.win_levels


def test_transitions_carry_the_whole_animation():
    """design-thesis section 3 undercounted this: it is not one frame per action."""
    a, b, c = (np.zeros((64, 64)), np.ones((64, 64)), np.full((64, 64), 2))
    ep = Episode("g", 0, GameMetadata("g", 2))
    ep.observations = [
        _obs([a]),
        _obs([b, c], action=Action.simple(1)),
    ]
    (t,) = ep.transitions()
    assert np.array_equal(t.before, a)
    assert np.array_equal(t.after, c)
    assert t.n_intermediate == 1
    assert not t.is_noop


def test_transitions_carry_prev_grid_through_a_noop():
    a = np.zeros((64, 64))
    ep = Episode("g", 0, GameMetadata("g", 2))
    ep.observations = [
        _obs([a]),
        _obs([], action=Action.simple(1)),  # no-op
        _obs([np.ones((64, 64))], action=Action.simple(2)),
    ]
    t1, t2 = ep.transitions()
    assert t1.is_noop and np.array_equal(t1.after, a)
    # The second transition must start from the last *real* grid, not from nothing.
    assert np.array_equal(t2.before, a)


def test_transitions_skip_errored_steps():
    a = np.zeros((64, 64))
    ep = Episode("g", 0, GameMetadata("g", 2))
    ep.observations = [
        _obs([a]),
        _obs([], action=Action.simple(1), error="boom"),
        _obs([np.ones((64, 64))], action=Action.simple(2)),
    ]
    assert len(ep.transitions()) == 1
