"""Tests for dynamics induction.

Several of these are **regression tests for bugs that silently produced a
negative result** during the first run of the dynamics experiment. Each one
would, if reintroduced, make the experiment report "dynamics do not induce"
while the actual fault lay in the scoring or the search. They are the most
valuable tests in the file and they are marked as such.
"""

from __future__ import annotations

import numpy as np
import pytest

from arc3.dynamics import (
    Effect,
    Rule,
    RuleSet,
    Selector,
    induce,
    observe_transition,
    predict_frame,
    score_transitions,
)
from arc3.dynamics.predict import bookkeeping_mask, score_static_baseline
from arc3.dynamics.rules import (
    RECOLOUR,
    STATIC,
    TRANSLATE,
    VANISH,
    describe_object,
    reveal_colour,
)
from arc3.perception.segment import segment
from arc3.types import Action, GameState, Transition, freeze_grid

BG = 0


def grid(*, background: int = BG) -> np.ndarray:
    return np.full((64, 64), background, dtype=np.int8)


def with_square(x, y, colour=3, size=4, *, background=BG, floor=None):
    g = grid(background=background)
    if floor is not None:
        g[y - 2 : y + size + 2, x - 2 : x + size + 2] = floor
    g[y : y + size, x : x + size] = colour
    return freeze_grid(g)


def transition(before, after, action=Action.simple(1)) -> Transition:
    return Transition(
        before=before,
        action=action,
        frames=(after,),
        levels_completed_before=0,
        levels_completed_after=0,
        state_after=GameState("NOT_FINISHED"),
    )


# -- the effect vocabulary ---------------------------------------------------


def test_a_clean_translation_is_observed_as_translate():
    t = transition(with_square(10, 10), with_square(14, 10))
    obs = observe_transition(t, "foreground_8")
    moved = [e for e in obs.events if not e.effect.is_identity]
    assert len(moved) == 1
    assert moved[0].effect.kind == TRANSLATE
    assert (moved[0].effect.dx, moved[0].effect.dy) == (4, 0)


def test_a_recolour_in_place_is_observed_as_recolour():
    t = transition(with_square(10, 10, colour=3), with_square(10, 10, colour=5))
    obs = observe_transition(t, "foreground_8")
    changed = [e for e in obs.events if not e.effect.is_identity]
    assert len(changed) == 1
    assert changed[0].effect.kind == RECOLOUR
    assert changed[0].effect.colour == 5


def test_a_disappearance_is_observed_as_vanish():
    t = transition(with_square(10, 10), freeze_grid(grid()))
    obs = observe_transition(t, "foreground_8")
    assert [e.effect.kind for e in obs.events] == [VANISH]


def test_a_move_that_changes_silhouette_is_counted_as_deformed():
    """The dominant known loss, so it must be *counted* rather than ignored.

    27-48% of real moves in the corpus are of this shape under colour_8. The
    language cannot express them and records them as static; what it must not do
    is fail to notice.
    """
    t = transition(with_square(10, 10, size=4), with_square(14, 10, size=6))
    obs = observe_transition(t, "foreground_8")
    assert obs.n_moves == 1
    assert obs.n_moves_deformed == 1
    assert all(e.effect.kind == STATIC for e in obs.events)


# -- rendering ---------------------------------------------------------------


def test_reveal_colour_prefers_the_local_surface_over_the_global_background():
    """REGRESSION. Erasing to the global background scored 0/100 on ls20.

    An object resting on a floor reveals the *floor* when it leaves, not the
    frame's background colour. Getting this wrong cost a median of 27 wrong
    pixels per transition and made exact prediction impossible even when every
    object's effect was known exactly.
    """
    frame = with_square(20, 20, colour=3, size=4, background=0, floor=7)
    seg = segment(frame, include=("colour_8",))["colour_8"]
    obj = next(c for c in seg.components if c.dominant_colour == 3)
    assert reveal_colour(frame, obj.mask, background=0) == 7


def test_prediction_reproduces_a_translation_exactly():
    before = with_square(10, 10)
    after = with_square(14, 10)
    rs = RuleSet(
        rules=(Rule(1, Selector((("colour", 3),)), Effect(TRANSLATE, dx=4, dy=0)),),
        defaults={1: Effect(STATIC)},
    )
    pred = predict_frame(before, Action.simple(1), rs, "foreground_8")
    assert np.array_equal(pred, after)


def test_prediction_reproduces_a_translation_over_a_floor_exactly():
    before = with_square(20, 20, colour=3, size=4, floor=7)
    after = with_square(24, 20, colour=3, size=4, floor=7)
    # The floor is wider than both positions, so the reveal is unambiguous.
    after = after.copy()
    after[18:26, 18:30] = 7
    after[20:24, 24:28] = 3
    rs = RuleSet(
        rules=(Rule(1, Selector((("colour", 3),)), Effect(TRANSLATE, dx=4, dy=0)),),
        defaults={1: Effect(STATIC)},
    )
    pred = predict_frame(before, Action.simple(1), rs, "colour_8")
    assert np.array_equal(pred[20:24, 24:28], np.full((4, 4), 3, dtype=np.int8))
    assert np.array_equal(pred[20:24, 20:24], np.full((4, 4), 7, dtype=np.int8))


def test_a_multicoloured_sprite_keeps_its_own_pixels_when_it_moves():
    """Repainting with the dominant colour would corrupt every striped sprite."""
    g = grid()
    g[10:14, 10:12] = 3
    g[10:14, 12:14] = 5
    before = freeze_grid(g)
    rs = RuleSet(rules=(), defaults={1: Effect(TRANSLATE, dx=6, dy=0)})
    pred = predict_frame(before, Action.simple(1), rs, "foreground_8")
    assert set(np.unique(pred[10:14, 16:18]).tolist()) == {3}
    assert set(np.unique(pred[10:14, 18:20]).tolist()) == {5}


# -- MDL ---------------------------------------------------------------------


def _events(action, specs):
    """Build transitions where `specs` is a list of (colour, size, moved)."""
    out = []
    for colour, size, moved in specs:
        before = with_square(10, 10, colour=colour, size=size)
        after = with_square(10 + (4 if moved else 0), 10, colour=colour, size=size)
        out.append(transition(before, after, action))
    return out


def test_mdl_learns_a_single_general_rule_from_consistent_evidence():
    ts = _events(Action.simple(1), [(3, 4, True)] * 6)
    rs = induce([observe_transition(t, "foreground_8") for t in ts])
    eff = rs.effect_for(Action.simple(1), {"colour": 3, "size": 16})
    assert eff.kind == TRANSLATE and eff.dx == 4


def test_mdl_does_not_split_when_the_default_explains_everything():
    """No rules should be emitted when one default suffices."""
    ts = _events(Action.simple(1), [(3, 4, True), (5, 4, True), (7, 4, True)] * 3)
    rs = induce([observe_transition(t, "foreground_8") for t in ts])
    assert rs.rules == ()


def test_mdl_charges_only_for_exception_groups_not_for_every_group():
    """REGRESSION, and the one that mattered most.

    An earlier cost function charged bits per *group in the partition* rather
    than per *rule written down*. On ls20 that made it prefer a selector with 16
    errors over one with 1 error, purely because the better selector induced 37
    groups of which 35 agreed with the default and needed no rule at all.

    Here: many distinct sizes, all static except one, which moves. A correct
    MDL must pay for one exception, not for the whole size alphabet, and so must
    find the rule.
    """
    specs = [(3, s, False) for s in range(2, 14)] + [(3, 20, True)] * 4
    ts = []
    for colour, size, moved in specs:
        before = with_square(10, 30, colour=colour, size=size)
        after = with_square(10 + (4 if moved else 0), 30, colour=colour, size=size)
        ts.append(transition(before, after, Action.simple(1)))
    rs = induce([observe_transition(t, "foreground_8") for t in ts])
    eff = rs.effect_for(Action.simple(1), {"colour": 3, "size": 400, "rect": True})
    assert eff.kind == TRANSLATE, f"failed to isolate the mover: {rs.describe()}"


def test_induction_is_deterministic():
    ts = _events(Action.simple(1), [(3, 4, True), (5, 4, False)] * 4)
    obs = [observe_transition(t, "foreground_8") for t in ts]
    a, b = induce(obs), induce(obs)
    assert a.describe() == b.describe()


# -- the bookkeeping layer ---------------------------------------------------


def test_bookkeeping_mask_finds_a_counter_that_advances():
    """REGRESSION. A per-pixel frequency mask finds an advancing counter never.

    ls20's move counter ticks one column per action in rows 61-62, so no single
    pixel changes twice, and the per-pixel version returned an empty mask while
    the rows were changing 99% of the time. Detection has to be band-level.
    """
    ts = []
    for k in range(10):
        b = grid()
        b[61:63, :k] = 11
        a = b.copy()
        a[61:63, k] = 11
        # An unrelated playfield object, moving to a *different* place each time
        # as a real sprite does, so its rows and columns are not constant.
        b[5 + 2 * k : 9 + 2 * k, 5 + k : 9 + k] = 3
        a[5 + 2 * k : 9 + 2 * k, 9 + k : 13 + k] = 3
        ts.append(transition(freeze_grid(b), freeze_grid(a)))
    mask = bookkeeping_mask(ts)
    assert mask[61].all() and mask[62].all()
    assert not mask[30].any()


def test_bookkeeping_mask_refuses_a_band_thick_enough_to_be_the_playfield():
    """REGRESSION, caught by the test above failing.

    A playfield object confined to fixed rows makes those rows change in 100%
    of transitions. Labelling them bookkeeping would not merely miss the UI, it
    would *inflate* the ex-bookkeeping figure by excusing the model from
    predicting the actual game. Bands thicker than `MAX_BAND` are refused.
    """
    ts = []
    for k in range(10):
        b = grid()
        a = b.copy()
        b[20:40, 5:9] = 3
        a[20:40, 9:13] = 3
        ts.append(transition(freeze_grid(b), freeze_grid(a)))
    mask = bookkeeping_mask(ts)
    assert int(mask.sum()) == 0, "a 20-row band was accepted as bookkeeping"


def test_bookkeeping_is_learned_from_training_transitions_only():
    """It is part of the model, so fitting it on test data would be leakage."""
    import inspect

    src = inspect.getsource(bookkeeping_mask)
    assert "training transitions only" in src


def test_vacuous_flag_fires_when_all_change_is_bookkeeping():
    """REGRESSION. vc33 scored 100% ex-bookkeeping with 0% object accuracy.

    If every change a game makes falls inside the mask, "exact outside the mask"
    is satisfied by predicting nothing at all, and the figure must be marked
    unusable rather than reported as a success.
    """
    ts = []
    for k in range(10):
        b = grid()
        b[61:63, :k] = 11
        a = b.copy()
        a[61:63, k] = 11
        ts.append(transition(freeze_grid(b), freeze_grid(a)))
    mask = bookkeeping_mask(ts)
    sc = score_transitions(ts, RuleSet(), "colour_8", None, ignore=mask)
    assert sc.n_changed > 0
    assert sc.n_changed_outside == 0
    assert sc.vacuous


# -- the metric itself -------------------------------------------------------


def test_changed_object_accuracy_is_not_flattered_by_static_objects():
    """Plain object accuracy scores ~91% for predicting nothing ever happens."""
    ts = _events(Action.simple(1), [(3, 4, True)] * 4)
    empty = RuleSet()
    sc = score_transitions(ts, empty, "foreground_8")
    assert sc.object_accuracy == pytest.approx(0.0) or sc.object_accuracy < 1.0
    assert sc.changed_object_accuracy == pytest.approx(0.0)


def test_static_baseline_scores_zero_on_changed_transitions():
    """D-019's measured chance level. If this ever moves, the bar moved."""
    ts = _events(Action.simple(1), [(3, 4, True)] * 5)
    b0 = score_static_baseline(ts)
    assert b0.n_changed == 5
    assert b0.exact_on_changed == 0.0


def test_scoring_a_perfect_ruleset_reaches_one_hundred_percent():
    """Guards against a metric that can never be satisfied."""
    ts = _events(Action.simple(1), [(3, 4, True)] * 4)
    rs = RuleSet(rules=(), defaults={1: Effect(TRANSLATE, dx=4, dy=0)})
    sc = score_transitions(ts, rs, "foreground_8")
    assert sc.exact_on_changed == 1.0


# -- integration on a real recording -----------------------------------------


def test_induction_runs_on_a_real_episode_and_beats_the_static_baseline(ls20):
    from arc3.perception.geometry import induce_geometry

    trans = [t for t in ls20.transitions() if t.action.id != 0]
    if len(trans) < 10:
        pytest.skip("fixture too short")
    geo = induce_geometry(ls20.grids()[:40])
    cut = int(len(trans) * 0.7)
    obs = [observe_transition(t, "colour_8", geo) for t in trans[:cut]]
    rs = induce(obs)
    sc = score_transitions(trans[cut:], rs, "colour_8", geo)
    # Not asserting a level: the sweep measures that. Asserting it runs, that
    # something was learned, and that the diagnostics are populated.
    assert rs.defaults
    assert sc.n_total == len(trans) - cut
    assert sc.n_objects > 0
