"""v3 fact layer on synthetic frames where the right answers are known."""

from __future__ import annotations

import numpy as np

from arc3.types import Action
from arc3.v3.facts import FactBook, diff_objects, object_at


def frame(px, py, lamp=7, extra=True):
    """A 4x4 player (colour 3) with a 2x4 hat (colour 12) riding on top, two
    identical walls (colour 5), and a lamp (colour lamp)."""
    f = np.zeros((64, 64), dtype=np.int8)
    f[py:py + 4, px:px + 4] = 3
    f[py - 2:py, px:px + 4] = 12
    f[40:44, 10:30] = 5
    f[50:54, 10:30] = 5
    if extra:
        f[5:8, 55:58] = lamp
    return f


def book_with_moves():
    fb = FactBook()
    fb.observe(frame(20, 20))
    fb.observe(frame(20, 16), Action.simple(1))   # up
    fb.observe(frame(24, 16), Action.simple(4))   # right
    fb.observe(frame(24, 20), Action.simple(2))   # down
    fb.observe(frame(24, 20, lamp=9), Action.click(56, 6))  # click the lamp: recolour
    return fb


def test_ids_are_stable_and_changes_are_typed():
    fb = book_with_moves()
    s = fb.steps[0]
    moved = {c.id: (c.dx, c.dy) for c in s.changes if "moved" in c.kinds}
    assert sorted(moved.values()) == [(0, -4), (0, -4)]  # player and hat, same ids after
    last = fb.steps[-1]
    assert last.clicked is not None
    rec = [c for c in last.changes if "recoloured" in c.kinds]
    assert len(rec) == 1 and (rec[0].colour_from, rec[0].colour_to) == (7, 9)


def test_relations_static_groups_tandem_player():
    fb = book_with_moves()
    walls = [o.id for o in fb.objects.values() if o.colour == 5]
    assert set(walls) <= set(fb.static_ids())
    assert sorted(fb.colour_groups()[5]) == sorted(walls)
    assert any(sorted(ids) == sorted(walls) for ids in fb.shape_groups().values())
    tandem = fb.tandem()
    assert tandem and tandem[0][2] == 3  # player and hat moved together 3 times
    pcs = fb.player_candidates()
    assert pcs and pcs[0]["moves"]["ACTION1"] == (0, -4) and pcs[0]["consistency"] == 1.0


def test_action_effects_key_clicks_by_clicked_colour():
    eff = book_with_moves().action_effects()
    assert "CLICK colour 7" in eff
    assert eff["ACTION1"]["tries"] == 1 and eff["ACTION1"]["noops"] == 0


def test_describe_is_exact_text():
    t = book_with_moves().describe()
    assert "MOVE TOGETHER" in t and "SAME COLOUR" in t and "CONTROLLED-LOOKING" in t
    assert "recoloured 7->9" in t


def test_appear_vanish_and_click_target():
    a = FactBook()._objects_of(frame(20, 20))
    b = FactBook()._objects_of(frame(20, 20, extra=False))
    # Different books assign ids independently; diff within one book instead.
    fb = FactBook()
    fb.observe(frame(20, 20))
    step = fb.observe(frame(20, 20, extra=False), Action.simple(5))
    assert [c.kinds for c in step.changes] == [("vanished",)]
    assert object_at(fb.steps[0].before, 56, 6) is not None
    assert len(a) == len(b) + 1
    assert diff_objects(fb.objects, fb.objects) == []
