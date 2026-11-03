"""v5 fact layer (D-046): deformation is not motion, and interface classification
separates a counter from things that move or grow with the action."""

import numpy as np

from arc3.types import Action
from arc3.v5.facts import FactBook5

BG = 0


def frame(cells):
    """cells: list of (colour, [(x, y), ...])."""
    g = np.full((64, 64), BG, dtype=np.int64)
    g[10, 10] = 1          # a fixed landmark so the background is unambiguous
    for col, pts in cells:
        for x, y in pts:
            g[y, x] = col
    return g


def bar(x0, x1, y, col=3):
    return (col, [(x, y) for x in range(x0, x1)])


def box(x, y, w, h, col):
    return (col, [(x + i, y + j) for i in range(w) for j in range(h)])


ACTS = [Action.simple(a) for a in (1, 2, 3, 4, 1, 2, 3, 4)]
DIRS = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


def run(frames):
    fb = FactBook5()
    fb.observe(frames[0], None)
    for a, f in zip(ACTS, frames[1:]):
        fb.observe(f, a)
    return fb


def test_counter_is_classified_and_not_reported_as_moving():
    # A bar at the bottom shrinking from the left by 1 per action, whatever the
    # action; a player moving with the action.
    frames, px, py = [], 30, 30
    for k in range(len(ACTS) + 1):
        if k:
            dx, dy = DIRS[ACTS[k - 1].id]
            px, py = px + 2 * dx, py + 2 * dy
        frames.append(frame([bar(20 + k, 50, 60), box(px, py, 2, 2, 5)]))
    fb = run(frames)
    counters = fb.counters()
    assert len(counters) == 1
    (cid, desc), = counters.items()
    assert fb.objects[cid].colour == 3 and "-1 cells per action" in desc
    assert all("moved" not in c.kinds for s in fb.steps for c in s.changes if c.id == cid)
    assert "INTERFACE" in fb.describe() and cid not in fb.movers()


def test_growing_snake_following_the_action_is_not_a_counter():
    # Grows by one cell per action, but on the side the action points to.
    cells = [(30, 30)]
    frames = [frame([(7, list(cells))])]
    for a in ACTS:
        dx, dy = DIRS[a.id]
        hx, hy = cells[-1]
        cells.append((hx + dx * (1 if a.id in (2, 4) else 1), hy + dy))
        frames.append(frame([(7, list(dict.fromkeys(cells)))]))
    fb = run(frames)
    assert fb.counters() == {}


def test_moving_and_reshaping_object_is_not_a_counter():
    # Translates by 3 each step (no overlap) and alternates its shape.
    frames = []
    for k in range(len(ACTS) + 1):
        w = 2 if k % 2 else 3
        frames.append(frame([box(5 + 3 * k * 2, 20, w, 1, 9)]))
    fb = run(frames)
    assert fb.counters() == {}


def test_autonomous_mover_is_flagged_but_stays_on_the_sheet():
    frames = [frame([box(5 + 2 * k, 40, 2, 2, 6)]) for k in range(len(ACTS) + 1)]
    fb = run(frames)
    movers = fb.movers()
    assert list(movers.values()) == [(2, 0)]
    (mid,) = movers
    assert mid not in fb.counters()
    assert "MOVES ON ITS OWN" in fb.object_sheet()


def test_names_are_kept_and_shown():
    frames = [frame([box(30, 30, 2, 2, 5)]), frame([box(30, 28, 2, 2, 5)])]
    fb = FactBook5()
    fb.observe(frames[0], None)
    fb.observe(frames[1], Action.simple(1))
    (i,) = [o.id for o in fb.objects.values() if o.colour == 5]
    fb.names[i] = "player"
    assert f"#{i} 'player'" in fb.object_sheet()
