"""v3 world model: sandbox, verifier counterexamples, planner shortest paths."""

from __future__ import annotations

import numpy as np

from arc3.types import Action
from arc3.v3.facts import FactBook
from arc3.v3.world import State, WorldModel, parse_action, plan, verify


def room(px, py, wall=True):
    """8x8 room of 4-pixel cells, colour-3 block, a wall column at cell x=4
    except a gap at cell y=6, and a colour-2 goal at cell (7,0)."""
    f = np.zeros((64, 64), dtype=np.int8)
    if wall:
        for cy in range(8):
            if cy != 6:
                f[4 * cy:4 * cy + 4, 16:20] = 5
    f[0:4, 28:32] = 2
    f[4 * py:4 * py + 4, 4 * px:4 * px + 4] = 3
    return f


MOVES = '''
MOVES = {"ACTION1": (0, -4), "ACTION2": (0, 4), "ACTION3": (-4, 0), "ACTION4": (4, 0)}
def predict(state, action):
    if action not in MOVES:
        return state
    b = state.of_colour(3)[0]
    dx, dy = MOVES[action]
    if state.free(b["x"] + dx, b["y"] + dy, 4, 4, passable={0, 2}, ignore=[b["id"]]):
        state.move(b["id"], dx, dy)
    return state
'''
GOAL = '''
def goal(state):
    b, g = state.of_colour(3), state.of_colour(2)
    if not b or not g: return 0.0
    d = abs(b[0]["x"] - g[0]["x"]) + abs(b[0]["y"] - g[0]["y"])
    return 1.0 - d / 128
'''
NO_WALLS = MOVES.replace('passable={0, 2}', 'passable={0, 2, 5}')


def recorded():
    fb = FactBook()
    fb.observe(room(1, 1))
    moves = [(4, (2, 1)), (4, (3, 1)), (4, (3, 1)), (2, (3, 2))]  # third ACTION4 blocked by wall
    for a, (x, y) in moves:
        fb.observe(room(x, y), Action.simple(a))
    return fb


def test_correct_model_verifies_exactly():
    fb = recorded()
    chk = verify(WorldModel.compile(MOVES, GOAL), fb.steps, background=fb.background)
    assert chk.n == 4 and chk.exact == 4


def test_wrong_model_yields_exact_counterexample():
    fb = recorded()
    chk = verify(WorldModel.compile(NO_WALLS, GOAL), fb.steps, background=fb.background)
    assert chk.exact == 3
    ce = chk.counterexamples[0]
    assert "t2 ACTION4" in ce and "actually x12 y4" in ce


def test_planner_finds_shortest_path_around_the_wall():
    fb = FactBook()
    fb.observe(room(1, 1))
    wm = WorldModel.compile(MOVES, GOAL)
    start = State.from_objects(fb.objects, fb.grid, fb.masks, fb.background)
    p = plan(wm, start, [1, 2, 3, 4], click=False)
    # Through the gap at cell y=6: down 5, right 6, up 6 = 17 moves.
    assert p.reached and len(p.actions) == 17
    assert p.actions.count("ACTION4") == 6


def test_sandbox_rejects_imports_and_survives_crashes():
    assert WorldModel.compile("import os\ndef predict(s, a):\n    return s").error
    wm = WorldModel.compile("def predict(state, action):\n    return 1 / 0\n")
    fb = recorded()
    s = State.before(fb.steps[0])
    assert wm.predict(s, "ACTION1").key() == s.key()   # failing model: no change
    assert "ZeroDivisionError" in wm.error


def test_parse_action():
    assert parse_action("ACTION3") == (3, None, None)
    assert parse_action("click(10, 20)") == (6, 10, 20)
    assert parse_action("jump") is None
