"""The movement simulator Nemotron wrote for ls20 in debug run (d) (2026-10-30,
call 2), with its newlines restored and sent as lines, validates on real ls20
play and code searches it."""

import pytest

from arc3.v5 import sim as simmod

pytest.importorskip("arc_agi")

LINES = [
    "def predict(state, action):",
    "    p = state.get(7)",
    "    f = state.get(8)",
    "    if not p or not f:",
    "        return state",
    "    delta = {'ACTION1': (0,-5), 'ACTION2': (0,5), 'ACTION3': (-5,0), 'ACTION4': (5,0)}.get(action)",
    "    if delta is None:",
    "        return state",
    "    dx, dy = delta",
    "    if state.free(p['x'] + dx, p['y'] + dy, 5, 5, passable=(3,), ignore=(7, 8)):",
    "        state.move(7, dx, dy)",
    "        state.move(8, dx, dy)",
    "    return state",
    "",
    "def goal(state):",
    "    p = state.get(7)",
    "    return bool(p and p['x'] == 14 and p['y'] == 30)",
]


@pytest.mark.engine
def test_the_models_ls20_simulator_validates_and_is_searched():
    from arc3.client.env import EnvClient
    from arc3.types import Action
    from arc3.v5.facts import FactBook5

    try:
        c = EnvClient("ls20", seed=0, record=False)
    except Exception as exc:  # no environment files here
        pytest.skip(str(exc))
    fb = FactBook5()
    fb.observe(c.current.last, None)
    for a in (1, 2, 3, 4, 4, 1, 1, 3):
        o = c.step(Action.simple(a))
        fb.observe(o.last, Action.simple(a))
    code = simmod.source_of({"lines": LINES, "tracked": ["#7", "#8"], "goal_id": "A1"})
    s = simmod.build(code, ["#7", "#8"], "A1", fb)
    assert s.model.error is None, s.why
    assert s.check.n == 8 and s.check.accuracy >= 0.8, s.why
    assert s.valid
    r = simmod.search(s, fb, [1, 2, 3, 4], False)
    assert r.reached and 0 < len(r.actions) <= 12, r.why
