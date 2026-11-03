"""D-044: the fixes from the ls20 forensic analysis, and subgoals walked by code.

Each test names the failure it guards against. Frames are synthetic; the real
ls20 check is the probe recorded in D-044.
"""

from __future__ import annotations

import json

import numpy as np

from arc3.llm import ScriptedLLM
from arc3.types import Action, GameState, Observation
from arc3.v4.agent import V4Agent
from arc3.v4.expect import Expectation, check
from arc3.v4.facts import FactBook4, reduce_scale
from arc3.v4.route import Subgoal, route
from arc3.v4.store import AssumptionStore

from test_v4 import WalledRoom, _exp, _item, run


def frame(cells):
    f = np.zeros((64, 64), dtype=np.int8)
    for (x, y, w, h), c in cells.items():
        f[y:y + h, x:x + w] = c
    return f


def _id_of(fb, colour):
    return next(o.id for o in fb.objects.values() if o.colour == colour)


# -- fact layer -------------------------------------------------------------------


def test_a_mover_over_a_floor_does_not_reshape_the_floor():
    """ls20: '#1 reshaped' on every move; the model built a rule on it."""
    fb = FactBook4()
    fb.observe(frame({(0, 0, 40, 40): 3, (8, 8, 4, 4): 9}))
    s = fb.observe(frame({(0, 0, 40, 40): 3, (12, 8, 4, 4): 9}), Action.simple(4))
    assert [c.kinds for c in s.changes] == [("moved",)]
    assert fb.occluded == 1


def _bar_game(n, acts=(1, 2)):
    """A block that moves and a bar that loses one cell on every action."""
    fb = FactBook4()
    fb.observe(frame({(4, 4, 4, 4): 3, (0, 60, 40, 2): 11}))
    x, y = 4, 4
    steps = []
    for k in range(n):
        a = acts[k % len(acts)]
        y += 4 if a == 2 else -4
        steps.append(fb.observe(frame({(x, y, 4, 4): 3, (0, 60, 39 - k, 2): 11}),
                                Action.simple(a)))
    fb.last_xy = (x, y)
    return fb, steps


def test_a_bar_that_changes_on_every_action_is_a_ticker_and_exempt_from_unchanged():
    fb, steps = _bar_game(6)
    bar = _id_of(fb, 11)
    assert bar in fb.tickers()
    assert "CHANGE ON EVERY ACTION" in fb.describe()
    exp = _exp([_item("colour 3", "moved", 0, 4)], "unchanged")
    assert check(exp, steps[-1], "continue", ignore=frozenset(fb.tickers())).ok
    assert not check(exp, steps[-1], "continue").ok          # strict without the exemption


def test_an_action_that_changes_only_the_ticker_is_a_noop():
    fb, steps = _bar_game(5)
    x, y = fb.last_xy
    s = fb.observe(frame({(x, y, 4, 4): 3, (0, 60, 34, 2): 11}), Action.simple(3))
    assert fb.is_noop(s)
    assert "ACTION3: tried 1, no effect 1" in fb.describe()


def test_an_object_back_at_its_level_start_is_reported():
    """ls20: a lost life put the player back at its start; the model read a capture."""
    fb = FactBook4()
    fb.observe(frame({(4, 4, 4, 4): 3, (40, 40, 4, 4): 5}))
    fb.observe(frame({(8, 4, 4, 4): 3, (40, 40, 4, 4): 5}), Action.simple(4))
    s = fb.observe(frame({(4, 4, 4, 4): 3, (40, 40, 4, 4): 5}), Action.simple(3))
    assert any("level-start position" in n for n in fb.notes[s.t])


def test_shapes_equal_up_to_scale_and_rotation():
    """ls20: the goal glyph is a 2x rotated copy of the key glyph; v3 saw no link."""
    ell = np.array([[1, 0], [1, 0], [1, 1]], dtype=bool)
    big = np.kron(np.rot90(ell), np.ones((2, 2), dtype=bool))
    assert reduce_scale(big)[1] == 2
    f = np.zeros((64, 64), dtype=np.int8)
    f[2:5, 2:4][ell] = 9
    f[20:20 + big.shape[0], 20:20 + big.shape[1]][big] = 9
    fb = FactBook4()
    fb.observe(f)
    sim = fb.similar_shapes()
    assert len(sim) == 1 and "x2" in sim[0]


def test_blocking_colours_are_learned_and_routed_around():
    env = WalledRoom()
    fb = FactBook4()
    fb.observe(env.frame())
    for a in (4, 2, 1, 4, 4):       # the last move hits the wall at x=4
        env.step(Action.simple(a))
        fb.observe(env.frame(), Action.simple(a))
    assert 5 in fb.blocked and 5 not in fb.entered
    sg = Subgoal.parse({"object": f"#{_id_of(fb, 3)}", "to": f"#{_id_of(fb, 2)}",
                        "relation": "overlap"})
    path, why = route(fb, sg, ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
    assert path is not None, why
    x, y = env.x, env.y
    for a, dx, dy in path:
        x, y = x + dx // 4, y + dy // 4
        assert not (x == 4 and y != 6), "route goes through the wall"
    assert (x, y) == (7, 0)


# -- store -------------------------------------------------------------------------


def test_correct_move_predictions_do_not_credit_goals():
    """ls20: a wrong goal reached 0.91 on 'the bar ticked as predicted' supports."""
    s = AssumptionStore()
    s.apply([{"op": "add", "statement": "ACTION1 moves up", "kind": "dynamics", "confidence": 0.5},
             {"op": "add", "statement": "fill the bar", "kind": "goal", "confidence": 0.5}], "t0")
    s.supported_by_prediction(["A1", "A2"], "t1")
    assert s.items["A1"].confidence > 0.5 and s.items["A2"].confidence == 0.5
    s.goals_failed(["A1", "A2"], "t2", "no level")
    assert s.items["A2"].confidence < 0.5 and s.items["A1"].contra == 0


def test_no_rules_on_other_and_goals_keep_their_attain_condition():
    s = AssumptionStore()
    rule = {"action": "ACTION4", "expect": [_item("any", "none")]}
    s.apply([{"op": "add", "statement": "probe right", "kind": "other", "rule": rule},
             {"op": "add", "statement": "reach the box", "kind": "goal",
              "attain": {"object": "#7", "to": "#3", "relation": "overlap"}}], "t0")
    assert s.items["A1"].rule is None
    assert s.items["A2"].attain["to"] == "#3" and "attained when" in s.describe()
    assert "GOAL HYPOTHESES by weight" in s.describe()


# -- the controller ---------------------------------------------------------------------


def test_an_unmentioned_side_effect_is_now_a_wrong_prediction():
    agent = V4Agent(llm=None)
    fb = FactBook4()
    fb.observe(frame({(4, 4, 4, 4): 3, (40, 40, 4, 4): 2}))
    step = fb.observe(frame({(8, 4, 4, 4): 3, (40, 40, 4, 4): 9}), Action.simple(4))
    agent.fb = fb
    agent.queue = [("next", None, [], None)]
    agent._pending = (_exp([_item("colour 3", "moved", 4, 0)], "unchanged"), ["A1"], None)
    agent._judge(step, "continue")
    assert agent.stats["predictions_wrong"] == 1 and not agent.queue and agent.need_update


def test_a_goal_whose_condition_holds_without_a_win_loses_confidence():
    env = WalledRoom()
    agent = V4Agent(llm=None)
    agent.fb.observe(env.frame())
    wall = next(o.id for o in agent.fb.objects.values() if o.colour == 5)
    me = _id_of(agent.fb, 3)
    agent.store.apply([{"op": "add", "statement": "touching the wall wins", "kind": "goal",
                        "confidence": 0.8,
                        "attain": {"object": f"#{me}", "to": f"#{wall}", "relation": "touch"}}], "t0")
    for a in (4, 4):
        env.step(Action.simple(a))
        agent._judge(agent.fb.observe(env.frame(), Action.simple(a)), "continue")
    g = agent.store.items["A1"]
    assert g.confidence < 0.8 and g.contra == 1         # once per arrival, not per step
    assert any("GOAL TEST" in e for e in agent.events)


class Planner:
    """A scripted model that explores two moves, then hands code a subgoal."""

    def __init__(self, env: WalledRoom) -> None:
        self.env = env
        self.agent = None

    def __call__(self, messages) -> str:
        text = messages[-1]["content"]
        text = text[0]["text"] if isinstance(text, list) else text
        if "TASK: UPDATE" in text:
            return json.dumps({"ops": [], "summary": "ok"})
        fb = self.agent.fb
        if not fb.player_candidates() and not fb.prior_moves:
            return json.dumps({"steps": [
                {"action": "ACTION2", "expect": {"changes": [_item("colour 3", "moved", 0, 4)],
                                                 "rest": "unchanged", "outcome": "continue"}},
                {"action": "ACTION4", "expect": {"changes": [_item("colour 3", "moved", 4, 0)],
                                                 "rest": "unchanged", "outcome": "continue"}},
                {"action": "ACTION1", "expect": {"changes": [_item("colour 3", "moved", 0, -4)],
                                                 "rest": "unchanged", "outcome": "continue"}}],
                "plan": [], "purpose": "learn the moves", "tests": []})
        return json.dumps({"steps": [], "plan": [
            {"object": f"#{_id_of(fb, 3)}", "to": f"#{_id_of(fb, 2)}", "relation": "overlap",
             "on_arrival": {"changes": [], "rest": "any", "outcome": "level_complete"},
             "goal": ""}], "purpose": "go to the square", "tests": []})


def test_v4_walks_a_subgoal_with_code_routes_and_few_calls():
    env = WalledRoom()
    mind = Planner(env)
    agent = V4Agent(llm=ScriptedLLM(mind))
    mind.agent = agent
    n = run(agent, env, cap=200)
    assert env.levels == 2, agent.describe()
    assert agent.stats.get("explore_steps", 0) == 0
    assert agent.stats.get("routes", 0) >= 2
    # A walked route costs no calls: far fewer act calls than actions.
    assert agent.stats["calls_act"] * 3 < n, (agent.stats, n)
    # Level 2 is routed with level 1's moves: the three exploratory steps happen once.
    assert agent.stats["llm_steps"] == 3, agent.stats


def test_periodic_updates_are_skipped_only_while_predictions_hold():
    env = WalledRoom()
    seen = []

    def mind(messages):
        text = messages[-1]["content"]
        seen.append("update" if "TASK: UPDATE" in text else "act")
        if seen[-1] == "update":
            return json.dumps({"ops": [], "summary": "ok"})
        return json.dumps({"steps": [], "plan": [], "purpose": "", "tests": []})
    agent = V4Agent(llm=ScriptedLLM(mind), update_every=3)
    obs = env.obs()
    agent.fb.observe(obs.last)
    agent.need_update, agent.since_update = False, 5
    agent._recent = [True] * 5
    agent._decide(obs)
    assert seen == ["act"] and agent.stats["updates_skipped"] == 1
    agent.since_update = 5
    agent._recent = [True, True, False, True, True]
    agent._decide(obs)
    assert seen[1:] == ["update", "act"]
