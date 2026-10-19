"""v4: the expectation checker, the assumption store, and the controller (D-042).

The controller test plays the synthetic walled room with a scripted "model"
that knows the room's layout only after it is surprised by the wall: it first
predicts a straight run right, the third step hits the wall, the checker reports
the surprise, the queue is dropped, the update call records the wall, and the
model routes through the gap. That exercises update -> act -> check -> surprise
-> update -> act -> win -> carry the store to level 2.
"""

from __future__ import annotations

import json

import numpy as np

from arc3.llm import ScriptedLLM, Truncated
from arc3.types import Action, GameState, Observation
from arc3.v3.facts import FactBook
from arc3.v4.agent import V4Agent
from arc3.v4.expect import Expectation, check
from arc3.v4.store import AssumptionStore

# -- expect -----------------------------------------------------------------------


def _step(before_cells, after_cells, action=Action.simple(1)):
    """A Step from two frames given as {(x, y, w, h): colour}."""
    fb = FactBook()

    def frame(cells):
        f = np.zeros((64, 64), dtype=np.int8)
        for (x, y, w, h), c in cells.items():
            f[y:y + h, x:x + w] = c
        return f
    fb.observe(frame(before_cells))
    return fb.observe(frame(after_cells), action), fb


def _exp(items, rest="any", outcome="continue"):
    return Expectation.parse({"changes": items, "rest": rest, "outcome": outcome})


def _item(target, change, dx=None, dy=None, colour=None):
    return {"target": target, "change": change, "dx": dx, "dy": dy, "colour": colour}


def test_move_prediction_holds_and_fails_on_the_wrong_displacement():
    step, _ = _step({(4, 4, 4, 4): 3, (40, 40, 4, 4): 2}, {(8, 4, 4, 4): 3, (40, 40, 4, 4): 2})
    assert check(_exp([_item("colour 3", "moved", 4, 0)], "unchanged"), step, "continue").ok
    v = check(_exp([_item("colour 3", "moved", 0, -4)]), step, "continue")
    assert not v.ok and "moved (+4,+0)" in v.describe()


def test_rest_unchanged_catches_an_unpredicted_change():
    step, _ = _step({(4, 4, 4, 4): 3, (40, 40, 4, 4): 2}, {(8, 4, 4, 4): 3, (40, 40, 4, 4): 9})
    v = check(_exp([_item("colour 3", "moved", 4, 0)], "unchanged"), step, "continue")
    assert not v.ok and v.unpredicted and "recoloured 2->9" in v.unpredicted[0]
    assert check(_exp([_item("colour 3", "moved", 4, 0)], "any"), step, "continue").ok


def test_none_and_outcome_items():
    step, _ = _step({(4, 4, 4, 4): 3}, {(4, 4, 4, 4): 3})
    assert check(_exp([_item("any", "none")], "unchanged"), step, "continue").ok
    assert not check(_exp([], "any", "level_complete"), step, "continue").ok
    assert not _exp([], "any").informative


def test_object_id_targets():
    step, fb = _step({(4, 4, 4, 4): 3}, {(4, 8, 4, 4): 3})
    oid = next(iter(fb.objects))
    assert check(_exp([_item(f"#{oid}", "moved", 0, 4)], "unchanged"), step, "continue").ok
    assert not check(_exp([_item(f"#{oid}", "none")]), step, "continue").ok


# -- store -------------------------------------------------------------------------


def test_store_applies_edits_and_skips_bad_ones():
    s = AssumptionStore()
    done = s.apply([
        {"op": "add", "statement": "colour 3 is the player", "kind": "ontology", "confidence": 0.7},
        {"op": "add", "statement": "reach colour 2", "kind": "goal", "scope": "game"},
        {"op": "revise", "id": "A1", "confidence": 0.9},
        {"op": "retire", "id": "A2", "note": "touched it, nothing happened"},
        {"op": "support", "id": "A99"},
        "garbage",
    ], "t0")
    assert s.items["A1"].confidence == 0.9 and s.items["A2"].status == "retired"
    assert any("unknown id" in d for d in done)
    assert "RETIRED" in s.describe() and "touched it" in s.describe()


def test_rules_are_checked_by_code_and_soften_never_delete():
    s = AssumptionStore()
    s.apply([{"op": "add", "statement": "ACTION1 moves colour 3 right by 4", "kind": "dynamics",
              "confidence": 0.8, "rule": {"action": "ACTION1",
                                          "expect": [_item("colour 3", "moved", 4, 0)]}}], "t0")
    a = s.items["A1"]
    ok, _ = _step({(4, 4, 4, 4): 3}, {(8, 4, 4, 4): 3})
    assert s.observe(ok, "continue") == [] and a.support == 1 and a.confidence > 0.8
    blocked, _ = _step({(4, 4, 4, 4): 3}, {(4, 4, 4, 4): 3})
    for _ in range(4):
        notes = s.observe(blocked, "continue")
    assert notes and a.contra == 4 and a.status == "doubtful"   # softened, still there
    other, _ = _step({(4, 4, 4, 4): 3}, {(4, 4, 4, 4): 3}, Action.simple(2))
    s.observe(other, "continue")
    assert a.contra == 4                                          # rule is for ACTION1 only


def test_new_level_marks_everything_carried():
    s = AssumptionStore()
    s.apply([{"op": "add", "statement": "x", "scope": "level", "confidence": 1.0},
             {"op": "add", "statement": "y", "scope": "game", "confidence": 1.0}], "t0")
    s.new_level(1)
    assert all(a.carried for a in s.live())
    assert s.items["A1"].confidence < s.items["A2"].confidence < 1.0
    assert "CARRIED" in s.describe()


# -- the controller ------------------------------------------------------------------


class WalledRoom:
    """8x8 cells of 4px; wall column at cell x=4 with a gap at y=6; goal at (7,0).
    Two identical levels. Same room as test_v3_agent."""

    def __init__(self) -> None:
        self.levels = 0
        self.x, self.y = 1, 1

    def frame(self) -> np.ndarray:
        f = np.zeros((64, 64), dtype=np.int8)
        for cy in range(8):
            if cy != 6:
                f[4 * cy:4 * cy + 4, 16:20] = 5
        f[0:4, 28:32] = 2
        f[4 * self.y:4 * self.y + 4, 4 * self.x:4 * self.x + 4] = 3
        return f

    def obs(self) -> Observation:
        st = GameState.WIN if self.levels >= 2 else GameState.NOT_FINISHED
        return Observation(frames=(self.frame(),), state=st, levels_completed=self.levels,
                           win_levels=2, available_actions=(1, 2, 3, 4))

    def step(self, a: Action) -> Observation:
        if a.is_reset:
            self.x, self.y = 1, 1
            return self.obs()
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}.get(a.id, (0, 0))
        nx, ny = self.x + dx, self.y + dy
        if 0 <= nx < 8 and 0 <= ny < 8 and not (nx == 4 and ny != 6):
            self.x, self.y = nx, ny
        if (self.x, self.y) == (7, 0):
            self.levels += 1
            self.x, self.y = 1, 1
        return self.obs()


MOVES = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}


class Mind:
    """A scripted model that reads the room's state directly (this tests the
    controller's mechanics, not intelligence). Naive until it learns the wall."""

    def __init__(self, env: WalledRoom) -> None:
        self.env = env
        self.knows_wall = False
        self.updates: list[str] = []

    def route(self) -> list[int]:
        x, y = self.env.x, self.env.y
        if not self.knows_wall:
            return [4] * (7 - x)
        out = []
        while (x, y) != (7, 0):
            if x < 4 and y != 6:
                a = 2 if y < 6 else 1
            elif x < 7:
                a = 4
            else:
                a = 1
            dx, dy = MOVES[a]
            x, y = x + dx, y + dy
            out.append(a)
        return out

    def __call__(self, messages) -> str:
        user = messages[-1]["content"]
        text = user[0]["text"] if isinstance(user, list) else user
        if "TASK: UPDATE" in text:
            self.updates.append(text)
            ops = []
            if "A1 [" not in text:
                ops.append({"op": "add", "statement": "ACTION4 moves the colour-3 block right 4",
                            "kind": "dynamics", "scope": "game", "confidence": 0.6,
                            "rule": {"action": "ACTION4",
                                     "expect": [_item("colour 3", "moved", 4, 0)]}})
                ops.append({"op": "add", "statement": "reaching the colour-2 square wins",
                            "kind": "goal", "scope": "game", "confidence": 0.5})
            if "SURPRISE" in text and not self.knows_wall:
                self.knows_wall = True
                ops.append({"op": "add", "statement": "the colour-5 column blocks movement "
                            "except at row 24", "kind": "dynamics", "confidence": 0.8})
                ops.append({"op": "revise", "id": "A1", "statement": "ACTION4 moves the block "
                            "right 4 unless a wall is in the way", "rule": None})
            return json.dumps({"ops": ops, "summary": "ok"})
        steps = []
        x, y = self.env.x, self.env.y
        for a in self.route()[:6]:
            dx, dy = MOVES[a]
            x, y = x + dx, y + dy
            win = (x, y) == (7, 0)
            steps.append({"action": f"ACTION{a}", "expect": {
                "changes": [] if win else [_item("colour 3", "moved", 4 * dx, 4 * dy)],
                "rest": "any" if win else "unchanged",
                "outcome": "level_complete" if win else "continue"}})
        return json.dumps({"steps": steps, "purpose": "go to the colour-2 square",
                           "tests": ["A1", "A2"]})


def run(agent, env, cap=300):
    obs = env.obs()
    for i in range(cap):
        if obs.state == GameState.WIN:
            return i
        obs = env.step(agent.act(obs))
    return cap


def test_v4_wins_both_levels_through_a_surprise():
    env = WalledRoom()
    mind = Mind(env)
    agent = V4Agent(llm=ScriptedLLM(mind))
    n = run(agent, env)
    assert env.levels == 2, agent.describe()
    st = agent.stats
    assert st.get("predictions_wrong", 0) == 1           # the wall, once
    assert st.get("predictions_ok", 0) >= 20
    assert st.get("explore_steps", 0) == 0                # every action was the model's
    # Updates: game start, the surprise, the new level. Not one per action.
    assert st["calls_update"] <= 6 < st["calls_act"] + st["calls_update"]
    assert n <= 2 + 4 * 1 + 17 + 17, n                    # 2 wasted moves into the wall region
    assert agent.store.items["A1"].rule is None and "unless" in agent.store.items["A1"].statement
    assert agent.store.items["A3"].carried is False or agent.level_notes
    assert any("NEW LEVEL 2" in u for u in mind.updates)
    assert "level 1: won" in agent.level_notes[0].lower() and "you predicted it" in agent.level_notes[0]


def test_v4_sends_the_frame_as_an_image_and_the_reasoning_to_update():
    env = WalledRoom()
    seen = []

    class WithReasoning(ScriptedLLM):
        last: dict = {}

        def complete(self, messages, **kw):
            seen.append(messages)
            self.last = {"reasoning": "I suspect the grey column is a wall."}
            return super().complete(messages, **kw)
    agent = V4Agent(llm=WithReasoning(Mind(env)))
    run(agent, env, cap=12)
    first_user = seen[0][-1]["content"]
    assert first_user[1]["type"] == "image_url"
    assert first_user[1]["image_url"]["url"].startswith("data:image/png;base64,")
    later_updates = [m[-1]["content"][0]["text"] for m in seen[1:]
                     if "TASK: UPDATE" in m[-1]["content"][0]["text"]]
    assert later_updates and "grey column is a wall" in later_updates[0]


def test_v4_retries_a_truncated_answer_with_more_room():
    env = WalledRoom()
    mind = Mind(env)
    budgets = []

    class CutOnce(ScriptedLLM):
        def complete(self, messages, *, max_tokens=512, **kw):
            budgets.append(max_tokens)
            if len(budgets) == 1:
                raise Truncated("cut")
            return super().complete(messages, max_tokens=max_tokens, **kw)
    agent = V4Agent(llm=CutOnce(mind), update_tokens=1000)
    run(agent, env, cap=3)
    assert budgets[:2] == [1000, 2000] and agent.stats["truncated"] == 1
    assert agent.calls[0]["ok"]


def test_v4_without_llm_still_plays():
    env = WalledRoom()
    agent = V4Agent(llm=None)
    assert run(agent, env, cap=150) <= 150
    assert agent.stats.get("explore_steps", 0) > 0


def test_an_add_with_the_claim_in_the_note_is_kept():
    s = AssumptionStore()
    s.apply([{"op": "add", "id": "grey_is_", "note": "grey #1 is a wall", "kind": "ontology"}], "t0")
    assert s.items["A1"].statement == "grey #1 is a wall"


def test_an_unmentioned_side_effect_does_not_drop_the_sequence():
    """ls20, 2026-10-19: every move also shrinks a bar the model left out."""
    from arc3.v4.agent import V4Agent as A
    agent = A(llm=None)
    step, fb = _step({(4, 4, 4, 4): 3, (40, 60, 20, 2): 11}, {(8, 4, 4, 4): 3, (41, 60, 19, 2): 11})
    agent.fb = fb
    agent.queue = [("next", None, [])]
    agent._pending = (_exp([_item("colour 3", "moved", 4, 0)], "unchanged"), ["A1"])
    agent._judge(step, "continue")
    assert agent.queue and agent.stats["predictions_partial"] == 1 and agent.need_update
    agent._pending = (_exp([_item("colour 3", "moved", 0, 4)], "unchanged"), ["A1"])
    agent._judge(step, "continue")
    assert not agent.queue and agent.stats["predictions_wrong"] == 1
