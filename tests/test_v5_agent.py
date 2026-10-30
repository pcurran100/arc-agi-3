"""The v5 controller (D-046): a simulator the model writes is validated and searched
by code; small surprises do not stop a plan; call tiers switch thinking."""

from __future__ import annotations

import json

from arc3.llm import ScriptedLLM
from arc3.v5.agent import V5Agent, _thinking_backend
from test_v4 import WalledRoom, run

SIM = """
def predict(state, action):
    ps = state.of_colour(3)
    if not ps:
        return state
    p = ps[0]
    d = {"ACTION1": (0, -4), "ACTION2": (0, 4), "ACTION3": (-4, 0), "ACTION4": (4, 0)}.get(action)
    if d and state.free(p["x"] + d[0], p["y"] + d[1], 4, 4, passable=(0, 2), ignore=(p["id"],)):
        state.move(p["id"], d[0], d[1])
    return state

def goal(state):
    p, t = state.of_colour(3), state.of_colour(2)
    return bool(p and t and state.overlaps(p[0], t[0]))
"""

# Ignores walls: right four times, then up; the wall stops it at the first move.
NAIVE = SIM.replace('passable=(0, 2)', 'passable=(0, 2, 5)')


class SimMind:
    """Writes a simulator at every THINK; the first one optionally naive."""

    def __init__(self, first=SIM):
        self.first = first
        self.thinks = 0
        self.texts: list[str] = []

    def __call__(self, messages) -> str:
        text = messages[-1]["content"]
        self.texts.append(text)
        assert text.endswith("TASK: THINK"), "this mind only expects think calls"
        self.thinks += 1
        code = self.first if self.thinks == 1 else SIM
        return json.dumps({
            "ops": [{"op": "add", "id": "", "statement": "reaching colour 2 wins",
                     "kind": "goal", "scope": "game", "confidence": 0.6, "note": "",
                     "attain": None}] if self.thinks == 1 else [],
            "names": [], "steps": [], "plan": [], "purpose": "search my simulator",
            "simulator": {"code": code, "tracked": [], "goal": "A1"}})


def test_v5_wins_both_levels_by_searching_its_simulator():
    env = WalledRoom()
    mind = SimMind()
    agent = V5Agent(llm=ScriptedLLM(mind))
    n = run(agent, env)
    assert env.levels == 2, agent.describe()
    assert agent.stats.get("explore_steps", 0) == 0
    assert mind.thinks == 2, agent.describe()            # game start + new level, no more
    assert agent.stats["probe_steps"] == 4               # each action once, game start only
    assert n <= 4 + 2 * 17 + 3, n                        # probes, then ~the shortest path twice
    assert agent.stats["sim_plan_steps"] >= 34
    assert "NEW LEVEL 2" in mind.texts[1]


def test_a_wrong_simulator_is_caught_and_repaired():
    env = WalledRoom()
    mind = SimMind(first=NAIVE)
    agent = V5Agent(llm=ScriptedLLM(mind))
    run(agent, env)
    assert env.levels == 2, agent.describe()
    assert agent.stats.get("sim_wrong", 0) >= 1
    assert any("SIMULATOR WRONG" in t for t in mind.texts[1:])
    assert agent.stats.get("explore_steps", 0) == 0


class PlanMind:
    """Names the player only as 'player' and asks code to route to the goal cell;
    it never mentions the wall, which code must find and route around."""

    def __init__(self):
        self.calls = 0

    def __call__(self, messages) -> str:
        self.calls += 1
        return json.dumps({
            "ops": [], "names": [], "simulator": None, "steps": [], "purpose": "go",
            "plan": [{"object": "player", "to": "29,1", "relation": "overlap",
                      "on_arrival": {"changes": [], "rest": "any", "outcome": "level_complete"},
                      "goal": ""}]})


def test_a_plan_naming_the_player_is_routed_and_rerouted_around_a_wall():
    env = WalledRoom()
    mind = PlanMind()
    agent = V5Agent(llm=ScriptedLLM(mind))
    run(agent, env, cap=150)
    assert env.levels == 2, agent.describe()
    assert agent.stats.get("reroutes", 0) >= 1          # the wall, found by bumping into it
    assert agent.stats.get("explore_steps", 0) == 0
    assert mind.calls <= 4, agent.describe()            # start + new level (+ slack)


def test_thinking_is_switched_per_call_and_restored():
    class FakeBackend:
        base_url = "http://127.0.0.1:1"
        name = "fake"

        def __init__(self):
            self.extra = {"chat_template_kwargs": {"enable_thinking": True,
                                                   "reasoning_budget": 12000}, "top_p": 0.9}
            self.seen = []
            self.last = {}

        def complete(self, messages, **kw):
            self.seen.append(self.extra["chat_template_kwargs"]["enable_thinking"])
            return json.dumps({"steps": [], "plan": [], "purpose": "x"})

    be = FakeBackend()
    assert _thinking_backend(be) is be
    agent = V5Agent(llm=be)
    agent._call("next", [], {}, 100, False)
    agent._call("think", [], {}, 100, True)
    assert be.seen == [False, True]
    # Other template options (a soft reasoning budget on NIM) survive the switch.
    assert be.extra == {"chat_template_kwargs": {"enable_thinking": True,
                                                 "reasoning_budget": 12000}, "top_p": 0.9}
    assert [c["thinking"] for c in agent.calls] == [False, True]


def test_a_lost_network_halts_the_game_instead_of_exploring():
    from arc3.llm import NetworkDown

    def down(messages):
        raise NetworkDown("no connection for 600s")

    env = WalledRoom()
    agent = V5Agent(llm=ScriptedLLM(down))
    obs = env.obs()
    spent = 0
    for _ in range(40):
        a = agent.act(obs)
        if agent.halted:
            break
        obs = env.step(a)
        spent += 1
    assert agent.halted.startswith("network down")
    assert spent == 4                                    # the code probes only
    assert agent.stats.get("explore_steps", 0) == 0
    assert "INVALID" in agent.describe()
