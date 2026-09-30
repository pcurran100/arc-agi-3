"""The v3 controller on a synthetic two-level game with a scripted LLM.

The scripted "model" behaves as a good LLM should: its first world model is
wrong about walls, the planned path hits a wall, the surprise is reported, and
its second assessment is right. This exercises probe -> assess -> verify ->
plan -> surprise -> re-assess -> plan -> win -> carry the model to level 2.
"""

from __future__ import annotations

import json

import numpy as np

from arc3.agent import GraphAgent
from arc3.llm import ScriptedLLM
from arc3.types import Action, GameState, Observation
from arc3.v3.agent import V3Agent

MOVES = '''
MOVES = {"ACTION1": (0, -4), "ACTION2": (0, 4), "ACTION3": (-4, 0), "ACTION4": (4, 0)}
def predict(state, action):
    if action not in MOVES:
        return state
    b = state.of_colour(3)[0]
    dx, dy = MOVES[action]
    if state.free(b["x"] + dx, b["y"] + dy, 4, 4, passable=PASS, ignore=[b["id"]]):
        state.move(b["id"], dx, dy)
    return state
'''
GOAL = '''
def goal(state):
    b, g = state.of_colour(3), state.of_colour(2)
    if not b: return 0.0
    if not g: return 1.0
    d = abs(b[0]["x"] - g[0]["x"]) + abs(b[0]["y"] - g[0]["y"])
    return 1.0 if d == 0 else 1.0 - d / 128
'''
RIGHT = MOVES.replace("PASS", "{0, 2}")
NO_WALLS = MOVES.replace("PASS", "{0, 2, 5}")


def answer(code, kind="plan"):
    return json.dumps({
        "thoughts": "the colour-3 block is the player; ACTION1-4 move it 4 cells",
        "hypotheses": [{"kind": "player", "claim": "colour 3 block", "confidence": 0.9},
                       {"kind": "goal", "claim": "reach the colour-2 square", "confidence": 0.7}],
        "ignore_ids": [], "predict_code": code, "goal_code": GOAL,
        "next": {"kind": kind, "actions": [], "purpose": "go to the goal"}})


class WalledRoom:
    """8x8 cells; wall column at cell x=4 with a gap at y=6; goal at cell (7,0).
    Two identical levels."""

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
        wall = nx == 4 and ny != 6
        if 0 <= nx < 8 and 0 <= ny < 8 and not wall:
            self.x, self.y = nx, ny
        if (self.x, self.y) == (7, 0):
            self.levels += 1
            self.x, self.y = 1, 1
        return self.obs()


def run(agent, cap=400):
    env = WalledRoom()
    obs = env.obs()
    first = None
    for i in range(cap):
        if obs.state == GameState.WIN:
            return first, i
        obs = env.step(agent.act(obs))
        if first is None and env.levels >= 1:
            first = i + 1
    return first, cap


def scripted():
    answers = iter([answer(NO_WALLS), answer(RIGHT)])
    return ScriptedLLM(lambda messages: next(answers, ""))


def test_v3_wins_both_levels_through_a_surprise():
    agent = V3Agent(llm=scripted())
    first, total = run(agent)
    assert first is not None and total < 400, agent.describe()
    assert agent.stats.get("surprises", 0) >= 1            # the wall contradicted NO_WALLS
    assert agent.stats.get("carried_plans", 0) >= 1         # level 2 planned with level 1's model
    kinds = [c["kind"] for c in agent.calls]
    assert kinds.count("assess") >= 2


def test_v3_is_far_more_efficient_than_blind_v1():
    v3 = run(V3Agent(llm=scripted()))
    v1 = run(GraphAgent(seed=0, edge_band=1.0))
    assert v3[1] < v1[1], (v3, v1)
    # 4 probe steps + a wall surprise + the 17-move optimal path, then 17 for level 2.
    assert v3[1] <= 70, v3


def test_v3_without_llm_still_plays():
    first, total = run(V3Agent(llm=None), cap=200)
    assert total == 200 or first is not None   # explores without crashing
