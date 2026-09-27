"""v2 on a synthetic 2-D game: a right hypothesis must beat blind exploration."""

from __future__ import annotations

import numpy as np

from arc3.agent import GraphAgent
from arc3.agent_v2 import HypothesisAgent
from arc3.llm import ScriptedLLM
from arc3.types import Action, GameState, Observation

RIGHT = '''```python
# GOAL: the colour-3 block reaches the colour-2 square
def progress(scene):
    p = scene.of_colour(3)
    g = scene.of_colour(2)
    if not p or not g:
        return 1.0 if p and not g else 0.0
    d = abs(p[0].cx - g[0].cx) + abs(p[0].cy - g[0].cy)
    return 1.0 - d / 128
```'''

WRONG = '''```python
# GOAL: the block goes to the top-left corner
def progress(scene):
    p = scene.of_colour(3)
    return 1.0 - (p[0].cx + p[0].cy) / 128 if p else 0.0
```'''


class Room:
    """A 12x12 room of 4-pixel cells. ACTION1-4 move the colour-3 block; the
    level is won when it reaches the colour-2 goal square in the far corner.
    Two levels, same goal, so carried hypotheses can be tested too."""

    N = 12

    def __init__(self) -> None:
        self.levels = 0
        self.reset()

    def reset(self) -> None:
        self.x = self.y = 5

    def frame(self) -> np.ndarray:
        f = np.zeros((64, 64), dtype=np.int8)
        gx = gy = self.N - 1
        f[4 * gy : 4 * gy + 4, 4 * gx : 4 * gx + 4] = 2
        f[4 * self.y : 4 * self.y + 4, 4 * self.x : 4 * self.x + 4] = 3
        return f

    def obs(self, action=None) -> Observation:
        state = GameState.WIN if self.levels >= 2 else GameState.NOT_FINISHED
        return Observation(frames=(self.frame(),), state=state, levels_completed=self.levels,
                           win_levels=2, available_actions=(1, 2, 3, 4), action=action)

    def step(self, a: Action) -> Observation:
        if a.is_reset:
            self.reset()
            return self.obs(a)
        dx, dy = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[a.id]
        self.x = min(self.N - 1, max(0, self.x + dx))
        self.y = min(self.N - 1, max(0, self.y + dy))
        if (self.x, self.y) == (self.N - 1, self.N - 1):
            self.levels += 1
            self.reset()
        return self.obs(a)


def play(agent, cap=600):
    game = Room()
    obs = game.obs()
    first = None
    for i in range(cap):
        if obs.state == GameState.WIN:
            return first, i
        obs = game.step(agent.act(obs))
        if first is None and game.levels >= 1:
            first = i + 1
    return first, cap


def test_no_llm_is_exactly_v1():
    # v2 confines the band to edge rows; with that one setting matched it is v1.
    v1 = play(GraphAgent(seed=0, edge_band=1.0))
    v2 = play(HypothesisAgent(seed=0, llm=None))
    assert v1 == v2


def test_right_hypothesis_beats_blind_exploration():
    blind = play(GraphAgent(seed=0))
    guided_agent = HypothesisAgent(seed=0, llm=ScriptedLLM([RIGHT, WRONG, WRONG, WRONG]))
    guided = play(guided_agent)
    assert guided[1] < blind[1], (guided, blind)
    assert guided_agent.stats.get("carried_hypotheses", 0) >= 1


def test_wrong_hypothesis_is_down_weighted_by_the_win():
    agent = HypothesisAgent(seed=0, llm=ScriptedLLM([RIGHT + "\n" + WRONG]))
    game = Room()
    obs = game.obs()
    while game.levels < 1:
        obs = game.step(agent.act(obs))
    agent.act(obs)  # the agent sees the level change and scores the win
    # After the win the population is rebuilt from carried hypotheses only.
    goals = [h.goal for h in agent.pop.hypotheses]
    assert goals == ["the colour-3 block reaches the colour-2 square"]
