"""GraphAgent on a synthetic deterministic game: no engine required."""

from __future__ import annotations

import numpy as np

from arc3.agent import GraphAgent, state_key
from arc3.types import Action, GameState, Observation


class Corridor:
    """A 1-D corridor of ``n`` cells drawn as a block. ACTION1 = left, ACTION2 =
    right, ACTION3 does nothing. Reaching the right end clears the level; a
    move counter in row 63 ticks every action, which the agent must see through.
    Stepping left from cell 0 is death (GAME_OVER), cleared only by RESET.
    """

    def __init__(self, n: int = 10) -> None:
        self.n, self.pos, self.ticks, self.levels, self.dead = n, 0, 0, 0, False

    def frame(self) -> np.ndarray:
        f = np.zeros((64, 64), dtype=np.int8)
        f[10:14, 4 + 5 * self.pos : 8 + 5 * self.pos] = 3
        f[63, : self.ticks % 64] = 5
        return f

    def obs(self, action=None) -> Observation:
        state = GameState.GAME_OVER if self.dead else GameState.NOT_FINISHED
        if self.levels >= 1:
            state = GameState.WIN
        return Observation(frames=(self.frame(),), state=state,
                           levels_completed=self.levels, win_levels=1,
                           available_actions=(1, 2, 3), action=action)

    def step(self, a: Action) -> Observation:
        self.ticks += 1
        if a.is_reset:
            self.pos, self.dead = 0, False
        elif not self.dead:
            if a.id == 1:
                if self.pos == 0:
                    self.dead = True
                else:
                    self.pos -= 1
            elif a.id == 2:
                self.pos += 1
                if self.pos == self.n - 1:
                    self.levels = 1
        return self.obs(a)


def run(game: Corridor, agent: GraphAgent, cap: int) -> int:
    obs = game.obs()
    for i in range(cap):
        if obs.state == GameState.WIN:
            return i
        obs = game.step(agent.act(obs))
    return cap


def test_clears_corridor_despite_ticking_counter():
    game = Corridor(10)
    used = run(game, GraphAgent(seed=0, use_model=False), 200)
    assert game.levels == 1
    # Nine moves right are needed; exploration overhead should stay modest.
    assert used < 60


def test_resets_after_game_over_and_keeps_going():
    game = Corridor(6)
    agent = GraphAgent(seed=1, use_model=False)
    run(game, agent, 200)
    assert game.levels == 1


def test_model_path_runs():
    game = Corridor(8)
    run(game, GraphAgent(seed=0, use_model=True), 200)
    assert game.levels == 1


def test_state_key_ignores_band_rows():
    a = np.zeros((64, 64), dtype=np.int8)
    b = a.copy()
    b[63, :5] = 5
    band = np.zeros((64, 64), dtype=bool)
    assert state_key(a, band) != state_key(b, band)
    band[63, :] = True
    assert state_key(a, band) == state_key(b, band)


def test_empty_observation_is_a_noop_not_a_reset():
    """Zero frames means the action changed nothing; it must not trigger RESET."""
    game = Corridor(6)
    agent = GraphAgent(seed=0, use_model=False)
    first = agent.act(game.obs())
    empty = Observation(frames=(), state=GameState.NOT_FINISHED, levels_completed=0,
                        win_levels=1, available_actions=(1, 2, 3), action=first)
    assert not agent.act(empty).is_reset
