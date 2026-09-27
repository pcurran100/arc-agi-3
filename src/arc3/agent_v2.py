"""v2: GraphAgent steered by LLM-proposed goal hypotheses (D-035).

`GraphAgent` (D-034) knows how to act, remember and navigate, and has no idea
where to go. This subclass supplies the missing part: a population of goal
hypotheses (`hypothesis.Population`) proposed as code by an LLM, which gives
every known state a value, and the value steers the frontier choice — a
best-first search over the exact state graph, guided by the hypotheses.

Where the pieces meet:

- **Proposing.** The LLM is asked for hypotheses (`propose.goal_messages`) at
  three moments: ``WARMUP`` actions into a level, so there are observed effects
  to report; every ``STUCK_EVERY`` actions without a win; and never more than
  ``MAX_LEVEL_CALLS`` per level or ``MAX_GAME_CALLS`` per game.
- **Testing.** Every visited state is shown to the population
  (`observe_state`: states a hypothesis calls "reached" that do not end the
  level count against it); a win scores every hypothesis on the state the
  winning action was taken from (`observe_win`).
- **Deciding.** `_node_value` is the population's weighted progress. It lowers
  the frontier cost of promising states, and `_prefer_frontier` leaves the
  current state for a known state whose value is clearly higher, even if
  untried actions remain here.
- **Transfer.** Hypotheses that gained weight on a win are carried to the next
  level at the prior weight, since levels of one game tend to share a goal. The
  rest are dropped with the level.

With no LLM, or before the first proposal, every value is 0 and this is exactly
v1.

**What this does not do.**

- **The LLM never picks actions** and never sees anything but the scene text.
- **No dynamics hypotheses.** Effects are reported to the LLM as evidence, but
  it is not asked to predict them; the state graph is still the only dynamics
  model in use.
- **Values are for visited states only.** A hypothesis cannot value a state the
  agent has not reached, because nothing predicts unseen frames reliably yet
  (the induced model hurt, D-034). The search is best-first over what exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .agent import GraphAgent
from .hypothesis import PRIOR_WEIGHT, Hypothesis, Population
from .llm import LLM
from .propose import PROMPT_VERSION, Evidence, describe_effect, goal_messages
from .scene import Scene, make_scene
from .types import CLICK, RESET, Action, Observation

WARMUP = 10
STUCK_EVERY = 150
MAX_LEVEL_CALLS = 4
MAX_GAME_CALLS = 16


MAX_TOKENS = 900

#: Weight ceiling for a hypothesis carried into the next level.
CARRY_CAP = 4.0


@dataclass
class HypothesisAgent(GraphAgent):
    llm: Optional[LLM] = None
    #: Value units worth one step of walking (the frontier trades them off).
    goal_weight: float = 100.0
    #: v2 needs the band confined to edge rows. Unconfined, a player moving
    #: sideways within the same rows gets those rows masked as a "counter"
    #: (measured on the synthetic room in tests/test_agent_v2.py, and the same
    #: failure as tr87's 489 re-keys, D-034) — and then the hypotheses cannot
    #: see the player at all. Neutral on v1's score on ten games.
    edge_band: float = 1.0
    pop: Population = field(default_factory=Population, init=False)
    _scenes: dict = field(default_factory=dict, init=False)
    _scene_sig: bytes = field(default=b"", init=False)
    _effects: list = field(default_factory=list, init=False)
    _prev: Optional[tuple[bytes, np.ndarray, Action]] = field(default=None, init=False)
    _level_actions: int = field(default=0, init=False)
    _level_calls: int = field(default=0, init=False)
    _since_call: int = field(default=0, init=False)
    _previous_win: Optional[str] = field(default=None, init=False)
    _available: tuple = field(default=(), init=False)
    calls: list = field(default_factory=list, init=False)  # (level, actions, n hyps, n compiled)

    # -- scenes ------------------------------------------------------------

    def _scene(self, key: bytes) -> Scene:
        # Scenes hide the band, so a band change makes every cached scene stale.
        if self._scene_sig != self._band_sig:
            self._scenes, self._scene_sig = {}, self._band_sig
        s = self._scenes.get(key)
        if s is None:
            node = self._nodes.get(key)
            frame = node.frame if node is not None else self._prev[1]
            s = make_scene(frame, self._mask())
            self._scenes[key] = s
        return s

    # -- hooks -------------------------------------------------------------

    def _node_value(self, key: bytes) -> float:
        v = self.pop.value(key, lambda: self._scene(key))
        return 0.0 if v is None else v

    def _prefer_frontier(self, here: bytes) -> bool:
        """Best-first: leave if a known state beats this one by more than the walk.

        A fixed value margin was tried first (0.15) and failed on the synthetic
        room: one step backwards from a 0.969 state cost 0.06, under the margin,
        so the agent never returned and drifted away from a goal one move off.
        Trading value against path length (``goal_weight`` value units per
        step) is what best-first search needs.
        """
        if not self.pop.live():
            return False
        here_v = self._node_value(here)
        dist = self._distances(here)
        for k, n in self._nodes.items():
            if k == here or n.dead or not n.untried() or k not in dist:
                continue
            if self.goal_weight * (self._node_value(k) - here_v) > dist[k]:
                self._bump("goal_detours")
                return True
        return False

    def _distances(self, src: bytes) -> dict[bytes, int]:
        """Steps from ``src`` over known edges, with RESET as a one-step jump to the start."""
        from collections import deque
        dist = {src: 0}
        q = deque([src])
        if self._start is not None and self._start not in dist:
            dist[self._start] = 1
            q.append(self._start)
        while q:
            u = q.popleft()
            n = self._nodes.get(u)
            if n is None or n.dead:
                continue
            for v in n.edges.values():
                if v not in dist and v in self._nodes:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def _on_state(self, key: bytes, frame: np.ndarray, obs: Observation) -> None:
        self._available = obs.available_actions
        self._level_actions += 1
        self._since_call += 1
        if self.llm is None:
            # Scenes and effect lines are prompt material; without a model they
            # are pure cost, and the no-LLM fallback should run at v1's speed.
            return
        if self._prev is not None and self._prev[2] is not None and not self._prev[2].is_reset:
            pkey = self._prev[0]
            if pkey in self._nodes:
                line = describe_effect(self._scene(pkey), self._scene(key), str(self._prev[2]))
                if not self._effects or self._effects[-1] != line:
                    self._effects.append(line)
                    self._effects = self._effects[-40:]
        self.pop.observe_state(key, lambda: self._scene(key))
        self._maybe_propose(key)

    def _on_level_won(self, key: bytes, frame: np.ndarray, action: Action) -> None:
        scene = make_scene(frame, self._mask())
        self.pop.observe_win(key, lambda: scene)
        self._previous_win = (f"{action} was taken from a state with "
                              + scene.describe(with_grid=False).replace("\n", "; "))
        keep = [h for h in self.pop.hypotheses if h.fn is not None and h.weight > PRIOR_WEIGHT]
        self.pop = Population()
        for h in keep:
            # Carried at the weight it earned (capped), not the prior: entering
            # level 2 at weight 1 beside a fresh wrong guess at weight 1 left
            # the two pulling in opposite directions (synthetic room, measured).
            self.pop.hypotheses.append(Hypothesis(
                id=self.pop._next, goal=h.goal, source=h.source, fn=h.fn,
                origin=f"carried from level {self._level + 1}",
                weight=min(h.weight, CARRY_CAP), wins_confirmed=h.wins_confirmed))
            self.pop._next += 1
        for _ in keep:
            self._bump("carried_hypotheses")
        self._scenes, self._effects = {}, []
        self._level_actions = self._level_calls = self._since_call = 0

    # -- proposing -----------------------------------------------------------

    def _maybe_propose(self, key: bytes) -> None:
        if self.llm is None or len(self.calls) >= MAX_GAME_CALLS:
            return
        if self._level_calls >= MAX_LEVEL_CALLS:
            return
        first = self._level_calls == 0 and self._level_actions >= WARMUP
        stuck = self._level_calls > 0 and self._since_call >= STUCK_EVERY
        if not (first or stuck):
            return
        actions = [f"ACTION{a}" for a in sorted(self._available) if a not in (RESET, CLICK)]
        if CLICK in self._available:
            actions.append("CLICK(x, y) on any cell")
        ranked = sorted(self.pop.hypotheses, key=lambda h: -h.weight)
        ev = Evidence(scene=self._scene(key), actions=actions, effects=list(self._effects),
                      level=self._level, actions_used=self._level_actions,
                      previous_win=self._previous_win, hypotheses=ranked)
        try:
            text = self.llm.complete(goal_messages(ev), max_tokens=MAX_TOKENS,
                                     temperature=0.0, seed=self.seed + len(self.calls))
        except Exception:
            self._bump("llm_errors")
            text = ""
        added = self.pop.add_from_text(
            text, origin=f"{PROMPT_VERSION} level {self._level + 1} call {self._level_calls + 1}")
        self.calls.append((self._level, self._level_actions, len(added),
                           sum(1 for h in added if h.fn is not None)))
        self._level_calls += 1
        self._since_call = 0
        self._bump("llm_calls")

    # -- bookkeeping -----------------------------------------------------------

    def act(self, obs: Observation) -> Action:
        a = super().act(obs)
        self._prev = (self._last[0], self._last[1], self._last[2]) if self._last else None
        return a

    def describe(self) -> str:
        n = len(self.pop.hypotheses)
        live = len(self.pop.live())
        return f"{super().describe()}, hyps={n} live={live}"
