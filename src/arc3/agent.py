"""The first end-to-end agent: act, observe, remember, learn, act again (D-034).

Everything before this module was a measurement instrument. This one plays.

**What it does, in one paragraph.** It builds the game's state graph as it goes.
A state is the resting frame with the bookkeeping band masked out, so a move
counter ticking does not make every state unique. From the current state it
takes an action it has not yet tried there; when there are none left worth
trying, it walks the known graph to the nearest state that still has some, or
presses `RESET` when the level start is the shorter route there. Untried actions
are ranked by two things: the explorer's Thompson estimate of how often that
*kind* of action moves the playfield (D-031), and the induced dynamics model's
one-step prediction (D-025, D-033) — an action the model predicts leads to an
already-visited state is pushed down, and one it predicts does nothing is pushed
further.

**Why a graph.** The engine is byte-deterministic (D-009), so every observed
transition is an exact, permanent fact about the game. On visited states the
graph *is* a perfect dynamics model; the induced rules exist to generalise to
states the graph has not reached. The graph is the model-based thesis in its
most literal form, and it is the part that can plan today: navigating to a
frontier is shortest-path search against a model, done entirely internally.

**What this does not do.**

- **No goal.** It explores the state space; it does not know what winning looks
  like. Levels are cleared when systematic exploration stumbles onto the win
  state. The goal subsystem of the design thesis is not built, and this agent is
  the baseline it must beat.
- **No multi-step use of the rules.** The dynamics model ranks one step ahead.
  Rollouts through predicted frames are not attempted, because multi-step
  prediction accuracy has never been measured (state-of-play, objective 6).
- **No transfer across levels** beyond the explorer's arm statistics and the
  rule corpus. The state graph is discarded on each level change, since states
  from level *n* never recur in level *n+1*.
- **No hidden state.** If the same masked frame and action lead to different
  outcomes, the latest observation overwrites the edge. A game whose response
  depends on history will look to this agent like a graph that keeps changing.
- **Bookkeeping by rows only.** The band (`explore.OnlineBand`) is a set of rows.
  A counter drawn in a column, or inside the playfield, still splits states.
"""

from __future__ import annotations

import hashlib
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .dynamics.induce import TransitionObservation, induce, observe_transition
from .dynamics.rules import (
    STATIC,
    VANISH,
    RuleSet,
    apply_effect,
    describe_object,
    reveal_colour,
)
from .explore import DECAY, OnlineBand, _Arm
from .perception.segment import Segmentation, background_colour, segment
from .types import CLICK, RESET, Action, GameState, Observation, Transition

#: Carving the dynamics model is induced and evaluated under. D-025's best mean.
MODEL_CARVING = "colour_4"

#: Carving click targets are drawn from. The explorer's measured choice (D-031).
CLICK_CARVING = "colour_8"

#: Re-induce the rule set after this many new transitions.
INDUCE_EVERY = 10

#: Cap on transitions fed to induction. D-025 found the model saturates at ~50.
INDUCE_WINDOW = 150

#: Cap on click candidates per state, keeping the largest distinct targets.
#: bp35 proposes 196 targets for 8 true outcomes (D-021); beyond this the arms
#: decide which colours are worth the rest.
MAX_CLICKS = 64

#: Ranking penalties from the model's one-step prediction.
PENALTY_KNOWN = 0.35   # predicted to reach a state already in the graph
PENALTY_NOOP = 0.6     # predicted to change nothing at all

#: Rows from the top or bottom edge a bookkeeping band may occupy (edge_band).
EDGE_ROWS = 4

#: How many of the arm-ranked untried actions the model is consulted on.
MODEL_TOP_K = 8

#: Extra path cost per unit of missing priority when choosing a frontier.
FRONTIER_WEIGHT = 4.0


def state_key(frame: np.ndarray, band: np.ndarray) -> bytes:
    """Hash of the frame with bookkeeping rows blanked."""
    f = frame.copy() if band.any() else frame
    if band.any():
        f[band] = 0
    return hashlib.blake2b(np.ascontiguousarray(f).tobytes(), digest_size=12).digest()


@dataclass
class _Node:
    frame: np.ndarray
    candidates: list[tuple[Any, Action]]  # (arm key, action)
    tried: set[Action] = field(default_factory=set)
    edges: dict[Action, bytes] = field(default_factory=dict)
    dead: bool = False  # GAME_OVER state; only RESET leaves it

    def untried(self) -> list[tuple[Any, Action]]:
        return [(k, a) for k, a in self.candidates if a not in self.tried]


@dataclass
class GraphAgent:
    """Explores a level's state graph, ranked by arms and the induced model.

    Feed it observations with :meth:`act`; it returns the next action. It keeps
    no reference to any client, so the same object drives the local engine, a
    replay, or the Kaggle harness through an adapter.
    """

    seed: int = 0
    use_model: bool = False  # the model lowers RHAE 3.46% -> 1.18% (D-034)
    #: Leave a state whose untried actions all belong to arms with a mean below
    #: this, if a frontier elsewhere offers better. 0 disables.
    local_min: float = 0.3
    #: Restrict the bookkeeping band to the top and bottom EDGE_ROWS rows.
    edge_band: float = 0.0
    #: Priority bonus for repeating the arm of the last action, if it moved the
    #: playfield. Biases exploration towards straight lines. 0 disables.
    repeat: float = 0.0
    _moved_key: Any = field(default=None, init=False)
    _rng: random.Random = field(init=False)
    band: OnlineBand = field(default_factory=OnlineBand, init=False)
    _band_sig: bytes = field(default=b"", init=False)
    _nodes: dict[bytes, _Node] = field(default_factory=dict, init=False)
    _start: Optional[bytes] = field(default=None, init=False)
    _arms: dict[Any, _Arm] = field(default_factory=dict, init=False)
    _history: list[tuple[np.ndarray, Action, np.ndarray]] = field(default_factory=list, init=False)
    _obs_cache: list[TransitionObservation] = field(default_factory=list, init=False)
    _ruleset: Optional[RuleSet] = field(default=None, init=False)
    _since_induce: int = field(default=0, init=False)
    _last: Optional[tuple[bytes, np.ndarray, Action, Any]] = field(default=None, init=False)
    _level: int = field(default=0, init=False)
    _path: list[Action] = field(default_factory=list, init=False)
    stats: dict[str, int] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # -- bookkeeping ---------------------------------------------------------

    def _bump(self, k: str) -> None:
        self.stats[k] = self.stats.get(k, 0) + 1

    def _arm(self, key: Any) -> _Arm:
        return self._arms.setdefault(key, _Arm())

    def _rekey_if_band_changed(self) -> None:
        """Re-hash the graph when the bookkeeping band estimate moves.

        Keys depend on the band, so a band change silently orphans every node.
        Rebuilding from stored frames is cheap and keeps the graph coherent.
        """
        mask = self._mask()
        sig = np.packbits(mask).tobytes()
        if sig == self._band_sig:
            return
        self._band_sig = sig
        old = self._nodes
        remap = {k: state_key(n.frame, mask) for k, n in old.items()}
        self._nodes = {}
        for k, n in old.items():
            nk = remap[k]
            n.edges = {a: remap.get(t, t) for a, t in n.edges.items()}
            if nk in self._nodes:  # two states merged by the new mask
                m = self._nodes[nk]
                m.tried |= n.tried
                m.edges.update(n.edges)
                m.dead = m.dead and n.dead
            else:
                self._nodes[nk] = n
        if self._start is not None:
            self._start = remap.get(self._start, self._start)
        if self._last is not None:
            k, f, a, ak = self._last
            self._last = (remap.get(k, k), f, a, ak)
        self._path = []
        self._bump("rekeys")

    def _mask(self) -> np.ndarray:
        """The bookkeeping band, optionally restricted to the frame's edge rows.

        The row-frequency estimate cannot tell a counter from a playfield
        region that happens to change on about half of all actions. On tr87
        it flips rows 48-60 in and out of the band every step and the graph is
        re-keyed 489 times in 500 actions. Every counter measured so far sits
        at an edge (sp80 row 0, ls20 rows 61-62), where the UI layer is drawn.
        """
        m = self.band.mask()
        if self.edge_band:
            m = m.copy()
            m[EDGE_ROWS:-EDGE_ROWS, :] = False
        return m

    def _key(self, frame: np.ndarray) -> bytes:
        return state_key(frame, self._mask())

    def _node(self, key: bytes, frame: np.ndarray, available: tuple[int, ...]) -> _Node:
        n = self._nodes.get(key)
        if n is None:
            n = _Node(frame=frame, candidates=self._candidates(frame, available))
            self._nodes[key] = n
        return n

    def _candidates(self, frame: np.ndarray, available: tuple[int, ...]) -> list[tuple[Any, Action]]:
        out: list[tuple[Any, Action]] = []
        for a in sorted(a for a in available if a not in (RESET, CLICK)):
            out.append((("act", a), Action.simple(a)))
        if CLICK in available:
            seg = segment(frame, include=(CLICK_CARVING,))[CLICK_CARVING]
            bg = seg.background_colour
            comps = [c for c in seg.components if bg is None or c.dominant_colour != bg]
            comps.sort(key=lambda c: -c.size)
            seen: set[tuple[int, int]] = set()
            for c in comps[:MAX_CLICKS]:
                xy = c.click_target
                if xy in seen:
                    continue
                seen.add(xy)
                out.append((("click", c.dominant_colour), Action.click(*xy)))
        return out

    # -- learning ------------------------------------------------------------

    def _learn(self, obs: Observation) -> None:
        if self._last is None:
            return
        pkey, pframe, action, akey = self._last
        after = obs.last if obs.last is not None else pframe
        if action.is_reset or obs.levels_completed != self._level:
            # A restart or a level change is not the game's dynamics.
            return
        self.band.update(pframe, after)
        moved = self.band.moved_playfield(pframe, after)
        self._moved_key = akey if moved else None
        if akey is not None:
            self._arm(akey).update(moved)
            for arm in self._arms.values():
                arm.decay(DECAY)
        self._history.append((pframe, action, after))
        self._since_induce += 1
        self._bump("playfield_moves" if moved else "playfield_still")

    def _maybe_induce(self) -> None:
        if not self.use_model or self._since_induce < INDUCE_EVERY:
            return
        self._since_induce = 0
        recent = self._history[-INDUCE_WINDOW:]
        # Observations are cached by position; the window slides, so rebuild the
        # tail that is new and drop what fell out.
        need = len(recent)
        cached = self._obs_cache[-need:] if self._obs_cache else []
        missing = recent[len(cached):]
        for before, action, after in missing:
            t = Transition(before=before, action=action, frames=(after,),
                           levels_completed_before=0, levels_completed_after=0,
                           state_after=GameState.NOT_FINISHED)
            cached.append(observe_transition(t, MODEL_CARVING))
        self._obs_cache = cached[-INDUCE_WINDOW:]
        self._ruleset = induce(self._obs_cache)
        self._bump("inductions")

    def _predict(self, frame: np.ndarray, seg: Segmentation, action: Action) -> np.ndarray:
        """`predict_frame`, reusing one segmentation across candidate actions."""
        assert self._ruleset is not None
        bg = seg.background_colour if seg.background_colour is not None else background_colour(frame)
        canvas = frame.copy()
        effects = [(c, self._ruleset.effect_for(action, describe_object(c, action)))
                   for c in seg.components]
        for c, e in effects:
            if e.kind != STATIC:
                canvas[c.mask] = reveal_colour(frame, c.mask, bg)
        for c, e in effects:
            if e.kind not in (STATIC, VANISH):
                apply_effect(canvas, frame, c, e, bg, self._ruleset.shapes)
        return canvas

    # -- choosing ------------------------------------------------------------

    def _priority(self, key: bytes, node: _Node) -> list[tuple[float, Any, Action]]:
        """Untried actions at ``node``, each with a priority in roughly [0, 1]."""
        untried = node.untried()
        if not untried:
            return []
        seg = None
        if self.use_model and self._ruleset is not None:
            seg = segment(node.frame, include=(MODEL_CARVING,))[MODEL_CARVING]
        mask = self._mask()
        def base(k: Any) -> float:
            p = self._arm(k).sample(self._rng)
            if self.repeat and k is not None and k == self._moved_key:
                p += self.repeat
            return p

        scored = sorted(((base(k), k, a) for k, a in untried), key=lambda t: -t[0])
        out = []
        for i, (p, akey, action) in enumerate(scored):
            # Predicting is the expensive part; only the front of the queue can
            # change the choice, so only it is consulted.
            if seg is not None and i < MODEL_TOP_K:
                pred = self._predict(node.frame, seg, action)
                changed = bool(((pred != node.frame) & ~mask).any())
                if not changed:
                    p -= PENALTY_NOOP
                elif state_key(pred, mask) in self._nodes:
                    p -= PENALTY_KNOWN
            out.append((p, akey, action))
        out.sort(key=lambda t: -t[0])
        return out

    def _frontier_path(self, here: bytes) -> list[Action]:
        """Shortest known route to a state with worthwhile untried actions.

        Breadth-first over observed edges from ``here`` and, at a cost of one
        extra action, from the level start via `RESET`. Candidates are scored
        by path length plus a penalty for low best-arm mean, so a long walk is
        only taken for an action class that has been paying off.
        """
        def bfs(src: bytes) -> dict[bytes, list[Action]]:
            paths = {src: []}
            q = deque([src])
            while q:
                u = q.popleft()
                n = self._nodes.get(u)
                if n is None or n.dead:
                    continue
                for a, v in n.edges.items():
                    if v not in paths and v in self._nodes:
                        paths[v] = paths[u] + [a]
                        q.append(v)
            return paths

        options: dict[bytes, list[Action]] = bfs(here)
        if self._start is not None and self._start != here:
            for k, p in bfs(self._start).items():
                rp = [Action.simple(RESET)] + p
                if k not in options or len(rp) < len(options[k]):
                    options[k] = rp

        best: Optional[tuple[float, list[Action]]] = None
        for k, p in options.items():
            if not p:
                continue
            n = self._nodes[k]
            if n.dead:
                continue
            u = n.untried()
            if not u:
                continue
            q = max(self._arm(ak).mean for ak, _ in u)
            cost = len(p) + FRONTIER_WEIGHT * (1.0 - q)
            if best is None or cost < best[0]:
                best = (cost, p)
        return best[1] if best else []

    def act(self, obs: Observation) -> Action:
        """The next action. **Returning it is free; the caller spends it.**"""
        self._learn(obs)

        if obs.levels_completed > self._level:
            # New level: its states are all new. Keep arms and rules.
            self._level = obs.levels_completed
            self._nodes, self._start, self._path = {}, None, []
            self._bump("levels")

        frame = obs.last if obs.last is not None else (self._last[1] if self._last else None)
        if frame is None:
            self._last = None
            return Action.simple(RESET)

        self._rekey_if_band_changed()
        key = self._key(frame)
        node = self._node(key, frame, obs.available_actions)
        if self._start is None or obs.full_reset or (
            self._last is not None and self._last[2].is_reset
        ):
            self._start = key

        # Record the edge we just took.
        if self._last is not None:
            pkey, _, paction, _ = self._last
            prev = self._nodes.get(pkey)
            if prev is not None and not paction.is_reset:
                prev.edges[paction] = key
                prev.tried.add(paction)

        if obs.state == GameState.GAME_OVER:
            node.dead = True
            self._path = []
            return self._emit(key, frame, Action.simple(RESET), None)
        if obs.state == GameState.WIN:
            return self._emit(key, frame, Action.simple(RESET), None)

        self._maybe_induce()

        # Follow a planned route while it still matches the graph.
        if self._path:
            a = self._path.pop(0)
            if a.is_reset or a in node.edges or a in {c for _, c in node.candidates}:
                self._bump("nav_steps")
                return self._emit(key, frame, a, None)
            self._path = []

        ranked = self._priority(key, node)
        local_best = max((self._arm(k).mean for _, k, _ in ranked), default=0.0)
        if ranked and ranked[0][0] > -0.2 and local_best >= self.local_min:
            _, akey, a = ranked[0]
            self._bump("explore_steps")
            return self._emit(key, frame, a, akey)

        path = self._frontier_path(key)
        if path:
            self._path = path[1:]
            self._bump("nav_steps")
            return self._emit(key, frame, path[0], None)

        if ranked:
            _, akey, a = ranked[0]
            self._bump("explore_steps")
            return self._emit(key, frame, a, akey)

        # Graph exhausted: every reachable state fully tried. Retry at random
        # rather than stall, since hidden state may make a retry informative.
        self._bump("exhausted")
        akey, a = self._rng.choice(node.candidates) if node.candidates else (None, Action.simple(RESET))
        return self._emit(key, frame, a, akey)

    def _emit(self, key: bytes, frame: np.ndarray, action: Action, akey: Any) -> Action:
        if akey is None and not action.is_reset:
            node = self._nodes.get(key)
            if node is not None:
                akey = next((k for k, a in node.candidates if a == action), None)
        self._last = (key, frame, action, akey)
        return action

    def describe(self) -> str:
        return (f"{len(self._nodes)} states, level {self._level}, "
                + ", ".join(f"{k}={v}" for k, v in sorted(self.stats.items())))
