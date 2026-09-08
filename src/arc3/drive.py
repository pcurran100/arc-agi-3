"""Driving policies — how a rollout decides what to do next.

This is not an agent. It has no goal, no model and no memory of consequences.
Its only job is to generate a *corpus of transitions* for dynamics induction to
learn from, and the thing that matters about it is the **distribution** of
transitions it produces, not whether it wins anything.

Why this module exists at all (D-019, D-020): every perception number in the
repository was measured under uniformly random play, whose transitions turn out
to be small and low-diversity — 1–58 changed pixels of 4,096, heavily dominated
by recolouring. Inducing dynamics from that corpus would mean learning rules for
trivia. Random play is kept here as a named baseline so the improvement can be
measured rather than assumed.

**The action space is not uniform across games**, and the driver has to respect
that. Measured over all 25 public demo games at level 1:

- ``RESET`` (action 0) **never appears** in ``available_actions``. Cycling over
  the published mask therefore cannot accidentally restart a level, and nothing
  here needs to special-case it.
- **6 games publish only ``[6]``** — the click is the only input they accept, so
  the action axis is degenerate and *where you click is the entire policy*.
- **6 games never publish 6 at all** and are keyboard-only, so click targeting is
  irrelevant to them.
- 19 games publish the click at some point; the mask was stable within a game
  over 30 steps.

That split is why click targeting is a first-class axis here rather than a
detail. For a quarter of the corpus it is the only source of variation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np

from .perception.segment import segment
from .types import CLICK, GRID_H, GRID_W, Action, Observation

#: Canonical order the round-robin cursor walks. RESET is deliberately absent:
#: it is never offered by the engine, and stepping it would silently restart the
#: level and corrupt the transition corpus with a discontinuity.
ACTION_CYCLE: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)


class Driver(Protocol):
    """Chooses the next action. Stateful, deterministic given its seed."""

    @property
    def name(self) -> str: ...

    def next_action(self, obs: Observation) -> Optional[Action]:
        """The next action, or ``None`` when the rollout should stop."""
        ...


def _legal(obs: Observation) -> list[int]:
    """Available action ids, RESET excluded, in ascending order."""
    return sorted(a for a in obs.available_actions if a != 0)


@dataclass
class RandomDriver:
    """Uniform random over legal actions, with uniform random click pixels.

    The policy every session-1 measurement used. Kept so that "round-robin is
    better" is a comparison against something reproducible rather than a claim.
    """

    seed: int = 0
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    @property
    def name(self) -> str:
        return "random"

    def next_action(self, obs: Observation) -> Optional[Action]:
        acts = _legal(obs)
        if not acts or obs.state.is_terminal:
            return None
        a = self._rng.choice(acts)
        if a == CLICK:
            return Action.click(self._rng.randrange(GRID_W), self._rng.randrange(GRID_H))
        return Action.simple(a)


def lattice_targets(step: int = 8) -> list[tuple[int, int]]:
    """A fixed coarse grid of click points covering the frame.

    The null hypothesis for click targeting: no perception involved, just sweep
    the frame. If object-targeting cannot beat this, the pruning is not earning
    anything on the click-only games and we should know that.
    """
    half = step // 2
    return [(x, y) for y in range(half, GRID_H, step) for x in range(half, GRID_W, step)]


def object_targets(frame: np.ndarray) -> list[tuple[int, int]]:
    """One click point per candidate object, unioned across all segmentations.

    Deliberately the union and deliberately without a ``Geometry``: the driver
    only needs a plausible pixel *inside* something, not the best possible
    carving, and making rollout generation depend on geometry — which finds
    structure on only 4 of 25 games — would couple the corpus to the weakest
    component in the pipeline.
    """
    return segment(frame).union_click_targets()


@dataclass
class RoundRobinDriver:
    """Cycle the legal actions in order; cycle click targets independently.

    Two cursors, advanced separately, because the two axes carry different
    amounts of information in different games. On a keyboard-only game the
    target cursor never moves; on a click-only game the action cursor never
    moves and the target cursor is doing all the work.

    Round-robin rather than sampling because it **guarantees coverage**: every
    legal action gets tried a comparable number of times, which is what a
    dynamics corpus needs. Uniform sampling leaves coverage to luck and, over the
    short rollouts we can afford, reliably under-samples something.

    This is a deliberate placeholder. It has no notion of which actions did
    anything, so it will keep spending actions on inputs already known to be
    inert. Replacing it with something that follows information is future work
    (D-020).
    """

    target_policy: str = "objects"  # "objects" | "lattice"
    seed: int = 0
    lattice_step: int = 8
    _action_cursor: int = field(default=0, init=False)
    _target_cursor: int = field(default=0, init=False)

    @property
    def name(self) -> str:
        return f"round_robin[{self.target_policy}]"

    def next_action(self, obs: Observation) -> Optional[Action]:
        acts = _legal(obs)
        if not acts or obs.state.is_terminal:
            return None

        # Walk the canonical order until we land on something legal, so coverage
        # stays even even if the published mask changes between steps.
        chosen = None
        for _ in range(len(ACTION_CYCLE)):
            candidate = ACTION_CYCLE[self._action_cursor % len(ACTION_CYCLE)]
            self._action_cursor += 1
            if candidate in acts:
                chosen = candidate
                break
        if chosen is None:  # pragma: no cover - _legal() non-empty makes this unreachable
            chosen = acts[0]

        if chosen != CLICK:
            return Action.simple(chosen)

        frame = obs.last
        targets: Sequence[tuple[int, int]]
        if self.target_policy == "lattice" or frame is None:
            targets = lattice_targets(self.lattice_step)
        else:
            targets = object_targets(frame) or lattice_targets(self.lattice_step)

        x, y = targets[self._target_cursor % len(targets)]
        self._target_cursor += 1
        return Action.click(int(x), int(y))


def make_driver(name: str, seed: int = 0) -> Driver:
    """Build a driver by name, for scripts that take one as an argument."""
    if name == "random":
        return RandomDriver(seed=seed)
    if name in ("round_robin", "objects"):
        return RoundRobinDriver(target_policy="objects", seed=seed)
    if name == "lattice":
        return RoundRobinDriver(target_policy="lattice", seed=seed)
    raise ValueError(f"unknown driver {name!r}; have random, round_robin, lattice")
