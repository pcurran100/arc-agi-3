"""Driving policies — how a rollout decides what to do next.

This is not an agent. It has no goal, no model and no memory of consequences.
Its only job is to generate a *corpus of transitions* for dynamics induction to
learn from, and the thing that matters about it is the **distribution** of
transitions it produces, not whether it wins anything.

Why this module exists at all (D-019, D-020, D-026): every perception number in
the repository was measured under uniformly random play, whose transitions turn
out to be small and low-diversity — 1–58 changed pixels of 4,096, heavily
dominated by recolouring.

The history is worth keeping straight, because the obvious fix was tried and
failed before the real one was found. Round-robin over the action mask was
proposed as the improvement and **measured worse** than random on event
diversity (D-020), so random was retained. D-021 later mapped the click space
exhaustively and showed that the thing actually being wasted was not the action
choice but the *click placement*: object-derived targets reach 100% of a game's
distinct click outcomes on 13 of 15 live games, while as few as 2.3% of raw
pixels do anything at all. `RandomObjectDriver` is the combination those two
results point at, and it did not exist until D-026.

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

from .perception.segment import background_colour, segment
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

    **Includes the background**, because ``colour_4``, ``colour_8``,
    ``colour_mass`` and ``bbox`` all partition the entire frame, so the
    background region is a component like any other and contributes its own
    click target — usually near the middle of the frame. That is correct for
    D-005's purpose, where the cost of dropping a live target is worse than the
    cost of carrying a dead one, and it is why D-021's proposal counts are as
    large as they are. A driver that wants to *spend* actions should prefer
    :func:`foreground_object_targets`.
    """
    return segment(frame).union_click_targets()


def foreground_object_targets(frame: np.ndarray) -> list[tuple[int, int]]:
    """Object click points excluding the frame's background colour.

    The version a driver should use. `object_targets` includes the background
    region's own centroid, and on most frames that is one of the least
    informative pixels available — it was caught by a test asserting that this
    driver's clicks land inside objects, which failed on (32,32).

    Not a hard exclusion of background clicks from the action space, which would
    be a Guardrail 1 rejection of a whole region on no evidence: it is a
    preference applied when generating a corpus, and the caller falls back to
    the full set when nothing survives. Whether background clicks ever matter is
    answerable from the click map (D-021) and has not been checked.
    """
    bg = background_colour(frame)
    out = [(x, y) for (x, y) in object_targets(frame) if int(frame[y, x]) != bg]
    return out


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


@dataclass
class RandomObjectDriver:
    """Uniform random over legal actions, with clicks aimed at segmented objects.

    The missing cell in the driver design space, and the one the evidence
    actually points at. Two axes were varied independently across D-020 and
    D-021 and never combined:

    - **Action selection.** D-020 measured uniform random against a cyclic
      round-robin over all 25 games and random won on the metric that matters
      for a learning corpus, event diversity (40.1 distinct events per game
      against 36.9). The likely mechanism, offered there as a hypothesis, is
      that a deterministic engine plus a strictly periodic policy falls into
      limit cycles that a random walk breaks out of.
    - **Click placement.** D-021 then mapped every one of the 4,096 pixels on
      all 19 click-publishing games and found object-derived targets reach
      **100% of a game's distinct click outcomes on 13 of 15** live games,
      against as little as 33% for a blind lattice sweep. It also found the live
      fraction can be tiny: 7.0% of pixels on ft09, 2.3% on lp85.

    So random *actions* and object-targeted *clicks* are each the measured
    winner on their own axis, and `RandomDriver` — the driver every corpus in
    this repository was recorded with — combines random actions with **random
    pixels**. On a game where 2.3% of pixels do anything, that throws away
    roughly forty-nine clicks in fifty.

    That matters beyond the no-op rate. The rule language's only relational
    feature is ``clicked`` (D-024), which is true when the action landed inside
    a given object. Learning anything from it requires hitting the *same*
    object repeatedly, and scattering clicks uniformly over 4,096 pixels almost
    never does. Under `RandomDriver` the corpus is close to structurally
    incapable of teaching a click rule, which is a confound in D-025's finding
    that every click-only game scored zero.

    One change from `RandomDriver`, deliberately: the click target is drawn from
    the object proposal set instead of the pixel grid. Nothing else moves, so a
    comparison between the two isolates that single variable.

    **What this does not do.** It does not prefer targets it has not tried, does
    not avoid objects already known to be inert, and does not merge targets that
    produced identical outcomes — all of which D-021 decision 2 argues for and
    all of which are information-seeking behaviour that D-020 deliberately
    deferred until there is a model to seek information *for*.
    """

    seed: int = 0
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    @property
    def name(self) -> str:
        return "random_objects"

    def next_action(self, obs: Observation) -> Optional[Action]:
        acts = _legal(obs)
        if not acts or obs.state.is_terminal:
            return None
        a = self._rng.choice(acts)
        if a != CLICK:
            return Action.simple(a)
        frame = obs.last
        targets: Sequence[tuple[int, int]] = []
        if frame is not None:
            # Prefer real objects; fall back to the full union (which includes
            # the background region) before falling back to raw pixels.
            targets = foreground_object_targets(frame) or object_targets(frame)
        if not targets:
            # No segmentation to aim at; fall back to the pixel grid rather than
            # skipping the action, so the action distribution is unchanged.
            return Action.click(self._rng.randrange(GRID_W), self._rng.randrange(GRID_H))
        x, y = self._rng.choice(targets)
        return Action.click(int(x), int(y))


def make_driver(name: str, seed: int = 0) -> Driver:
    """Build a driver by name, for scripts that take one as an argument."""
    if name == "random":
        return RandomDriver(seed=seed)
    if name in ("round_robin", "objects"):
        return RoundRobinDriver(target_policy="objects", seed=seed)
    if name == "lattice":
        return RoundRobinDriver(target_policy="lattice", seed=seed)
    if name in ("random_objects", "random_obj"):
        return RandomObjectDriver(seed=seed)
    raise ValueError(
        f"unknown driver {name!r}; have random, random_objects, round_robin, lattice"
    )
