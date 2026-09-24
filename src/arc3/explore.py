"""Exploration that tries to make the game move, rather than sampling blindly.

The first component in this repository that is agent-like rather than a research
instrument: it learns from what comes back and changes what it does next.

**Why this exists.** D-028 decomposed the sixteen zero-scoring games and found
that *half* of them are not a modelling failure at all — the agent simply never
makes the game do anything. D-030 then measured that the moves are there to be
found: vc33 moves the playfield on **2 of 256 probes at the level start and 254
of 256 five actions later**, and merely aiming clicks at objects rather than
random pixels multiplies playfield activity **2x to 20x** across all eight
starved games. Nothing here needs a better learner. It needs a policy that
notices what works.

Three things follow from the measurements, and each is a design constraint:

1. **Count playfield changes, not frame changes.** A move counter ticking is not
   the game responding. vc33 changed its frame on 99 of 99 actions and its
   playfield on **zero** of them (D-028); a policy rewarded for frame change
   would have been perfectly satisfied. The bookkeeping band has to be estimated
   online, which :class:`OnlineBand` does.

2. **Arms are object *signatures*, not pixels.** Objects move, so a pixel that
   paid off last turn means nothing this turn. An arm here is the attribute
   vector the rule language already selects on, so experience transfers as
   objects move around, and the explorer and the learner are looking at the same
   features. This is the same signature D-026 used to measure whether a corpus
   could teach a click rule.

3. **Evidence decays.** Responsiveness is *state-dependent*, measured on eight of
   nineteen games for click liveness (D-021) and confirmed for playfield effect
   on vc33 (D-030). A stationary estimate would learn "clicking does nothing"
   at the level start and never revise it five actions later when everything
   works. Counts are discounted every step so the policy tracks the game rather
   than averaging over it.

**Guardrail 1 is respected by construction.** Thompson sampling from a Beta
posterior never assigns zero probability to an arm, so nothing is ever hard
rejected — an arm that has failed twenty times is heavily down-weighted and
still occasionally retried, which is exactly what a state-gated game requires.

**What this does not do.**

- **No goal.** It maximises "something happened in the playfield", which is a
  proxy for interestingness and *not* a proxy for winning. It is the exploration
  half of D-027's target architecture; the half that acts on hypotheses towards
  a goal is not built.
- **No model.** It does not know *what* an action does, only whether it tends to
  do something. It is deliberately dumber than the dynamics learner and exists to
  feed it.
- **No RESET policy.** Some games get *less* responsive with depth (cd82 and
  ka59 fall from 100% to ~6% by twenty actions, D-021), so a policy that noticed
  a dry spell and restarted the level would sometimes be right. `RESET` costs an
  action and does not rewind the action count, so this needs the RHAE cost model
  from D-022 and is left out of the first version deliberately.
- **No credit assignment across steps.** An action that unlocks responsiveness
  five moves later gets no credit for it, which is precisely what vc33 rewards.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .drive import foreground_object_targets, object_targets
from .dynamics.rules import describe_object
from .perception.segment import segment
from .types import CLICK, GRID_H, GRID_W, Action, Observation

#: Rows changing in at least this fraction of observed changes are bookkeeping.
BAND_FRACTION = 0.5

#: A band may not be thicker than this, or it is playfield (see
#: `dynamics.predict.bookkeeping_mask`, which uses the same cap for the same
#: reason: without it the estimator swallows the game).
MAX_BAND_ROWS = 4

#: Observations needed before the band estimate is used at all. Below this the
#: policy counts every change as playfield, which over-credits early and
#: corrects itself rather than starting with a wrong band.
BAND_WARMUP = 8

#: Per-step multiplier on accumulated evidence. Responsiveness is state
#: dependent, so old observations must fade or the policy cannot follow a game
#: that changes underneath it.
DECAY = 0.97


class OnlineBand:
    """Incremental estimate of which rows are bookkeeping rather than game.

    The online counterpart of `dynamics.predict.bookkeeping_mask`, which needs a
    whole corpus. Same rule, same thresholds, updated one transition at a time.
    """

    def __init__(self, height: int = GRID_H, width: int = GRID_W) -> None:
        self._rows = np.zeros(height, dtype=np.float64)
        self._n = 0.0
        self._w = width
        self._h = height

    def update(self, before: np.ndarray, after: np.ndarray) -> None:
        diff = before != after
        if not diff.any():
            return
        self._n += 1.0
        self._rows[np.unique(np.nonzero(diff)[0])] += 1.0

    @property
    def ready(self) -> bool:
        return self._n >= BAND_WARMUP

    def mask(self) -> np.ndarray:
        m = np.zeros((self._h, self._w), dtype=bool)
        if not self.ready:
            return m
        rows = np.nonzero(self._rows >= BAND_FRACTION * self._n)[0]
        if len(rows) and len(rows) <= MAX_BAND_ROWS:
            m[rows, :] = True
        return m

    def moved_playfield(self, before: np.ndarray, after: np.ndarray) -> bool:
        """Did this transition change anything the band does not explain?"""
        return bool(((before != after) & ~self.mask()).any())


@dataclass
class _Arm:
    """Beta posterior over "this choice moves the playfield"."""

    a: float = 1.0
    b: float = 1.0

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(self.a, self.b)

    def update(self, success: bool) -> None:
        if success:
            self.a += 1.0
        else:
            self.b += 1.0

    def decay(self, gamma: float) -> None:
        # Decay towards the prior, never below it, so an arm always retains some
        # optimism and Guardrail 1's "never hard reject" survives arbitrarily
        # long runs of failure.
        self.a = 1.0 + (self.a - 1.0) * gamma
        self.b = 1.0 + (self.b - 1.0) * gamma

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)


@dataclass
class ExploringDriver:
    """Chooses actions by Thompson sampling on "does this move the playfield".

    Learns from the observation stream alone: it remembers the frame and action
    it last emitted and scores them when the next observation arrives, so it
    satisfies the plain `Driver` protocol and needs no feedback channel.
    """

    seed: int = 0
    carving: str = "colour_8"
    decay: float = DECAY
    _rng: random.Random = field(init=False)
    _arms: dict[Any, _Arm] = field(default_factory=dict, init=False)
    band: OnlineBand = field(default_factory=OnlineBand, init=False)
    _last_frame: Optional[np.ndarray] = field(default=None, init=False)
    _last_key: Optional[Any] = field(default=None, init=False)
    n_playfield: int = field(default=0, init=False)
    n_steps: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    @property
    def name(self) -> str:
        return "explore"

    # -- learning ----------------------------------------------------------

    def _arm(self, key: Any) -> _Arm:
        return self._arms.setdefault(key, _Arm())

    def _learn(self, frame: Optional[np.ndarray]) -> None:
        """Score the previous action against what came back."""
        if self._last_frame is None or self._last_key is None or frame is None:
            return
        self.band.update(self._last_frame, frame)
        moved = self.band.moved_playfield(self._last_frame, frame)
        self._arm(self._last_key).update(moved)
        self.n_playfield += int(moved)
        self.n_steps += 1
        for arm in self._arms.values():
            arm.decay(self.decay)

    # -- acting ------------------------------------------------------------

    @staticmethod
    def _click_key(attrs: dict) -> tuple:
        # Colour and size, not the full signature: shape is too specific to
        # generalise across frames, and the pair was enough to isolate the mover
        # on ls20 (D-024's worked example).
        return ("click", attrs["colour"], attrs["size"])

    def next_action(self, obs: Observation) -> Optional[Action]:
        frame = obs.last
        self._learn(frame)

        acts = sorted(a for a in obs.available_actions if a != 0)
        if not acts or obs.state.is_terminal:
            self._last_frame, self._last_key = None, None
            return None

        candidates: list[tuple[float, Any, Action]] = []
        for a in acts:
            if a == CLICK:
                continue
            key = ("act", a)
            candidates.append((self._arm(key).sample(self._rng), key, Action.simple(a)))

        if CLICK in acts and frame is not None:
            seg = segment(frame, include=(self.carving,))[self.carving]
            bg = seg.background_colour
            for comp in seg.components:
                if bg is not None and comp.dominant_colour == bg:
                    continue
                x, y = comp.click_target
                key = self._click_key(describe_object(comp, Action.click(x, y)))
                candidates.append((self._arm(key).sample(self._rng), key, Action.click(x, y)))
        elif CLICK in acts:
            x, y = self._rng.randrange(GRID_W), self._rng.randrange(GRID_H)
            candidates.append((self._arm(("click", -1, -1)).sample(self._rng),
                               ("click", -1, -1), Action.click(x, y)))

        if not candidates:
            a = self._rng.choice(acts)
            action = (
                Action.click(self._rng.randrange(GRID_W), self._rng.randrange(GRID_H))
                if a == CLICK
                else Action.simple(a)
            )
            self._last_frame, self._last_key = (
                None if frame is None else frame.copy(),
                ("act", a),
            )
            return action

        _, key, action = max(candidates, key=lambda c: c[0])
        self._last_frame = None if frame is None else frame.copy()
        self._last_key = key
        return action

    # -- inspection --------------------------------------------------------

    @property
    def playfield_rate(self) -> float:
        return self.n_playfield / self.n_steps if self.n_steps else 0.0

    def describe(self, limit: int = 8) -> str:
        rows = sorted(self._arms.items(), key=lambda kv: -kv[1].mean)
        out = [f"{len(self._arms)} arms, playfield rate {100 * self.playfield_rate:.0f}%"]
        for k, v in rows[:limit]:
            out.append(f"  {str(k):28s} mean={v.mean:.2f}  a={v.a:.1f} b={v.b:.1f}")
        return "\n".join(out)
