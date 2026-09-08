"""Action accounting and offline RHAE scoring.

The human baselines are in every game's ``metadata.json`` (D-009), so we can
compute the real metric locally, per level, from day one. Do it on every run -
``docs/benchmark.md`` warns that local development permits infinite retries
while RHAE punishes them quadratically, and the only defence is to make the
score visible from the start rather than at the end.

**Two aggregation details are not pinned down by the published methodology**, and
are exposed here as explicit parameters rather than guessed silently:

- The stated "per-level score is capped at 1.15x the human baseline" could mean
  the *score* is capped at 1.15, or the *efficiency ratio* is, giving a score cap
  of 1.15**2 = 1.3225. We default to capping the score, the more literal reading.
- The stated example "completing 4 of 5 levels cannot exceed 66.7%" does not
  follow from a plain mean over attempted levels (which gives 80%) or from
  squaring it (64%), so the game-level aggregation rule is not recoverable from
  the text. We use the mean over *all* levels in the game, scoring 0 for levels
  never completed, and flag the result as approximate.

Anything reported from this module should say which convention it used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

#: Per-level score cap, as the more literal reading of the published rule.
DEFAULT_SCORE_CAP = 1.15


def level_score(
    baseline_actions: int,
    ai_actions: int,
    *,
    cap: float = DEFAULT_SCORE_CAP,
    cap_ratio: bool = False,
) -> float:
    """RHAE for one completed level: ``(baseline / ai) ** 2``, capped.

    ``cap_ratio=True`` caps the efficiency ratio at ``cap`` instead of the score,
    which is the alternative reading of the published rule (see module docstring).
    """
    if ai_actions <= 0:
        raise ValueError("ai_actions must be positive for a completed level")
    if baseline_actions <= 0:
        raise ValueError("baseline_actions must be positive")
    ratio = baseline_actions / ai_actions
    if cap_ratio:
        return min(ratio, cap) ** 2
    return min(ratio**2, cap)


@dataclass(frozen=True, slots=True)
class LevelResult:
    """One level's outcome. ``actions`` counts only actions spent on this level."""

    index: int
    actions: int
    completed: bool
    baseline: Optional[int] = None

    def score(self, *, cap: float = DEFAULT_SCORE_CAP, cap_ratio: bool = False) -> Optional[float]:
        """RHAE for this level, or None if the baseline is unavailable.

        ``None`` is a real answer, not a failure: under D-012 the metadata
        provider may withhold baselines, and the caller must cope.
        """
        if not self.completed:
            return 0.0
        if self.baseline is None:
            return None
        return level_score(self.baseline, self.actions, cap=cap, cap_ratio=cap_ratio)


@dataclass(frozen=True, slots=True)
class GameResult:
    game_id: str
    levels: tuple[LevelResult, ...]
    total_levels: int
    actions_used: int

    @property
    def levels_completed(self) -> int:
        return sum(1 for lv in self.levels if lv.completed)

    def score(
        self, *, cap: float = DEFAULT_SCORE_CAP, cap_ratio: bool = False
    ) -> Optional[float]:
        """Mean level score over *all* levels in the game, or None.

        Levels never reached score 0, which is what makes an incomplete game
        cost more than the levels it did finish. Returns ``None`` if any
        completed level lacks a baseline, since a partial score would silently
        mix conventions.
        """
        if self.total_levels <= 0:
            return None
        total = 0.0
        for lv in self.levels:
            s = lv.score(cap=cap, cap_ratio=cap_ratio)
            if s is None:
                return None
            total += s
        return total / self.total_levels


class ActionBudget:
    """Tracks actions per level and reports RHAE as a run proceeds.

    Feed it every observation. It segments the run into levels on
    ``levels_completed`` increments and on full resets, so per-level action
    counts come out without the caller bookkeeping anything.
    """

    def __init__(self, game_id: str, total_levels: int, baselines: Optional[Sequence[int]] = None):
        self.game_id = game_id
        self.total_levels = total_levels
        self.baselines: Optional[tuple[int, ...]] = (
            tuple(int(b) for b in baselines) if baselines is not None else None
        )
        self.actions_used = 0
        self._level_index = 0
        self._actions_this_level = 0
        self._completed: list[LevelResult] = []

    @property
    def level_index(self) -> int:
        return self._level_index

    @property
    def actions_this_level(self) -> int:
        return self._actions_this_level

    def baseline_here(self) -> Optional[int]:
        """Human baseline for the current level, or None if withheld."""
        if self.baselines is None or not 0 <= self._level_index < len(self.baselines):
            return None
        return self.baselines[self._level_index]

    def remaining_at_par(self) -> Optional[int]:
        """Actions left before this level's score starts dropping below 1.0.

        ``None`` when baselines are withheld - the planner then has no cost
        model and must fall back on something else (D-012).
        """
        b = self.baseline_here()
        return None if b is None else b - self._actions_this_level

    def observe(self, obs) -> None:
        """Record one step. Errors and no-ops count: the environment charges them."""
        if obs.action is None:
            return
        self.actions_used += 1
        self._actions_this_level += 1
        if obs.full_reset:
            # Back to level 0 with a clean slate; prior level results stand.
            self._level_index = 0
            self._actions_this_level = 0
            return
        if obs.levels_completed > self._level_index:
            # One or more levels cleared by this action.
            while self._level_index < obs.levels_completed:
                base = (
                    self.baselines[self._level_index]
                    if self.baselines is not None and self._level_index < len(self.baselines)
                    else None
                )
                self._completed.append(
                    LevelResult(
                        index=self._level_index,
                        actions=self._actions_this_level,
                        completed=True,
                        baseline=base,
                    )
                )
                self._level_index += 1
                self._actions_this_level = 0

    def result(self) -> GameResult:
        """The run so far, with unreached levels recorded as incomplete."""
        levels = list(self._completed)
        done = {lv.index for lv in levels}
        for i in range(self.total_levels):
            if i not in done:
                base = (
                    self.baselines[i]
                    if self.baselines is not None and i < len(self.baselines)
                    else None
                )
                actions = self._actions_this_level if i == self._level_index else 0
                levels.append(LevelResult(index=i, actions=actions, completed=False, baseline=base))
        levels.sort(key=lambda lv: lv.index)
        return GameResult(
            game_id=self.game_id,
            levels=tuple(levels),
            total_levels=self.total_levels,
            actions_used=self.actions_used,
        )
