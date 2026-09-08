"""Core typed values shared by every subsystem.

Design notes that matter downstream:

- An :class:`Observation` is *one action's worth* of return, which is a
  **sequence** of 0..N grids, not a single grid. Sequences of up to 42 frames
  occur (sb26), and the intermediate frames show the trajectory of motion rather
  than just its endpoints. Anything that reduces an observation to its last
  frame is discarding the richest free evidence the environment gives us.
  See ``docs/benchmark.md`` and D-009.
- Zero frames is a legal, meaningful return: the action was accepted and did
  nothing. It is distinct from an error.
- Grids are immutable. Perception is exact (D-006) and every consumer assumes it
  can hold a reference without defensive copying.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Optional, Sequence

import numpy as np

GRID_H = 64
GRID_W = 64
N_COLOURS = 16

# Action ids, mirroring arcengine.GameAction without importing it, so that
# replayed episodes and pure-perception tests never need the engine present.
RESET = 0
CLICK = 6  # ACTION6, the only parameterised action: takes (x, y)
ALL_ACTION_IDS = (0, 1, 2, 3, 4, 5, 6, 7)


class GameState(str, Enum):
    NOT_PLAYED = "NOT_PLAYED"
    NOT_FINISHED = "NOT_FINISHED"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"

    @property
    def is_terminal(self) -> bool:
        return self in (GameState.WIN, GameState.GAME_OVER)


def freeze_grid(a: Any) -> np.ndarray:
    """Return ``a`` as a validated, read-only 64x64 int8 grid."""
    arr = np.asarray(a, dtype=np.int8)
    if arr.shape != (GRID_H, GRID_W):
        raise ValueError(f"expected a {GRID_H}x{GRID_W} grid, got {arr.shape}")
    if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= N_COLOURS):
        raise ValueError(
            f"colours out of range [0, {N_COLOURS}): {int(arr.min())}..{int(arr.max())}"
        )
    out = arr.copy() if arr.flags.writeable else arr
    out.flags.writeable = False
    return out


@dataclass(frozen=True, slots=True)
class Action:
    """An environment action. Constructing one is free; *spending* one is not."""

    id: int
    x: Optional[int] = None
    y: Optional[int] = None

    def __post_init__(self) -> None:
        if self.id not in ALL_ACTION_IDS:
            raise ValueError(f"unknown action id {self.id}")
        if self.id == CLICK:
            if self.x is None or self.y is None:
                raise ValueError("ACTION6 requires x and y")
            if not (0 <= self.x < GRID_W and 0 <= self.y < GRID_H):
                raise ValueError(f"click out of bounds: ({self.x}, {self.y})")
        elif self.x is not None or self.y is not None:
            raise ValueError(f"action {self.id} takes no coordinates")

    @property
    def is_click(self) -> bool:
        return self.id == CLICK

    @property
    def is_reset(self) -> bool:
        return self.id == RESET

    @property
    def data(self) -> Optional[dict[str, int]]:
        """The ``data`` payload the engine expects, or None."""
        if self.is_click:
            assert self.x is not None and self.y is not None
            return {"x": self.x, "y": self.y}
        return None

    def __str__(self) -> str:
        if self.is_click:
            return f"ACTION6({self.x},{self.y})"
        return "RESET" if self.is_reset else f"ACTION{self.id}"

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id}
        if self.is_click:
            d["x"], d["y"] = self.x, self.y
        return d

    @staticmethod
    def from_json(d: dict[str, Any]) -> "Action":
        return Action(id=int(d["id"]), x=d.get("x"), y=d.get("y"))

    @staticmethod
    def click(x: int, y: int) -> "Action":
        return Action(CLICK, x, y)

    @staticmethod
    def simple(action_id: int) -> "Action":
        return Action(action_id)


@dataclass(frozen=True, slots=True)
class GameMetadata:
    """Per-game facts the engine supplies outside the frame stream.

    ``tags`` and ``baseline_actions`` are permitted agent inputs (D-012) but are
    served through a provider that may withhold them, because we cannot verify
    that the private environments expose them. **Every consumer must work when
    they are None.**
    """

    game_id: str
    win_levels: int
    title: Optional[str] = None
    tags: Optional[tuple[str, ...]] = None
    baseline_actions: Optional[tuple[int, ...]] = None

    def baseline_for_level(self, level_index: int) -> Optional[int]:
        """Human baseline actions for a zero-indexed level, or None."""
        if self.baseline_actions is None:
            return None
        if not 0 <= level_index < len(self.baseline_actions):
            return None
        return self.baseline_actions[level_index]

    def withheld(self) -> "GameMetadata":
        """This metadata with the unverifiable fields removed (D-012 test path)."""
        return GameMetadata(
            game_id=self.game_id, win_levels=self.win_levels, title=self.title
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "win_levels": self.win_levels,
            "title": self.title,
            "tags": list(self.tags) if self.tags is not None else None,
            "baseline_actions": (
                list(self.baseline_actions) if self.baseline_actions is not None else None
            ),
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> "GameMetadata":
        tags = d.get("tags")
        base = d.get("baseline_actions")
        return GameMetadata(
            game_id=d["game_id"],
            win_levels=int(d["win_levels"]),
            title=d.get("title"),
            tags=tuple(tags) if tags is not None else None,
            baseline_actions=tuple(int(b) for b in base) if base is not None else None,
        )


@dataclass(frozen=True, slots=True)
class Observation:
    """Everything one environment step returned.

    ``frames`` may be empty. That is not an error - it means the action was
    accepted and changed nothing observable, which is a wasted action and a fact
    worth learning from.
    """

    frames: tuple[np.ndarray, ...]
    state: GameState
    levels_completed: int
    win_levels: int
    available_actions: tuple[int, ...]
    full_reset: bool = False
    action: Optional[Action] = None
    actions_used: int = 0
    error: Optional[str] = None

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter(self.frames)

    @property
    def is_noop(self) -> bool:
        """The action was accepted but produced no frame."""
        return not self.frames and self.error is None

    @property
    def is_error(self) -> bool:
        return self.error is not None

    @property
    def last(self) -> Optional[np.ndarray]:
        """The resting grid after the action, or None if nothing was drawn."""
        return self.frames[-1] if self.frames else None

    @property
    def first(self) -> Optional[np.ndarray]:
        return self.frames[0] if self.frames else None

    def to_json(self, include_frames: bool = True) -> dict[str, Any]:
        d: dict[str, Any] = {
            "state": self.state.value,
            "levels_completed": self.levels_completed,
            "win_levels": self.win_levels,
            "available_actions": list(self.available_actions),
            "full_reset": self.full_reset,
            "action": self.action.to_json() if self.action else None,
            "actions_used": self.actions_used,
            "error": self.error,
            "n_frames": len(self.frames),
        }
        if include_frames:
            d["frames"] = [f.tolist() for f in self.frames]
        return d

    @staticmethod
    def from_json(d: dict[str, Any]) -> "Observation":
        act = d.get("action")
        return Observation(
            frames=tuple(freeze_grid(f) for f in d.get("frames", ())),
            state=GameState(d["state"]),
            levels_completed=int(d["levels_completed"]),
            win_levels=int(d["win_levels"]),
            available_actions=tuple(int(a) for a in d["available_actions"]),
            full_reset=bool(d.get("full_reset", False)),
            action=Action.from_json(act) if act else None,
            actions_used=int(d.get("actions_used", 0)),
            error=d.get("error"),
        )


@dataclass(frozen=True, slots=True)
class Transition:
    """A (before, action, frames, after) tuple - one dynamics example.

    ``frames`` is the whole animation, not just the endpoints. A transition with
    a long ``frames`` sequence carries far more information than its endpoints
    do, which is the correction to design-thesis section 3's "one labelled
    example per action" (D-009).
    """

    before: np.ndarray
    action: Action
    frames: tuple[np.ndarray, ...]
    levels_completed_before: int
    levels_completed_after: int
    state_after: GameState

    @property
    def after(self) -> np.ndarray:
        """The resting grid after the action - ``before`` if nothing was drawn."""
        return self.frames[-1] if self.frames else self.before

    @property
    def is_noop(self) -> bool:
        return not self.frames or bool(np.array_equal(self.before, self.after))

    @property
    def advanced_level(self) -> bool:
        return self.levels_completed_after > self.levels_completed_before

    @property
    def n_intermediate(self) -> int:
        """Frames strictly between ``before`` and ``after``."""
        return max(0, len(self.frames) - 1)


@dataclass
class Episode:
    """A recorded run: metadata, seed, and every observation in order."""

    game_id: str
    seed: int
    metadata: GameMetadata
    observations: list[Observation] = field(default_factory=list)
    engine_version: str = ""
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def actions_used(self) -> int:
        return sum(1 for o in self.observations if o.action is not None)

    @property
    def final(self) -> Observation:
        return self.observations[-1]

    def grids(self) -> list[np.ndarray]:
        """Every grid in the episode, in order, flattened across observations."""
        return [f for obs in self.observations for f in obs.frames]

    def resting_grids(self) -> list[np.ndarray]:
        """One grid per observation that drew anything - the settled states."""
        return [obs.last for obs in self.observations if obs.last is not None]

    def transitions(self) -> list[Transition]:
        """Consecutive observations as dynamics examples.

        Skips errored steps, and carries the previous resting grid forward
        through no-ops so that ``before`` is always a real observed grid.
        """
        out: list[Transition] = []
        prev: Optional[np.ndarray] = None
        prev_levels = 0
        for obs in self.observations:
            if obs.is_error:
                continue
            if obs.action is not None and prev is not None:
                out.append(
                    Transition(
                        before=prev,
                        action=obs.action,
                        frames=obs.frames,
                        levels_completed_before=prev_levels,
                        levels_completed_after=obs.levels_completed,
                        state_after=obs.state,
                    )
                )
            if obs.last is not None:
                prev = obs.last
            prev_levels = obs.levels_completed
        return out


def as_sequence(frames: Sequence[Any]) -> tuple[np.ndarray, ...]:
    return tuple(freeze_grid(f) for f in frames)
