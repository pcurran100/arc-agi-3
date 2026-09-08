"""Exact structural diff between frames.

Not "the grid changed" but which cells, which regions, and how (D-007). This
module deliberately stops short of object-level attribution - saying *which
object* moved is :mod:`arc3.perception.track`'s job, and needs a segmentation to
even be well posed. What lives here is everything derivable from a pair of grids
alone, exactly and cheaply.

The most useful thing here is :func:`diff_sequence`. A single action can return
up to 42 frames, and the intermediate frames of an animation show the trajectory
of motion rather than just its endpoints - by some distance the richest free
evidence the environment gives us, and something design-thesis section 3
undercounted (D-009). :class:`SequenceDiff` summarises a whole animation,
including whether it is a steady drift in one direction, which is the shape a
sprite translation takes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

from .components import CONN8, label
from .geometry import Rect, bounding_rect


@dataclass(frozen=True, slots=True)
class ChangeRegion:
    """One connected region of changed cells."""

    mask: np.ndarray
    bbox: Rect
    size: int
    colours_before: tuple[int, ...]
    colours_after: tuple[int, ...]

    @property
    def is_recolour(self) -> bool:
        """The region changed colour without changing shape context.

        True when exactly one colour left and one arrived - the signature of a
        toggle or a state flip rather than movement.
        """
        return len(self.colours_before) == 1 and len(self.colours_after) == 1


@dataclass(frozen=True, slots=True)
class FrameDiff:
    """What changed between two grids."""

    before: np.ndarray
    after: np.ndarray
    changed: np.ndarray
    transitions: Counter[tuple[int, int]]
    regions: tuple[ChangeRegion, ...]

    @property
    def n_changed(self) -> int:
        return int(self.changed.sum())

    @property
    def is_empty(self) -> bool:
        return self.n_changed == 0

    @property
    def bbox(self) -> Rect:
        return bounding_rect(self.changed)

    @property
    def colour_delta(self) -> dict[int, int]:
        """Net change in the number of cells of each colour."""
        delta: Counter[int] = Counter()
        for (old, new), n in self.transitions.items():
            delta[old] -= n
            delta[new] += n
        return {c: d for c, d in delta.items() if d != 0}

    def translation(self) -> Optional[tuple[int, int]]:
        """``(dx, dy)`` if the whole frame is a rigid shift of ``before``.

        Checks only shifts small enough to leave overlap worth testing. Returns
        ``None`` for anything that is not an exact whole-frame translation - a
        scrolling camera is the case this catches, and it matters because it
        makes every object appear to move at once.
        """
        if self.is_empty:
            return (0, 0)
        h, w = self.before.shape
        for dy in range(-h + 1, h):
            for dx in range(-w + 1, w):
                if dx == 0 and dy == 0:
                    continue
                ys0, ys1 = max(0, dy), min(h, h + dy)
                xs0, xs1 = max(0, dx), min(w, w + dx)
                if ys1 - ys0 < h // 2 or xs1 - xs0 < w // 2:
                    continue
                src = self.before[ys0 - dy : ys1 - dy, xs0 - dx : xs1 - dx]
                dst = self.after[ys0:ys1, xs0:xs1]
                if np.array_equal(src, dst):
                    return (dx, dy)
        return None


def diff(before: np.ndarray, after: np.ndarray, *, connectivity: int = CONN8) -> FrameDiff:
    """Exact structural diff of two grids."""
    b = np.asarray(before)
    a = np.asarray(after)
    if b.shape != a.shape:
        raise ValueError(f"shape mismatch: {b.shape} vs {a.shape}")
    changed = b != a

    transitions: Counter[tuple[int, int]] = Counter()
    if changed.any():
        pairs = np.stack([b[changed], a[changed]], axis=1)
        for old, new in pairs:
            transitions[(int(old), int(new))] += 1

    regions: list[ChangeRegion] = []
    if changed.any():
        labels, n = label(
            changed.astype(np.int8), connectivity=connectivity, equal_values=False
        )
        for i in range(n):
            m = labels == i
            regions.append(
                ChangeRegion(
                    mask=m,
                    bbox=bounding_rect(m),
                    size=int(m.sum()),
                    colours_before=tuple(sorted(int(v) for v in np.unique(b[m]))),
                    colours_after=tuple(sorted(int(v) for v in np.unique(a[m]))),
                )
            )
        regions.sort(key=lambda r: (-r.size, r.bbox.y0, r.bbox.x0))

    return FrameDiff(
        before=b, after=a, changed=changed, transitions=transitions, regions=tuple(regions)
    )


@dataclass(frozen=True, slots=True)
class SequenceDiff:
    """The whole animation one action produced.

    ``steps`` are the consecutive diffs. ``net`` is the endpoint-to-endpoint
    diff, which is what an agent that only looked at the final frame would see -
    keeping both makes it obvious how much information the intermediate frames
    are carrying.
    """

    frames: tuple[np.ndarray, ...]
    steps: tuple[FrameDiff, ...]
    net: Optional[FrameDiff]
    ever_changed: np.ndarray
    per_step_translation: tuple[Optional[tuple[int, int]], ...] = field(default=())

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def is_static(self) -> bool:
        return not bool(self.ever_changed.any())

    @property
    def is_animated(self) -> bool:
        """More than one frame, and something moved between them."""
        return len(self.steps) > 0 and any(not s.is_empty for s in self.steps)

    @property
    def transient_mask(self) -> np.ndarray:
        """Cells that changed during the animation but ended where they began.

        These are invisible to an endpoint-only view and are exactly what a
        planner would miss - a sprite that passed through a cell, a flash, a
        counter that ticked and reset.
        """
        if self.net is None:
            return np.zeros_like(self.ever_changed)
        return self.ever_changed & ~self.net.changed

    def steady_drift(self) -> Optional[tuple[int, int]]:
        """``(dx, dy)`` if every step is the same non-zero whole-frame shift."""
        shifts = [t for t in self.per_step_translation]
        if not shifts or any(s is None or s == (0, 0) for s in shifts):
            return None
        first = shifts[0]
        return first if all(s == first for s in shifts) else None


def diff_sequence(
    frames: Sequence[np.ndarray] | Iterable[np.ndarray],
    *,
    previous: Optional[np.ndarray] = None,
    connectivity: int = CONN8,
    detect_translation: bool = False,
) -> SequenceDiff:
    """Diff every consecutive pair in an action's frame sequence.

    Args:
        frames: the frames the action returned. May be empty, in which case the
            action was a no-op and the result is static.
        previous: the resting grid before the action. Supply it - without it the
            first frame of the sequence has nothing to be compared against, and
            the first and often largest change is lost.
        detect_translation: also test each step for a whole-frame shift. Off by
            default because it is O(h*w) shifts per step and only pays off on
            games with a scrolling camera.
    """
    seq = [np.asarray(f) for f in frames]
    chain = ([np.asarray(previous)] if previous is not None else []) + seq
    if not chain:
        raise ValueError("diff_sequence needs at least one frame or a previous grid")

    steps = tuple(diff(chain[i], chain[i + 1], connectivity=connectivity) for i in range(len(chain) - 1))
    net = diff(chain[0], chain[-1], connectivity=connectivity) if len(chain) > 1 else None

    stack = np.stack(chain)
    ever = (stack != stack[0]).any(axis=0)

    shifts: tuple[Optional[tuple[int, int]], ...] = ()
    if detect_translation:
        shifts = tuple(s.translation() for s in steps)

    return SequenceDiff(
        frames=tuple(seq),
        steps=steps,
        net=net,
        ever_changed=ever,
        per_step_translation=shifts,
    )
