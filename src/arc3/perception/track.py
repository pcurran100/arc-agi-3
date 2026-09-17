"""Object identity across frames.

The hard component and the acknowledged tarpit risk (D-007). The job: given a
segmentation of frame *t* and of frame *t+1*, say which component in one *is*
which component in the other, through movement, colour change, occlusion,
appearance and disappearance.

Two commitments shape the design:

- **No hard rejection** (Guardrail 1, D-003). A match is scored, never asserted.
  A track that loses its object keeps a decaying confidence and a memory of what
  it looked like, so it can be reacquired after an occlusion rather than being
  destroyed by one bad frame. Hidden state means a correct-but-conditional
  hypothesis will sometimes look wrong, and deleting on that is unsound.
- **Matching is global, not greedy per component.** Greedy nearest-match breaks
  exactly where it matters - two identical sprites passing each other - so
  assignment is done by cost minimisation over the whole frame pair. With at
  most a few dozen components the Hungarian algorithm is affordable, and
  internal compute is free.

Cost between two components is built from evidence that is exact, in decreasing
order of how much it should count: identical shape and colouring, identical
shape after recolouring, mask overlap, size similarity, and distance moved.
Nothing here is learned or fitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .segment import Component, Segmentation

#: Cost assigned to leaving a component unmatched. A pairing only happens when
#: it is cheaper than both sides going unexplained, so this sets the bar for
#: "these are the same object" without being a tuned similarity threshold:
#: it is the price of the alternative explanation, appearance-plus-disappearance.
UNMATCHED_COST = 1.0


@dataclass(frozen=True, slots=True)
class Match:
    """One component in frame *t* identified with one in frame *t+1*."""

    prev_index: int
    next_index: int
    cost: float
    displacement: tuple[int, int]
    same_shape: bool
    same_colour: bool

    @property
    def moved(self) -> bool:
        return self.displacement != (0, 0)

    @property
    def recoloured(self) -> bool:
        return self.same_shape and not self.same_colour

    @property
    def confidence(self) -> float:
        """1.0 for a perfect identification, falling to 0 at the unmatched bar."""
        return max(0.0, 1.0 - self.cost / UNMATCHED_COST)


@dataclass(frozen=True, slots=True)
class FrameMatching:
    """The full correspondence between two consecutive segmentations."""

    matches: tuple[Match, ...]
    appeared: tuple[int, ...]
    disappeared: tuple[int, ...]

    @property
    def n_matched(self) -> int:
        return len(self.matches)

    def by_prev(self) -> dict[int, Match]:
        return {m.prev_index: m for m in self.matches}

    def by_next(self) -> dict[int, Match]:
        return {m.next_index: m for m in self.matches}

    def rigid_motions(self) -> dict[tuple[int, int], int]:
        """How many objects moved by each displacement.

        A single dominant non-zero displacement is the signature of "the action
        moved everything" (a scrolling camera); a single object with a non-zero
        displacement while everything else is still is the signature of "this is
        me". Both are ontology's business, but this is where the evidence is.
        """
        counts: dict[tuple[int, int], int] = {}
        for m in self.matches:
            counts[m.displacement] = counts.get(m.displacement, 0) + 1
        return counts


def _pair_cost(a: Component, b: Component) -> float:
    """Cost of identifying ``a`` in frame *t* with ``b`` in frame *t+1*.

    Zero means indistinguishable and in the same place; costs at or above
    :data:`UNMATCHED_COST` mean the pairing is not worth making.
    """
    if a.shape_key == b.shape_key:
        base = 0.0
    else:
        # Same silhouette, different colouring - a toggle or state flip.
        same_silhouette = (
            a.bbox.width == b.bbox.width
            and a.bbox.height == b.bbox.height
            and a.size == b.size
            and np.array_equal(
                a.mask[a.bbox.y0 : a.bbox.y1, a.bbox.x0 : a.bbox.x1],
                b.mask[b.bbox.y0 : b.bbox.y1, b.bbox.x0 : b.bbox.x1],
            )
        )
        if same_silhouette:
            base = 0.25
        else:
            overlap = float(np.logical_and(a.mask, b.mask).sum())
            union = float(np.logical_or(a.mask, b.mask).sum())
            iou = overlap / union if union else 0.0
            size_ratio = min(a.size, b.size) / max(a.size, b.size)
            # Deformation is real (growing, shrinking, merging), so it is priced
            # rather than forbidden - but it must cost more than a clean move.
            base = 0.5 + 0.5 * (1.0 - iou) * (1.0 - size_ratio)

    dx = b.centroid[0] - a.centroid[0]
    dy = b.centroid[1] - a.centroid[1]
    distance = (dx * dx + dy * dy) ** 0.5
    # Movement is expected and cheap, but not free: between two otherwise
    # identical candidates the nearer one wins. Saturates so that a long jump is
    # still preferred over inventing an appearance and a disappearance.
    move_penalty = 0.4 * (1.0 - 1.0 / (1.0 + distance / 8.0))
    return base + move_penalty


def _hungarian(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost assignment (Jonker-Volgenant style shortest augmenting path).

    Rectangular input; returns ``(row, col)`` pairs for the ``min(n, m)`` rows
    that get assigned. Written out rather than pulled from scipy to keep the
    offline dependency set to what ``arc-agi`` already installs.
    """
    n, m = cost.shape
    if n == 0 or m == 0:
        return []
    transposed = n > m
    c = cost.T if transposed else cost
    n, m = c.shape

    INF = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)  # p[j] = row assigned to column j
    way = np.zeros(m + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = c[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if j1 < 0:
                break
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    pairs = []
    for j in range(1, m + 1):
        if p[j] > 0:
            row, col = p[j] - 1, j - 1
            pairs.append((col, row) if transposed else (row, col))
    return pairs


def match_frames(prev: Segmentation, nxt: Segmentation) -> FrameMatching:
    """Identify the components of ``prev`` with those of ``nxt``.

    Pairings costing at least :data:`UNMATCHED_COST` are discarded, which is what
    lets objects genuinely appear and disappear rather than being forced into a
    correspondence that does not exist.
    """
    a, b = prev.components, nxt.components
    if not a or not b:
        return FrameMatching(
            matches=(),
            appeared=tuple(range(len(b))),
            disappeared=tuple(range(len(a))),
        )

    cost = np.array([[_pair_cost(x, y) for y in b] for x in a], dtype=float)
    matches: list[Match] = []
    used_prev: set[int] = set()
    used_next: set[int] = set()
    for i, j in _hungarian(cost):
        if cost[i, j] >= UNMATCHED_COST:
            continue
        ca, cb = a[i], b[j]
        matches.append(
            Match(
                prev_index=i,
                next_index=j,
                cost=float(cost[i, j]),
                displacement=(
                    int(round(cb.centroid[0] - ca.centroid[0])),
                    int(round(cb.centroid[1] - ca.centroid[1])),
                ),
                same_shape=ca.shape_key == cb.shape_key
                or (ca.size == cb.size and ca.bbox.width == cb.bbox.width),
                same_colour=ca.dominant_colour == cb.dominant_colour,
            )
        )
        used_prev.add(i)
        used_next.add(j)

    matches.sort(key=lambda m: m.prev_index)
    return FrameMatching(
        matches=tuple(matches),
        appeared=tuple(j for j in range(len(b)) if j not in used_next),
        disappeared=tuple(i for i in range(len(a)) if i not in used_prev),
    )


@dataclass
class Track:
    """One object's identity carried through time.

    ``confidence`` decays while unseen instead of the track being deleted
    (Guardrail 1). A track below :data:`REACQUIRE_FLOOR` is dormant, not gone,
    and will be revived if something matching its remembered appearance shows up
    - which is exactly what happens when an object emerges from behind another.
    """

    track_id: int
    first_frame: int
    last_seen_frame: int
    component: Component
    history: list[tuple[int, Component]] = field(default_factory=list)
    confidence: float = 1.0
    frames_unseen: int = 0

    @property
    def is_active(self) -> bool:
        return self.frames_unseen == 0

    @property
    def positions(self) -> list[tuple[float, float]]:
        return [c.centroid for _, c in self.history]

    def displacement_since(self, frames: int = 1) -> Optional[tuple[float, float]]:
        if len(self.history) <= frames:
            return None
        a = self.history[-1 - frames][1].centroid
        b = self.history[-1][1].centroid
        return (b[0] - a[0], b[1] - a[1])

    @property
    def total_displacement(self) -> tuple[float, float]:
        if len(self.history) < 2:
            return (0.0, 0.0)
        a, b = self.history[0][1].centroid, self.history[-1][1].centroid
        return (b[0] - a[0], b[1] - a[1])

    @property
    def is_stationary(self) -> bool:
        return all(c.centroid == self.history[0][1].centroid for _, c in self.history)


#: Confidence below which a track is dormant. It is never deleted, so this only
#: controls whether a track is offered as a current object, not whether it is
#: remembered.
REACQUIRE_FLOOR = 0.1

#: Multiplied into confidence for each frame an object goes unseen.
DECAY = 0.6


class Tracker:
    """Maintains :class:`Track` identities across a sequence of segmentations.

    Feed it one segmentation per frame, in order. The segmentation must be
    produced by the same rule each time - mixing ``colour_4`` and
    ``foreground_8`` frame to frame would make identity meaningless. That
    requirement used to be a docstring only: ``segmentation_name`` was stored
    and never read, so the error it warns about was undetectable. It is now
    checked in :meth:`update` (D-015).

    The default is ``colour_8``, which is what D-014's 0.8% id-switch rate was
    measured on. It was ``foreground_8`` until D-015 pointed out that the
    measured carving and the default carving were not the same one.
    """

    def __init__(self, segmentation_name: str = "colour_8") -> None:
        self.segmentation_name = segmentation_name
        self.tracks: dict[int, Track] = {}
        self._next_id = 0
        self._frame = -1
        self._prev: Optional[Segmentation] = None
        self._prev_track_of: dict[int, int] = {}

    @property
    def active_tracks(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.is_active]

    @property
    def live_tracks(self) -> list[Track]:
        """Active tracks plus dormant ones still above the reacquisition floor."""
        return [t for t in self.tracks.values() if t.confidence >= REACQUIRE_FLOOR]

    def update(self, seg: Segmentation) -> FrameMatching:
        """Advance one frame. Returns the frame-to-frame matching."""
        if seg.name != self.segmentation_name:
            raise ValueError(
                f"tracker is running on {self.segmentation_name!r} but was fed "
                f"{seg.name!r} at frame {self._frame + 1}; identity across a "
                "mixed carving is meaningless"
            )
        self._frame += 1
        if self._prev is None:
            self._prev = seg
            self._prev_track_of = {}
            for c in seg.components:
                t = self._new_track(c)
                self._prev_track_of[c.index] = t.track_id
            return FrameMatching(
                matches=(), appeared=tuple(range(len(seg.components))), disappeared=()
            )

        matching = match_frames(self._prev, seg)
        next_track_of: dict[int, int] = {}

        for m in matching.matches:
            tid = self._prev_track_of.get(m.prev_index)
            if tid is None:
                continue
            track = self.tracks[tid]
            comp = seg.components[m.next_index]
            track.component = comp
            track.history.append((self._frame, comp))
            track.last_seen_frame = self._frame
            track.frames_unseen = 0
            # Confidence follows the quality of the identification, and recovers
            # rather than ratcheting down permanently.
            track.confidence = min(1.0, 0.5 * track.confidence + 0.5 * m.confidence + 0.25)
            next_track_of[m.next_index] = tid

        for i in matching.disappeared:
            tid = self._prev_track_of.get(i)
            if tid is None:
                continue
            track = self.tracks[tid]
            track.frames_unseen += 1
            track.confidence *= DECAY  # down-weight, never delete

        for j in matching.appeared:
            comp = seg.components[j]
            revived = self._try_reacquire(comp)
            if revived is not None:
                revived.component = comp
                revived.history.append((self._frame, comp))
                revived.last_seen_frame = self._frame
                revived.frames_unseen = 0
                revived.confidence = min(1.0, revived.confidence + 0.4)
                next_track_of[j] = revived.track_id
            else:
                next_track_of[j] = self._new_track(comp).track_id

        self._prev = seg
        self._prev_track_of = next_track_of
        return matching

    def _try_reacquire(self, comp: Component) -> Optional[Track]:
        """Match a newly appeared component against dormant tracks.

        This is what makes occlusion survivable: an object hidden behind another
        for three frames comes back as the same identity instead of a new one.
        """
        best: Optional[Track] = None
        best_cost = UNMATCHED_COST
        for track in self.tracks.values():
            if track.is_active or track.confidence < REACQUIRE_FLOOR:
                continue
            cost = _pair_cost(track.component, comp)
            if cost < best_cost:
                best, best_cost = track, cost
        return best

    def _new_track(self, comp: Component) -> Track:
        track = Track(
            track_id=self._next_id,
            first_frame=self._frame,
            last_seen_frame=self._frame,
            component=comp,
            history=[(self._frame, comp)],
        )
        self.tracks[self._next_id] = track
        self._next_id += 1
        return track


def track_sequence(
    segmentations: Sequence[Segmentation], *, name: str = "colour_8"
) -> Tracker:
    """Run a tracker over a whole sequence of segmentations."""
    tracker = Tracker(segmentation_name=name)
    for seg in segmentations:
        tracker.update(seg)
    return tracker
