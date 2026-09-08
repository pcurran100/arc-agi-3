"""Segmentation into candidate object sets.

The right notion of "object" is environment-dependent, so this module refuses to
pick one. It produces a :class:`SegmentationSet` of several segmentations of the
same frame, computed by different rules, and leaves the choice to whatever can
actually evaluate it - which at this stage of the project is the evaluation
scripts, and later the ontology subsystem.

That is deliberate and is the one place the living survey attributes a standalone
**20-30 point** advantage (``docs/field-review.md`` section 4). It also stays on
the right side of Guardrail 2: these are general representational rules, not
guesses about what kinds of games exist.

Segmentations are computed in **cell space** when
:mod:`arc3.perception.geometry` found verified block structure, and in pixel
space otherwise. On the public demo games that is roughly a half-and-half split,
so both paths are load-bearing and both are exercised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

from .components import CONN4, CONN8, label
from .geometry import Geometry, Rect, bounding_rect


@dataclass(frozen=True, slots=True)
class Component:
    """One candidate object in a frame.

    ``mask`` is in **display pixel** coordinates whatever space the segmentation
    was computed in, so that a click target derived from it is always directly
    usable. ``shape_key`` identifies the component's form independently of where
    it sits, which is what cross-frame matching keys on.
    """

    index: int
    mask: np.ndarray
    bbox: Rect
    size: int
    colours: tuple[int, ...]
    dominant_colour: int
    shape_key: bytes
    centroid: tuple[float, float]

    @property
    def is_rectangular(self) -> bool:
        return self.size == self.bbox.area

    @property
    def click_target(self) -> tuple[int, int]:
        """A pixel inside this component, for ACTION6.

        The centroid when it lands inside the mask, otherwise the first mask
        pixel in raster order - never a pixel outside the object, which would
        make the click land on something else.
        """
        cx, cy = int(round(self.centroid[0])), int(round(self.centroid[1]))
        if 0 <= cy < self.mask.shape[0] and 0 <= cx < self.mask.shape[1] and self.mask[cy, cx]:
            return cx, cy
        ys, xs = np.nonzero(self.mask)
        return int(xs[0]), int(ys[0])


def _shape_key(mask: np.ndarray, bbox: Rect, colours: np.ndarray) -> bytes:
    """Translation-invariant identity: the cropped mask plus its colouring."""
    crop = mask[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1]
    cols = np.where(crop, colours[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1], -1).astype(np.int8)
    return bytes([bbox.height, bbox.width]) + cols.tobytes()


@dataclass(frozen=True, slots=True)
class Segmentation:
    """One way of carving a frame into objects."""

    name: str
    components: tuple[Component, ...]
    cell: int
    background_colour: Optional[int]

    def __len__(self) -> int:
        return len(self.components)

    def __iter__(self) -> Iterator[Component]:
        return iter(self.components)

    @property
    def covered(self) -> int:
        return sum(c.size for c in self.components)

    def click_targets(self) -> list[tuple[int, int]]:
        """One click candidate per component - D-005's pruning, concretely."""
        return [c.click_target for c in self.components]

    def describe(self) -> str:
        return f"{self.name}: {len(self.components)} components, cell={self.cell}"


@dataclass(frozen=True, slots=True)
class SegmentationSet:
    """Several segmentations of the same frame, computed by different rules."""

    frame: np.ndarray
    segmentations: tuple[Segmentation, ...]
    geometry: Optional[Geometry] = None

    def __getitem__(self, name: str) -> Segmentation:
        for s in self.segmentations:
            if s.name == name:
                return s
        raise KeyError(f"no segmentation named {name!r}; have {[s.name for s in self]}")

    def __iter__(self) -> Iterator[Segmentation]:
        return iter(self.segmentations)

    def __len__(self) -> int:
        return len(self.segmentations)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.segmentations)

    def union_click_targets(self) -> list[tuple[int, int]]:
        """Click candidates proposed by *any* segmentation, de-duplicated.

        The union rather than one segmentation's list, because D-005 warns that
        pruning away the winning action is the failure mode to avoid, and we do
        not yet know which notion of object is right for a given game.
        """
        seen: dict[tuple[int, int], None] = {}
        for seg in self.segmentations:
            for t in seg.click_targets():
                seen.setdefault(t, None)
        return list(seen)

    def describe(self) -> str:
        return "\n".join(s.describe() for s in self.segmentations)


def background_colour(grid: np.ndarray, mask: Optional[np.ndarray] = None) -> int:
    """The most common colour, which is the best available background guess.

    A guess, not a fact: a game whose playfield is mostly one object colour will
    fool it. Every segmentation that uses it is offered alongside one that does
    not, so a wrong guess costs a candidate rather than the whole set.
    """
    values = grid[mask] if mask is not None else grid
    counts = np.bincount(np.asarray(values).ravel().astype(np.int64), minlength=16)
    return int(np.argmax(counts))


def _components_from_labels(
    labels: np.ndarray, n: int, colours: np.ndarray, scale: int, ox: int, oy: int
) -> tuple[Component, ...]:
    """Build display-space components from a labelling in cell space."""
    out: list[Component] = []
    for i in range(n):
        cell_mask = labels == i
        if not cell_mask.any():
            continue
        if scale > 1:
            mask = np.zeros(
                (cell_mask.shape[0] * scale + oy, cell_mask.shape[1] * scale + ox), dtype=bool
            )
            up = np.repeat(np.repeat(cell_mask, scale, axis=0), scale, axis=1)
            mask[oy : oy + up.shape[0], ox : ox + up.shape[1]] = up
            mask = _pad_to_frame(mask)
            pix_colours = _pad_to_frame(
                np.repeat(np.repeat(colours, scale, axis=0), scale, axis=1), offset=(ox, oy)
            )
        else:
            mask = _pad_to_frame(cell_mask)
            pix_colours = _pad_to_frame(colours)
        bbox = bounding_rect(mask)
        vals = pix_colours[mask]
        uniq, counts = np.unique(vals, return_counts=True)
        ys, xs = np.nonzero(mask)
        out.append(
            Component(
                index=len(out),
                mask=mask,
                bbox=bbox,
                size=int(mask.sum()),
                colours=tuple(int(v) for v in uniq),
                dominant_colour=int(uniq[int(np.argmax(counts))]),
                shape_key=_shape_key(mask, bbox, pix_colours),
                centroid=(float(xs.mean()), float(ys.mean())),
            )
        )
    out.sort(key=lambda c: (c.bbox.y0, c.bbox.x0))
    return tuple(
        Component(
            index=i,
            mask=c.mask,
            bbox=c.bbox,
            size=c.size,
            colours=c.colours,
            dominant_colour=c.dominant_colour,
            shape_key=c.shape_key,
            centroid=c.centroid,
        )
        for i, c in enumerate(out)
    )


def _pad_to_frame(arr: np.ndarray, offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Place ``arr`` into a 64x64 frame at ``offset``, cropping any overflow."""
    from ..types import GRID_H, GRID_W

    ox, oy = offset
    out = np.zeros((GRID_H, GRID_W), dtype=arr.dtype)
    h = min(GRID_H - oy, arr.shape[0])
    w = min(GRID_W - ox, arr.shape[1])
    if h > 0 and w > 0:
        out[oy : oy + h, ox : ox + w] = arr[:h, :w]
    return out


def segment(
    frame: np.ndarray,
    geometry: Optional[Geometry] = None,
    *,
    include: Optional[tuple[str, ...]] = None,
) -> SegmentationSet:
    """Produce several candidate segmentations of ``frame``.

    The rules, all general and none game-specific:

    - ``colour_4`` / ``colour_8`` - maximal same-colour regions. The classic ARC
      notion of an object.
    - ``foreground_4`` / ``foreground_8`` - regions of anything that is not the
      background colour, so a multi-coloured sprite stays one object rather than
      fragmenting into its colours.
    - ``colour_mass`` - all cells of one colour as a single object regardless of
      connectivity, which is what a scattered collectible set looks like.
    - ``bbox`` - the bounding box of each same-colour region filled in, so a
      hollow outline reads as the region it encloses.

    Pass ``geometry`` whenever you have it: with verified block structure the
    segmentation runs in cell space, which is both faster and correct on the
    upscaled games where pixel-space components are ``cell**2`` times too large.
    """
    frame = np.asarray(frame)
    cell, ox, oy = 1, 0, 0
    work = frame
    if geometry is not None and geometry.block.is_structured:
        block = geometry.block
        work, ox, oy = block.tile_frame(frame)
        cell = block.cell

    bg = background_colour(work)
    segs: list[Segmentation] = []

    def add(name: str, labels: np.ndarray, n: int) -> None:
        if include is not None and name not in include:
            return
        segs.append(
            Segmentation(
                name=name,
                components=_components_from_labels(labels, n, work, cell, ox, oy),
                cell=cell,
                background_colour=bg,
            )
        )

    for conn, suffix in ((CONN4, "4"), (CONN8, "8")):
        labels, n = label(work, connectivity=conn)
        add(f"colour_{suffix}", labels, n)

        # Always emitted, even when the frame is entirely background and the
        # result is empty. A missing key is a worse failure mode than an empty
        # segmentation: callers select a segmentation by name once and use it
        # for a whole run, and a name that vanishes on a blank frame turns a
        # legitimate "nothing is visible" into a KeyError mid-episode.
        fg_mask = work != bg
        flabels, fn = label(fg_mask.astype(np.int8), connectivity=conn, equal_values=False)
        add(f"foreground_{suffix}", flabels, fn)

    # One object per colour, ignoring connectivity.
    if include is None or "colour_mass" in include:
        mass = np.full(work.shape, -1, dtype=np.int32)
        n = 0
        for colour in sorted(int(c) for c in np.unique(work)):
            m = work == colour
            mass[m] = n
            n += 1
        add("colour_mass", mass, n)

    # Filled bounding boxes of same-colour regions.
    if include is None or "bbox" in include:
        labels, n = label(work, connectivity=CONN8)
        boxed = np.full(work.shape, -1, dtype=np.int32)
        for i in range(n):
            m = labels == i
            r = bounding_rect(m)
            boxed[r.y0 : r.y1, r.x0 : r.x1] = i
        add("bbox", boxed, n)

    return SegmentationSet(frame=frame, segmentations=tuple(segs), geometry=geometry)
