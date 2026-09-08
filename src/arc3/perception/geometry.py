"""Frame geometry: recovering the cell structure underneath the pixels.

Why this exists (D-011): the 64x64 frame is frequently an integer upscale of a
much smaller camera. On bp35 and lf52 the camera is 8x8, so a naive connected
component is 64x its true area and most "cells" are rendering artefacts.

What first contact with the frames taught us, and which D-011 did not
anticipate: **there is no single scale**. ARCEngine composites a scaled camera
view with a UI layer drawn last at full 64x64 resolution, and on bp35, lf52 and
m0r0 the UI *is* most of what you see - the camera underneath is nearly empty.
Asking for "the upscale factor" produces a confident wrong answer exactly where
it would do most damage.

So this module answers a different and more useful question: **at what
granularity does this game's content actually move?** That is what segmentation
and click enumeration need, and unlike the camera's internal scale it is always
well defined.

The method is propose-then-verify, and neither half has a tuned threshold:

1. **Propose** the cell size as the GCD of every interior colour run inside the
   region that changes across frames. A genuine ``s``-times upscale makes every
   run length in that region a multiple of ``s``.
2. **Verify** that every cell the proposal implies holds exactly one colour in
   *every* observed frame. If not, fall back through the proposal's divisors
   until it does. A cell size of 1 always verifies, so the pipeline is total and
   its failure mode is "no structure claimed", never "wrong structure claimed".

An MDL-over-largest-rectangle formulation was tried first and rejected; see
D-013 for the measurement and why it fails.

Everything here is exact and deterministic (D-006). Nothing is fitted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from math import gcd
from typing import Iterable, Optional, Sequence

import numpy as np

from ..types import GRID_H, GRID_W


@dataclass(frozen=True, slots=True)
class Rect:
    """Half-open pixel rectangle ``[x0, x1) x [y0, y1)``."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def height(self) -> int:
        return max(0, self.y1 - self.y0)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def is_empty(self) -> bool:
        return self.area == 0

    def mask(self) -> np.ndarray:
        m = np.zeros((GRID_H, GRID_W), dtype=bool)
        m[self.y0 : self.y1, self.x0 : self.x1] = True
        return m

    def contains(self, x: int, y: int) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def snapped(self, cell: int, origin_x: int, origin_y: int) -> "Rect":
        """Grown outward to the nearest cell boundaries of the given grid."""
        if cell <= 1:
            return self

        def down(v: int, o: int) -> int:
            return o + ((v - o) // cell) * cell

        def up(v: int, o: int) -> int:
            return o + -((o - v) // cell) * cell

        return Rect(
            max(0, down(self.x0, origin_x)),
            max(0, down(self.y0, origin_y)),
            min(GRID_W, up(self.x1, origin_x)),
            min(GRID_H, up(self.y1, origin_y)),
        )

    def __str__(self) -> str:
        return f"({self.x0},{self.y0})-({self.x1},{self.y1})"


FULL_FRAME = Rect(0, 0, GRID_W, GRID_H)


@dataclass(frozen=True, slots=True)
class BlockGrid:
    """A verified cell size and origin covering ``rect``.

    ``cell == 1`` is the honest null result: no block structure was found, and
    callers must work in pixel space. Check :attr:`is_structured` before using
    cell coordinates for anything.
    """

    cell: int
    origin_x: int
    origin_y: int
    rect: Rect
    proposed_cell: int = 1

    @property
    def is_structured(self) -> bool:
        return self.cell > 1

    @property
    def was_reduced(self) -> bool:
        """The GCD proposal failed verification and had to be coarsened down."""
        return self.proposed_cell != self.cell

    @property
    def cols(self) -> int:
        return self.rect.width // self.cell

    @property
    def rows(self) -> int:
        return self.rect.height // self.cell

    def cell_of(self, x: int, y: int) -> Optional[tuple[int, int]]:
        """Display pixel -> cell coordinate, or None if outside ``rect``."""
        if not self.rect.contains(x, y):
            return None
        return ((x - self.rect.x0) // self.cell, (y - self.rect.y0) // self.cell)

    def centre_of(self, col: int, row: int) -> tuple[int, int]:
        """Cell coordinate -> the display pixel to click for that cell."""
        return (
            self.rect.x0 + col * self.cell + self.cell // 2,
            self.rect.y0 + row * self.cell + self.cell // 2,
        )

    def downsample(self, grid: np.ndarray) -> np.ndarray:
        """``rect`` as a ``(rows, cols)`` grid of cell colours."""
        sub = grid[
            self.rect.y0 : self.rect.y0 + self.rows * self.cell,
            self.rect.x0 : self.rect.x0 + self.cols * self.cell,
        ]
        return np.ascontiguousarray(sub[:: self.cell, :: self.cell])

    def tile_frame(self, grid: np.ndarray) -> tuple[np.ndarray, int, int]:
        """The **whole frame** as cells on this grid's lattice.

        Returns ``(cells, ox, oy)`` where ``ox, oy`` is the display-space origin
        of cell ``(0, 0)``.

        Unlike :meth:`downsample` this is not restricted to :attr:`rect`, because
        static scenery is made of objects too and segmenting only the changing
        region would lose every wall in the game. The cost is that cells outside
        the verified region are represented by their top-left pixel and may not
        be uniform - so downstream code gets a defined answer everywhere and an
        exact one only inside :attr:`rect`.
        """
        ox, oy = self.origin_x, self.origin_y
        rows = (GRID_H - oy) // self.cell
        cols = (GRID_W - ox) // self.cell
        sub = grid[oy : oy + rows * self.cell, ox : ox + cols * self.cell]
        return np.ascontiguousarray(sub[:: self.cell, :: self.cell]), ox, oy

    def __str__(self) -> str:
        if not self.is_structured:
            return "BlockGrid(unstructured)"
        note = f", proposed {self.proposed_cell}" if self.was_reduced else ""
        return (
            f"BlockGrid(cell={self.cell}, origin=({self.origin_x},{self.origin_y}), "
            f"rect={self.rect}, {self.rows}x{self.cols} cells{note})"
        )


UNSTRUCTURED = BlockGrid(cell=1, origin_x=0, origin_y=0, rect=FULL_FRAME, proposed_cell=1)


def _interior_runs(row: np.ndarray) -> list[tuple[int, int]]:
    """``(start, length)`` for each run of equal values, edge runs excluded.

    Runs touching either end are dropped: the frame truncates them, so their
    length is a lower bound rather than a measurement, and feeding a truncated
    length into a GCD would wrongly collapse it toward 1.
    """
    if row.size < 3:
        return []
    changes = np.flatnonzero(row[1:] != row[:-1]) + 1
    if changes.size == 0:
        return []
    edges = np.concatenate(([0], changes, [row.size]))
    return [
        (int(a), int(b - a))
        for a, b in pairwise(edges.tolist())
        if a != 0 and b != row.size
    ]


def propose_cell(
    grids: Sequence[np.ndarray], mask: Optional[np.ndarray] = None
) -> tuple[int, int, int]:
    """Propose ``(cell, origin_x, origin_y)`` from run structure inside ``mask``.

    The cell size is the GCD of every interior run length; the origins are the
    most common residue of the run boundaries. This is a *proposal* - it has not
    been verified that the implied cells are single-coloured. Use
    :func:`verify_cell` or, better, :func:`induce_geometry`.
    """
    g = 0
    x_starts: Counter[int] = Counter()
    y_starts: Counter[int] = Counter()

    for grid in grids:
        for transposed in (False, True):
            arr = grid.T if transposed else grid
            m = None if mask is None else (mask.T if transposed else mask)
            for i, row in enumerate(arr):
                if m is not None:
                    valid = m[i]
                    if not valid.any():
                        continue
                for start, length in _interior_runs(row):
                    if m is not None and not valid[start : start + length].all():
                        continue
                    g = gcd(g, length)
                    (y_starts if transposed else x_starts)[start] = (
                        y_starts if transposed else x_starts
                    )[start] + 1

    cell = g or 1
    if cell <= 1:
        return 1, 0, 0
    ox = _dominant_residue(x_starts, cell)
    oy = _dominant_residue(y_starts, cell)
    return cell, ox, oy


def _dominant_residue(starts: Counter[int], cell: int) -> int:
    if not starts:
        return 0
    residues: Counter[int] = Counter()
    for pos, n in starts.items():
        residues[pos % cell] += n
    return residues.most_common(1)[0][0]


def _block_view(arr: np.ndarray, cell: int, ox: int, oy: int) -> Optional[np.ndarray]:
    """``arr`` reshaped to ``(rows, cell, cols, cell)`` anchored at ``(ox, oy)``."""
    rows = (GRID_H - oy) // cell
    cols = (GRID_W - ox) // cell
    if rows < 1 or cols < 1:
        return None
    sub = arr[oy : oy + rows * cell, ox : ox + cols * cell]
    return sub.reshape(rows, cell, cols, cell)


def verify_cell(
    grids: Sequence[np.ndarray],
    cell: int,
    ox: int,
    oy: int,
    mask: Optional[np.ndarray] = None,
) -> bool:
    """Is every cell of the implied grid a single colour, in every frame?

    This is the property downstream code actually depends on: a cell that spans
    two colours makes a downsampled grid a lie, and every component and track
    built on it inherits the lie.

    Only cells lying **entirely** inside ``mask`` are judged. A cell straddling
    the mask boundary mixes content we are modelling with content we are not, so
    it is not evidence either way - counting it would let a single-pixel UI
    element veto a real 8x upscale, which is exactly the failure that made the
    first version of this function reject every game.
    """
    if cell <= 1:
        return True
    for grid in grids:
        blocks = _block_view(np.asarray(grid), cell, ox, oy)
        if blocks is None:
            return False
        uniform = (blocks == blocks[:, :1, :, :1]).all(axis=(1, 3))
        if mask is not None:
            mblocks = _block_view(mask, cell, ox, oy)
            assert mblocks is not None
            judged = mblocks.all(axis=(1, 3))
        else:
            judged = np.ones_like(uniform, dtype=bool)
        if not judged.any():
            return False
        if not bool(uniform[judged].all()):
            return False
    return True


def _divisors_descending(n: int) -> list[int]:
    return [d for d in range(n, 0, -1) if n % d == 0]


@dataclass(frozen=True, slots=True)
class Geometry:
    """What can be said about a frame's structure from one or more observations.

    ``static_mask`` and ``active_rect`` are ``None`` when only one frame was
    available. They are by far the strongest signal here and they cost nothing
    beyond frames we already have, so prefer passing a whole rollout.
    """

    block: BlockGrid
    border_colour: Optional[int]
    border_rect: Rect
    static_mask: Optional[np.ndarray] = None
    active_rect: Optional[Rect] = None
    n_frames: int = 1

    @property
    def is_structured(self) -> bool:
        return self.block.is_structured

    @property
    def playfield(self) -> Rect:
        """Best estimate of where the game happens.

        Change evidence beats block structure beats the trimmed border, in that
        order, because that is the order of how much each can be trusted.
        """
        if self.active_rect is not None and not self.active_rect.is_empty:
            return self.active_rect
        if self.block.is_structured:
            return self.block.rect
        return self.border_rect

    def click_targets(self) -> list[tuple[int, int]]:
        """One display pixel per playfield cell - D-005's pruning, made concrete.

        Restated in playfield cells as D-011 requires. Returns an empty list when
        geometry found no structure, so every caller must keep a pixel-space
        path; on the public demo games that is roughly half of them.
        """
        if not self.block.is_structured:
            return []
        pf = self.playfield
        out: list[tuple[int, int]] = []
        for row in range(self.block.rows):
            for col in range(self.block.cols):
                x, y = self.block.centre_of(col, row)
                if pf.contains(x, y):
                    out.append((x, y))
        return out

    def describe(self) -> str:
        parts = [str(self.block)]
        if self.border_colour is not None:
            parts.append(f"border={self.border_colour}@{self.border_rect}")
        if self.active_rect is not None:
            parts.append(f"active={self.active_rect}")
        parts.append(f"n_frames={self.n_frames}")
        return "  ".join(parts)


def _trim_uniform_border(grid: np.ndarray) -> tuple[Optional[int], Rect]:
    """Strip a solid single-colour frame border, if there is one."""
    x0, y0, x1, y1 = 0, 0, GRID_W, GRID_H
    colour: Optional[int] = None
    while x1 - x0 > 2 and y1 - y0 > 2:
        edges = np.concatenate(
            [
                grid[y0, x0:x1].ravel(),
                grid[y1 - 1, x0:x1].ravel(),
                grid[y0:y1, x0].ravel(),
                grid[y0:y1, x1 - 1].ravel(),
            ]
        )
        vals = np.unique(edges)
        if vals.size != 1 or (colour is not None and int(vals[0]) != colour):
            break
        colour = int(vals[0])
        x0, y0, x1, y1 = x0 + 1, y0 + 1, x1 - 1, y1 - 1
    return colour, Rect(x0, y0, x1, y1)


def bounding_rect(mask: np.ndarray) -> Rect:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return Rect(0, 0, 0, 0)
    return Rect(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def induce_geometry(
    frames: Sequence[np.ndarray] | Iterable[np.ndarray],
) -> Geometry:
    """Induce geometry from one or more frames of the same game.

    Pass every frame available, including the intermediate frames of animated
    transitions - they are free, and they move exactly the things that matter.
    """
    grids = [np.asarray(f) for f in frames]
    if not grids:
        raise ValueError("induce_geometry needs at least one frame")

    reference = grids[0]
    border_colour, border_rect = _trim_uniform_border(reference)

    static_mask: Optional[np.ndarray] = None
    active_rect: Optional[Rect] = None
    changed: Optional[np.ndarray] = None
    if len(grids) > 1:
        stack = np.stack(grids)
        changed = (stack != stack[0]).any(axis=0)
        static_mask = ~changed
        if changed.any():
            active_rect = bounding_rect(changed)
        else:
            changed = None

    # Propose from the *mask* of changing pixels, not its bounding box: on games
    # whose UI layer paints at full resolution the box contains 1-pixel detail
    # that drags the GCD to 1, while the mask itself does not.
    #
    # Deliberately the whole changing mask, not its largest connected piece.
    # Restricting to the biggest region was tried: it rescues cn04, where a
    # small animating UI element vetoes the real 3x upscale, but it breaks ft09
    # (2 -> 6) and sc25 (1 -> 2) because a smaller region contains fewer runs to
    # constrain the GCD, and the estimate drifts upward. Overestimating is the
    # dangerous direction - a too-coarse cell silently merges genuinely distinct
    # game cells - so the variant with fewer overestimates wins. See D-013.
    cell, ox, oy = propose_cell(grids, changed)

    chosen, proposed = 1, cell
    for candidate in _divisors_descending(cell):
        if candidate == 1 or verify_cell(grids, candidate, ox, oy, changed):
            chosen = candidate
            break

    if chosen > 1:
        rect = active_rect if active_rect is not None else border_rect
        if rect.is_empty:
            rect = FULL_FRAME
        block = BlockGrid(
            cell=chosen,
            origin_x=ox % chosen,
            origin_y=oy % chosen,
            rect=rect.snapped(chosen, ox % chosen, oy % chosen),
            proposed_cell=proposed,
        )
    else:
        block = UNSTRUCTURED

    return Geometry(
        block=block,
        border_colour=border_colour,
        border_rect=border_rect,
        static_mask=static_mask,
        active_rect=active_rect,
        n_frames=len(grids),
    )
