"""The symbolic scene: what the LLM is shown, and what its code runs against (D-035).

One object, two uses, deliberately the same:

- :meth:`Scene.describe` renders it as text for a prompt.
- Hypothesis code the LLM writes receives a :class:`Scene` and computes on it.

So the model writes code against exactly the representation it was shown, and
no field appears in one that is absent from the other. Perception stays exact
(D-006): objects are `colour_8` connected components, not a vision model's
reading of a picture.

**What this does not do.**

- **One carving only.** Objects are same-colour 8-connected regions. A
  multi-coloured sprite arrives as several objects (the 2.55 fragmentation of
  §3 in architecture.md); the grid text is there so the model can see through
  that, the object list will not.
- **No identity across frames.** Object ``id`` is an index into this frame's
  list. Tracking (`perception.track`) exists but is not wired into scenes yet.
- **The bookkeeping band is removed**, when known, from both the objects and
  the grid. A goal that is literally "make the counter reach N" is therefore
  invisible to hypotheses, which is a deliberate trade against the far more
  common case of the counter polluting every state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .perception.segment import segment

#: Hex digit per colour, the grid encoding used in prompts.
HEX = "0123456789abcdef"

#: Objects listed in a prompt, largest first. Hypothesis code sees all of them.
PROMPT_OBJECTS = 40

#: Objects smaller than this are omitted from the prompt list (still in code).
PROMPT_MIN_SIZE = 2


@dataclass(frozen=True)
class Obj:
    """One object: a same-colour 8-connected region."""

    id: int
    colour: int
    size: int
    x0: int
    y0: int
    x1: int  # exclusive
    y1: int  # exclusive
    cx: float
    cy: float

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    def describe(self) -> str:
        return (f"#{self.id} colour {self.colour} size {self.size} "
                f"box x{self.x0}-{self.x1 - 1} y{self.y0}-{self.y1 - 1} ({self.w}x{self.h})")


@dataclass
class Scene:
    """A frame as objects plus the raw grid. Built by :func:`make_scene`."""

    grid: np.ndarray            # 64x64 int, band rows set to -1
    background: int
    objects: list[Obj]
    band_rows: tuple[int, ...] = ()
    _by_colour: dict = field(default_factory=dict, repr=False)

    # -- helpers the LLM's code may use -----------------------------------------

    def of_colour(self, colour: int) -> list[Obj]:
        if not self._by_colour:
            for o in self.objects:
                self._by_colour.setdefault(o.colour, []).append(o)
        return list(self._by_colour.get(colour, []))

    def count(self, colour: int) -> int:
        """Pixels of ``colour`` outside the band."""
        return int((self.grid == colour).sum())

    def colours(self) -> list[int]:
        return sorted(int(c) for c in np.unique(self.grid) if c >= 0)

    # -- rendering -------------------------------------------------------------------

    def crop(self) -> tuple[int, int, int, int]:
        """Bounding box (x0, y0, x1, y1) of everything that is not background."""
        fg = (self.grid != self.background) & (self.grid >= 0)
        if not fg.any():
            return 0, 0, self.grid.shape[1], self.grid.shape[0]
        ys, xs = np.nonzero(fg)
        return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    def grid_text(self) -> str:
        """The non-background crop as hex rows, with coordinates."""
        x0, y0, x1, y1 = self.crop()
        lines = [f"(grid crop x{x0}-{x1 - 1}, y{y0}-{y1 - 1}; one hex digit per cell, "
                 f"'.' = background {self.background}, '_' = hidden UI row)"]
        for y in range(y0, y1):
            row = self.grid[y, x0:x1]
            if (row < 0).all():
                lines.append(f"y{y:02d} " + "_" * (x1 - x0))
                continue
            lines.append(f"y{y:02d} " + "".join(
                "." if v == self.background else ("_" if v < 0 else HEX[v]) for v in row))
        return "\n".join(lines)

    def describe(self, with_grid: bool = True) -> str:
        shown = [o for o in self.objects if o.size >= PROMPT_MIN_SIZE][:PROMPT_OBJECTS]
        parts = [f"background colour {self.background}; {len(self.objects)} objects "
                 f"(showing {len(shown)} largest)"]
        parts += ["  " + o.describe() for o in shown]
        if with_grid:
            parts.append(self.grid_text())
        return "\n".join(parts)


def make_scene(frame: np.ndarray, band: Optional[np.ndarray] = None) -> Scene:
    """Build a scene from a resting frame. ``band`` is a 64x64 bool mask of UI rows."""
    grid = np.asarray(frame, dtype=np.int16).copy()
    rows: tuple[int, ...] = ()
    if band is not None and band.any():
        rows = tuple(int(r) for r in np.nonzero(band.any(axis=1))[0])
        grid[band] = -1
    seg = segment(np.asarray(frame), include=("colour_8",))["colour_8"]
    bg = seg.background_colour if seg.background_colour is not None else 0
    objs: list[Obj] = []
    comps = [c for c in seg.components if c.dominant_colour != bg]
    if band is not None and band.any():
        comps = [c for c in comps if not band[c.mask].all()]
    comps.sort(key=lambda c: (-c.size, c.bbox.y0, c.bbox.x0))
    for i, c in enumerate(comps):
        objs.append(Obj(
            id=i, colour=int(c.dominant_colour), size=int(c.size),
            x0=int(c.bbox.x0), y0=int(c.bbox.y0), x1=int(c.bbox.x1), y1=int(c.bbox.y1),
            cx=round(float(c.centroid[0]), 1), cy=round(float(c.centroid[1]), 1),
        ))
    return Scene(grid=grid, background=int(bg), objects=objs, band_rows=rows)
