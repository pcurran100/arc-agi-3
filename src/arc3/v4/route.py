"""Subgoals and the routes code computes to them (D-044).

The model says *where* to go ("bring #7 onto #3") and what should happen on
arrival; code works out *how*: a breadth-first search over the positions of the
controlled object, using only what the fact layer has measured -

- each action's displacement of that object (``FactBook4.player_candidates``:
  the dominant displacement per action, from moves that happened), and
- which colours stop it: colours that were in the way of a move that did not
  happen and were never entered by one that did (``blocked - entered``).
  Colours never met either way are treated as passable; a wrong guess is a
  failed expectation on the step that meets it, which re-plans with one more
  colour learned.
- and the exact places where a move was refused (``blocked_moves``: the
  object's position and the action), because colour is not always what blocks.

A subgoal is

    {"object": "#7", "to": "#3" | "x,y", "relation": "overlap" | "touch" | "at",
     "on_arrival": <expectation, expect.py>, "goal": "A5" | ""}

- ``overlap``: the object's group box intersects the target's box;
- ``touch``: it intersects the target's box grown by one cell;
- ``at``: the object's top-left corner is exactly at the target (x,y).

**What this does not do.**

- *Only translation.* Actions that rotate, resize or teleport the object are not
  modelled; a subgoal that needs them gets no route and goes back to the model.
- *No moving obstacles,* no pushing, no state other than the object's position:
  the search runs against the current frame as if everything else stays put.
- *No clicks.* Click games plan in the model's own steps.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .expect import Expectation
from .facts import FactBook4, cells_of

MAX_NODES = 20000


@dataclass
class Subgoal:
    obj: int
    to_obj: Optional[int] = None
    to_xy: Optional[tuple[int, int]] = None
    relation: str = "overlap"
    on_arrival: Expectation = Expectation()
    goal: str = ""
    why: str = ""

    @staticmethod
    def parse(d) -> Optional["Subgoal"]:
        if not isinstance(d, dict):
            return None
        obj = _id(d.get("object"))
        if obj is None:
            return None
        to = str(d.get("to", "")).strip()
        to_obj, to_xy = _id(to), None
        if to_obj is None:
            try:
                x, y = (int(v) for v in to.replace("(", "").replace(")", "").split(","))
                to_xy = (x, y)
            except ValueError:
                return None
        rel = d.get("relation") if d.get("relation") in ("overlap", "touch", "at") else "overlap"
        return Subgoal(obj, to_obj, to_xy, rel, Expectation.parse(d.get("on_arrival")),
                       str(d.get("goal", "") or "")[:8], str(d.get("why", "") or "")[:200])

    def describe(self) -> str:
        tgt = f"#{self.to_obj}" if self.to_obj is not None else f"({self.to_xy[0]},{self.to_xy[1]})"
        return (f"bring #{self.obj} to {self.relation} {tgt}"
                + (f", expecting on arrival: {self.on_arrival.describe()}"
                   if self.on_arrival.informative else ""))


def _id(v) -> Optional[int]:
    s = str(v or "").strip()
    if s.startswith("#"):
        try:
            return int(s[1:])
        except ValueError:
            return None
    return None


def _box(cells) -> tuple[int, int, int, int]:
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def _hit(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def reached(fb: FactBook4, sg: Subgoal) -> bool:
    group = fb.group_of(sg.obj)
    if sg.obj not in fb.objects:
        return False
    cells = set()
    for g in group:
        cells |= cells_of(fb.objects[g], fb.masks.get(g))
    return _satisfied(fb, sg, cells, (0, 0))


def _satisfied(fb: FactBook4, sg: Subgoal, cells: set, off: tuple[int, int]) -> bool:
    ox, oy = off
    if sg.relation == "at" and sg.to_xy is not None:
        o = fb.objects[sg.obj]
        return (o.x + ox, o.y + oy) == sg.to_xy
    x0, y0, x1, y1 = _box(cells)
    box = (x0 + ox, y0 + oy, x1 + ox, y1 + oy)
    if sg.to_obj is not None:
        t = fb.objects.get(sg.to_obj)
        if t is None:
            return False
        tb = (t.x, t.y, t.x + t.w, t.y + t.h)
    else:
        tb = (sg.to_xy[0], sg.to_xy[1], sg.to_xy[0] + 1, sg.to_xy[1] + 1)
    if sg.relation == "touch":
        tb = (tb[0] - 1, tb[1] - 1, tb[2] + 1, tb[3] + 1)
    return _hit(box, tb)


def blocking_colours(fb: FactBook4) -> set:
    return {c for c in fb.blocked if c not in fb.entered}


def route(fb: FactBook4, sg: Subgoal, actions: list[str]) -> tuple[Optional[list[tuple[str, int, int]]], str]:
    """Shortest list of (action, dx, dy) bringing the subgoal's object to its
    target, or (None, reason)."""
    if sg.obj not in fb.objects or fb.grid is None:
        return None, f"#{sg.obj} is not on screen"
    if sg.to_obj is not None and sg.to_obj not in fb.objects:
        return None, f"#{sg.to_obj} is not on screen"
    pc = {p["id"]: p for p in fb.player_candidates()}
    group = fb.group_of(sg.obj)
    p = next((pc[g] for g in group if g in pc), None)
    if fb.prior_moves:
        # The previous level's displacements, as a prior for actions this level
        # has not yet shown: every step of the route is still checked, and a
        # wrong one is a surprise like any other.
        p = {"id": p["id"] if p else sg.obj, "moves": {**fb.prior_moves, **(p["moves"] if p else {})}}
    if p is None:
        return None, (f"#{sg.obj} has not been seen to move under two different actions, so "
                      "code has no movement model for it; give explicit steps instead")
    moves = [(a, int(d[0]), int(d[1])) for a, d in sorted(p["moves"].items())
             if a in actions and (d[0] or d[1])]
    if not moves:
        return None, f"no available action is known to move #{sg.obj}"
    own = set()
    for g in group:
        own |= cells_of(fb.objects[g], fb.masks.get(g))
    grid = fb.grid
    H, W = grid.shape
    stop = blocking_colours(fb)
    exempt = set()
    if sg.to_obj is not None:                # the target itself may be entered
        exempt = cells_of(fb.objects[sg.to_obj], fb.masks.get(sg.to_obj))

    def free(off) -> bool:
        ox, oy = off
        for (x, y) in own:
            nx, ny = x + ox, y + oy
            if not (0 <= nx < W and 0 <= ny < H):
                return False
            if (nx, ny) in own or (nx, ny) in exempt:
                continue
            if int(grid[ny, nx]) in stop:
                return False
        return True

    start = (0, 0)
    if _satisfied(fb, sg, own, start):
        return [], "already there"
    anchor = fb.objects[p["id"]]
    refused = fb.blocked_moves
    prev = {start: None}
    q = deque([start])
    while q and len(prev) < MAX_NODES:
        cur = q.popleft()
        for a, dx, dy in moves:
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in prev or (anchor.x + cur[0], anchor.y + cur[1], a) in refused \
                    or not free(nxt):
                continue
            prev[nxt] = (cur, a, dx, dy)
            if _satisfied(fb, sg, own, nxt):
                path = []
                n = nxt
                while prev[n] is not None:
                    c, a2, ddx, ddy = prev[n]
                    path.append((a2, ddx, ddy))
                    n = c
                return path[::-1], f"{len(path)} steps"
            q.append(nxt)
    known = ", ".join(f"colour {c}" for c in sorted(stop)) or "none known yet"
    return None, (f"no route found for #{sg.obj} using moves "
                  + ", ".join(f"{a}({dx:+d},{dy:+d})" for a, dx, dy in moves)
                  + f"; colours treated as blocking: {known}")
