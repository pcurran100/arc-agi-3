"""The LLM's world model: its state, its sandbox, its verifier, its planner (D-039).

The LLM writes two functions in Python::

    def predict(state, action):     # how the game works
        ...                         # mutate `state` with its helpers and return it
        return state

    def goal(state):                # what winning is
        return 0.0 .. 1.0           # 1.0 = level won (a bool also works)

`predict` runs on a :class:`State` — the objects of one frame (the same ids the
fact layer tracks) plus the frame itself for collision queries. This is the
code-world-model approach: a program that can be *executed*, so it can be
checked against every recorded transition (:func:`verify`), repaired from the
exact mismatches it produces, and *searched* for the shortest path to its own
goal (:func:`plan`) — internal planning, costing no actions.

**What this does not do.**

- *Object level only.* Predictions are compared on each object's box and colour
  and on which objects exist; the exact cells inside a box are not checked. An
  occluded maze whose mask changes as the player passes over it therefore does
  not count as a mispredicted maze.
- *Collision is against the frame the step started from* (`colours_in_box`),
  with the moving objects' own cells removed. Two objects moved in one
  simulated step do not see each other unless the model checks boxes itself.
- *Search is bounded* (`max_nodes`). A goal further than the budget reaches is
  reported as not found, not as unreachable.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Any, Optional

import numpy as np

from ..hypothesis import _SAFE_BUILTINS, _budgeted
from .facts import Obj, Step

#: Line events one predict() or goal() call may execute.
CALL_BUDGET = 20_000


class State:
    """Objects of one frame, mutable, plus the frame for collision queries.

    Objects are plain dicts with keys id, colour, x, y, w, h, size, shape, so
    the model's code can read and write them directly. The helpers are the
    operations games are made of: move, place, recolour, remove, add, and
    queries about what occupies a region.
    """

    def __init__(self, objects: dict[int, dict], grid: Optional[np.ndarray],
                 masks: Optional[dict], background: int, meta: Optional[dict] = None):
        self.objects = objects
        self.grid = grid
        self.masks = masks or {}
        self.background = background
        self.meta = meta if meta is not None else {}   # free-form: counters, flags
        self._origin = {i: (o["x"], o["y"]) for i, o in objects.items()}
        self._next_new = -1

    # -- construction --------------------------------------------------------

    @staticmethod
    def from_objects(objs: dict[int, Obj], grid=None, masks=None, background: int = 0) -> "State":
        return State({i: {"id": o.id, "colour": o.colour, "x": o.x, "y": o.y, "w": o.w,
                          "h": o.h, "size": o.size, "shape": o.shape} for i, o in objs.items()},
                     grid, masks, background)

    @staticmethod
    def before(step: Step, background: int = 0) -> "State":
        return State.from_objects(step.before, step.grid_before, step.masks_before, background)

    def copy(self) -> "State":
        s = State({i: dict(o) for i, o in self.objects.items()}, self.grid, self.masks,
                  self.background, dict(self.meta))
        s._origin = self._origin
        s._next_new = self._next_new
        return s

    def key(self) -> tuple:
        return tuple(sorted((i, o["colour"], o["x"], o["y"], o["w"], o["h"])
                            for i, o in self.objects.items())) + tuple(sorted(
                                (k, v) for k, v in self.meta.items()
                                if isinstance(v, (int, float, str, bool))))

    # -- queries the model may use ---------------------------------------------

    def get(self, i: int) -> Optional[dict]:
        return self.objects.get(i)

    def of_colour(self, c: int) -> list[dict]:
        return sorted((o for o in self.objects.values() if o["colour"] == c), key=lambda o: o["id"])

    def at(self, x: int, y: int) -> Optional[dict]:
        hits = [o for o in self.objects.values()
                if o["x"] <= x < o["x"] + o["w"] and o["y"] <= y < o["y"] + o["h"]]
        return min(hits, key=lambda o: o["size"]) if hits else None

    def colours_in_box(self, x: int, y: int, w: int, h: int, ignore=()) -> set:
        """Colours of the frame inside a box, with the ORIGINAL cells of the
        ``ignore`` ids removed (so a moving object does not collide with itself).
        Out-of-frame area reports -1."""
        out: set = set()
        if self.grid is None:
            return out
        H, W = self.grid.shape
        if x < 0 or y < 0 or x + w > W or y + h > H:
            out.add(-1)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x0 >= x1 or y0 >= y1:
            return out
        region = self.grid[y0:y1, x0:x1]
        keep = np.ones(region.shape, dtype=bool)
        for i in ignore:
            m = self.masks.get(i)
            ox, oy = self._origin.get(i, (None, None))
            if m is None or ox is None:
                continue
            mh, mw = m.shape
            ax0, ay0 = max(x0, ox), max(y0, oy)
            ax1, ay1 = min(x1, ox + mw), min(y1, oy + mh)
            if ax0 < ax1 and ay0 < ay1:
                keep[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0] &= ~m[ay0 - oy:ay1 - oy, ax0 - ox:ax1 - ox]
        out.update(int(v) for v in np.unique(region[keep]))
        return out

    def free(self, x: int, y: int, w: int, h: int, passable=None, ignore=()) -> bool:
        """True if the box holds only ``passable`` colours (default: background)."""
        ok = set(passable) if passable is not None else {self.background}
        return self.colours_in_box(x, y, w, h, ignore) <= ok

    @staticmethod
    def overlaps(a: dict, b: dict) -> bool:
        return (a["x"] < b["x"] + b["w"] and b["x"] < a["x"] + a["w"]
                and a["y"] < b["y"] + b["h"] and b["y"] < a["y"] + a["h"])

    # -- mutations -----------------------------------------------------------------

    def move(self, i: int, dx: int, dy: int) -> "State":
        o = self.objects.get(i)
        if o is not None:
            o["x"] += int(dx)
            o["y"] += int(dy)
        return self

    def place(self, i: int, x: int, y: int) -> "State":
        o = self.objects.get(i)
        if o is not None:
            o["x"], o["y"] = int(x), int(y)
        return self

    def recolour(self, i: int, c: int) -> "State":
        o = self.objects.get(i)
        if o is not None:
            o["colour"] = int(c)
        return self

    def resize(self, i: int, w: int, h: int) -> "State":
        o = self.objects.get(i)
        if o is not None:
            o["w"], o["h"] = int(w), int(h)
            o["size"] = int(w) * int(h)
        return self

    def remove(self, i: int) -> "State":
        self.objects.pop(i, None)
        return self

    def add(self, colour: int, x: int, y: int, w: int, h: int) -> "State":
        """A new object; its id is negative until the real frame names it."""
        i = self._next_new
        self._next_new -= 1
        self.objects[i] = {"id": i, "colour": int(colour), "x": int(x), "y": int(y),
                           "w": int(w), "h": int(h), "size": int(w) * int(h), "shape": "new"}
        return self


# -- the sandboxed model ---------------------------------------------------------


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def strip_fences(code: str) -> str:
    m = _FENCE.search(code or "")
    return m.group(1) if m else (code or "")


def click_xy(action) -> Optional[tuple]:
    """(x, y) of a "CLICK(x,y)" action string, else None. Provided to model code:
    in the 2026-10-30 v5 run a vc33 simulator imported ``re`` to parse clicks and
    did not compile."""
    m = re.fullmatch(r"\s*CLICK\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*", str(action))
    return (int(m.group(1)), int(m.group(2))) if m else None


@dataclass
class WorldModel:
    """`predict` and `goal` compiled from the LLM's code, run in the sandbox."""

    source: str
    predict_fn: Optional[Callable] = None
    goal_fn: Optional[Callable] = None
    error: Optional[str] = None

    @staticmethod
    def compile(predict_src: str, goal_src: str = "") -> "WorldModel":
        src = strip_fences(predict_src) + "\n\n" + strip_fences(goal_src)
        wm = WorldModel(source=src)
        if re.search(r"^\s*(import|from)\s", src, re.M) or "__" in src:
            wm.error = "imports and dunder access are not allowed (np and math are provided)"
            return wm
        ns: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS, "np": np, "math": math,
                              "click_xy": click_xy}
        try:
            exec(compile(src, "<hypothesis>", "exec"), ns)
        except Exception as exc:
            wm.error = f"{type(exc).__name__}: {exc}"
            return wm
        wm.predict_fn = ns.get("predict") if callable(ns.get("predict")) else None
        wm.goal_fn = ns.get("goal") if callable(ns.get("goal")) else None
        if wm.predict_fn is None:
            wm.error = "no predict(state, action) function"
        return wm

    def predict(self, state: State, action: str) -> State:
        """One simulated step. Never raises: a failing model predicts no change."""
        if self.predict_fn is None:
            return state.copy()
        s = state.copy()
        try:
            out = _budgeted(lambda st: self.predict_fn(st, action), s, CALL_BUDGET)
        except Exception as exc:
            self.error = f"predict raised {type(exc).__name__}: {exc}"
            return state.copy()
        return out if isinstance(out, State) else s

    def goal(self, state: State) -> float:
        if self.goal_fn is None:
            return 0.0
        try:
            v = _budgeted(self.goal_fn, state, CALL_BUDGET)
            v = float(v)
            return min(1.0, max(0.0, v)) if math.isfinite(v) else 0.0
        except Exception:
            return 0.0


# -- verification -----------------------------------------------------------------


@dataclass
class Check:
    """How well a world model explains the recorded steps of a level."""

    n: int = 0
    exact: int = 0
    counterexamples: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.exact / self.n if self.n else 0.0

    def describe(self) -> str:
        return f"{self.exact}/{self.n} steps predicted exactly ({100 * self.accuracy:.0f}%)"


def compare(pred: State, after: dict[int, Obj], ignore=()) -> list[str]:
    """Mismatches between a predicted state and the observed objects, as text."""
    ign = set(ignore)
    out = []
    new_pred = [o for i, o in pred.objects.items() if i < 0]
    for i, a in after.items():
        if i in ign:
            continue
        p = pred.objects.get(i)
        if p is None:
            # An object the model did not keep: it was either not predicted to
            # survive, or it is new. New ones may match a predicted add().
            hit = next((q for q in new_pred if q["colour"] == a.colour and q["x"] == a.x
                        and q["y"] == a.y), None)
            if hit is not None:
                new_pred.remove(hit)
                continue
            out.append(f"#{i} (colour {a.colour}) exists at x{a.x} y{a.y}; predicted absent")
            continue
        diffs = []
        if (p["x"], p["y"]) != (a.x, a.y):
            diffs.append(f"at x{p['x']} y{p['y']} -> actually x{a.x} y{a.y}")
        if p["colour"] != a.colour:
            diffs.append(f"colour {p['colour']} -> actually {a.colour}")
        if (p["w"], p["h"]) != (a.w, a.h):
            diffs.append(f"size {p['w']}x{p['h']} -> actually {a.w}x{a.h}")
        if diffs:
            out.append(f"#{i} predicted " + "; ".join(diffs))
    for i, p in pred.objects.items():
        if i >= 0 and i not in after and i not in ign:
            out.append(f"#{i} (colour {p['colour']}) predicted present; actually vanished")
    for q in new_pred:
        out.append(f"predicted a new colour-{q['colour']} object at x{q['x']} y{q['y']}; none appeared")
    return out


def verify(model: WorldModel, steps: list[Step], ignore=(), background: int = 0,
           max_examples: int = 6) -> Check:
    """Replay the model on every recorded non-reset step of the level."""
    chk = Check()
    for s in steps:
        if s.reset or s.grid_before is None:
            continue
        chk.n += 1
        pred = model.predict(State.before(s, background), s.action)
        miss = compare(pred, s.after, ignore)
        if not miss:
            chk.exact += 1
        elif len(chk.counterexamples) < max_examples:
            chk.counterexamples.append(f"t{s.t} {s.action}: " + "; ".join(miss[:4]))
    return chk


# -- planning -----------------------------------------------------------------------


@dataclass
class Plan:
    actions: list[str]
    reached: bool          # does the model say the goal is met at the end?
    progress: float        # the goal's value at the end state
    start_progress: float
    nodes: int


def action_space(state: State, simple: list[int], click: bool, max_clicks: int = 40) -> list[str]:
    acts = [f"ACTION{a}" for a in simple]
    if click:
        objs = sorted(state.objects.values(), key=lambda o: -o["size"])[:max_clicks]
        acts += [f"CLICK({o['x'] + o['w'] // 2},{o['y'] + o['h'] // 2})" for o in objs]
    return acts


def plan(model: WorldModel, start: State, simple: list[int], click: bool,
         max_nodes: int = 4000, max_depth: int = 120, bfs_nodes: int = 20_000) -> Plan:
    """The shortest path to the model's goal, searched entirely inside the model.

    Phase 1 is breadth-first up to ``bfs_nodes`` expansions: the first goal
    state it reaches is at the fewest actions, which is what RHAE's quadratic
    penalty rewards. Greedy best-first was the first version and returned 19
    moves where 17 were optimal (tests/test_v3_world.py). Phase 2, only when the
    state space outgrows phase 1, is best-first on the goal's progress, which
    finds *a* path in large spaces without promising the shortest; if that
    finds none either, the path to the best state seen is returned, flagged
    not reached.
    """
    found = _bfs(model, start, simple, click, bfs_nodes, max_depth)
    if found is not None:
        return found
    p0 = model.goal(start)
    best = (p0, [])
    seen = {start.key()}
    heap: list = [(-p0, 0, 0, [], start)]
    tie = 0
    nodes = 0
    while heap and nodes < max_nodes:
        negp, depth, _, path, s = heappop(heap)
        nodes += 1
        if -negp >= 0.99:
            return Plan(path, True, -negp, p0, nodes)
        if depth >= max_depth:
            continue
        for a in action_space(s, simple, click):
            t = model.predict(s, a)
            k = t.key()
            if k in seen:
                continue
            seen.add(k)
            p = model.goal(t)
            tie += 1
            np_ = path + [a]
            if p > best[0] or (p == best[0] and len(np_) < len(best[1]) and best[1]):
                best = (p, np_)
            heappush(heap, (-p, depth + 1, tie, np_, t))
    return Plan(best[1], best[0] >= 0.99, best[0], p0, nodes)


def _bfs(model: WorldModel, start: State, simple: list[int], click: bool,
         max_nodes: int, max_depth: int) -> Optional[Plan]:
    """Breadth-first to the nearest goal state; None if the budget ran out first.

    Returns a not-reached Plan (empty path) when the reachable space is
    exhausted without a goal: the model then says the goal is unreachable,
    which is itself evidence against model or goal.
    """
    from collections import deque

    p0 = model.goal(start)
    if p0 >= 0.99:
        return Plan([], True, p0, p0, 0)
    seen = {start.key()}
    q = deque([(start, [])])
    nodes = 0
    while q:
        if nodes >= max_nodes:
            return None
        s, path = q.popleft()
        nodes += 1
        if len(path) >= max_depth:
            continue
        for a in action_space(s, simple, click):
            t = model.predict(s, a)
            k = t.key()
            if k in seen:
                continue
            seen.add(k)
            p = model.goal(t)
            if p >= 0.99:
                return Plan(path + [a], True, p, p0, nodes)
            q.append((t, path + [a]))
    return Plan([], False, p0, p0, nodes)


def parse_action(a: str):
    """'ACTION3' -> (3, None, None); 'CLICK(10,20)' -> (6, 10, 20); else None."""
    a = a.strip().upper().replace(" ", "")
    m = re.fullmatch(r"ACTION([1-7])", a)
    if m:
        return int(m.group(1)), None, None
    m = re.fullmatch(r"CLICK\((\d+),(\d+)\)", a)
    if m:
        return 6, int(m.group(1)), int(m.group(2))
    if a == "RESET":
        return 0, None, None
    return None
