"""v4's fact layer: v3's FactBook plus the facts the first live runs lacked (D-044).

Every addition here comes from a measured failure on ls20 (2026-10-19, see
docs/experiments/nim/v4_ls20_80_2026-10-19/), stated generically - none of it
knows what a key, a life or a counter is (Guardrail 2):

- *Occlusion is not a change.* When a moving object covers or uncovers cells of
  a still object, the still object's component changes shape; v3 reported
  "#1 reshaped" on every move and the model built (and retired a correct rule
  over) that artefact. A change whose changed cells all lie under a mover's old
  or new footprint is dropped from the step and counted in ``occluded``.
- *Tickers.* An object that changed in size, colour or shape on every one of the
  last few steps, under at least two different actions, is listed separately:
  what it does does not depend on the action. A step on which only tickers
  changed is a **no-op**: the action itself did nothing.
- *Return to the level start.* An object that moved and now sits exactly where
  it was when the level began, and objects whose whole state is again exactly
  the level-start state, are reported as such (on ls20 this was a lost life,
  read by the model as "capturing a target").
- *Shapes equal up to scale, rotation and reflection.* v3 matched only identical
  shapes, so a 2x rotated copy of a glyph was invisible.
- *What a controlled-looking object would enter.* For each action with a known
  displacement, the colours under the footprint the object would occupy; and
  across the level, which colours were entered by moves that happened and which
  were in the way of moves that did not.

**What this does not do.**

- *No interpretation.* It says "changes on every action whatever the action",
  not "this is a timer".
- *Identity is still the tracker's.* When a bar empties and refills, the tracker
  can swap ids between its spent and live parts (seen on ls20); tickers are
  found per id over a short window, so a swapped id is re-detected within a few
  steps rather than corrected.
- *Footprints are boxes of the object's own mask.* A mover that changes shape
  while moving is judged by its new mask only.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..types import Action
from ..v3.facts import Change, FactBook, Obj, Step

TICKER_WINDOW = 4        # a ticker changed on each of the last this-many steps
SIZE_CHANGES = ("grew", "shrank", "recoloured", "reshaped")

#: The eight symmetries of the square, as (name, function on a 2-D array).
DIHEDRAL = (
    ("", lambda m: m),
    ("rotated 90", lambda m: np.rot90(m, 1)),
    ("rotated 180", lambda m: np.rot90(m, 2)),
    ("rotated 270", lambda m: np.rot90(m, 3)),
    ("mirrored", lambda m: m[:, ::-1]),
    ("mirrored and rotated 90", lambda m: np.rot90(m[:, ::-1], 1)),
    ("mirrored and rotated 180", lambda m: np.rot90(m[:, ::-1], 2)),
    ("mirrored and rotated 270", lambda m: np.rot90(m[:, ::-1], 3)),
)


def reduce_scale(m: np.ndarray) -> tuple[np.ndarray, int]:
    """The mask at the largest integer scale k that it is an exact k-fold
    upscale of, and k."""
    h, w = m.shape
    for k in range(min(h, w), 1, -1):
        if h % k or w % k:
            continue
        small = m[::k, ::k]
        if np.array_equal(np.kron(small, np.ones((k, k), dtype=m.dtype)), m):
            return small, k
    return m, 1


def _key(m: np.ndarray) -> tuple:
    return (m.shape, m.tobytes())


def _effect(c: Change) -> str:
    """A change without exact sizes: 'moved (+5,+0)', 'shrank', 'recoloured 3->8'."""
    bits = []
    for k in c.kinds:
        if k == "moved":
            bits.append(f"moved ({c.dx:+d},{c.dy:+d})")
        elif k == "recoloured":
            bits.append(f"recoloured {c.colour_from}->{c.colour_to}")
        else:
            bits.append(k)
    return ", ".join(bits)


def cells_of(o: Obj, mask: Optional[np.ndarray]) -> set:
    if mask is None:
        return {(o.x + c, o.y + r) for r in range(o.h) for c in range(o.w)}
    ys, xs = np.nonzero(mask)
    return {(o.x + int(c), o.y + int(r)) for r, c in zip(ys, xs)}


@dataclass
class FactBook4(FactBook):
    start: dict = field(default_factory=dict)        # id -> Obj at the level's first frame
    start_colours: set = field(default_factory=set)
    start_shapes: set = field(default_factory=set)
    seen_colours: set = field(default_factory=set)
    seen_shapes: set = field(default_factory=set)
    occluded: int = 0
    notes: dict = field(default_factory=dict)        # step t -> [fact lines about that step]
    entered: Counter = field(default_factory=Counter)   # colours moved into by moves that happened
    blocked: Counter = field(default_factory=Counter)   # colours in the way of moves that did not
    blocked_moves: set = field(default_factory=set)     # (x, y, action) of a mover where it failed

    # -- observing ---------------------------------------------------------

    def observe(self, frame: np.ndarray, action: Optional[Action] = None) -> Optional[Step]:
        prev_masks, prev_grid, prev_objs = self.masks, self.grid, self.objects
        step = super().observe(frame, action)
        if step is None:
            if not self.start:
                self.start = dict(self.objects)
                self.seen_colours = {o.colour for o in self.objects.values()}
                self.seen_shapes = {o.shape for o in self.objects.values()}
            return None
        if step.reset:
            return step
        self._drop_occlusion(step, prev_masks)
        self._learn_passability(step, prev_grid, prev_objs, prev_masks)
        notes = []
        notes += self._returns(step)
        novel = self._novelty(step)
        notes += novel
        self.notes[step.t] = notes
        step.novel = bool(novel)          # type: ignore[attr-defined]
        return step

    def _drop_occlusion(self, step: Step, prev_masks: dict) -> None:
        movers = [c for c in step.changes if "moved" in c.kinds or c.kinds in (("appeared",), ("vanished",))]
        if not movers:
            return
        under: set = set()
        for c in movers:
            b, a = step.before.get(c.id), step.after.get(c.id)
            if b is not None:
                under |= cells_of(b, prev_masks.get(c.id))
            if a is not None:
                under |= cells_of(a, self.masks.get(c.id))
        keep = []
        for c in step.changes:
            if c in movers or not set(c.kinds) <= {"grew", "shrank", "reshaped"}:
                keep.append(c)
                continue
            b, a = step.before[c.id], step.after[c.id]
            diff = cells_of(b, prev_masks.get(c.id)) ^ cells_of(a, self.masks.get(c.id))
            if diff and diff <= under:
                self.occluded += 1
                continue
            keep.append(c)
        step.changes = keep

    def _learn_passability(self, step: Step, prev_grid, prev_objs: dict, prev_masks: dict) -> None:
        """For controlled-looking objects: which colours did a move enter, and which
        were in the way when the action that usually moves it did not."""
        if prev_grid is None or step.action.startswith("CLICK"):
            return
        pcs = {p["id"]: p for p in self.player_candidates()}
        for i, p in pcs.items():
            d = p["moves"].get(step.action)
            if d is None or i not in prev_objs:
                continue
            group = self.group_of(i, prev_objs)
            ch = next((c for c in step.changes if c.id == i and "moved" in c.kinds), None)
            cols = self._entered_colours(prev_grid, prev_objs, prev_masks, group, d)
            if ch is not None and (ch.dx, ch.dy) == tuple(d):
                self.entered.update(cols)
            elif ch is None:
                self.blocked.update(cols)
                # Colour is not always what blocks (ls20: a move refused with
                # floor-coloured cells ahead), so the place is remembered too.
                o = prev_objs[i]
                self.blocked_moves.add((o.x, o.y, step.action))

    # -- derived facts ---------------------------------------------------------

    def player_candidates(self) -> list[dict]:
        """v3's candidates, minus objects that move the same way whatever the
        action: on ls20 a bar shrinking from the left 'moved (+1,0)' under all
        four actions and outranked the real mover."""
        return [p for p in super().player_candidates()
                if len(set(map(tuple, p["moves"].values()))) >= 2]

    def action_effects(self) -> dict[str, dict]:
        """v3's per-action summary without the tickers (listed on their own line)
        and without exact sizes, so one effect is one line, not one per size."""
        tk = self.tickers()
        out: dict[str, dict] = {}
        for s in self.play_steps():
            key = s.action
            if s.action.startswith("CLICK"):
                col = s.before[s.clicked].colour if s.clicked in s.before else None
                key = f"CLICK colour {col}" if col is not None else "CLICK background"
            e = out.setdefault(key, {"tries": 0, "noops": 0, "effects": Counter()})
            e["tries"] += 1
            real = [c for c in s.changes if c.id not in tk]
            if not real:
                e["noops"] += 1
            for c in real:
                who = s.before.get(c.id) or s.after.get(c.id)
                tag = f"colour-{who.colour}" if who else f"#{c.id}"
                e["effects"][f"{tag} " + _effect(c)] += 1
        return out

    def group_of(self, i: int, objs: Optional[dict] = None) -> list[int]:
        objs = self.objects if objs is None else objs
        mates = {b if a == i else a for a, b, _ in self.tandem() if i in (a, b)}
        return sorted({i} | {m for m in mates if m in objs})

    def _entered_colours(self, grid, objs, masks, group, d) -> Counter:
        dx, dy = d
        own = set()
        for g in group:
            if g in objs:
                own |= cells_of(objs[g], masks.get(g))
        out: Counter = Counter()
        H, W = grid.shape
        for (x, y) in own:
            nx, ny = x + dx, y + dy
            if (nx, ny) in own:
                continue
            if not (0 <= nx < W and 0 <= ny < H):
                out[-1] += 1
            else:
                out[int(grid[ny, nx])] += 1
        return out

    def tickers(self) -> dict[int, int]:
        """id -> streak: objects that changed size/colour/shape (not by moving
        alone) on each of the last TICKER_WINDOW steps, under >= 2 actions."""
        steps = self.play_steps()
        if len(steps) < TICKER_WINDOW:
            return {}
        pcs = {p["id"] for p in self.player_candidates()}
        out = {}
        for i in self.objects:
            if i in pcs:
                continue
            streak, acts = 0, set()
            for s in reversed(steps):
                c = next((c for c in s.changes if c.id == i), None)
                if c is None or not set(c.kinds) & set(SIZE_CHANGES):
                    break
                streak += 1
                acts.add(s.action)
            if streak >= TICKER_WINDOW and len(acts) >= 2:
                out[i] = streak
        return out

    def is_noop(self, step: Step) -> bool:
        """Only tickers changed (or nothing did)."""
        tk = self.tickers()
        return all(c.id in tk for c in step.changes)

    def _returns(self, step: Step) -> list[str]:
        out = []
        back = []
        for c in step.changes:
            a, s0 = step.after.get(c.id), self.start.get(c.id)
            if a is None or s0 is None:
                continue
            if "moved" in c.kinds and (a.x, a.y) == (s0.x, s0.y):
                out.append(f"#{c.id} moved ({c.dx:+d},{c.dy:+d}) and is now exactly at its "
                           f"level-start position x{a.x} y{a.y}")
            if (a.colour, a.x, a.y, a.w, a.h, a.size, a.shape) == \
                    (s0.colour, s0.x, s0.y, s0.w, s0.h, s0.size, s0.shape):
                back.append(c.id)
        if len(back) >= 2:
            out.append("RESTORED: " + " ".join(f"#{i}" for i in back)
                       + " are again exactly as they were at the start of the level")
        return out

    def _novelty(self, step: Step) -> list[str]:
        out = []
        for c in step.changes:
            a = step.after.get(c.id)
            if a is None:
                continue
            if a.colour not in self.seen_colours:
                out.append(f"NEW COLOUR: #{c.id} has colour {a.colour}, not seen before on this level")
            elif "appeared" in c.kinds and a.shape not in self.seen_shapes:
                out.append(f"NEW SHAPE: #{c.id} (colour {a.colour}) appeared with a shape not seen "
                           "before on this level")
        for o in step.after.values():
            self.seen_colours.add(o.colour)
            self.seen_shapes.add(o.shape)
        return out

    def similar_shapes(self, min_cells: int = 3) -> list[str]:
        """Pairs of objects whose masks are equal up to integer scale, rotation and
        reflection but not identical. Solid rectangles are left out (every one is
        'similar' to every other)."""
        canon: dict[tuple, list[tuple[int, int, np.ndarray]]] = defaultdict(list)
        for i, o in self.objects.items():
            m = self.masks.get(i)
            if m is None or m.all():
                continue
            small, k = reduce_scale(m)
            if int(small.sum()) < min_cells:
                continue
            ck = min(_key(f(small)) for _, f in DIHEDRAL)
            canon[ck].append((i, k, small))
        out = []
        for members in canon.values():
            if len(members) < 2:
                continue
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    (i, ki, mi), (j, kj, mj) = members[a], members[b]
                    oi, oj = self.objects[i], self.objects[j]
                    if oi.shape == oj.shape:
                        continue                 # already in SAME SHAPE
                    how = next((name for name, f in DIHEDRAL if _key(f(mj)) == _key(mi)), "")
                    scale = (f"x{ki // kj} larger" if ki > kj and ki % kj == 0
                             else f"x{kj // ki} smaller" if kj > ki and kj % ki == 0
                             else f"scale {ki}:{kj}" if ki != kj else "same size")
                    out.append(f"#{i} ~ #{j} ({scale}" + (f", {how}" if how else "") + ")")
        return out[:12]

    def neighbourhood(self) -> list[str]:
        """For the most controlled-looking object: the colours each known move would enter."""
        pcs = self.player_candidates()
        if not pcs or self.grid is None:
            return []
        p = pcs[0]
        group = self.group_of(p["id"])
        lines = []
        for a, d in sorted(p["moves"].items()):
            cols = self._entered_colours(self.grid, self.objects, self.masks, group, d)
            txt = ", ".join(("off-grid" if c == -1 else f"colour {c}") + f" x{n}"
                            for c, n in cols.most_common())
            lines.append(f"  {a} -> ({d[0]:+d},{d[1]:+d}) would enter: {txt or 'nothing new'}")
        head = "group " + " ".join(f"#{g}" for g in group)
        return [f"NEXT TO the most controlled-looking object ({head}):"] + lines

    # -- rendering ---------------------------------------------------------------

    def describe(self, max_objects: int = 40, max_effects: int = 6) -> str:
        lines = [super().describe(max_objects, max_effects)]
        tk = self.tickers()
        if tk:
            lines.append("CHANGE ON EVERY ACTION, WHATEVER THE ACTION (excluded from no-op checks): "
                         + "; ".join(f"#{i} (last {n} steps)" for i, n in sorted(tk.items())))
        sim = self.similar_shapes()
        if sim:
            lines.append("SAME SHAPE UP TO SCALE/ROTATION/REFLECTION: " + "; ".join(sim))
        lines += self.neighbourhood()
        if self.entered or self.blocked:
            fmt = lambda cnt: ", ".join(("off-grid" if c == -1 else f"colour {c}") + f" x{n}"
                                        for c, n in cnt.most_common(6)) or "none"
            lines.append(f"MOVES THAT HAPPENED entered: {fmt(self.entered)}. MOVES THAT DID NOT "
                         f"HAPPEN had in the way: {fmt(self.blocked)}.")
        if self.occluded:
            lines.append(f"({self.occluded} shape changes of still objects were only a mover "
                         "covering or uncovering them, and are not reported as changes)")
        return "\n".join(lines)
