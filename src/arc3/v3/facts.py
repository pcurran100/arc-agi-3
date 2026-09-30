"""Exact facts about the frames, recomputed every step (D-039, part 1).

The LLM reasons; this module sees. Everything here is computed, never guessed:
it is the evidence the model's hypotheses must explain and the referee that
scores them.

**What it computes.**

- *Objects with stable ids.* `colour_8` components (background excluded,
  bookkeeping band excluded) tracked across frames by the global-assignment
  tracker (D-014), so "object #3" means the same thing from step to step.
- *Changes per step,* typed: ``moved`` (with dx, dy), ``recoloured`` (from,
  to), ``grew`` / ``shrank`` (size before, after), ``reshaped``, ``appeared``,
  ``vanished``. One object can have several at once (moved and reshaped).
- *Relations:* same-colour groups, same-shape groups (translation-invariant
  shape identity), objects never changed so far (static), pairs that moved with
  the same displacement on the same step (co-motion / tandem).
- *Action effects:* for every action (clicks keyed by the clicked object), how
  often it was tried, how often it did nothing, and what it did to what.
- *Player candidates:* objects moved by two or more different actions with a
  consistent displacement per action — the signature of something controlled.

**What this does not do.**

- *No interpretation.* It will say "#3 moved (0,-5) on all 7 ACTION1 steps and
  never moved otherwise"; it will not say "#3 is the player". That is a
  hypothesis, and hypotheses are the LLM's (Guardrail 2).
- *One carving.* Multi-coloured sprites arrive as several objects; co-motion
  facts are how the model can see they belong together.
- *No animation frames.* Each step is judged on its resting frame.
- *Identity is the tracker's best assignment,* not ground truth: 0.8% switch
  rate measured on resting frames (D-014).
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from ..perception.segment import Segmentation, segment
from ..perception.track import Tracker
from ..types import Action

CARVING = "colour_8"


@dataclass(frozen=True)
class Obj:
    """One object in one frame. Also the element of a world-model state."""

    id: int
    colour: int
    x: int          # bounding box, x1/y1 exclusive
    y: int
    w: int
    h: int
    size: int
    shape: str      # short translation-invariant shape identity

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def describe(self) -> str:
        return (f"#{self.id} colour {self.colour} at x{self.x} y{self.y} "
                f"size {self.w}x{self.h} ({self.size} cells) shape {self.shape}")


@dataclass(frozen=True)
class Change:
    id: int
    kinds: tuple[str, ...]      # subset of KINDS
    dx: int = 0
    dy: int = 0
    colour_from: Optional[int] = None
    colour_to: Optional[int] = None
    size_from: Optional[int] = None
    size_to: Optional[int] = None

    def describe(self) -> str:
        bits = []
        for k in self.kinds:
            if k == "moved":
                bits.append(f"moved ({self.dx:+d},{self.dy:+d})")
            elif k == "recoloured":
                bits.append(f"recoloured {self.colour_from}->{self.colour_to}")
            elif k in ("grew", "shrank"):
                bits.append(f"{k} {self.size_from}->{self.size_to}")
            else:
                bits.append(k)
        return f"#{self.id} " + ", ".join(bits)


KINDS = ("moved", "recoloured", "grew", "shrank", "reshaped", "appeared", "vanished")


@dataclass
class Step:
    t: int
    action: str
    clicked: Optional[int]          # id of the clicked object, for clicks
    before: dict[int, Obj]
    after: dict[int, Obj]
    changes: list[Change]
    reset: bool = False

    @property
    def noop(self) -> bool:
        return not self.changes


def _shape_id(key: bytes) -> str:
    return hashlib.blake2b(key, digest_size=3).hexdigest()


def action_name(a: Action) -> str:
    return f"CLICK({a.x},{a.y})" if a.is_click else ("RESET" if a.is_reset else f"ACTION{a.id}")


@dataclass
class FactBook:
    """All facts for one level. Feed it every resting frame with :meth:`observe`."""

    band: Optional[np.ndarray] = None       # bool mask of bookkeeping rows
    tracker: Tracker = field(default_factory=lambda: Tracker(CARVING))
    steps: list[Step] = field(default_factory=list)
    objects: dict[int, Obj] = field(default_factory=dict)
    background: int = 0
    _t: int = 0

    # -- observing ---------------------------------------------------------

    def _objects_of(self, frame: np.ndarray) -> dict[int, Obj]:
        seg = segment(np.asarray(frame), include=(CARVING,))[CARVING]
        bg = seg.background_colour if seg.background_colour is not None else 0
        self.background = int(bg)
        keep = []
        for c in seg.components:
            if c.dominant_colour == bg:
                continue
            if self.band is not None and self.band.any() and self.band[c.mask].all():
                continue
            keep.append(c)
        comps = [replace(c, index=i) for i, c in enumerate(keep)]
        self.tracker.update(Segmentation(name=CARVING, components=tuple(comps),
                                         cell=seg.cell, background_colour=bg))
        ids = self.tracker.current_ids()
        out = {}
        for c in comps:
            tid = ids.get(c.index)
            if tid is None:
                continue
            out[tid] = Obj(id=tid, colour=int(c.dominant_colour), x=int(c.bbox.x0),
                           y=int(c.bbox.y0), w=int(c.bbox.x1 - c.bbox.x0),
                           h=int(c.bbox.y1 - c.bbox.y0), size=int(c.size),
                           shape=_shape_id(c.shape_key))
        return out

    def observe(self, frame: np.ndarray, action: Optional[Action] = None) -> Optional[Step]:
        """Record the frame that followed ``action`` (None for the first frame)."""
        before = self.objects
        after = self._objects_of(frame)
        self.objects = after
        if action is None:
            return None
        changes = diff_objects(before, after)
        clicked = None
        if action.is_click:
            clicked = object_at(before, action.x, action.y)
        step = Step(t=self._t, action=action_name(action), clicked=clicked, before=before,
                    after=after, changes=changes, reset=action.is_reset)
        self._t += 1
        self.steps.append(step)
        return step

    # -- relations ---------------------------------------------------------

    def play_steps(self) -> list[Step]:
        return [s for s in self.steps if not s.reset]

    def static_ids(self) -> list[int]:
        """Objects present since they appeared and never changed on any step."""
        changed = {c.id for s in self.play_steps() for c in s.changes}
        return sorted(i for i in self.objects if i not in changed)

    def colour_groups(self) -> dict[int, list[int]]:
        g: dict[int, list[int]] = defaultdict(list)
        for o in self.objects.values():
            g[o.colour].append(o.id)
        return {c: sorted(ids) for c, ids in g.items() if len(ids) > 1}

    def shape_groups(self) -> dict[str, list[int]]:
        g: dict[str, list[int]] = defaultdict(list)
        for o in self.objects.values():
            g[o.shape].append(o.id)
        return {s: sorted(ids) for s, ids in g.items() if len(ids) > 1}

    def tandem(self, min_times: int = 2) -> list[tuple[int, int, int]]:
        """(a, b, n): a and b moved with the same displacement on n steps, and
        never moved on a step without the other."""
        together: Counter = Counter()
        moved_steps: dict[int, set[int]] = defaultdict(set)
        for s in self.play_steps():
            moves = {c.id: (c.dx, c.dy) for c in s.changes if "moved" in c.kinds}
            for i in moves:
                moved_steps[i].add(s.t)
            ids = sorted(moves)
            for i, a in enumerate(ids):
                for b in ids[i + 1:]:
                    if moves[a] == moves[b]:
                        together[(a, b)] += 1
        out = []
        for (a, b), n in together.items():
            if n >= min_times and moved_steps[a] == moved_steps[b]:
                out.append((a, b, n))
        return sorted(out, key=lambda t: -t[2])

    def action_effects(self) -> dict[str, dict]:
        """Per action key: tries, no-ops, and a Counter of change descriptions.

        Clicks are keyed by the clicked object's colour ("CLICK colour 9"), so
        experience with one blue square transfers to the other blue squares.
        """
        out: dict[str, dict] = {}
        for s in self.play_steps():
            key = s.action
            if s.action.startswith("CLICK"):
                col = s.before[s.clicked].colour if s.clicked in s.before else None
                key = f"CLICK colour {col}" if col is not None else "CLICK background"
            e = out.setdefault(key, {"tries": 0, "noops": 0, "effects": Counter()})
            e["tries"] += 1
            if s.noop:
                e["noops"] += 1
            for c in s.changes:
                who = s.before.get(c.id) or s.after.get(c.id)
                tag = f"colour-{who.colour}" if who else f"#{c.id}"
                e["effects"][f"{tag} " + c.describe().split(" ", 1)[1]] += 1
        return out

    def player_candidates(self) -> list[dict]:
        """Objects moved by >= 2 distinct non-click actions, each with one dominant
        displacement: what a controlled object looks like, not a claim that it is."""
        per: dict[int, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        for s in self.play_steps():
            if s.action.startswith("CLICK"):
                continue
            for c in s.changes:
                if "moved" in c.kinds:
                    per[c.id][s.action][(c.dx, c.dy)] += 1
        out = []
        for i, acts in per.items():
            if len(acts) < 2:
                continue
            summary = {a: cnt.most_common(1)[0][0] for a, cnt in acts.items()}
            consistency = sum(cnt.most_common(1)[0][1] for cnt in acts.values()) / \
                sum(sum(cnt.values()) for cnt in acts.values())
            out.append({"id": i, "moves": summary, "consistency": round(consistency, 2)})
        return sorted(out, key=lambda d: (-len(d["moves"]), -d["consistency"]))

    # -- rendering -----------------------------------------------------------

    def describe(self, max_objects: int = 40, max_effects: int = 6) -> str:
        """The facts as text for a prompt. Every line is exact."""
        objs = sorted(self.objects.values(), key=lambda o: -o.size)
        static = set(self.static_ids())
        moved_n = Counter(c.id for s in self.play_steps() for c in s.changes)
        lines = [f"OBJECTS now ({len(objs)}; background colour {self.background}):"]
        for o in objs[:max_objects]:
            tag = "static" if o.id in static else f"changed on {moved_n[o.id]} steps"
            lines.append(f"  {o.describe()} [{tag}]")
        if len(objs) > max_objects:
            lines.append(f"  ... and {len(objs) - max_objects} smaller objects")
        cg, sg = self.colour_groups(), self.shape_groups()
        if cg:
            lines.append("SAME COLOUR: " + "; ".join(
                f"colour {c}: " + " ".join(f"#{i}" for i in ids[:12]) for c, ids in cg.items()))
        if sg:
            lines.append("SAME SHAPE: " + "; ".join(
                " ".join(f"#{i}" for i in ids[:12]) for ids in list(sg.values())[:8]))
        td = self.tandem()
        if td:
            lines.append("MOVE TOGETHER: " + "; ".join(
                f"#{a} and #{b} ({n} times)" for a, b, n in td[:8]))
        eff = self.action_effects()
        if eff:
            lines.append("ACTION EFFECTS so far:")
            for k, e in sorted(eff.items()):
                top = ", ".join(f"{d} x{n}" for d, n in e["effects"].most_common(max_effects))
                lines.append(f"  {k}: tried {e['tries']}, no effect {e['noops']}"
                             + (f"; {top}" if top else ""))
        pc = self.player_candidates()
        if pc:
            lines.append("CONTROLLED-LOOKING objects (moved by several actions):")
            for p in pc[:3]:
                mv = ", ".join(f"{a}->({d[0]:+d},{d[1]:+d})" for a, d in sorted(p["moves"].items()))
                lines.append(f"  #{p['id']}: {mv} (consistency {p['consistency']})")
        recent = [s for s in self.steps[-6:]]
        if recent:
            lines.append("LAST STEPS:")
            for s in recent:
                ch = "; ".join(c.describe() for c in s.changes[:5]) or "nothing changed"
                if len(s.changes) > 5:
                    ch += f"; +{len(s.changes) - 5} more"
                lines.append(f"  t{s.t} {s.action}: {ch}")
        return "\n".join(lines)


def diff_objects(before: dict[int, Obj], after: dict[int, Obj]) -> list[Change]:
    out = []
    for i, a in after.items():
        b = before.get(i)
        if b is None:
            out.append(Change(id=i, kinds=("appeared",)))
            continue
        kinds = []
        dx, dy = a.x - b.x, a.y - b.y
        if dx or dy:
            kinds.append("moved")
        if a.colour != b.colour:
            kinds.append("recoloured")
        if a.size > b.size:
            kinds.append("grew")
        elif a.size < b.size:
            kinds.append("shrank")
        elif a.shape != b.shape:
            kinds.append("reshaped")
        if kinds:
            out.append(Change(id=i, kinds=tuple(kinds), dx=dx, dy=dy,
                              colour_from=b.colour, colour_to=a.colour,
                              size_from=b.size, size_to=a.size))
    for i in before:
        if i not in after:
            out.append(Change(id=i, kinds=("vanished",)))
    return sorted(out, key=lambda c: c.id)


def object_at(objs: dict[int, Obj], x: Optional[int], y: Optional[int]) -> Optional[int]:
    """The smallest object whose box contains (x, y) — the likeliest click target."""
    if x is None or y is None:
        return None
    hits = [o for o in objs.values() if o.x <= x < o.x + o.w and o.y <= y < o.y + o.h]
    return min(hits, key=lambda o: o.size).id if hits else None
