"""v5's fact layer: object permanence, interface classification, an object sheet (D-046).

v4 handed the model an event log ("#0 moved (-1,+0), grew 9->10" on every step)
and let it rebuild what each object is. The 2026-10-29 five-game run showed the
cost: the model reasoned about a step counter as a puzzle piece, and spent its
thinking on per-step bookkeeping. This layer keeps the state of every object in
code and shows the model one line per object.

**What it adds to ``FactBook4``.**

- *Deformation is not motion.* v3 reports ``moved`` whenever the bounding box's
  corner shifts, so a bar growing leftwards "moved (-1,+0)". Measured with
  ``scripts/eval_identity.py`` (80 scripted actions per game, seed 0, v4 fact
  layer): **1269 of 2478 reported moves were deformations in place**. A change
  whose old and new cells overlap and whose shape changed is now reported as
  grew/shrank/reshaped with the sides that extended or retracted, never as a
  move.
- *Splits and merges* are reported as events ("#5 split: #23 appeared out of
  it"), not as an unexplained appearance.
- *Names.* The model can name an id ("#14 = move counter"); code keeps the name
  for the level and shows it wherever the id appears.
- *Interface classification*, from evidence only (Guardrail 2: no game kinds):
  - **counter**: stays in place (no translation), changes size by the same amount
    on every one of the last ``WINDOW`` steps, extending or retracting on the
    same side each time, under at least two different actions. "The same change
    whatever the action" is what separates it from a growing snake or a moving
    sprite, whose change follows the action.
  - **autonomous mover**: translates on every one of the last ``WINDOW`` steps
    under at least two different actions, with the same displacement whatever
    the action. Game-relevant: it stays on the sheet, flagged.
  Counters leave the object sheet and appear on one INTERFACE line; they are
  auto-predicted (never count against a prediction).
- *An action-effect table* keyed by action, and for clicks by the clicked
  object, without counters and autonomous movers.

**What this does not do.**

- *No claims about meaning.* A counter is "changes the same way whatever you
  do", not "a timer"; the model is told it is probably interface, and may argue
  otherwise.
- *Recolour-only counters* (a digit that changes colour each step) are not
  classified; only size changes count.
- *Identity is still a frame-to-frame assignment.* The tracker now prefers
  same-coloured in-place overlap (perception/track.py), but an object that
  vanishes for several frames and returns elsewhere can still get a new id.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from ..types import Action
from ..v3.facts import Change, Obj, Step
from ..v4.facts import FactBook4, cells_of

WINDOW = 4               # steps of evidence before an object is classified
SHEET_MAX = 30           # objects listed one per line; the rest are grouped


def _sides(b: Obj, a: Obj) -> str:
    """Which edges of the box moved: 'extended left', 'retracted top', ..."""
    out = []
    for name, d_lo, d_hi in (("left", b.x - a.x, 0), ("top", b.y - a.y, 0),
                             ("right", 0, (a.x + a.w) - (b.x + b.w)),
                             ("bottom", 0, (a.y + a.h) - (b.y + b.h))):
        d = d_lo or d_hi
        if d > 0:
            out.append(f"extended {name}")
        elif d < 0:
            out.append(f"retracted {name}")
    return ", ".join(out)


@dataclass
class FactBook5(FactBook4):
    names: dict = field(default_factory=dict)          # id -> name the model gave
    roles: dict = field(default_factory=dict)          # id -> role guess the model gave
    shape_notes: dict = field(default_factory=dict)    # (t, id) -> "extended left"
    events: list = field(default_factory=list)         # (t, text): splits, merges, returns
    _start_size: dict = field(default_factory=dict)

    # -- observing ---------------------------------------------------------

    def observe(self, frame: np.ndarray, action: Optional[Action] = None) -> Optional[Step]:
        prev_masks = dict(self.masks)
        step = super().observe(frame, action)
        if step is None:
            if not self._start_size:
                self._start_size = {i: o.size for i, o in self.objects.items()}
            return None
        if step.reset:
            return step
        self._unmove(step, prev_masks)
        self._splits(step, prev_masks)
        for n in self.notes.get(step.t, []):
            if "level-start" in n or n.startswith("RESTORED"):
                self.events.append((step.t, n))
        return step

    def _unmove(self, step: Step, prev_masks: dict) -> None:
        """A change whose old and new cells overlap and whose shape changed is a
        deformation in place, not a motion."""
        out = []
        for c in step.changes:
            if "moved" not in c.kinds:
                out.append(c)
                continue
            b, a = step.before.get(c.id), step.after.get(c.id)
            if b is None or a is None or b.shape == a.shape:
                out.append(c)
                continue
            bc = cells_of(b, prev_masks.get(c.id))
            ac = cells_of(a, self.masks.get(c.id))
            if not bc & ac:
                out.append(c)
                continue
            kinds = tuple(k for k in c.kinds if k != "moved")
            if not kinds:
                kinds = ("reshaped",)
            self.shape_notes[(step.t, c.id)] = _sides(b, a)
            out.append(replace(c, kinds=kinds, dx=0, dy=0))
        step.changes = out

    def _splits(self, step: Step, prev_masks: dict) -> None:
        appeared = [c.id for c in step.changes if c.kinds == ("appeared",)]
        vanished = [c.id for c in step.changes if c.kinds == ("vanished",)]
        for j in appeared:
            a = step.after[j]
            ac = cells_of(a, self.masks.get(j))
            for i, b in step.before.items():
                if i == j or b.colour != a.colour or i not in step.after:
                    continue
                if cells_of(b, prev_masks.get(i)) & ac:
                    self.events.append((step.t, f"#{i} split: #{j} appeared out of it"))
                    break
        for i in vanished:
            b = step.before[i]
            bc = cells_of(b, prev_masks.get(i))
            for k, a in step.after.items():
                if k == i or a.colour != b.colour:
                    continue
                if len(bc & cells_of(a, self.masks.get(k))) * 2 >= len(bc):
                    self.events.append((step.t, f"#{i} merged into #{k}"))
                    break

    # -- classification ------------------------------------------------------

    def counters(self) -> dict[int, str]:
        """id -> description, for objects that change the same way whatever the action."""
        steps = self.play_steps()[-WINDOW:]
        if len(steps) < WINDOW or len({s.action for s in steps}) < 2:
            return {}
        out = {}
        for i in self.objects:
            sig = set()
            for s in steps:
                c = next((c for c in s.changes if c.id == i), None)
                if c is None or "moved" in c.kinds or not ({"grew", "shrank"} & set(c.kinds)):
                    break
                delta = (c.size_to or 0) - (c.size_from or 0)
                sig.add((delta, self.shape_notes.get((s.t, i), "")))
            else:
                if len(sig) == 1:
                    delta, side = next(iter(sig))
                    out[i] = f"{delta:+d} cells per action" + (f" ({side})" if side else "")
        return out

    def movers(self) -> dict[int, tuple[int, int]]:
        """id -> (dx, dy), for objects that translate the same way whatever the action."""
        steps = self.play_steps()[-WINDOW:]
        if len(steps) < WINDOW or len({s.action for s in steps}) < 2:
            return {}
        out = {}
        for i in self.objects:
            ds = set()
            for s in steps:
                c = next((c for c in s.changes if c.id == i), None)
                if c is None or "moved" not in c.kinds:
                    break
                ds.add((c.dx, c.dy))
            else:
                if len(ds) == 1:
                    out[i] = next(iter(ds))
        return out

    def tickers(self) -> dict[int, int]:
        """v4's interface: objects whose change does not depend on the action.
        Used by v4's checker and no-op test; here, counters and autonomous movers."""
        return {i: WINDOW for i in (*self.counters(), *self.movers())}

    # -- rendering ---------------------------------------------------------------

    def label(self, i: int) -> str:
        n = self.names.get(i)
        return f"#{i}" + (f" '{n}'" if n else "")

    def _history(self) -> dict[int, Counter]:
        """id -> Counter of 'ACTION2: moved (+0,+5)' style effects, counters excluded."""
        h: dict[int, Counter] = defaultdict(Counter)
        for s in self.play_steps():
            act = s.action
            for c in s.changes:
                if act.startswith("CLICK"):
                    key = "clicked" if c.id == s.clicked else (
                        f"CLICK on #{s.clicked}" if s.clicked is not None else "CLICK on background")
                else:
                    key = act
                desc = []
                for k in c.kinds:
                    if k == "moved":
                        desc.append(f"moved ({c.dx:+d},{c.dy:+d})")
                    elif k == "recoloured":
                        desc.append(f"recoloured {c.colour_from}->{c.colour_to}")
                    else:
                        side = self.shape_notes.get((s.t, c.id))
                        desc.append(k + (f" ({side})" if side else ""))
                h[c.id][f"{key}: {', '.join(desc)}"] += 1
        return h

    def interface_line(self) -> str:
        cs = self.counters()
        if not cs:
            return ""
        parts = []
        for i, d in sorted(cs.items()):
            o = self.objects[i]
            s0 = self._start_size.get(i)
            parts.append(f"{self.label(i)} colour {o.colour} at x{o.x} y{o.y} {o.w}x{o.h}, "
                         f"{o.size} cells now" + (f" ({s0} at level start)" if s0 else "")
                         + f", {d}")
        return ("INTERFACE (changes the same way on every action, whatever the action - most "
                "likely a move counter or timer drawn on the screen, not a puzzle piece; it is "
                "left out of the object sheet and never counts against a prediction): "
                + "; ".join(parts))

    def object_sheet(self) -> str:
        cs, mv = self.counters(), self.movers()
        hist = self._history()
        objs = [o for o in self.objects.values() if o.id not in cs]
        changed = [o for o in objs if hist.get(o.id)]
        still = [o for o in objs if not hist.get(o.id)]
        named_still = [o for o in still if o.id in self.names]
        rest = [o for o in still if o.id not in self.names]
        rest.sort(key=lambda o: -o.size)
        lines = [f"OBJECT SHEET ({len(objs)} objects; background colour {self.background}; "
                 "x = column, y = row of the top-left corner):"]
        for o in sorted(changed, key=lambda o: -sum(hist[o.id].values())) + named_still:
            bits = [f"{self.label(o.id)} colour {o.colour} at x{o.x} y{o.y} {o.w}x{o.h} "
                    f"({o.size} cells)"]
            if o.id in self.roles:
                bits.append(f"role: {self.roles[o.id]}")
            if o.id in mv:
                bits.append(f"MOVES ON ITS OWN ({mv[o.id][0]:+d},{mv[o.id][1]:+d}) every step")
            if hist.get(o.id):
                bits.append("; ".join(f"{k} x{n}" for k, n in hist[o.id].most_common(5)))
            lines.append("  " + " | ".join(bits))
        room = max(0, SHEET_MAX - len(lines) + 1)
        for o in rest[:room]:
            lines.append(f"  {self.label(o.id)} colour {o.colour} at x{o.x} y{o.y} {o.w}x{o.h} "
                         f"({o.size} cells) | never changed")
        if len(rest) > room:
            by_col = Counter(o.colour for o in rest[room:])
            lines.append("  ... and never-changed small objects: " + ", ".join(
                f"{n} of colour {c}" for c, n in by_col.most_common()))
        return "\n".join(lines)

    def effect_table(self, max_effects: int = 5) -> str:
        """{action (clicks: the clicked object) -> observed effects with counts}."""
        skip = set(self.counters())
        rows: dict[str, dict] = {}
        for s in self.play_steps():
            if s.action.startswith("CLICK"):
                key = (f"CLICK {self.label(s.clicked)}" if s.clicked is not None
                       else "CLICK background")
            else:
                key = s.action
            r = rows.setdefault(key, {"n": 0, "none": 0, "eff": Counter()})
            r["n"] += 1
            real = [c for c in s.changes if c.id not in skip]
            if not real:
                r["none"] += 1
            for c in real:
                d = []
                for k in c.kinds:
                    d.append(f"moved ({c.dx:+d},{c.dy:+d})" if k == "moved" else
                             f"recoloured {c.colour_from}->{c.colour_to}" if k == "recoloured"
                             else k)
                r["eff"][f"{self.label(c.id)} {', '.join(d)}"] += 1
        if not rows:
            return ""
        lines = ["ACTION-EFFECT TABLE (what each action did so far, counters left out):"]
        for k, r in sorted(rows.items()):
            top = "; ".join(f"{d} x{n}" for d, n in r["eff"].most_common(max_effects))
            lines.append(f"  {k}: tried {r['n']}, nothing happened {r['none']}"
                         + (f" | {top}" if top else ""))
        return "\n".join(lines)

    def describe(self, max_objects: int = 40, max_effects: int = 5) -> str:
        lines = []
        il = self.interface_line()
        if il:
            lines.append(il)
        lines.append(self.object_sheet())
        td = self.tandem()
        if td:
            lines.append("MOVE TOGETHER: " + "; ".join(
                f"{self.label(a)} and {self.label(b)} ({n} times)" for a, b, n in td[:6]))
        sg = self.shape_groups()
        if sg:
            lines.append("SAME SHAPE: " + "; ".join(
                " ".join(f"#{i}" for i in ids[:10]) for ids in list(sg.values())[:6]))
        sim = self.similar_shapes()
        if sim:
            lines.append("SAME SHAPE UP TO SCALE/ROTATION/REFLECTION: " + "; ".join(sim))
        pc = self.player_candidates()
        if pc:
            lines.append("CONTROLLED-LOOKING (moved differently by different actions): " + "; ".join(
                f"{self.label(p['id'])}: " + ", ".join(
                    f"{a}->({d[0]:+d},{d[1]:+d})" for a, d in sorted(p["moves"].items()))
                for p in pc[:2]))
        et = self.effect_table(max_effects)
        if et:
            lines.append(et)
        lines += self.neighbourhood()
        if self.entered or self.blocked:
            fmt = lambda cnt: ", ".join(("off-grid" if c == -1 else f"colour {c}") + f" x{n}"
                                        for c, n in cnt.most_common(6)) or "none"
            lines.append(f"MOVES THAT HAPPENED entered: {fmt(self.entered)}. MOVES THAT DID NOT "
                         f"HAPPEN had in the way: {fmt(self.blocked)}.")
        if self.prior_moves and not pc:
            lines.append("ON THE LAST LEVEL the controlled object moved: " + ", ".join(
                f"{a}->({d[0]:+d},{d[1]:+d})" for a, d in sorted(self.prior_moves.items())))
        ev = self.events[-6:]
        if ev:
            lines.append("EVENTS: " + "; ".join(f"t{t} {e}" for t, e in ev))
        return "\n".join(lines)
