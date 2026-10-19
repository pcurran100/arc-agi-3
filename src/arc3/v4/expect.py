"""Testable predictions: what the LLM says an action will do, checked exactly (D-042).

The act call must say, for every action it chooses, what it expects to happen,
in a small language over the fact layer's change types. Code then checks the
expectation against the step the action produced. A correct prediction is
evidence for the assumptions behind it; a wrong one is a *surprise*, and the
exact difference goes back to the model.

**The language.** An expectation is

    {"changes": [item, ...], "rest": "unchanged" | "any",
     "outcome": "continue" | "level_complete" | "game_over"}

and an item is

    {"target": "#12" | "colour 9" | "any",
     "change": "moved" | "recoloured" | "grew" | "shrank" | "reshaped"
               | "appeared" | "vanished" | "none",
     "dx": int | null, "dy": int | null, "colour": int | null}

- ``#12`` names one object (ids are stable within a level); ``colour 9`` is
  satisfied by *any* object of that colour; ``any`` by any object at all.
- ``change: "none"`` asserts the target did *not* change.
- ``dx``/``dy`` (for moved) and ``colour`` (the new colour, for recoloured, or
  the colour of the new object, for appeared) are checked only when given.
- ``rest: "unchanged"`` asserts nothing outside the named targets changed;
  ``"any"`` makes no claim about the rest.

The same item form is used by assumption *rules* (store.py), which code checks
on every step whose action matches, without asking the model.

**What this does not do.**

- *No positions, only changes.* "The block ends at x=20" is expressed as
  "moved (+4, 0)". Absolute targets would need the model to do coordinate
  arithmetic it is bad at, and changes are what the fact layer measures.
- *No partial credit.* An expectation holds or it does not; the verdict lists
  every failed item, so the model sees exactly which part was wrong.
- *No counting of cells inside an object.* grew/shrank says which way, not by how
  much; the fact text gives the sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..v3.facts import Change, Obj, Step

CHANGES = ("moved", "recoloured", "grew", "shrank", "reshaped", "appeared", "vanished", "none")
OUTCOMES = ("continue", "level_complete", "game_over")

ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {"type": "string", "maxLength": 24},
        "change": {"type": "string", "enum": list(CHANGES)},
        "dx": {"type": ["integer", "null"]},
        "dy": {"type": ["integer", "null"]},
        "colour": {"type": ["integer", "null"]},
    },
    "required": ["target", "change", "dx", "dy", "colour"],
}

EXPECT_SCHEMA = {
    "type": "object",
    "properties": {
        "changes": {"type": "array", "items": ITEM_SCHEMA, "maxItems": 8},
        "rest": {"type": "string", "enum": ["unchanged", "any"]},
        "outcome": {"type": "string", "enum": list(OUTCOMES)},
    },
    "required": ["changes", "rest", "outcome"],
}


@dataclass(frozen=True)
class Item:
    target: str
    change: str
    dx: Optional[int] = None
    dy: Optional[int] = None
    colour: Optional[int] = None

    @staticmethod
    def parse(d: dict) -> Optional["Item"]:
        if not isinstance(d, dict) or d.get("change") not in CHANGES:
            return None
        t = str(d.get("target", "any")).strip().lower()

        def num(k):
            v = d.get(k)
            return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        return Item(target=t, change=d["change"], dx=num("dx"), dy=num("dy"), colour=num("colour"))

    def describe(self) -> str:
        s = f"{self.target} {self.change}"
        if self.change == "moved" and (self.dx is not None or self.dy is not None):
            s += f" ({_fmt(self.dx)},{_fmt(self.dy)})"
        if self.colour is not None and self.change in ("recoloured", "appeared"):
            s += f" to colour {self.colour}" if self.change == "recoloured" else f" colour {self.colour}"
        return s


def _fmt(v: Optional[int]) -> str:
    return "?" if v is None else f"{v:+d}"


@dataclass(frozen=True)
class Expectation:
    items: tuple[Item, ...] = ()
    rest: str = "any"
    outcome: str = "continue"

    @staticmethod
    def parse(d) -> "Expectation":
        if not isinstance(d, dict):
            return Expectation()
        items = tuple(i for i in (Item.parse(x) for x in d.get("changes", []) or []) if i)
        rest = d.get("rest") if d.get("rest") in ("unchanged", "any") else "any"
        outcome = d.get("outcome") if d.get("outcome") in OUTCOMES else "continue"
        return Expectation(items, rest, outcome)

    @property
    def informative(self) -> bool:
        """Does it claim anything at all? An empty, rest-any 'continue' does not."""
        return bool(self.items) or self.rest == "unchanged" or self.outcome != "continue"

    def describe(self) -> str:
        parts = [i.describe() for i in self.items]
        if self.rest == "unchanged":
            parts.append("nothing else changes")
        if self.outcome != "continue":
            parts.append(self.outcome.replace("_", " "))
        return "; ".join(parts) or "(no prediction)"


@dataclass
class Verdict:
    ok: bool
    failures: list[str] = field(default_factory=list)
    unpredicted: list[str] = field(default_factory=list)   # changes outside the named targets

    def describe(self) -> str:
        if self.ok:
            return "as predicted"
        return "; ".join(self.failures + [f"unpredicted: {u}" for u in self.unpredicted])


def _matches(item: Item, c: Change, who: Optional[Obj]) -> bool:
    """Does change ``c`` (of object ``who``) satisfy ``item``, ignoring the target?"""
    if item.change not in c.kinds:
        return False
    if item.change == "moved":
        if item.dx is not None and c.dx != item.dx:
            return False
        if item.dy is not None and c.dy != item.dy:
            return False
    if item.colour is not None:
        if item.change == "recoloured" and c.colour_to != item.colour:
            return False
        if item.change == "appeared" and (who is None or who.colour != item.colour):
            return False
    return True


def _targets(item: Item, step: Step) -> Optional[set[int]]:
    """Ids the item may refer to; None for 'any'. Objects of a colour count if
    they had that colour before or after the step (a recoloured object, a new one)."""
    t = item.target
    if t.startswith("#"):
        try:
            return {int(t[1:])}
        except ValueError:
            return set()
    if t.startswith("colour") or t.startswith("color"):
        try:
            col = int(t.split()[-1])
        except ValueError:
            return set()
        objs = list(step.before.values()) + list(step.after.values())
        return {o.id for o in objs if o.colour == col}
    return None


def check(exp: Expectation, step: Step, outcome: str) -> Verdict:
    """Check ``exp`` against what ``step`` did. ``outcome`` is what the game reported
    after the step: continue, level_complete or game_over."""
    failures: list[str] = []
    by_id = {c.id: c for c in step.changes}
    covered: set[int] = set()
    for it in exp.items:
        ids = _targets(it, step)
        cands = [c for c in step.changes if ids is None or c.id in ids]
        if it.change == "none":
            if ids is not None:
                covered |= ids
            if cands:
                failures.append(f"expected {it.target} unchanged, but " +
                                ", ".join(c.describe() for c in cands[:3]))
            continue
        hit = [c for c in cands if _matches(it, c, step.after.get(c.id) or step.before.get(c.id))]
        if hit:
            covered |= {c.id for c in hit}
            if ids is not None and it.target.startswith("#"):
                covered |= ids
            continue
        if ids is not None and it.target.startswith("#"):
            covered |= ids
        got = ", ".join(c.describe() for c in cands[:3]) or "nothing changed there"
        failures.append(f"expected {it.describe()}, got: {got}")
    unpredicted: list[str] = []
    if exp.rest == "unchanged":
        unpredicted = [c.describe() for i, c in sorted(by_id.items()) if i not in covered][:6]
    if outcome != exp.outcome:
        failures.append(f"expected outcome {exp.outcome}, got {outcome}")
    return Verdict(ok=not failures and not unpredicted, failures=failures, unpredicted=unpredicted)
