"""The assumption store: what the agent currently believes, as editable records (D-042).

Every belief the LLM forms about the game - which object it controls, what an
action does, what an object is for, what winning requires - is an
:class:`Assumption` here, not a sentence lost in a reasoning trace. The update
call edits the store with explicit operations; the act call reads it to choose
actions; code adds evidence to it as actions play out.

**Who maintains what.**

- *The LLM* adds, revises and retires assumptions, sets their confidence, and
  links the predictions it makes to the assumptions they test.
- *Code* adds evidence: a prediction that came true supports the assumptions it
  was said to test, and an assumption with a machine-checkable ``rule`` ("every
  ACTION1 moves #12 by (0,-4)") is checked on every step whose action matches,
  with no model call.

**Soft, never hard (Guardrail 1).** Contradicting evidence lowers confidence
and, past a threshold, marks an assumption *doubtful*; only the model retires
one, and retired assumptions stay visible so they are not re-proposed blindly.

**Across levels.** At a level change every live assumption is marked *carried*
and level-scoped ones lose confidence: they are hypotheses about the new level
until something on it supports them again (D-042).

**What this does not do.**

- *No rival bundles.* Assumptions are individually weighted; there is no notion
  of a consistent set of them competing with another set. Contradictory
  assumptions can coexist, and the model sees both.
- *Rules check changes only* (expect.py's language), not conditions such as
  "unless a wall is in the way"; a conditional rule will collect contradictions,
  become doubtful, and the model is expected to restate it.
- *No automatic generalisation.* Code never invents an assumption from a fact;
  the fact text is in front of the model and inventing is the model's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..v3.facts import Step
from .expect import ITEM_SCHEMA, Expectation, Item, check

KINDS = ("ontology", "dynamics", "goal", "other")
SCOPES = ("level", "game")
SUPPORT_GAIN = 0.15      # confidence moves this fraction of the way to 1 on support
CONTRA_KEEP = 0.6        # and is multiplied by this on contradiction
DOUBTFUL_BELOW = 0.25
CARRY_LEVEL = 0.6        # level-scoped confidence multiplier at a level change
CARRY_GAME = 0.85        # game-scoped
MAX_EVIDENCE = 6         # evidence notes kept per assumption (counts are kept in full)

RULE_SCHEMA = {
    "type": ["object", "null"],
    "properties": {
        "action": {"type": "string", "maxLength": 32},
        "expect": {"type": "array", "items": ITEM_SCHEMA, "maxItems": 4},
    },
    "required": ["action", "expect"],
}


@dataclass
class Rule:
    """``action`` is "ACTION1", "CLICK" (any click), "CLICK colour 9" (a click on an
    object of colour 9) or "any"; ``expect`` holds for every step with that action."""

    action: str
    expect: tuple[Item, ...]

    @staticmethod
    def parse(d) -> Optional["Rule"]:
        if not isinstance(d, dict) or not d.get("action"):
            return None
        items = tuple(i for i in (Item.parse(x) for x in d.get("expect", []) or []) if i)
        return Rule(str(d["action"]).strip(), items) if items else None

    def applies(self, step: Step) -> bool:
        a = self.action.upper()
        if a == "ANY":
            return True
        if a.startswith("CLICK"):
            if not step.action.startswith("CLICK"):
                return False
            if a == "CLICK":
                return True
            try:
                col = int(a.split()[-1])
            except ValueError:
                return False
            o = step.before.get(step.clicked) if step.clicked is not None else None
            return o is not None and o.colour == col
        return step.action.upper() == a

    def describe(self) -> str:
        return f"on {self.action}: " + "; ".join(i.describe() for i in self.expect)


@dataclass
class Assumption:
    id: str
    statement: str
    kind: str = "other"
    scope: str = "level"
    confidence: float = 0.5
    status: str = "active"            # active | doubtful | retired
    carried: bool = False             # from an earlier level, not yet re-supported
    rule: Optional[Rule] = None
    support: int = 0
    contra: int = 0
    evidence: list[str] = field(default_factory=list)
    level: int = 0                    # level it was made on
    retired_because: str = ""

    def note(self, s: str) -> None:
        self.evidence = [*self.evidence, s][-MAX_EVIDENCE:]

    def supported(self, note: str) -> None:
        self.support += 1
        self.confidence += (1.0 - self.confidence) * SUPPORT_GAIN
        self.carried = False
        if self.status == "doubtful" and self.confidence >= DOUBTFUL_BELOW:
            self.status = "active"
        self.note("+ " + note)

    def contradicted(self, note: str) -> None:
        self.contra += 1
        self.confidence *= CONTRA_KEEP
        if self.status == "active" and self.confidence < DOUBTFUL_BELOW:
            self.status = "doubtful"
        self.note("- " + note)

    def describe(self) -> str:
        tags = [self.kind, self.scope + "-wide" if self.scope == "game" else "this level"]
        if self.carried:
            tags.append("CARRIED from an earlier level, unconfirmed here")
        if self.status != "active":
            tags.append(self.status.upper())
        s = (f"{self.id} [{', '.join(tags)}] conf {self.confidence:.2f} "
             f"(+{self.support}/-{self.contra}): {self.statement}")
        if self.rule:
            s += f"\n     rule: {self.rule.describe()}"
        if self.evidence:
            s += "\n     recent evidence: " + " | ".join(self.evidence[-3:])
        return s


@dataclass
class AssumptionStore:
    items: dict[str, Assumption] = field(default_factory=dict)
    level: int = 0
    _next: int = 1
    log: list[str] = field(default_factory=list)     # every edit, for the record

    # -- reading ----------------------------------------------------------

    def live(self) -> list[Assumption]:
        return [a for a in self.items.values() if a.status != "retired"]

    def describe(self, max_retired: int = 8) -> str:
        live = sorted(self.live(), key=lambda a: (KINDS.index(a.kind) if a.kind in KINDS else 9,
                                                   -a.confidence))
        lines = [a.describe() for a in live] or ["(none yet)"]
        retired = [a for a in self.items.values() if a.status == "retired"][-max_retired:]
        if retired:
            lines.append("RETIRED (do not re-propose without new evidence):")
            lines += [f"  {a.id}: {a.statement} - {a.retired_because}" for a in retired]
        return "\n".join(lines)

    # -- the model's edits ----------------------------------------------------

    def apply(self, ops: list, where: str) -> list[str]:
        """Apply the update call's operations. Returns one line per edit made.
        Malformed operations are skipped and reported, never raised."""
        done = []
        for op in ops or []:
            if not isinstance(op, dict):
                continue
            kind = op.get("op")
            a = self.items.get(str(op.get("id", "")).strip())
            note = str(op.get("note", "") or "")[:200]
            if kind == "add":
                # Models sometimes put the claim in the note (seen 2026-10-19,
                # Nemotron-VL: 12 of 12 adds); a claim is a claim wherever it is.
                st = str(op.get("statement", "") or "").strip() or note.strip()
                if not st:
                    continue
                new = Assumption(
                    id=f"A{self._next}", statement=st[:400],
                    kind=op.get("kind") if op.get("kind") in KINDS else "other",
                    scope=op.get("scope") if op.get("scope") in SCOPES else "level",
                    confidence=_conf(op.get("confidence"), 0.5),
                    rule=Rule.parse(op.get("rule")), level=self.level)
                if note and note.strip() != st:
                    new.note(f"{where}: {note}")
                self._next += 1
                self.items[new.id] = new
                done.append(f"added {new.id}: {new.statement}")
            elif a is None:
                if kind:
                    done.append(f"skipped {kind} on unknown id {op.get('id')!r}")
            elif kind == "revise":
                if str(op.get("statement", "") or "").strip():
                    a.statement = str(op["statement"]).strip()[:400]
                if op.get("kind") in KINDS:
                    a.kind = op["kind"]
                if op.get("scope") in SCOPES:
                    a.scope = op["scope"]
                if "rule" in op:
                    a.rule = Rule.parse(op.get("rule"))
                a.confidence = _conf(op.get("confidence"), a.confidence)
                if a.status == "doubtful" and a.confidence >= DOUBTFUL_BELOW:
                    a.status = "active"
                a.note(f"{where}: revised" + (f" - {note}" if note else ""))
                done.append(f"revised {a.id}: {a.statement}")
            elif kind == "retire":
                a.status = "retired"
                a.retired_because = note or "retired by the model"
                done.append(f"retired {a.id}: {a.retired_because}")
            elif kind == "support":
                a.supported(f"{where}: {note}" if note else where)
                done.append(f"supported {a.id}")
            elif kind == "weaken":
                a.contradicted(f"{where}: {note}" if note else where)
                done.append(f"weakened {a.id}")
        self.log += [f"[{where}] {d}" for d in done]
        return done

    # -- code's evidence --------------------------------------------------------

    def observe(self, step: Step, outcome: str) -> list[str]:
        """Check every live rule whose action matches ``step``. Returns notes on
        rules that were contradicted (supports are routine and not reported)."""
        out = []
        for a in self.live():
            if a.rule is None or not a.rule.applies(step):
                continue
            v = check(Expectation(a.rule.expect, "any", outcome), step, outcome)
            where = f"t{step.t} {step.action}"
            # The outcome is not part of a rule; only its change items are judged.
            fails = [f for f in v.failures if not f.startswith("expected outcome")]
            if fails:
                a.contradicted(f"{where}: " + "; ".join(fails)[:200])
                out.append(f"{a.id}'s rule failed at {where}: " + "; ".join(fails)[:200])
            else:
                a.supported(where)
        return out

    def supported_by_prediction(self, ids: list, where: str) -> None:
        for i in ids or []:
            a = self.items.get(str(i))
            if a is not None and a.status != "retired":
                a.supported(f"{where}: prediction came true")

    # -- levels -------------------------------------------------------------------

    def new_level(self, level: int) -> None:
        self.level = level
        for a in self.live():
            a.carried = True
            a.confidence *= CARRY_LEVEL if a.scope == "level" else CARRY_GAME
            a.note(f"carried into level {level + 1}")


def _conf(v, default: float) -> float:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return max(0.0, min(1.0, float(v)))
    return default
