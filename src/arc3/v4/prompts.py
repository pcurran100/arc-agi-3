"""The two LLM calls of v4: update the assumptions, then act on them (D-042).

- **update** — given what just happened (above all, any prediction that failed),
  edit the assumption store: add, revise, support, weaken, retire. This is
  where beliefs that would otherwise stay buried in reasoning are written down.
- **act** — given the store and the current frame, choose the next action or a
  short sequence, each step with a testable expectation (expect.py).

Both see the same exact context: the store, the fact layer's text, the recent
steps with what was predicted and what happened, notes from earlier levels, the
frame as text and (optionally) as an image. Both are expected to think at
length; the reasoning happens in the model's thinking channel and only the
edits or the actions come back as JSON, so the answer is short and cannot be
cut off by a long chain of thought.

The prompt describes the loop and the interfaces, never what kinds of games
exist (Guardrail 2).

Bump ``PROMPT_VERSION`` whenever wording changes: recorded calls replay only
while prompts are byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .expect import EXPECT_SCHEMA
from .render import image_part, legend
from .store import KINDS, RULE_SCHEMA, SCOPES

PROMPT_VERSION = "v4-2"
MAX_STEPS = 6

SYSTEM = f"""\
You are playing an unknown turn-based game on a 64x64 grid of 16 colours. Nobody
tells you the rules or the goal: you find them out by acting and watching. The
game has several levels; finishing one moves you to the next. Score depends on
finishing levels in as FEW ACTIONS as possible, compared with a human player -
thinking is free, actions are not.

You work in a loop with exact code:
- An ASSUMPTION STORE holds everything currently believed about this game:
  what the objects are (ontology), what actions do (dynamics), and what winning
  requires (goal). Each assumption has a confidence and an evidence count.
- Before every action you PREDICT what it will do. Code checks the prediction
  exactly against the next frame. A correct prediction is evidence for the
  assumptions behind it; a wrong one is a SURPRISE that you must explain.
- Code measures facts exactly (objects with stable ids, what changed on every
  step); trust those over your reading of the image.

Good play: keep several candidate explanations alive while the evidence is thin
and design actions that tell them apart; learn what actions do before betting
actions on a guess about the goal; once the rules are clear, go straight for the
goal. Levels of one game usually share rules but CAN DIFFER: objects may look
different, new objects may appear, things that were still may start moving.
Assumptions carried from an earlier level are hypotheses until this level
confirms them.

Coordinates: x is the column (0-63, left to right), y is the row (0-63, top to
bottom). CLICK(x,y) clicks the cell at column x, row y.
{legend()}
"""

UPDATE_TASK = """\
TASK: UPDATE THE ASSUMPTION STORE.
Think carefully about what the evidence above shows, especially any surprise.
Then return the edits as operations:
Every operation has op, id, statement and note. For add, leave id "" (code
assigns A1, A2, ...) and put the claim in statement; for the other operations,
id is an existing id and statement may be "" unless you are rewording it.
- add: a new assumption. One claim per assumption, concrete and testable, naming
  objects by id or colour. Give kind (ontology | dynamics | goal | other), scope
  (level, or game if you expect it to hold on every level), a confidence 0-1,
  and, when it is a regular effect of one action, a rule that code can check on
  every future use of that action.
- revise: change the statement, confidence, kind, scope or rule of an existing id.
- support / weaken: record evidence for or against an id (with a note).
- retire: the evidence has decisively ruled it out (with the reason).
Write down every belief your reasoning relies on, including the guesses about
the goal - an assumption that is not in the store does not exist next turn.
Keep competing goal assumptions side by side rather than picking one too early.
Do not retire an assumption on one contradiction if a condition could explain
it; revise it to state the condition instead.
"""

ACT_TASK = f"""\
TASK: CHOOSE WHAT TO DO NEXT.
Think it through, then return 1 to {MAX_STEPS} steps. Each step is an action
("ACTION1".."ACTION7" or "CLICK(x,y)") and an exact expectation of what it will
do. While you are unsure what will happen, return ONE step and make its
expectation say what you think; return several only when you are confident of
every one of them. The sequence stops at the first wrong expectation and you
are asked again, so a confident sequence costs nothing extra if it is wrong.
In "tests", list the assumption ids this choice relies on or is meant to test.
Expectation format: "changes" lists what will change ("target" is "#id",
"colour N" or "any"; "change" is moved/recoloured/grew/shrank/reshaped/appeared/
vanished/none; dx, dy for moves and colour for recolours or new objects, null
when you do not know); "rest" is "unchanged" if nothing else will change, else
"any"; "outcome" is continue, level_complete or game_over.
"""

OP_SCHEMA = {
    "type": "object",
    "properties": {
        "op": {"type": "string", "enum": ["add", "revise", "support", "weaken", "retire"]},
        "id": {"type": "string", "maxLength": 8,
               "description": "an existing id such as A3; empty for add (code assigns ids)"},
        "statement": {"type": "string", "maxLength": 300,
                      "description": "the claim itself; required for add"},
        "kind": {"type": "string", "enum": list(KINDS)},
        "scope": {"type": "string", "enum": list(SCOPES)},
        "confidence": {"type": "number"},
        "rule": RULE_SCHEMA,
        "note": {"type": "string", "maxLength": 200},
    },
    "required": ["op", "id", "statement", "note"],
}

UPDATE_SCHEMA = {
    "title": "update",
    "type": "object",
    "properties": {
        "ops": {"type": "array", "items": OP_SCHEMA, "maxItems": 16},
        "summary": {"type": "string", "maxLength": 300},
    },
    "required": ["ops", "summary"],
}

ACT_SCHEMA = {
    "title": "act",
    "type": "object",
    "properties": {
        "steps": {
            "type": "array", "minItems": 1, "maxItems": MAX_STEPS,
            "items": {"type": "object",
                      "properties": {"action": {"type": "string", "maxLength": 16},
                                     "expect": EXPECT_SCHEMA},
                      "required": ["action", "expect"]},
        },
        "purpose": {"type": "string", "maxLength": 300},
        "tests": {"type": "array", "items": {"type": "string", "maxLength": 8}, "maxItems": 8},
    },
    "required": ["steps", "purpose", "tests"],
}

@dataclass
class Context:
    level: int
    total_levels: Optional[int]
    actions_on_level: int
    actions_total: int
    human_baseline: Optional[int]
    available: list[str]
    store: str
    facts: str
    grid: str
    history: list[str] = field(default_factory=list)
    level_notes: list[str] = field(default_factory=list)
    event: str = ""                 # what triggered this call
    last_reasoning: str = ""        # tail of the previous act call's thinking
    image: Optional[dict] = None    # an image_url content part


def _body(c: Context) -> str:
    lv = f"level {c.level + 1}" + (f" of {c.total_levels}" if c.total_levels else "")
    base = (f"; a human needed about {c.human_baseline} actions for this level"
            if c.human_baseline else "")
    parts = [
        f"STATUS: {lv}; {c.actions_on_level} actions on this level, {c.actions_total} in total"
        f"{base}.\nAVAILABLE ACTIONS: {', '.join(c.available)}",
    ]
    if c.level_notes:
        parts.append("EARLIER LEVELS:\n" + "\n".join(c.level_notes))
    parts.append("ASSUMPTION STORE:\n" + c.store)
    parts.append("FACTS (exact):\n" + c.facts)
    if c.history:
        parts.append("RECENT STEPS (action | predicted | what happened):\n" + "\n".join(c.history))
    if c.event:
        parts.append("WHAT JUST HAPPENED:\n" + c.event)
    if c.last_reasoning:
        parts.append("YOUR REASONING WHEN YOU CHOSE THE LAST ACTIONS (extract any belief in it "
                     "that is not yet in the store):\n" + c.last_reasoning)
    parts.append("CURRENT FRAME (exact; the image shows the same frame):\n" + c.grid)
    return "\n\n".join(parts)


def _messages(c: Context, task: str) -> list[dict]:
    text = _body(c) + "\n\n" + task
    content = [{"type": "text", "text": text}, c.image] if c.image else text
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def update_messages(c: Context) -> list[dict]:
    return _messages(c, UPDATE_TASK)


def act_messages(c: Context) -> list[dict]:
    return _messages(c, ACT_TASK)


def with_image(c: Context, grid) -> Context:
    c.image = image_part(grid)
    return c
