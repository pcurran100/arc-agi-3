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
from .store import ATTAIN_SCHEMA, KINDS, RULE_SCHEMA, SCOPES

PROMPT_VERSION = "v4-5"
MAX_SUBGOALS = 4
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
  step, objects that change on every action whatever the action, shapes that
  are copies of each other up to scale or rotation); trust those over your
  reading of the image.
- Code can also WALK: name an object you control and a target, and code
  computes the shortest route with the moves it has seen work, then checks
  every step of it. Plan in subgoals ("first bring #7 onto #5, then onto #3")
  and let code find the moves.

Good play: keep several candidate explanations alive while the evidence is thin
and design actions that tell them apart; learn what actions do before betting
actions on a guess about the goal. Keep two or more GOAL hypotheses, each with
the condition that should complete the level, and TEST them: the cheapest
action that would show a goal hypothesis wrong is worth taking early. When a
goal's condition is met and the level does not complete, that goal is wrong or
incomplete - revise it (maybe something must be done first) or drop it. Once a
goal is likely, go straight for it. Levels of one game usually share rules but CAN DIFFER: objects may look
different, new objects may appear, things that were still may start moving.
Assumptions carried from an earlier level are hypotheses until this level
confirms them.

Only the actions listed under AVAILABLE ACTIONS exist in this game; each game
offers its own set. Coordinates: x is the column (0-63, left to right), y is the
row (0-63, top to bottom); when CLICK(x,y) is available it clicks that cell.
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
Assumptions are beliefs about how the game WORKS (what things are, what actions
do, what wins), not descriptions of where things are now: the FACTS show the
current positions every turn, so never store a snapshot of the current state.
Prefer revising an existing assumption to adding a near-duplicate, and retire
assumptions that a newer one replaces. Keep the store short enough to read.
Keep competing goal assumptions side by side rather than picking one too early.
Give every goal assumption an "attain" condition when you can: the object that
must get there, the target ("#id" or "x,y") and the relation (overlap, touch or
at). Code checks it: if the condition holds and the level does not complete,
the goal loses confidence. Goals gain confidence only when a level completes;
correct predictions about moves support only ontology and dynamics.
Do not retire an assumption on one contradiction if a condition could explain
it; revise it to state the condition instead.
Plans and intentions ("next I will go left") are not assumptions: they belong in
the act call, never in the store.
"""

ACT_TASK = f"""\
TASK: CHOOSE WHAT TO DO NEXT.
Think it through, then return up to {MAX_STEPS} steps (none if you give a
plan, below). Each step is an action
("ACTION1".."ACTION7" or "CLICK(x,y)") and an exact expectation of what it will
do. Use only the AVAILABLE ACTIONS. An action you have never tried is always
uncertain. While you are unsure what will happen, return ONE step and make its
expectation say what you think; return several only when you are confident of
every one of them. The sequence stops at the first wrong expectation and you
are asked again, so a confident sequence costs nothing extra if it is wrong.
In "tests", list the assumption ids this choice relies on or is meant to test.
Expectation format: "changes" lists what will change ("target" is "#id",
"colour N" or "any"; object ids can be reassigned when an object vanishes and a
similar one appears, or when a region is redrawn in another colour, so prefer
"colour N" for things that grow, shrink or recolour; "change" is moved/recoloured/grew/shrank/reshaped/appeared/
vanished/none; dx, dy for moves and colour for recolours or new objects, null
when you do not know); "rest" is "unchanged" if nothing else will change, else
"any" (objects listed as changing on every action never count against
"unchanged"); "outcome" is continue, level_complete or game_over.

Instead of (or after) explicit steps you may give a "plan": up to
{MAX_SUBGOALS} subgoals done in order. Each names an object you control
("object": "#7"), a target ("to": "#3" or "x,y"), a relation (overlap, touch or
at), what you expect on arrival ("on_arrival", an expectation as above; put
"level_complete" there when you think arriving wins), and the goal id it
serves. Code walks each subgoal with the moves it has seen work, checks every
step, and calls you back when the plan is done, a step surprises it, or no
route exists. Use a plan whenever you know where to go; use steps to try
things out.
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
        "attain": ATTAIN_SCHEMA,
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
            "type": "array", "minItems": 0, "maxItems": MAX_STEPS,
            "items": {"type": "object",
                      "properties": {"action": {"type": "string", "maxLength": 16},
                                     "expect": EXPECT_SCHEMA},
                      "required": ["action", "expect"]},
        },
        "plan": {
            "type": "array", "maxItems": MAX_SUBGOALS,
            "items": {"type": "object",
                      "properties": {"object": {"type": "string", "maxLength": 8},
                                     "to": {"type": "string", "maxLength": 12},
                                     "relation": {"type": "string",
                                                  "enum": ["overlap", "touch", "at"]},
                                     "on_arrival": EXPECT_SCHEMA,
                                     "goal": {"type": "string", "maxLength": 8}},
                      "required": ["object", "to", "relation", "on_arrival", "goal"]},
        },
        "purpose": {"type": "string", "maxLength": 300},
        "tests": {"type": "array", "items": {"type": "string", "maxLength": 8}, "maxItems": 8},
    },
    "required": ["steps", "plan", "purpose", "tests"],
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
    plan: str = ""                  # subgoals still to walk


def _body(c: Context) -> str:
    # D-044: "16 actions on this level ... a human needed about 22" was read on
    # ls20 as a budget ("1 action left"), and the model burned actions to
    # "finish" it. No baseline here, and the count says there is no limit.
    lv = f"level {c.level + 1}" + (f" of {c.total_levels}" if c.total_levels else "")
    parts = [
        f"STATUS: {lv}; actions used so far on this level: {c.actions_on_level} (there is "
        f"no fixed limit; every action lowers the score, so fewer is better)."
        f"\nAVAILABLE ACTIONS: {', '.join(c.available)}",
    ]
    if c.plan:
        parts.append("CURRENT PLAN (being walked by code):\n" + c.plan)
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
