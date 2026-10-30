"""The two call tiers of v5 (D-046).

- **think** (thinking on): at a level start, after a MAJOR surprise, when the
  simulator was invalidated or the goal disproved, and periodically. One call
  does what v4 split into update + act: edit the assumption store, name
  objects, write or repair the simulator and goal, and choose what to do.
- **next** (thinking off, short prompt): when the last plan finished as
  expected and only the next move is needed.

Everything that does not change between calls is in the system message, so a
server with prefix caching (vLLM) reuses it; the user message holds only the
current state and a one-line task.

Bump ``PROMPT_VERSION`` whenever wording changes: recorded calls replay only
while prompts are byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..v4.expect import EXPECT_SCHEMA
from ..v4.prompts import OP_SCHEMA
from ..v4.render import legend

PROMPT_VERSION = "v5-5"
MAX_STEPS = 8
MAX_SUBGOALS = 4

SYSTEM = f"""\
You are playing an unknown turn-based game on a 64x64 grid of 16 colours. Nobody
tells you the rules or the goal: you find them out by acting and watching. The
game has several levels; finishing one moves you to the next. Score depends on
finishing each level in as FEW ACTIONS as possible - thinking is free, actions
are not. There is no fixed action limit.

HOW YOU SEE THE GAME. Exact code tracks every object and keeps its state for you:
- The OBJECT SHEET has one line per object: a permanent id (#n; an object keeps
  its id while it moves, grows, shrinks or changes shape), its colour, box and
  size, the name and role you gave it, and what each action has done to it so
  far. Objects that never changed are listed briefly.
- The INTERFACE line lists objects that change the same way on every action
  whatever the action (e.g. a bar that loses a cell per action). These are
  almost always on-screen counters or timers, not puzzle pieces: do not try to
  "fill", "reach" or "move" them. Code predicts them for you.
- The ACTION-EFFECT TABLE says what each action (for clicks: each clicked
  object) did so far. EVENTS lists splits, merges and returns to the level-start
  position (an object jumping back to where the level began often means a
  life was lost or the level was reset).
- Coordinates: x is the column (0-63, left to right), y the row (0-63, top to
  bottom). Only the AVAILABLE ACTIONS exist; CLICK(x,y) clicks that cell.
- The CURRENT FRAME is there to glance at, NOT to count by hand: never count
  characters along a row or column. Positions and sizes are on the sheet, and
  code answers spatial questions exactly - in a simulator, state.free() and
  state.colours_in_box() check walls and paths for you, and code searches the
  routes. Spend your thinking on what things ARE and what WINS.

HOW YOU ACT. Prefer, in this order:
1. A SIMULATOR. Write Python defining predict(state, action) and goal(state).
   Code checks predict against every recorded step (only on the objects you
   list as "tracked"), then searches it for the SHORTEST action sequence that
   makes goal(state) true, executes it and checks every step. One good simulator
   replaces dozens of decisions. It may be partial: model what you understand
   (usually the object you control and what blocks it) and track only those
   objects. The state API (objects are dicts with id, colour, x, y, w, h, size):
     state.get(id), state.of_colour(c), state.at(x, y), state.objects (dict),
     state.colours_in_box(x, y, w, h, ignore=(ids)) -> set of colours there
       (-1 = off screen), state.free(x, y, w, h, passable=(colours), ignore=(ids)),
     state.overlaps(a, b), state.move(id, dx, dy), state.place(id, x, y),
     state.recolour(id, c), state.resize(id, w, h), state.remove(id),
     state.add(colour, x, y, w, h), state.meta (dict for your own counters).
   action is "ACTION1".."ACTION7" or "CLICK(x,y)"; click_xy(action) gives (x, y)
   for a click and None otherwise. There is NO import: a line starting with
   "import" or "from" makes the simulator fail. np (numpy) and math already exist.
   Send it as "simulator": {{"lines": [the source, ONE STRING PER LINE, indentation
   kept], "tracked": ["#7", ...], "goal_id": the store id of the goal goal()
   encodes}}. Both predict and goal go in "lines". Use ONLY single quotes in
   the Python ('ACTION1', never "ACTION1"): a double quote ends the JSON string.
   Example: def predict(state, action):
                p = state.get(7); d = {{'ACTION1': (0,-5), 'ACTION2': (0,5)}}.get(action)
                if p and d and state.free(p["x"]+d[0], p["y"]+d[1], p["w"], p["h"],
                                          passable=(3,), ignore=(7,)):
                    state.move(7, *d)
                return state
            def goal(state):
                p, t = state.get(7), state.get(3)
                return bool(p and t and state.overlaps(p, t))
2. A PLAN of subgoals: "object" you control, "to" a target ("#id" or "x,y"),
   "relation" overlap | touch | at. Code routes there with the moves it has
   seen work.
3. STEPS: explicit actions, to try something out. Each step's "expect" names
   ONLY the changes your current idea predicts (leave "rest" as "any"); code
   predicts counters and known effects itself. Expect format: "changes" lists
   items with "target" ("#id", "colour N" or "any"), "change"
   (moved/recoloured/grew/shrank/reshaped/appeared/vanished/none), dx/dy or
   colour when known; "outcome" is continue, level_complete or game_over.

SURPRISES. Code sorts what happens against what you expected. Small things (an
extra side effect, a size off by one) are written down and shown to you later;
they do not stop a plan. A MAJOR surprise stops the plan and calls you back: the
object you were driving did something else, the outcome differed (level
complete or game over), something new appeared, the simulator mispredicted a
tracked object, or the same small surprise happened twice.

THE ASSUMPTION STORE holds what is believed about the game: ontology (what
things are), dynamics (what actions do), goal (what wins). Keep two or more GOAL
hypotheses with a checkable "attain" condition each; a goal whose condition
holds without the level completing loses confidence. Test the cheapest goal
first, and once a goal is likely, go straight for it. Store beliefs, not
positions or plans. Levels usually share rules but can differ.

GOOD PLAY: learn what the actions do first (a few distinct actions, not the same
one repeated), name the objects (player, wall, key, door, target, ...), form a
goal, then let a simulator or plan do the walking. Never repeat an action whose
effect you already know unless it moves you toward a goal. If an action did
nothing (only the interface changed), do not try it again in the same place.
{legend()}

YOUR TASKS. Each request ends with one of:
- "TASK: THINK" - update the store (ops), name objects (names), write or repair
  the simulator (or null to keep the current one), then give steps and/or a plan
  (both may be empty if the simulator's goal should be searched).
- "TASK: NEXT" - the last plan finished as expected; give the next steps or plan.
- "TASK: FIX SIMULATOR" - your simulator failed to compile or to reproduce the
  recorded steps; send the corrected simulator (lines, tracked, goal_id),
  changing only what the problem needs.
"""

NAME_SCHEMA = {
    "type": "object",
    "properties": {"id": {"type": "string", "maxLength": 6},
                   "name": {"type": "string", "maxLength": 24},
                   "role": {"type": "string", "maxLength": 80}},
    "required": ["id", "name", "role"],
}

STEP_SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string", "maxLength": 16}, "expect": EXPECT_SCHEMA},
    "required": ["action", "expect"],
}

SUBGOAL_SCHEMA = {
    "type": "object",
    "properties": {"object": {"type": "string", "maxLength": 24,
                              "description": "#id (or a name you gave it)"},
                   "to": {"type": "string", "maxLength": 24,
                          "description": "#id, a name you gave, or x,y"},
                   "relation": {"type": "string", "enum": ["overlap", "touch", "at"]},
                   "on_arrival": EXPECT_SCHEMA,
                   "goal": {"type": "string", "maxLength": 8}},
    "required": ["object", "to", "relation", "on_arrival", "goal"],
}

# Code as a list of lines: in debug run (d) the model's one-string code arrived
# with every newline turned into a bare "n" ("def predict(state, action):n    p
# = ..."), and a field called "goal" was read as the goal's code.
SIM_SCHEMA = {
    "type": ["object", "null"],
    "properties": {"lines": {"type": "array", "maxItems": 150,
                             "items": {"type": "string", "maxLength": 200},
                             "description": "the Python source, one string per line"},
                   "tracked": {"type": "array", "maxItems": 12,
                               "items": {"type": "string", "maxLength": 12},
                               "description": "ids the simulator models, e.g. \"#7\""},
                   "goal_id": {"type": "string", "maxLength": 8,
                               "description": "the store id of the goal that goal() encodes, or \"\""}},
    "required": ["lines", "tracked", "goal_id"],
}

THINK_SCHEMA = {
    "title": "think",
    "type": "object",
    "properties": {
        "ops": {"type": "array", "items": OP_SCHEMA, "maxItems": 12},
        "names": {"type": "array", "items": NAME_SCHEMA, "maxItems": 16},
        "simulator": SIM_SCHEMA,
        "steps": {"type": "array", "items": STEP_SCHEMA, "maxItems": MAX_STEPS},
        "plan": {"type": "array", "items": SUBGOAL_SCHEMA, "maxItems": MAX_SUBGOALS},
        "purpose": {"type": "string", "maxLength": 300},
    },
    "required": ["ops", "names", "simulator", "steps", "plan", "purpose"],
}

NEXT_SCHEMA = {
    "title": "next",
    "type": "object",
    "properties": {
        "steps": {"type": "array", "items": STEP_SCHEMA, "maxItems": MAX_STEPS},
        "plan": {"type": "array", "items": SUBGOAL_SCHEMA, "maxItems": MAX_SUBGOALS},
        "purpose": {"type": "string", "maxLength": 200},
    },
    "required": ["steps", "plan", "purpose"],
}


FIX_SCHEMA = {**SIM_SCHEMA, "type": "object", "title": "fix"}


@dataclass
class Context:
    level: int
    total_levels: Optional[int]
    actions_on_level: int
    available: list[str]
    facts: str
    store: str = ""
    grid: str = ""
    sim: str = ""
    plan: str = ""
    history: list[str] = field(default_factory=list)
    minors: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    level_notes: list[str] = field(default_factory=list)


def _body(c: Context, full: bool) -> str:
    lv = f"level {c.level + 1}" + (f" of {c.total_levels}" if c.total_levels else "")
    parts = [f"STATUS: {lv}; actions used on this level: {c.actions_on_level}.\n"
             f"AVAILABLE ACTIONS: {', '.join(c.available)}"]
    if full and c.level_notes:
        parts.append("EARLIER LEVELS:\n" + "\n".join(c.level_notes))
    if c.store:
        parts.append("ASSUMPTION STORE:\n" + c.store)
    parts.append(c.facts)
    if c.sim:
        parts.append("YOUR SIMULATOR: " + c.sim)
    if c.plan:
        parts.append("CURRENT PLAN: " + c.plan)
    if c.history:
        parts.append("RECENT STEPS (action | your expectation | what happened | verdict):\n"
                     + "\n".join(c.history))
    if c.minors:
        parts.append("SMALL SURPRISES since you last thought (logged, did not stop the plan):\n"
                     + "\n".join(c.minors))
    if c.events:
        parts.append("WHAT HAPPENED (why you are being asked):\n" + "\n".join(c.events))
    if full and c.grid:
        parts.append("CURRENT FRAME (exact):\n" + c.grid)
    return "\n\n".join(parts)


def think_messages(c: Context) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": _body(c, True) + "\n\nTASK: THINK"}]


def next_messages(c: Context) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": _body(c, False) + "\n\nTASK: NEXT"}]


def fix_messages(c: Context, source: str, problem: str) -> list[dict]:
    """The cheap repair call (thinking off): the error and the source."""
    src = "\n".join(f"{k + 1:3d}| {ln}" for k, ln in enumerate(source.splitlines()[:150]))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": _body(c, False)
             + "\n\nYOUR SIMULATOR (numbered; the numbers are not part of the code):\n" + src
             + "\n\nWHAT IS WRONG WITH IT:\n" + problem + "\n\nTASK: FIX SIMULATOR"}]
