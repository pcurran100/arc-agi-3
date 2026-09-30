"""The LLM's calls in v3: each has one job, exact inputs and a schema (D-039).

Two calls, deliberately narrow:

- **assess** — form or revise the whole hypothesis set: what the goal is,
  which object is the player, what each action does, which objects are static
  or counters, what the others are for; write them as a world model
  (`predict`) and a goal test (`goal`); choose what to do next — plan to the
  goal, run an experiment that tests a hypothesis, or explore.
- **repair** — the world model mispredicted recorded steps; fix the code given
  the exact counterexamples. Nothing else is asked, so nothing else can drift.

Every input is exact: the fact layer's text, a compact map of the frame, the
verifier's score and counterexamples, the goals already refuted, what the last
plan did. The prompt describes the *interface* — what a state, an action and a
helper are — and never what kinds of games exist (Guardrail 2).

Bump ``PROMPT_VERSION`` whenever wording changes: recorded calls replay only
while prompts are byte-identical.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

PROMPT_VERSION = "v3-1"

API = """\
WORLD-MODEL API (your code runs in a sandbox with `np` and `math`; no imports):
  predict(state, action) -> state   # simulate ONE action; mutate `state` and return it
  goal(state) -> float in [0, 1]    # 1.0 exactly when the level is won; graded so
                                    # that states closer to winning score higher
  `action` is a string: "ACTION1".."ACTION7" or "CLICK(x,y)" (x = column, y = row).
  state.objects            dict id -> {"id","colour","x","y","w","h","size","shape"}
                           (x, y = top-left of the bounding box; ids are the object
                           ids in the facts, stable across steps within a level)
  state.get(id)            the object dict or None
  state.of_colour(c)       list of objects of colour c
  state.at(x, y)           smallest object whose box contains (x, y), or None
  state.colours_in_box(x, y, w, h, ignore=[ids])   set of frame colours in that
                           box (-1 if it leaves the frame), not counting the cells
                           the ignored objects occupied at the start of the step
  state.free(x, y, w, h, passable={colours}, ignore=[ids])  True if the box only
                           holds passable colours (default: the background)
  state.overlaps(a, b)     do two object dicts' boxes overlap
  state.move(id, dx, dy) / state.place(id, x, y) / state.recolour(id, c) /
  state.resize(id, w, h) / state.remove(id) / state.add(colour, x, y, w, h)
  state.meta               a dict for anything else you need to track (counters,
                           flags); it persists along a simulated path
Find objects robustly (by colour, shape or position) rather than hard-coding a
single id when an object might be replaced."""

SYSTEM = f"""\
You are the reasoning core of an agent playing an unknown turn-based puzzle game
on a 64x64 grid of 16 colours (0-15). There are no instructions: the rules and
the goal must be discovered by acting and observing. The level is scored on how
FEW actions it takes to win (compared with a human), so think carefully and act
purposefully: every action should either test a hypothesis or make progress.
RESET restarts the current level (it costs one action).

The agent's code tells you exact facts about the frames: objects with stable
ids, which ones never changed, same-colour and same-shape groups, objects that
move together, what each action has done so far, and objects that look
controlled. Use them. Your hypotheses must explain these facts.

You express your understanding as code the agent can execute:
- `predict(state, action)` is your theory of how the game works. The agent
  checks it against EVERY recorded step and reports exactly where it was wrong.
- `goal(state)` is your theory of what wins. The agent searches your predict()
  for the SHORTEST action sequence that makes goal() reach 1.0, and executes it.
So a correct predict() plus a correct goal() wins the level in the fewest
actions. Objects you believe are irrelevant to the rules (move counters, score
bars, decorations) can be listed in ignore_ids so they are not checked.

{API}

Answer with JSON only, matching the schema you are given."""


ASSESS_SCHEMA = {
    "title": "assess",
    "type": "object",
    "properties": {
        "thoughts": {"type": "string",
                     "description": "Brief reasoning: what the facts show and what is still uncertain."},
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["goal", "player", "action", "static",
                                                        "role", "counter", "other"]},
                    "claim": {"type": "string"},
                    "objects": {"type": "array", "items": {"type": "integer"}},
                    "confidence": {"type": "number"},
                },
                "required": ["kind", "claim", "confidence"],
            },
        },
        "ignore_ids": {"type": "array", "items": {"type": "integer"}},
        "predict_code": {"type": "string", "description": "Python defining predict(state, action)."},
        "goal_code": {"type": "string", "description": "Python defining goal(state)."},
        "next": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["plan", "experiment", "explore"]},
                "actions": {"type": "array", "items": {"type": "string"}},
                "purpose": {"type": "string"},
            },
            "required": ["kind", "purpose"],
        },
    },
    "required": ["thoughts", "hypotheses", "predict_code", "goal_code", "next"],
}

REPAIR_SCHEMA = {
    "title": "repair",
    "type": "object",
    "properties": {
        "thoughts": {"type": "string", "description": "What the counterexamples show was wrong."},
        "predict_code": {"type": "string"},
        "ignore_ids": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["thoughts", "predict_code"],
}


@dataclass
class Context:
    """Everything the agent knows, gathered for a prompt."""

    level: int
    actions_on_level: int
    human_baseline: Optional[int]
    available: list[str]
    facts: str
    grid: str
    hypotheses: list[dict] = field(default_factory=list)
    predict_code: str = ""
    goal_code: str = ""
    ignore_ids: list[int] = field(default_factory=list)
    check: str = ""                     # verifier summary for predict_code
    counterexamples: list[str] = field(default_factory=list)
    refuted_goals: list[str] = field(default_factory=list)
    last_outcome: str = ""              # what the last plan/experiment did
    previous_level: str = ""            # what won the previous level


def _common(ctx: Context) -> list[str]:
    p = [f"LEVEL {ctx.level + 1}. Actions used on this level: {ctx.actions_on_level}"
         + (f" (a human needed about {ctx.human_baseline})." if ctx.human_baseline else ".")]
    p.append("AVAILABLE ACTIONS: " + ", ".join(ctx.available))
    if ctx.previous_level:
        p.append("PREVIOUS LEVEL: " + ctx.previous_level)
    p.append("FACTS (exact, computed by the agent):\n" + ctx.facts)
    p.append("CURRENT FRAME:\n" + ctx.grid)
    return p


def assess_messages(ctx: Context) -> list[dict]:
    p = _common(ctx)
    if ctx.hypotheses:
        p.append("YOUR CURRENT HYPOTHESES:\n" + "\n".join(
            f"  [{h.get('kind')}] {h.get('claim')} (confidence {h.get('confidence')})"
            for h in ctx.hypotheses[:20]))
    if ctx.predict_code:
        p.append("YOUR CURRENT predict():\n" + ctx.predict_code.strip())
        p.append("CHECK AGAINST RECORDED STEPS: " + (ctx.check or "not yet checked"))
        if ctx.counterexamples:
            p.append("MISPREDICTIONS:\n" + "\n".join("  " + c for c in ctx.counterexamples))
    if ctx.goal_code:
        p.append("YOUR CURRENT goal():\n" + ctx.goal_code.strip())
    if ctx.refuted_goals:
        p.append("GOALS ALREADY REFUTED (their goal() reached 1.0 and the level did NOT end):\n"
                 + "\n".join("  - " + g for g in ctx.refuted_goals[-6:]))
    if ctx.last_outcome:
        p.append("WHAT HAPPENED LAST: " + ctx.last_outcome)
    p.append(
        "TASK: Update your hypotheses about this game (goal, player, what each action does, "
        "static objects, counters, roles of other objects). Write predict() and goal() that "
        "express them. Then choose next: 'plan' (the agent searches your model for the "
        "shortest path to goal() = 1 and executes it), 'experiment' (give up to 8 actions that "
        "will test the hypothesis you are least sure of; say what you expect), or 'explore' "
        "(let the agent try untested actions for a while). Prefer 'plan' only when predict() "
        "explains the recorded steps and you believe goal()."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(p)}]


def repair_messages(ctx: Context) -> list[dict]:
    p = _common(ctx)
    p.append("YOUR predict():\n" + ctx.predict_code.strip())
    p.append("CHECK AGAINST RECORDED STEPS: " + ctx.check)
    p.append("MISPREDICTIONS (predicted vs what actually happened):\n"
             + "\n".join("  " + c for c in ctx.counterexamples))
    p.append("TASK: Fix predict() so it explains these steps, keeping what already works. "
             "If a mismatch is only a counter or decoration you do not need to model, add its "
             "id to ignore_ids instead.")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(p)}]


_OBJ = re.compile(r"\{.*\}", re.S)


def parse_json(text: str) -> Optional[dict]:
    """The JSON object in a response, tolerating stray prose or code fences."""
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    for candidate in (t, *(_OBJ.findall(t)[:1])):
        try:
            v = json.loads(candidate)
            if isinstance(v, dict):
                return v
        except (ValueError, TypeError):
            continue
    return None
