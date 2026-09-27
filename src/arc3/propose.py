"""Prompts that ask the LLM for goal hypotheses (D-035).

The model is told only what the agent actually knows: the current scene, what
each action has been observed to do, what won the previous level (if one has
been won), and how the existing hypotheses have fared. It is asked for several
*different* hypotheses as code, never for an action.

Prompt text is part of the experiment: change it and recorded calls stop
replaying (`llm.ReplayLLM` keys on the exact messages). Bump ``PROMPT_VERSION``
whenever the wording changes so recordings say which prompt produced them.

**What this does not do.** It gives no hint about what kinds of games exist.
Guardrail 2 forbids hand-crafting the hypothesis space, so the prompt describes
the *interface* (what a scene is, what an action is) and nothing about genres,
typical goals or known games.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from .hypothesis import Hypothesis
from .scene import Scene

PROMPT_VERSION = "goal-v1"

SYSTEM = """\
You help an agent play an unknown turn-based puzzle game on a 64x64 grid of \
16 colours (0-15). Nobody knows the rules or the goal; the agent learns them by \
acting. Your job is to propose hypotheses about what state of the grid WINS the \
current level, written as Python functions the agent can evaluate on any state.

Write each hypothesis as a separate ```python code block that starts with a \
line `# GOAL: <one sentence>` and defines exactly one function:

    def progress(scene):
        return <float between 0.0 and 1.0>   # 1.0 = goal reached

It must be higher for states closer to the goal, so the agent can steer by it. \
Available inside the function: `np` (numpy) and `math`. No imports. The scene \
API is:
  scene.objects        list of objects, largest first. Each has: id, colour, size,
                       x0, y0, x1, y1 (box, x1/y1 exclusive), w, h, cx, cy (centre)
  scene.of_colour(c)   objects of colour c
  scene.count(c)       number of cells of colour c
  scene.colours()      colours present
  scene.grid           64x64 numpy int array, grid[y, x]; -1 marks hidden UI rows
  scene.background     the background colour
Objects are same-colour connected regions; ids are NOT stable between states, \
so find objects by colour, size or position, never by id.
Propose hypotheses that differ from each other. Keep each function short."""


@dataclass
class Evidence:
    """What the agent knows, for the prompt."""

    scene: Scene
    actions: Sequence[str]                      # e.g. ["ACTION1", "CLICK(x,y)"]
    effects: Sequence[str]                      # observed action -> effect lines
    level: int
    actions_used: int
    previous_win: Optional[str] = None          # what changed on the winning move
    hypotheses: Sequence[Hypothesis] = ()


def goal_messages(ev: Evidence, n: int = 3) -> list[dict]:
    parts = [f"Level {ev.level + 1}. Actions used so far on this level: {ev.actions_used}.",
             "Available actions: " + ", ".join(ev.actions) + "."]
    if ev.effects:
        parts.append("What actions have been observed to do (most recent last):")
        parts += ["  " + e for e in ev.effects[-12:]]
    if ev.previous_win:
        parts.append("The previous level was won. On the winning move: " + ev.previous_win)
    if ev.hypotheses:
        parts.append("Hypotheses already being tested, with their evidence so far:")
        for h in ev.hypotheses[:6]:
            parts.append("  " + h.describe())
        parts.append("Propose NEW hypotheses; do not repeat these. A high 'false' count "
                     "means the agent reached states that hypothesis called the goal, "
                     "and the level did not end.")
    parts.append("Current state:")
    parts.append(ev.scene.describe())
    parts.append(f"Propose {n} different goal hypotheses as described.")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(parts)}]


def describe_effect(before: Scene, after: Scene, action: str) -> str:
    """One line: what an action did, in object terms, for the prompt."""
    if (before.grid == after.grid).all():
        return f"{action}: nothing changed"
    moved, changed = [], 0
    b_by = {(o.colour, o.size): o for o in before.objects}
    a_by = {(o.colour, o.size): o for o in after.objects}
    for k, o in a_by.items():
        p = b_by.get(k)
        if p is not None and (p.x0, p.y0) != (o.x0, o.y0):
            moved.append(f"colour {o.colour} size {o.size} moved "
                         f"({o.x0 - p.x0:+d},{o.y0 - p.y0:+d})")
    changed = int(((before.grid != after.grid) & (after.grid >= 0)).sum())
    appeared = len(set(a_by) - set(b_by))
    vanished = len(set(b_by) - set(a_by))
    bits = moved[:3]
    if appeared or vanished:
        bits.append(f"{appeared} object(s) appeared, {vanished} vanished")
    bits.append(f"{changed} cells changed")
    return f"{action}: " + "; ".join(bits)
