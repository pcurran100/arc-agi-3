"""The model's simulator and goal as a program; code validates and searches it (D-046).

The model writes, in the sandboxed Python of v3's world model (``v3/world.py``)::

    def predict(state, action):   # what an action does, to the objects it models
        ...
        return state
    def goal(state):              # True / 1.0 when the level should be won
        ...

and names the ids its simulator is meant to get right (``tracked``). Code then

1. **validates** it on every recorded step of the level, comparing only the
   tracked objects - no action is spent. v3 failed here: it demanded that the
   simulator predict *every* object exactly before trusting it, which on real
   games never happened, so the blind explorer played. A partial simulator that
   gets the player right is enough to route the player.
2. **searches** it: breadth-first from the current state for the fewest actions
   that make ``goal`` true, with a node budget and a wall-clock cap.
3. **executes** the path, checking each real step against the simulator on the
   tracked objects. A mismatch invalidates the simulator (a MAJOR surprise); a
   goal that comes true without the level completing disproves the goal.

**What this does not do.**

- *Untracked objects are assumed unchanged* by the search. Whatever the model
  does not model is invisible to the plan; a plan that depends on them is the
  model's mistake to see in the next surprise.
- *Clicks are searched over object centres only* (v3's action space), at most
  ``MAX_CLICKS`` of them.
- *No probabilistic or hidden state* beyond ``state.meta``, which the model may
  use for counters it simulates itself.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from ..v3.world import Check, State, WorldModel, action_space, compare
from .facts import FactBook5

ACCEPT = 0.8             # tracked-object accuracy on recorded steps to trust a simulator
MAX_NODES = 6000
MAX_DEPTH = 80
MAX_SECONDS = 20.0
MAX_CLICKS = 24


@dataclass
class Simulator:
    model: WorldModel
    tracked: tuple[int, ...]
    goal_id: str = ""
    check: Optional[Check] = None
    valid: bool = False
    why: str = ""                  # the validation report, for the next prompt
    plans: int = 0

    def describe(self) -> str:
        t = " ".join(f"#{i}" for i in self.tracked) or "all objects"
        state = "TRUSTED" if self.valid else "NOT TRUSTED"
        return f"simulator for {t} ({state}): {self.why}"


def state_of(fb: FactBook5) -> State:
    return State.from_objects(fb.objects, fb.grid, fb.masks, fb.background)


def _ignore(fb: FactBook5, tracked: tuple[int, ...], steps) -> set:
    ids = set()
    for s in steps:
        ids |= set(s.before) | set(s.after)
    if tracked:
        return ids - set(tracked)
    return set(fb.counters()) | set(fb.movers())


def source_of(s) -> str:
    """The simulator's source from the model's answer: ``lines`` (one string per
    line), or a single ``code`` string as older prompts asked."""
    if not isinstance(s, dict):
        return ""
    lines = s.get("lines")
    if isinstance(lines, list):
        return "\n".join(str(x) for x in lines)
    return str(s.get("code", "") or "")


def build(code: str, tracked, goal_id: str, fb: FactBook5) -> Simulator:
    """Compile and validate. Never raises."""
    tr = tuple(int(m) for t in (tracked or []) for m in re.findall(r"\d+", str(t))[:1])
    wm = WorldModel.compile(code or "")
    sim = Simulator(wm, tr, str(goal_id or "")[:8])
    if wm.error:
        sim.why = f"does not compile: {wm.error}"
        return sim
    validate(sim, fb)
    return sim


def validate(sim: Simulator, fb: FactBook5) -> None:
    steps = [s for s in fb.play_steps() if s.grid_before is not None]
    ign = _ignore(fb, sim.tracked, steps)
    chk = Check()
    for s in steps:
        pred = sim.model.predict(State.before(s, fb.background), s.action)
        miss = compare(pred, s.after, ign)
        chk.n += 1
        if not miss:
            chk.exact += 1
        elif len(chk.counterexamples) < 5:
            chk.counterexamples.append(f"t{s.t} {s.action}: " + "; ".join(miss[:3]))
    sim.check = chk
    if sim.model.error:
        sim.valid, sim.why = False, f"raised while replaying history: {sim.model.error}"
        return
    if chk.n == 0:
        sim.valid, sim.why = True, "no recorded steps yet to check it against (tentative)"
    else:
        sim.valid = chk.accuracy >= ACCEPT
        sim.why = (f"reproduces {chk.exact}/{chk.n} recorded steps on its tracked objects"
                   + ("" if not chk.counterexamples else
                      "; mismatches: " + " | ".join(chk.counterexamples)))
    if sim.model.goal_fn is not None and sim.model.goal(state_of(fb)) >= 0.99:
        sim.why += ("; NOTE: goal() is already true in the current state but the level is "
                    "not complete, so goal() is wrong or incomplete")


@dataclass
class Route:
    actions: list[str]
    states: list[State]            # the predicted state after each action
    reached: bool
    nodes: int
    why: str = ""


def search(sim: Simulator, fb: FactBook5, simple: list[int], click: bool,
           max_nodes: int = MAX_NODES, max_seconds: float = MAX_SECONDS) -> Route:
    """Fewest actions to a state where goal() holds, inside the simulator."""
    m = sim.model
    if m.goal_fn is None:
        return Route([], [], False, 0, "the simulator has no goal() function")
    start = state_of(fb)
    if m.goal(start) >= 0.99:
        return Route([], [], True, 0, "goal() already holds")
    t0 = time.time()
    seen = {start.key()}
    q = deque([(start, [], [])])
    nodes = 0
    while q:
        if nodes >= max_nodes or time.time() - t0 > max_seconds:
            return Route([], [], False, nodes,
                         f"search stopped after {nodes} states without reaching goal()")
        s, path, states = q.popleft()
        nodes += 1
        if len(path) >= MAX_DEPTH:
            continue
        for a in action_space(s, simple, click, max_clicks=MAX_CLICKS):
            t = m.predict(s, a)
            k = t.key()
            if k in seen:
                continue
            seen.add(k)
            if m.goal(t) >= 0.99:
                return Route(path + [a], states + [t], True, nodes, f"{len(path) + 1} actions")
            q.append((t, path + [a], states + [t]))
    return Route([], [], False, nodes,
                 f"goal() is unreachable in the simulator ({nodes} states explored)")


def mismatch(sim: Simulator, predicted: State, fb: FactBook5) -> list[str]:
    """Differences on the tracked objects between a predicted and the real state."""
    ids = set(fb.objects) | set(predicted.objects)
    ign = (ids - set(sim.tracked)) if sim.tracked else (set(fb.counters()) | set(fb.movers()))
    return compare(predicted, fb.objects, ign)
