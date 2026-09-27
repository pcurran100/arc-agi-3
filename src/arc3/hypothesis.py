"""Goal hypotheses as code, held as a weighted population (D-035).

A **goal hypothesis** is a Python function the LLM writes::

    # GOAL: move the blue block onto the red square
    def progress(scene):
        ...
        return 0.4   # in [0, 1]; 1.0 means "the goal is reached"

It is a claim about what winning looks like *and* a heuristic for how close a
state is to it, which is what the planner needs: the frontier is ranked by the
population's weighted progress, so exploration heads towards states the live
hypotheses think are nearer a win.

**The population, and how evidence moves it** (Guardrail 1: soft, never
deleting, never trusting one observation):

- *Win.* When a level is won, every hypothesis is scored on the state the
  winning action was taken from. High progress there is confirmation
  (weight x2 at progress 1, scaled down linearly); low progress is
  disconfirmation (x0.5 at progress 0).
- *False goal.* A visited state a hypothesis scores as reached (>= 0.99) from
  which the level did *not* complete is disconfirmation, x0.7, once per
  distinct state.
- *Flat.* A hypothesis whose progress has been identical on every one of at
  least 20 distinct states says nothing about where to go. It is marked
  uninformative and excluded from ranking (not from the population) until its
  output varies.
- *Broken.* An exception or a non-numeric return scores 0 and costs x0.8, once
  per distinct state, capped so a broken function sinks but is kept for the
  record.

Weights are floored at ``MIN_WEIGHT`` so nothing ever reaches zero.

**The sandbox.** The code runs with a whitelist of builtins, no ``import``,
``numpy`` and ``math`` pre-bound, and a line-count budget enforced with
``sys.settrace``, so a runaway loop costs a bounded amount and raises. This is
protection against a small model's mistakes, not against an adversary: the
code comes from our own model, and the process has no network in any case.

**What this does not do.**

- **Goal hypotheses only.** Dynamics hypotheses from the LLM ("ACTION1 moves the
  player left") are not proposed or tested yet; dynamics still comes from the
  exact state graph (D-034).
- **No hypothesis revision.** A hypothesis is never edited. The LLM may propose
  new ones, which enter at the prior weight; old ones sink.
- **No credit for partial progress.** Only wins and false goals move weights;
  a hypothesis whose progress rises along the path that eventually wins gets
  nothing extra for that.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from .scene import Scene

PRIOR_WEIGHT = 1.0
MIN_WEIGHT = 0.01

#: Line events a single progress() call may execute before it is stopped.
STEP_BUDGET = 20_000

#: Distinct states needed before a constant-output hypothesis counts as flat.
FLAT_AFTER = 20

REACHED = 0.99

_SAFE_BUILTINS = {
    name: getattr(__builtins__, name) if not isinstance(__builtins__, dict) else __builtins__[name]
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int",
        "len", "list", "map", "max", "min", "range", "reversed", "round", "set",
        "sorted", "str", "sum", "tuple", "zip", "isinstance", "ValueError",
        "Exception", "ZeroDivisionError", "IndexError", "KeyError", "True", "False", "None",
    )
    if (name in __builtins__ if isinstance(__builtins__, dict) else hasattr(__builtins__, name))
}

_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
_GOAL = re.compile(r"#\s*GOAL:\s*(.+)")


class BudgetExceeded(RuntimeError):
    pass


def _budgeted(fn: Callable[[Scene], Any], scene: Scene, budget: int) -> Any:
    count = 0

    # Only the hypothesis's own lines are counted. Tracing into numpy or the
    # scene helpers charged the budget for trusted library code: one call to
    # scene.colours() exceeded 20,000 steps inside np.unique.
    def local(frame, event, arg):
        nonlocal count
        if event == "line":
            count += 1
            if count > budget:
                raise BudgetExceeded(f"more than {budget} steps")
        return local

    def tracer(frame, event, arg):
        return local if frame.f_code.co_filename == "<hypothesis>" else None

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        return fn(scene)
    finally:
        sys.settrace(old)


def compile_progress(source: str) -> Callable[[Scene], Any]:
    """Compile LLM code and return its ``progress`` function, or raise."""
    if re.search(r"^\s*(import|from)\s", source, re.M) or "__" in source:
        raise ValueError("imports and dunder access are not allowed")
    ns: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS, "np": np, "math": math}
    exec(compile(source, "<hypothesis>", "exec"), ns)
    fn = ns.get("progress")
    if not callable(fn):
        # Small models rename it (measured: Qwen2.5-Coder-3B wrote
        # `progress_same_colour`). Accept one progress-prefixed function.
        named = [v for k, v in ns.items() if k.startswith("progress") and callable(v)]
        if len(named) != 1:
            raise ValueError("no progress(scene) function defined")
        fn = named[0]
    return fn


def parse_hypotheses(text: str) -> list[tuple[str, str]]:
    """``(goal statement, source)`` per hypothesis in the fenced code blocks.

    One block may hold several hypotheses (small models do this); it is split
    at each ``# GOAL:`` line that is followed by a ``def progress...``.
    """
    out = []
    for block in _BLOCK.findall(text) or [text]:
        starts = [m.start() for m in _GOAL.finditer(block)]
        chunks = ([block[a:b] for a, b in zip(starts, [*starts[1:], len(block)])]
                  if starts else [block])
        for src in chunks:
            if not re.search(r"def\s+progress\w*\s*\(", src):
                continue
            m = _GOAL.search(src)
            out.append(((m.group(1).strip() if m else "(no GOAL line)"), src.strip()))
    return out


@dataclass
class Hypothesis:
    id: int
    goal: str
    source: str
    fn: Optional[Callable[[Scene], Any]]
    origin: str = ""                 # which call proposed it
    weight: float = PRIOR_WEIGHT
    compile_error: Optional[str] = None
    n_errors: int = 0
    wins_confirmed: int = 0
    wins_missed: int = 0
    false_goals: int = 0
    _seen: dict = field(default_factory=dict, repr=False)     # state key -> value
    _penalised: set = field(default_factory=set, repr=False)

    def value(self, key: bytes, scene_fn: Callable[[], Scene]) -> float:
        """Progress on the state ``key`` (cached; scenes are built lazily)."""
        if key in self._seen:
            return self._seen[key]
        v = 0.0
        if self.fn is not None:
            try:
                r = _budgeted(self.fn, scene_fn(), STEP_BUDGET)
                v = float(r)
                if not math.isfinite(v):
                    raise ValueError("non-finite progress")
                v = min(1.0, max(0.0, v))
            except Exception:
                self.n_errors += 1
                if key not in self._penalised and self.n_errors <= 5:
                    self._penalised.add(key)
                    self._scale(0.8)
                v = 0.0
        self._seen[key] = v
        return v

    @property
    def informative(self) -> bool:
        if self.fn is None:
            return False
        vals = set(self._seen.values())
        return not (len(self._seen) >= FLAT_AFTER and len(vals) <= 1)

    def _scale(self, f: float) -> None:
        self.weight = max(MIN_WEIGHT, self.weight * f)

    def describe(self) -> str:
        flag = "" if self.informative else " [flat]"
        return (f"H{self.id} w={self.weight:.2f}{flag} wins+{self.wins_confirmed}"
                f"/-{self.wins_missed} false={self.false_goals} err={self.n_errors}: {self.goal}")


@dataclass
class Population:
    """All goal hypotheses for the current level, with their weights."""

    hypotheses: list[Hypothesis] = field(default_factory=list)
    _next: int = 0

    def add_from_text(self, text: str, origin: str = "") -> list[Hypothesis]:
        added = []
        for goal, src in parse_hypotheses(text):
            try:
                fn, err = compile_progress(src), None
            except Exception as exc:  # kept for the record, never ranks
                fn, err = None, f"{type(exc).__name__}: {exc}"
            h = Hypothesis(id=self._next, goal=goal, source=src, fn=fn,
                           origin=origin, compile_error=err,
                           weight=PRIOR_WEIGHT if fn else MIN_WEIGHT)
            self._next += 1
            self.hypotheses.append(h)
            added.append(h)
        return added

    def live(self) -> list[Hypothesis]:
        return [h for h in self.hypotheses if h.informative]

    def value(self, key: bytes, scene_fn: Callable[[], Scene]) -> Optional[float]:
        """Weighted mean progress over informative hypotheses; None if there are none."""
        live = self.live()
        if not live:
            return None
        tw = sum(h.weight for h in live)
        return sum(h.weight * h.value(key, scene_fn) for h in live) / tw

    def observe_state(self, key: bytes, scene_fn: Callable[[], Scene]) -> None:
        """A state was visited and the level did not complete there."""
        for h in self.hypotheses:
            if h.fn is None:
                continue
            if h.value(key, scene_fn) >= REACHED and key not in h._penalised:
                h._penalised.add(key)
                h.false_goals += 1
                h._scale(0.7)

    def observe_win(self, key: bytes, scene_fn: Callable[[], Scene]) -> None:
        """The level was won by an action taken from state ``key``."""
        for h in self.hypotheses:
            if h.fn is None:
                continue
            v = h.value(key, scene_fn)
            h._scale(0.5 + 1.5 * v)   # x0.5 at v=0, x2 at v=1
            if v >= 0.5:
                h.wins_confirmed += 1
            else:
                h.wins_missed += 1

    def describe(self, limit: int = 8) -> str:
        hs = sorted(self.hypotheses, key=lambda h: -h.weight)[:limit]
        return "\n".join(h.describe() for h in hs) or "(no hypotheses)"
