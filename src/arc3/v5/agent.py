"""The v5 controller: think rarely, plan in code, tolerate small surprises (D-046).

v4 (D-042, D-044) asked the model before nearly every action: an exact
prediction for each step, the first slip dropping the rest of the queue, two
calls (update, then act) per surprise. On five games (2026-10-29) that was ~1.3
calls per action, ~10k thinking tokens per call and no level completed. v5
changes the loop, keeping v4's store, routing and checker:

1. **The fact layer is v5's** (facts.py): permanent ids, deformation not
   reported as motion, counters moved to an INTERFACE line, an object sheet and
   an action-effect table instead of an event log.
2. **Surprises are sorted.** MAJOR - the outcome differed, the driven object
   did something else, a new colour or shape appeared, an object returned to
   its level-start position, the simulator mispredicted a tracked object, or
   one small surprise repeated - stops the plan and calls *think*. MINOR -
   any other wrong or unpredicted detail - is logged, shown at the next call,
   and does not stop anything.
3. **Code plans, and its plan wins.** With a TRUSTED simulator, code searches
   it (sim.py) for the shortest path to the model's goal(), or failing that to
   the current subgoal, and that route replaces any explicit steps the model
   gave (counted in ``steps_overridden``). Without a trusted simulator or a
   route: the model's explicit steps, then a subgoal walk (v4's route.py).
   Only when code has nothing to do is the model called. (2026-10-30 run: ls20
   had a simulator trusted on 10/10 recorded steps and code walked the
   model's subgoals instead; zero actions came from a simulator search.)
4. **Cheap repair first.** A simulator that does not compile, raises, or no
   longer reproduces the recorded steps (including one that just mispredicted
   a tracked object) goes back to the model in a *fix* call - thinking off,
   the error and the source - at most ``MAX_FIXES`` times, before any think
   call. (2026-10-30: a tu93 SyntaxError and a vc33 ``import re`` each cost a
   2-4 minute think call to notice.)
5. **Two call tiers.** *think* (thinking on, full context including the frame)
   only at genuinely hard moments: a level start, a MAJOR surprise (outcome
   differed, game over, a jump back to the start, the driven object did
   something *other* than predicted), a disproved goal, a simulator that a
   cheap fix could not save, and every ``THINK_EVERY`` actions. *next*
   (thinking off, no frame) for everything routine: a plan finished, an
   exploratory step on the driven object that was simply *blocked* (nothing
   changed there), a subgoal code could not route, or a STOP. At most
   ``MAX_NEXT_RUN`` *next* calls in a row. (2026-10-30: 20 of 21 good calls
   were think calls, because every blocked test move and every NO ROUTE was
   MAJOR and level-start/new-shape notes from the probe forced think.)

Without a model the step goes to v1's graph explorer, counted in
``stats['explore_steps']`` so a run that used it is visibly not clean. A
*lost network* is different from having no model: the client retries
connection and DNS failures for ten minutes, and if the server is still
unreachable the agent sets ``halted`` and the caller stops the game, marked
invalid, rather than letting the explorer play blind (2026-10-30 run).

**What this does not do.**

- *No rival simulators.* One simulator at a time (Guardrail 1 is kept for goals
  in the store, not for simulators); a failed one is shown to the model with its
  counterexamples for repair.
- *No search over clicks beyond object centres* (sim.py).
- *The thinking budget is not capped* (D-044, user's instruction); a truncated
  answer is retried once with twice the room and counted.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from ..agent import GraphAgent
from ..llm import LLM, NetworkDown, Truncated
from ..scene import make_scene
from ..types import CLICK, RESET, Action, GameState, Observation
from ..v3.facts import Step, action_name
from ..v3.prompts import parse_json
from ..v3.world import parse_action
from ..v4.expect import Expectation, Item, check
from ..v4.route import Subgoal, reached, route
from ..v4.store import AssumptionStore
from . import sim as simmod
from .facts import FactBook5
from .prompts import (FIX_SCHEMA, MAX_STEPS, NEXT_SCHEMA, PROMPT_VERSION, THINK_SCHEMA,
                      Context, fix_messages, next_messages, think_messages)

HISTORY = 10
MINOR_SHOWN = 12
THINK_EVERY = 30          # actions without a think call before one is due anyway
MAX_NEXT_RUN = 6          # next calls in a row before a think call is forced
REPEAT_MAJOR = 2          # a small surprise seen this often becomes MAJOR (once)
MAX_REROUTES = 3          # blocked route steps re-routed by code before the model is asked
MAX_FIXES = 2             # cheap (thinking off) simulator repairs before a think call
re_id = re.compile(r"#\s*\d+")
re_xy = re.compile(r"^\(?\s*x?\s*-?\d+\s*[, ]\s*y?\s*-?\d+\s*\)?$")


def _thinking_backend(llm):
    """The innermost OpenAI-compatible backend (wrappers expose .inner / .fallback)."""
    seen = set()
    while llm is not None and id(llm) not in seen:
        seen.add(id(llm))
        if hasattr(llm, "extra") and hasattr(llm, "base_url"):
            return llm
        llm = getattr(llm, "inner", None) or getattr(llm, "fallback", None)
    return None


@dataclass
class V5Agent:
    llm: Optional[LLM] = None
    seed: int = 0
    baselines: Optional[tuple] = None
    total_levels: Optional[int] = None
    temperature: float = 0.6
    max_calls: int = 600
    think_tokens: int = 16000
    next_tokens: int = 3000
    deadline: float = 0.0
    graph: GraphAgent = field(default=None)  # type: ignore[assignment]
    fb: FactBook5 = field(default_factory=FactBook5)
    store: AssumptionStore = field(default_factory=AssumptionStore)
    sim: Optional[simmod.Simulator] = None
    level: int = 0
    level_actions: int = 0
    total_actions: int = 0
    queue: list = field(default_factory=list)      # [(Action, Expectation|None, State|None, tag)]
    plan: list = field(default_factory=list)       # [Subgoal]
    history: list = field(default_factory=list)
    minors: list = field(default_factory=list)
    events: list = field(default_factory=list)
    level_notes: list = field(default_factory=list)
    need_think: bool = True
    want_think: bool = False          # think at the next decision, without interrupting
    probed: bool = False
    since_think: int = 0
    next_run: int = 0
    calls: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    _sigs: Counter = field(default_factory=Counter)
    _escalated: set = field(default_factory=set)
    _last_action: Optional[Action] = None
    _pending: Optional[tuple] = None
    _attained: set = field(default_factory=set)
    _sim_goal_hit: bool = False
    _reroutes: int = 0
    _player_look: Optional[tuple] = None      # (colour, shape) of last level's controlled object
    halted: str = ""                  # set when the model server is unreachable: stop the game
    calls_why: str = ""               # what forced the last think call (for the log)
    _search_failed: tuple = ()        # (simulator, action count) of the last failed search
    repair_due: str = ""              # a simulator mispredicted: try a cheap fix first

    def __post_init__(self) -> None:
        if self.graph is None:
            self.graph = GraphAgent(seed=self.seed, edge_band=1.0)

    def _bump(self, k: str, n: int = 1) -> None:
        self.stats[k] = self.stats.get(k, 0) + n

    def _major(self, s: str) -> None:
        self.events = [*self.events, s][-20:]
        self.need_think = True
        self._bump("major")

    def _soft(self, s: str) -> None:
        """Worth seeing, not worth interrupting or thinking about: shown with the
        minors at the next call, whatever its tier. (Until 2026-10-30 this forced
        the next call to be a think call; the probe's level-start returns and
        new counters made every first call a think call for no reason.)"""
        self.minors = [*self.minors, s][-MINOR_SHOWN:]
        self._bump("soft")

    def _stop(self, s: str) -> None:
        """Routine: the plan cannot go on, but nothing about the game is in doubt
        (a blocked step, an unroutable subgoal). Stops the plan; the next call is
        a *next* call that sees why."""
        self.events = [*self.events, s][-20:]
        self._drop("stop")
        self._bump("stops")

    def _jumped(self, step: Step) -> list[str]:
        """Objects that moved further than any displacement seen for this action
        (a teleport back to the start, not a step)."""
        known: dict[int, set] = {}
        for p in self.fb.player_candidates():
            known[p["id"]] = {tuple(d) for d in p["moves"].values()}
        out = []
        for c in step.changes:
            if "moved" not in c.kinds:
                continue
            ds = known.get(c.id, set())
            lim = max((abs(dx) + abs(dy) for dx, dy in ds), default=0)
            if lim and abs(c.dx) + abs(c.dy) > lim:
                out.append(f"#{c.id} moved ({c.dx:+d},{c.dy:+d})")
        return out

    def _minor(self, s: str, sig: tuple) -> None:
        """Logged; escalated to MAJOR the REPEAT_MAJOR-th time, once per signature."""
        self._sigs[sig] += 1
        if self._sigs[sig] >= REPEAT_MAJOR and sig not in self._escalated:
            self._escalated.add(sig)
            self._major(f"REPEATED ({self._sigs[sig]} times): {s}")
            return
        self.minors = [*self.minors, s][-MINOR_SHOWN:]
        self._bump("minor")

    # -- the loop ------------------------------------------------------------------

    def act(self, obs: Observation) -> Action:
        """The next action. **Returning it is free; the caller spends it.**"""
        graph_choice = self.graph.act(obs)
        frame = obs.last if obs.last is not None else self.fb.grid
        if self.total_levels is None and obs.win_levels:
            self.total_levels = obs.win_levels
        if obs.levels_completed > self.level:
            self._finish_level(obs)
            if frame is not None:
                self.fb.observe(frame, None)
        elif frame is not None:
            step = self.fb.observe(frame, self._last_action)
            if step is not None and not step.reset:
                self._judge(step, "game_over" if obs.state == GameState.GAME_OVER else "continue")
        if obs.state == GameState.GAME_OVER:
            self._drop("game over")
            self._major(f"GAME OVER after {self.level_actions} actions on this level; the level "
                        "restarts. Think about what caused it.")
            return self._emit(Action.simple(RESET), None)
        if obs.state == GameState.WIN or frame is None:
            return self._emit(Action.simple(RESET), None)

        if not self.queue and not self.probed:
            self._probe(obs)
        if not self.queue and self.repair_due:
            why, self.repair_due = self.repair_due, ""
            if not self.need_think and not self._repair(obs, why):
                self._major(why + " (the cheap repair did not fix it)")
        if not self.queue and not self.need_think:
            self._code_plan(obs)
        if not self.queue:
            self._decide(obs)
        if self.halted:
            # The caller checks ``halted`` and must not spend this action.
            return Action.simple(RESET)
        if self.queue:
            a, exp, pred, tag = self.queue.pop(0)
            return self._emit(a, (exp, pred, tag))
        self._bump("explore_steps")
        return self._emit(graph_choice, None)

    def _emit(self, a: Action, pending: Optional[tuple]) -> Action:
        self.graph.took(a)
        self._last_action = a
        self._pending = pending
        if not a.is_reset:
            self.level_actions += 1
            self.total_actions += 1
            self.since_think += 1
        return a

    def _drop(self, why: str) -> None:
        if self.queue or self.plan:
            self._bump("plans_dropped")
        self.queue, self.plan = [], []

    # -- code planning -------------------------------------------------------------

    def _probe(self, obs: Observation) -> None:
        """Try each non-click action once at the start of the game, before the first
        think call. Debug run 2026-10-30: the model spent one think call (60-190 s)
        per action learning what the four actions do; code can learn that for four
        actions and no calls (Guardrail 4: early actions go to dynamics)."""
        self.probed = True
        simple = self._simple(obs)
        if self.level or self.level_actions or len(simple) < 2 or self.llm is None:
            return
        for a in simple:
            self.queue.append((Action.simple(a), None, None, ("probe", None)))
        self._bump("probe_steps", len(simple))

    def _simple(self, obs: Observation) -> list[int]:
        return [a for a in sorted(obs.available_actions) if a not in (RESET, CLICK)]

    def _walk(self, obs: Observation) -> None:
        names = [f"ACTION{a}" for a in self._simple(obs)]
        while self.plan:
            sg = self.plan[0]
            self.fb.focus = sg.obj
            path, why = route(self.fb, sg, names)
            if path is None:
                self._bump("routes_failed")
                self.plan = []
                self._stop(f"NO ROUTE for subgoal '{sg.describe()}': {why}.")
                return
            if not path:
                self.plan.pop(0)
                continue
            self._bump("routes")
            for k, (a, dx, dy) in enumerate(path):
                last = k == len(path) - 1
                move = (Item(f"#{sg.obj}", "moved", dx, dy),)
                exp = (Expectation(move + sg.on_arrival.items, "any", sg.on_arrival.outcome)
                       if last else Expectation(move, "any", "continue"))
                self.queue.append((Action.simple(int(a[6:])), exp, None,
                                   ("arrive", sg.obj) if last else ("route", sg.obj)))
            self._bump("plan_steps", len(path))
            return

    def _sim_ready(self) -> bool:
        return (self.sim is not None and self.sim.valid
                and self._search_failed != (id(self.sim), self.total_actions))

    def _code_plan(self, obs: Observation) -> int:
        """Code's own plan: a search of a trusted simulator, else a subgoal walk.
        Returns the number of actions queued."""
        if self._sim_ready() and self._search(obs):
            return len(self.queue)
        if self.plan:
            self._walk(obs)
        return len(self.queue)

    def _search(self, obs: Observation) -> int:
        """Search the trusted simulator for its goal(), then for the current
        subgoal (when the simulator models the subgoal's object). Queues the
        first route found; returns its length (0: nothing found)."""
        simple, click = self._simple(obs), CLICK in obs.available_actions
        tries = []
        if self.sim.model.goal_fn is not None:
            tries.append((None, "goal()", None))
        if self.plan and (not self.sim.tracked or self.plan[0].obj in self.sim.tracked):
            sg = self.plan[0]
            tries.append((simmod.subgoal_test(sg), f"subgoal '{sg.describe()}'", sg))
        why = []
        for target, what, sg in tries:
            r = simmod.search(self.sim, self.fb, simple, click, target=target, what=what)
            self._bump("searches")
            if not r.reached or not r.actions:
                why.append(r.why)
                continue
            acts = [self._to_action(a, obs) for a in r.actions]
            if any(a is None for a in acts):
                why.append(f"{what}: the route uses an unavailable action")
                continue
            self.sim.plans += 1
            self._bump("sim_plan_steps", len(acts))
            last = ("sim_goal", None) if sg is None else ("sim_arrive", sg.obj)
            for k, (act, st) in enumerate(zip(acts, r.states)):
                self.queue.append((act, None, st, last if k == len(acts) - 1 else ("sim", None)))
            return len(acts)
        self._search_failed = (id(self.sim), self.total_actions)
        if why:
            self.minors = [*self.minors, "SIMULATOR SEARCH: " + "; ".join(why)][-MINOR_SHOWN:]
        return 0

    # -- checking ------------------------------------------------------------------

    def _driven(self) -> set:
        ids = set()
        for p in self.fb.player_candidates()[:1]:
            ids |= set(self.fb.group_of(p["id"]))
        if self.fb.focus is not None:
            ids |= set(self.fb.group_of(self.fb.focus))
        if self.sim is not None:
            ids |= set(self.sim.tracked)
        return ids

    def _judge(self, step: Step, outcome: str) -> None:
        skip = set(self.fb.counters())
        real = [c for c in step.changes if c.id not in skip]
        actual = "; ".join(c.describe() for c in real[:5]) or "nothing (only the interface)"
        if outcome != "continue":
            actual += f"; {outcome.upper()}"
        jumped = self._jumped(step)
        for t, e in self.fb.events:
            if t != step.t:
                continue
            if ("level-start" in e or e.startswith("RESTORED")) and jumped:
                self._major(f"t{step.t} {step.action}: {e} - a jump, not an ordinary move "
                            f"({', '.join(jumped)})")
            else:
                # Debug run 2026-10-30: ACTION1 then ACTION2 put the player back on its
                # start square and was read as a lost life; a split/return without a
                # jump is noted and thought about when the current plan is done.
                self._soft(f"t{step.t}: {e}")
        if getattr(step, "novel", False):
            for n in self.fb.notes.get(step.t, []):
                if n.startswith("NEW"):
                    # Counters appear on the first action, before they can be
                    # classified (both debug games): think next, do not interrupt.
                    self._soft(f"t{step.t} {step.action}: {n}")
        for note in self.store.observe(step, outcome):
            self._minor(f"t{step.t} rule check failed: {note}", ("rule", note[:40]))
        self._check_goals(step, outcome)
        if self._pending is None:
            self._line(f"t{step.t} {step.action} | (explorer) | {actual} |")
            return
        exp, pred, tag = self._pending
        self._pending = None
        verdict = "ok"
        if pred is not None:                       # a simulator step
            miss = simmod.mismatch(self.sim, pred, self.fb) if self.sim else []
            if miss:
                verdict = "SIM WRONG"
                self._bump("sim_wrong")
                self.sim.valid = False
                self._drop("simulator")
                msg = f"SIMULATOR WRONG at t{step.t} {step.action}: " + "; ".join(miss[:4])
                self.events = [*self.events, msg][-20:]
                self.repair_due = msg
            if tag and tag[0] == "sim_goal" and outcome != "level_complete":
                verdict = "GOAL NOT WON"
                self._bump("goals_disproved")
                if self.sim and self.sim.goal_id:
                    self.store.goals_failed([self.sim.goal_id], f"t{step.t}",
                                            "the simulator's goal() held; the level did not complete")
                if self.sim:
                    self.sim.valid = False
                self._major(f"GOAL TEST at t{step.t}: your simulator's goal() was reached but the "
                            "level did NOT complete - the goal is wrong or needs something first.")
        elif exp is not None and exp.informative:
            v = check(exp, step, outcome, ignore=frozenset(self.fb.tickers()))
            driven = self._driven()
            if outcome != exp.outcome:
                verdict = "WRONG OUTCOME"
                self._bump("predictions_wrong")
                self._drop("outcome")
                self._major(f"t{step.t} {step.action}: expected {exp.outcome}, got {outcome}.")
            elif v.failures:
                bad = [f for f in v.failures]
                on_driven = any(any(f"#{i} " in f or f"#{i}," in f or f.endswith(f"#{i}")
                                    for i in driven) for f in bad)
                self._bump("predictions_wrong")
                if on_driven and tag and tag[0] in ("route", "arrive") and self.plan \
                        and self._reroutes < MAX_REROUTES:
                    # A blocked route step: the fact layer has just learned the blocking
                    # colour and place, so code re-routes before asking the model.
                    verdict = "blocked, re-routing"
                    self._reroutes += 1
                    self.queue = []
                    self._bump("reroutes")
                    self.minors = [*self.minors, f"t{step.t} {step.action}: route step blocked "
                                   f"({'; '.join(bad[:1])}); code re-routed"][-MINOR_SHOWN:]
                elif on_driven and all("nothing changed there" in f for f in bad):
                    # A move into a wall: the effect table and fact layer record it;
                    # it answers the test, it does not overturn a belief.
                    verdict = "blocked"
                    self._stop(f"t{step.t} {step.action}: BLOCKED - {'; '.join(bad[:2])}")
                elif on_driven:
                    verdict = "WRONG (driven object)"
                    self._drop("driven")
                    self._major(f"t{step.t} {step.action}: {'; '.join(bad[:3])}")
                else:
                    verdict = "minor miss"
                    for f in bad[:3]:
                        self._minor(f"t{step.t} {step.action}: {f}", ("miss", f.split(",")[0][:50]))
            else:
                self._bump("predictions_ok")
                if any(c.id not in skip for c in step.changes) or outcome != "continue":
                    self._bump("predictions_ok_real")
            for u in v.unpredicted:
                self._minor(f"t{step.t} {step.action}: also {u}", ("extra", u.split(" ", 1)[0]))
        if tag and tag[0] in ("arrive", "sim_arrive") and self.plan \
                and verdict in ("ok", "minor miss"):
            self.plan.pop(0)
            self._bump("subgoals_done")
        what = exp.describe() if exp else ("simulator" if pred is not None else "code probe")
        self._line(f"t{step.t} {step.action} | {what} | "
                   f"{actual} | {verdict}")

    def _check_goals(self, step: Step, outcome: str) -> None:
        for g in self.store.goals():
            sg = Subgoal.parse(g.attain) if g.attain else None
            if sg is None:
                continue
            try:
                ok = reached(self.fb, sg)
            except (KeyError, ValueError):
                ok = False
            if not ok:
                self._attained.discard(g.id)
                continue
            if g.id in self._attained or outcome == "level_complete":
                continue
            self._attained.add(g.id)
            self.store.goals_failed([g.id], f"t{step.t}",
                                    "its condition held and the level did not complete")
            self._bump("goal_tests_failed")
            self._major(f"GOAL TEST at t{step.t}: {g.id}'s condition holds and the level did NOT "
                        f"complete - {g.id} is wrong or needs something first.")

    def _line(self, s: str) -> None:
        self.history = [*self.history, s][-HISTORY:]

    # -- calling the model --------------------------------------------------------------

    def _model_available(self) -> bool:
        if self.halted or self.llm is None or len(self.calls) >= self.max_calls:
            return False
        return not (self.deadline and time.time() > self.deadline)

    def _decide(self, obs: Observation) -> None:
        if not self._model_available():
            return
        think = (self.need_think or self.want_think or self.since_think >= THINK_EVERY
                 or self.next_run >= MAX_NEXT_RUN)
        self.calls_why = ("need" if self.need_think else "want" if self.want_think else
                          "periodic" if self.since_think >= THINK_EVERY else
                          "next_run" if self.next_run >= MAX_NEXT_RUN else "")
        if think:
            self._think(obs)
            for c in reversed(self.calls):
                if c.get("kind") == "think":
                    c["why"] = self.calls_why
                    break
        else:
            self._next(obs)

    def _available_names(self, obs: Observation) -> list[str]:
        out = [f"ACTION{a}" for a in self._simple(obs)]
        if CLICK in obs.available_actions:
            out.append("CLICK(x,y)")
        return out

    def _context(self, obs: Observation, full: bool) -> Context:
        return Context(
            level=self.level, total_levels=self.total_levels,
            actions_on_level=self.level_actions, available=self._available_names(obs),
            facts=self.fb.describe(), store=self.store.describe(),
            grid=make_scene(self.fb.grid).grid_text() if full else "",
            sim=self.sim.describe() if self.sim else "",
            plan="; ".join(sg.describe() for sg in self.plan),
            history=list(self.history), minors=list(self.minors), events=list(self.events),
            level_notes=list(self.level_notes))

    def _call(self, kind: str, messages: list, schema: dict, tokens: int,
              thinking: bool) -> Optional[dict]:
        t0 = time.time()
        be = _thinking_backend(self.llm)
        saved = None
        if be is not None:
            saved = dict(be.extra)
            ctk = dict(be.extra.get("chat_template_kwargs", {}))
            ctk["enable_thinking"] = thinking
            be.extra = {**be.extra, "chat_template_kwargs": ctk}
        text, last = "", {}
        try:
            for attempt in range(2):
                try:
                    text = self.llm.complete(messages, max_tokens=tokens * (attempt + 1),
                                             temperature=self.temperature,
                                             seed=self.seed + len(self.calls), schema=schema)
                    last = getattr(self.llm, "last", None) or {}
                    break
                except Truncated:
                    self._bump("truncated")
                    text = ""
                except NetworkDown as exc:
                    self._bump("network_down")
                    self.halted = f"network down: {exc}"[:300]
                    self.calls.append({"kind": kind, "ok": False, "level": self.level,
                                       "at": self.level_actions, "error": self.halted,
                                       "seconds": round(time.time() - t0, 2)})
                    return None
                except Exception as exc:  # a dead server must not end the game
                    self._bump("llm_errors")
                    self.calls.append({"kind": kind, "ok": False, "level": self.level,
                                       "at": self.level_actions,
                                       "error": f"{type(exc).__name__}: {exc}"[:200],
                                       "seconds": round(time.time() - t0, 2)})
                    return None
        finally:
            if be is not None and saved is not None:
                be.extra = saved
        data = parse_json(text)
        usage = last.get("usage") or {}
        self.calls.append({"kind": kind, "thinking": thinking, "ok": data is not None,
                           "level": self.level, "at": self.level_actions,
                           "seconds": round(time.time() - t0, 2),
                           "completion_tokens": usage.get("completion_tokens"),
                           "prompt_tokens": usage.get("prompt_tokens"),
                           "prompt": PROMPT_VERSION})
        self._bump(f"calls_{kind}")
        return data

    def _repair(self, obs: Observation, why: str = "") -> bool:
        """Up to MAX_FIXES thinking-off calls that get the simulator's error and
        source back and return fixed code. True when the simulator is trusted."""
        for _ in range(MAX_FIXES):
            if self.sim is None or not self._model_available():
                return False
            problem = (why + "\n" if why else "") + self.sim.describe()
            data = self._call("fix", fix_messages(self._context(obs, False),
                                                  self.sim.model.source.strip(), problem),
                              FIX_SCHEMA, 2 * self.next_tokens, False)
            self._bump("sim_fixes")
            code = simmod.source_of(data) if data else ""
            if not code.strip():
                continue
            self.sim = simmod.build(code, data.get("tracked") or list(self.sim.tracked),
                                    data.get("goal_id") or self.sim.goal_id, self.fb)
            self.calls[-1]["sim"] = self.sim.describe()[:200]
            if self.sim.valid:
                self._bump("sims_fixed")
                return True
            why = ""
        return False

    def _think(self, obs: Observation) -> None:
        data = self._call("think", think_messages(self._context(obs, True)), THINK_SCHEMA,
                          self.think_tokens, True)
        if data is None:
            return
        rec = self.calls[-1]
        where = f"L{self.level + 1} t{self.level_actions}"
        self.store.apply(data.get("ops") or [], where)
        self._names(data.get("names") or [])
        s = data.get("simulator")
        code = simmod.source_of(s)
        if code.strip():
            self.sim = simmod.build(code, s.get("tracked") or [],
                                    s.get("goal_id", s.get("goal", "")), self.fb)
            self._bump("sims_written")
            if self.sim.valid:
                self._bump("sims_trusted")
        elif self.sim is not None:
            was = self.sim.valid
            simmod.validate(self.sim, self.fb)        # re-check against the new evidence
            code = self.sim.model.source if was else ""
        self.events, self.minors = [], []
        self.need_think, self.want_think, self.since_think, self.next_run = False, False, 0, 0
        self.repair_due = ""
        rec["sim"] = self.sim.describe()[:200] if self.sim else None
        if self.sim is not None and not self.sim.valid and code.strip():
            self._repair(obs)
        self._take(data, obs, rec)
        rec["purpose"] = str(data.get("purpose", ""))[:200]

    def _next(self, obs: Observation) -> None:
        data = self._call("next", next_messages(self._context(obs, False)), NEXT_SCHEMA,
                          self.next_tokens, False)
        self.next_run += 1
        if data is None:
            return
        self.events = []
        self._take(data, obs)
        self.calls[-1]["purpose"] = str(data.get("purpose", ""))[:200]

    def _names(self, names: list) -> None:
        for n in names:
            if not isinstance(n, dict):
                continue
            s = str(n.get("id", "")).lstrip("#")
            if s.isdigit():
                i = int(s)
                if n.get("name"):
                    self.fb.names[i] = str(n["name"])[:24]
                if n.get("role"):
                    self.fb.roles[i] = str(n["role"])[:80]

    def _take(self, data: dict, obs: Observation, rec: Optional[dict] = None) -> None:
        q, bad = [], None
        for s in (data.get("steps") or [])[:MAX_STEPS]:
            if not isinstance(s, dict):
                continue
            a = self._to_action(str(s.get("action", "")), obs)
            if a is None:
                bad = str(s.get("action", ""))[:20]
                break
            q.append((a, Expectation.parse(s.get("expect")), None, None))
        if bad:
            self._minor(f"action {bad!r} is not available or not understood; available: "
                        f"{', '.join(self._available_names(obs))}", ("bad_action", bad))
        raw = [self._resolve_subgoal(d) for d in (data.get("plan") or []) if isinstance(d, dict)]
        self.plan = [sg for sg in (Subgoal.parse(d) for d in raw) if sg]
        self._reroutes = 0
        if len(self.plan) < len(raw):
            self._minor(f"{len(raw) - len(self.plan)} subgoal(s) not understood: 'object' must be "
                        "'#id' and 'to' '#id' or 'x,y'", ("bad_subgoal",))
        rec = rec if rec is not None else self.calls[-1]
        rec["steps"], rec["subgoals"] = len(q), len(self.plan)
        # A route through a trusted simulator takes precedence over explicit steps.
        self.queue = []
        n = self._search(obs) if self._sim_ready() else 0
        if n:
            rec["sim_route"] = n
            if q:
                rec["overridden"] = len(q)
                self._bump("steps_overridden", len(q))
            return
        self.queue = q
        self._bump("llm_steps", len(q))
        if not q and self.plan:
            self._walk(obs)
        if not self.queue and not self.need_think:
            self._major("Your last answer left nothing to do: no steps, no routable subgoal, "
                        "and no route to goal() in a trusted simulator. Give steps, or write "
                        "or repair the simulator.")

    def _resolve(self, s: str) -> str:
        """'#7' stays; a name or role the model gave ('player', 'player_body') becomes
        its '#id'; 'player'/'me' with no such name becomes the most controlled-looking
        object. Debug run (d): 5 of 6 plans named objects by name and were dropped."""
        t = str(s or "").strip()
        if re_id.fullmatch(t) or re_xy.search(t):
            return t
        m = re_id.search(t)
        if m and t.startswith(("#", "player #", "object #")):
            return m.group(0).replace(" ", "")
        key = t.lower().strip(" #")
        for table in (self.fb.names, self.fb.roles):
            for i, n in table.items():
                if i in self.fb.objects and n.lower() == key:
                    return f"#{i}"
        for table in (self.fb.names, self.fb.roles):
            for i, n in table.items():
                if i in self.fb.objects and key and key in n.lower():
                    return f"#{i}"
        if key.startswith(("player", "me", "avatar", "controlled")):
            pcs = self.fb.player_candidates()
            if pcs:
                return f"#{pcs[0]['id']}"
            # A new level, nothing moved yet: the object that looks like the last
            # level's controlled object.
            if self._player_look is not None:
                for o in self.fb.objects.values():
                    if (o.colour, o.shape) == self._player_look:
                        return f"#{o.id}"
        return t

    def _resolve_subgoal(self, d: dict) -> dict:
        return {**d, "object": self._resolve(d.get("object")), "to": self._resolve(d.get("to"))}

    def _to_action(self, s: str, obs: Observation) -> Optional[Action]:
        if s.strip().upper() == "RESET":
            return Action.simple(RESET)
        p = parse_action(s)
        if p is None:
            return None
        aid, x, y = p
        if aid not in obs.available_actions:
            return None
        if aid == CLICK:
            return Action.click(max(0, min(63, x)), max(0, min(63, y)))
        return Action.simple(aid)

    # -- levels ------------------------------------------------------------------------------

    def _finish_level(self, obs: Observation) -> None:
        won_with = action_name(self._last_action) if self._last_action else "?"
        how = "an explorer step"
        if self._pending is not None:
            exp, pred, tag = self._pending
            if pred is not None:
                how = "your simulator's plan" + (" (goal reached as searched)"
                                                 if tag and tag[0] == "sim_goal" else "")
                if self.sim and self.sim.goal_id:
                    self.store.goals_won([self.sim.goal_id], f"L{self.level + 1} win")
            elif exp is not None:
                how = ("a step you predicted would win" if exp.outcome == "level_complete"
                       else "a step you did NOT expect to win")
        self._pending = None
        top = sorted(self.store.live(), key=lambda a: -a.confidence)[:5]
        self.level_notes.append(
            f"Level {self.level + 1}: won in {self.level_actions} actions by {won_with}, "
            f"{how}. Most confident then: " + "; ".join(f"{a.id} {a.statement}" for a in top)
            + (f". Simulator then: {self.sim.model.source[:600]}" if self.sim else ""))
        self._bump("levels")
        self.level = obs.levels_completed
        self.level_actions = 0
        pcs = self.fb.player_candidates()
        if pcs and pcs[0]["id"] in self.fb.objects:
            o = self.fb.objects[pcs[0]["id"]]
            self._player_look = (o.colour, o.shape)
        self.fb = FactBook5(prior_moves=dict(pcs[0]["moves"]) if pcs else {})
        self._last_action, self.queue, self.plan, self.history = None, [], [], []
        self._sigs, self._escalated, self._attained = Counter(), set(), set()
        if self.sim is not None:
            self.sim.valid = False        # ids differ on a new level; re-validated at think
        self.store.new_level(self.level)
        self._major(f"NEW LEVEL {self.level + 1}. Object ids are new. Assumptions and the "
                    "simulator are carried as hypotheses: check them against this level, and "
                    "re-write the simulator with this level's ids.")

    # -- reporting ----------------------------------------------------------------------------

    def accuracy(self) -> dict:
        s = self.stats
        ok, bad = s.get("predictions_ok", 0), s.get("predictions_wrong", 0)
        return {"all": round(ok / (ok + bad), 3) if ok + bad else None, "n": ok + bad,
                "sim_steps": s.get("sim_plan_steps", 0), "sim_wrong": s.get("sim_wrong", 0)}

    def describe(self) -> str:
        clean = "CLEAN" if not self.stats.get("explore_steps") else \
            f"NOT CLEAN ({self.stats['explore_steps']} explorer steps)"
        if self.halted:
            clean = f"INVALID ({self.halted[:80]}), {clean}"
        kinds = Counter(c.get("kind") for c in self.calls)
        return (f"level {self.level}, calls {len(self.calls)} (think {kinds['think']}, "
                f"next {kinds['next']}, fix {kinds['fix']}), {clean}, acc {self.accuracy()}, "
                + ", ".join(f"{k}={v}" for k, v in sorted(self.stats.items())))
