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
3. **Code plans.** In order: a queued step; a subgoal route (v4's route.py);
   a search of the model's simulator for the shortest path to its goal
   (sim.py); and only then a model call.
4. **Two call tiers.** *think* (thinking on, full context including the frame)
   after anything MAJOR, at a level start, when the simulator or goal failed,
   and every ``THINK_EVERY`` actions; *next* (thinking off, no frame) when a
   plan finished without trouble. At most ``MAX_NEXT_RUN`` *next* calls in a
   row.

Without a model the step goes to v1's graph explorer, counted in
``stats['explore_steps']`` so a run that used it is visibly not clean.

**What this does not do.**

- *No rival simulators.* One simulator at a time (Guardrail 1 is kept for goals
  in the store, not for simulators); a failed one is shown to the model with its
  counterexamples for repair.
- *No search over clicks beyond object centres* (sim.py).
- *The thinking budget is not capped* (D-044, user's instruction); a truncated
  answer is retried once with twice the room and counted.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from ..agent import GraphAgent
from ..llm import LLM, Truncated
from ..scene import make_scene
from ..types import CLICK, RESET, Action, GameState, Observation
from ..v3.facts import Step, action_name
from ..v3.prompts import parse_json
from ..v3.world import parse_action
from ..v4.expect import Expectation, check
from ..v4.route import Subgoal, reached, route
from ..v4.store import AssumptionStore
from . import sim as simmod
from .facts import FactBook5
from .prompts import (MAX_STEPS, NEXT_SCHEMA, PROMPT_VERSION, THINK_SCHEMA, Context,
                      next_messages, think_messages)

HISTORY = 10
MINOR_SHOWN = 12
THINK_EVERY = 30          # actions without a think call before one is due anyway
MAX_NEXT_RUN = 3          # next calls in a row before a think call is forced
REPEAT_MAJOR = 2          # a small surprise seen this often becomes MAJOR (once)


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

    def __post_init__(self) -> None:
        if self.graph is None:
            self.graph = GraphAgent(seed=self.seed, edge_band=1.0)

    def _bump(self, k: str, n: int = 1) -> None:
        self.stats[k] = self.stats.get(k, 0) + n

    def _major(self, s: str) -> None:
        self.events = [*self.events, s][-20:]
        self.need_think = True
        self._bump("major")

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

        if not self.queue and self.plan:
            self._walk(obs)
        if not self.queue and not self.need_think and self.sim is not None and self.sim.valid:
            self._search(obs)
        if not self.queue:
            self._decide(obs)
            if not self.queue and self.plan:
                self._walk(obs)
            if not self.queue and self.sim is not None and self.sim.valid:
                self._search(obs)
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
                self._major(f"NO ROUTE for subgoal '{sg.describe()}': {why}.")
                return
            if not path:
                self.plan.pop(0)
                continue
            self._bump("routes")
            for k, (a, dx, dy) in enumerate(path):
                last = k == len(path) - 1
                exp = sg.on_arrival if last else Expectation()
                self.queue.append((Action.simple(int(a[6:])), exp, None,
                                   ("arrive", sg.obj) if last else ("route", sg.obj)))
            self._bump("plan_steps", len(path))
            return

    def _search(self, obs: Observation) -> None:
        r = simmod.search(self.sim, self.fb, self._simple(obs), CLICK in obs.available_actions)
        self._bump("searches")
        if not r.reached or not r.actions:
            self.sim.valid = False
            self._major(f"SIMULATOR SEARCH: {r.why}. Repair predict()/goal() or give steps.")
            return
        self.sim.plans += 1
        self._bump("sim_plan_steps", len(r.actions))
        for k, (a, st) in enumerate(zip(r.actions, r.states)):
            act = self._to_action(a, obs)
            if act is None:
                break
            tag = ("sim_goal", None) if k == len(r.actions) - 1 else ("sim", None)
            self.queue.append((act, None, st, tag))

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
        for t, e in self.fb.events:
            if t == step.t:
                if "level-start" in e or e.startswith("RESTORED"):
                    self._major(f"t{step.t} {step.action}: {e}")
                else:
                    self.minors = [*self.minors, f"t{step.t}: {e}"][-MINOR_SHOWN:]
        if getattr(step, "novel", False):
            for n in self.fb.notes.get(step.t, []):
                if n.startswith("NEW"):
                    self._major(f"t{step.t} {step.action}: {n}")
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
                self._major(f"SIMULATOR WRONG at t{step.t} {step.action}: " + "; ".join(miss[:4]))
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
                if on_driven:
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
        if tag and tag[0] == "arrive" and self.plan and verdict in ("ok", "minor miss"):
            self.plan.pop(0)
            self._bump("subgoals_done")
        self._line(f"t{step.t} {step.action} | {exp.describe() if exp else 'simulator'} | "
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
        if self.llm is None or len(self.calls) >= self.max_calls:
            return False
        return not (self.deadline and time.time() > self.deadline)

    def _decide(self, obs: Observation) -> None:
        if not self._model_available():
            return
        think = (self.need_think or self.since_think >= THINK_EVERY
                 or self.next_run >= MAX_NEXT_RUN or len(self.minors) >= MINOR_SHOWN)
        if think:
            self._think(obs)
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

    def _think(self, obs: Observation) -> None:
        data = self._call("think", think_messages(self._context(obs, True)), THINK_SCHEMA,
                          self.think_tokens, True)
        if data is None:
            return
        where = f"L{self.level + 1} t{self.level_actions}"
        self.store.apply(data.get("ops") or [], where)
        self._names(data.get("names") or [])
        s = data.get("simulator")
        if isinstance(s, dict) and str(s.get("code", "")).strip():
            self.sim = simmod.build(str(s["code"]), s.get("tracked") or [], s.get("goal", ""),
                                    self.fb)
            self._bump("sims_written")
            if self.sim.valid:
                self._bump("sims_trusted")
        elif self.sim is not None:
            simmod.validate(self.sim, self.fb)        # re-check against the new evidence
        self.events, self.minors = [], []
        self.need_think, self.since_think, self.next_run = False, 0, 0
        self._take(data, obs)
        self.calls[-1]["purpose"] = str(data.get("purpose", ""))[:200]
        self.calls[-1]["sim"] = self.sim.describe()[:200] if self.sim else None

    def _next(self, obs: Observation) -> None:
        data = self._call("next", next_messages(self._context(obs, False)), NEXT_SCHEMA,
                          self.next_tokens, False)
        self.next_run += 1
        if data is None:
            return
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

    def _take(self, data: dict, obs: Observation) -> None:
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
        self.queue = q
        raw = [d for d in (data.get("plan") or []) if isinstance(d, dict)]
        self.plan = [sg for sg in (Subgoal.parse(d) for d in raw) if sg]
        if len(self.plan) < len(raw):
            self._minor(f"{len(raw) - len(self.plan)} subgoal(s) not understood: 'object' must be "
                        "'#id' and 'to' '#id' or 'x,y'", ("bad_subgoal",))
        self._bump("llm_steps", len(q))
        self.calls[-1]["steps"] = len(q)
        self.calls[-1]["subgoals"] = len(self.plan)

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
        think = sum(1 for c in self.calls if c.get("kind") == "think")
        return (f"level {self.level}, calls {len(self.calls)} (think {think}, "
                f"next {len(self.calls) - think}), {clean}, acc {self.accuracy()}, "
                + ", ".join(f"{k}={v}" for k, v in sorted(self.stats.items())))
