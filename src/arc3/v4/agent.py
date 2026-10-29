"""The v4 controller: predict -> act -> check -> update assumptions (D-042, D-044).

Per step, deterministically:

1. The fact layer observes the new frame (``FactBook4``: v3's tracked objects
   and typed changes, without occlusion artefacts, with tickers, returns to the
   level start, novelty and passability).
2. Every assumption with a machine-checkable rule for the action just taken is
   checked (store.observe); the expectation attached to that action is checked
   (expect.check). A correct prediction supports the ontology and dynamics
   assumptions it named. A wrong one is a **surprise**: the queued steps and the
   plan are dropped and an update is due. Goal assumptions are tested by code:
   a goal whose ``attain`` condition holds without the level completing, or a
   failed ``level_complete`` prediction, weakens the goals involved.
3. If actions are queued, the next one is taken; if a plan has subgoals left,
   code routes to the next one (route.py) and queues the route.
4. Otherwise the model is asked:
   - **update** — after a surprise, a no-op, a novelty, at a level start, after a
     game over, and every ``update_every`` actions unless the last
     ``TRUSTED_RUN`` predictions all held and nothing new appeared;
   - **act** — steps to take, each with an expectation, and/or a plan of
     subgoals for code to walk.

Without a model (none configured, call budget spent, or a call that fails) the
step goes to v1's graph explorer, which is kept in sync on every step; every
such step is counted in ``stats['explore_steps']`` so a run that used it is
visibly not a clean run of the model.

Across levels the store is carried: every assumption is marked *carried* and
level-scoped ones lose confidence; code describes how the new level differs
from the last (colours, shapes, object count, actions) for the update call.

**What this does not do.**

- *No world-model code.* Routes use only measured displacements and colours
  that blocked; anything subtler (pushing, keys, state) is the model's to plan
  in steps or subgoals.
- *No search over rival action choices.* One act call, one choice.
- *Expectations are about changes, not positions* (expect.py); subgoals are the
  only place positions enter, and code, not the model, does the arithmetic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ..agent import GraphAgent
from ..llm import LLM, Truncated
from ..scene import make_scene
from ..types import CLICK, RESET, Action, GameState, Observation
from ..v3.facts import Step, action_name
from ..v3.prompts import parse_json
from ..v3.world import parse_action
from .expect import Expectation, Item, check
from .facts import FactBook4
from .prompts import (ACT_SCHEMA, MAX_STEPS, PROMPT_VERSION, UPDATE_SCHEMA, Context,
                      act_messages, update_messages, with_image)
from .route import Subgoal, reached, route
from .store import AssumptionStore

HISTORY = 14              # recent steps shown to the model
REASONING_TAIL = 4000     # characters of the last act call's thinking passed to update
EVENT_LINES = 30
MAX_LIVE = 25            # live assumptions before the update call is asked to consolidate
TRUSTED_RUN = 5          # predictions in a row that must hold before a periodic update is skipped


@dataclass
class V4Agent:
    llm: Optional[LLM] = None
    seed: int = 0
    baselines: Optional[tuple] = None
    total_levels: Optional[int] = None
    temperature: float = 0.6
    image: bool = False               # D-044: text only by default; the exact grid is in the text
    max_calls: int = 800              # per game
    update_every: int = 10            # actions without an update before one is due
    update_tokens: int = 16000        # thinking included
    act_tokens: int = 16000
    deadline: float = 0.0             # epoch seconds after which the model is not called
    graph: GraphAgent = field(default=None)  # type: ignore[assignment]
    fb: FactBook4 = field(default_factory=FactBook4)
    store: AssumptionStore = field(default_factory=AssumptionStore)
    level: int = 0
    level_actions: int = 0
    total_actions: int = 0
    queue: list = field(default_factory=list)       # [(Action, Expectation, tests, tag)]
    plan: list = field(default_factory=list)        # [Subgoal] still to walk
    plan_tests: list = field(default_factory=list)
    history: list = field(default_factory=list)
    level_notes: list = field(default_factory=list)
    events: list = field(default_factory=list)      # since the last update
    need_update: bool = True
    since_update: int = 0
    last_reasoning: str = ""
    calls: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    _last_action: Optional[Action] = None
    _pending: Optional[tuple] = None                # (Expectation, tests, tag) for the last action
    _level_start: Optional[dict] = None
    _prev_level_start: Optional[dict] = None
    _credited: set = field(default_factory=set)     # ids credited for the current decision
    _available: tuple = ()
    _recent: list = field(default_factory=list)     # True/False per judged prediction
    _attained: set = field(default_factory=set)     # goal ids whose condition currently holds

    def __post_init__(self) -> None:
        if self.graph is None:
            self.graph = GraphAgent(seed=self.seed, edge_band=1.0)

    def _bump(self, k: str, n: int = 1) -> None:
        self.stats[k] = self.stats.get(k, 0) + n

    def _event(self, s: str) -> None:
        self.events = [*self.events, s][-EVENT_LINES:]

    # -- the loop ------------------------------------------------------------------

    def act(self, obs: Observation) -> Action:
        """The next action. **Returning it is free; the caller spends it.**"""
        graph_choice = self.graph.act(obs)
        frame = obs.last if obs.last is not None else self.fb.grid
        self._available = tuple(obs.available_actions)
        if self.total_levels is None and obs.win_levels:
            self.total_levels = obs.win_levels

        if obs.levels_completed > self.level:
            self._finish_level(obs)
            if frame is not None:
                self.fb.observe(frame, None)
            self._describe_new_level(obs)
        elif frame is not None:
            step = self.fb.observe(frame, self._last_action)
            if step is not None and not step.reset:
                self._judge(step, "game_over" if obs.state == GameState.GAME_OVER else "continue")
            elif self.fb.grid is not None and self._level_start is None:
                self._level_start = self._snapshot(obs)

        if obs.state == GameState.GAME_OVER:
            self._drop_plan()
            self._event(f"GAME OVER after {self.level_actions} actions on this level; "
                        "RESET restarts the level.")
            self.need_update = True
            return self._emit(Action.simple(RESET), None)
        if obs.state == GameState.WIN or frame is None:
            return self._emit(Action.simple(RESET), None)

        if not self.queue and self.plan:
            self._walk(obs)
        if not self.queue:
            self._decide(obs)
            if not self.queue and self.plan:
                self._walk(obs)
        if self.queue:
            a, exp, tests, tag = self.queue.pop(0)
            return self._emit(a, (exp, tests, tag))
        self._bump("explore_steps")
        return self._emit(graph_choice, None)

    def _emit(self, a: Action, pending: Optional[tuple]) -> Action:
        self.graph.took(a)
        self._last_action = a
        self._pending = pending
        if not a.is_reset:
            self.level_actions += 1
            self.total_actions += 1
            self.since_update += 1
        return a

    def _drop_plan(self) -> None:
        if self.plan:
            self._event("The rest of the plan was dropped: "
                        + "; ".join(sg.describe() for sg in self.plan))
        self.queue, self.plan = [], []

    # -- walking subgoals ---------------------------------------------------------------

    def _walk(self, obs: Observation) -> None:
        """Route to the next subgoal and queue the route. Code only, no call."""
        while self.plan:
            sg = self.plan[0]
            self.fb.focus = sg.obj
            path, why = route(self.fb, sg, [n for n in self._available_names(obs)
                                            if n.startswith("ACTION")])
            if path is None:
                self._bump("routes_failed")
                self._event(f"NO ROUTE for subgoal '{sg.describe()}': {why}.")
                self._drop_plan()
                self.need_update = True
                return
            if not path:
                self._event(f"Subgoal '{sg.describe()}' already holds; moving to the next.")
                self.plan.pop(0)
                continue
            self._bump("routes")
            tests = list(dict.fromkeys(([sg.goal] if sg.goal else []) + self.plan_tests))
            for k, (a, dx, dy) in enumerate(path):
                exp = Expectation((Item(f"#{sg.obj}", "moved", dx, dy),), "any", "continue")
                tag = None
                if k == len(path) - 1:
                    arr = sg.on_arrival
                    exp = Expectation(exp.items + arr.items, arr.rest if arr.items else "any",
                                      arr.outcome)
                    tag = "arrive"
                self.queue.append((Action.simple(int(a[6:])), exp, tests, tag))
            self._bump("plan_steps", len(path))
            return

    # -- checking ----------------------------------------------------------------------

    def _judge(self, step: Step, outcome: str) -> None:
        actual = "; ".join(c.describe() for c in step.changes[:6]) or "nothing changed"
        if len(step.changes) > 6:
            actual += f"; +{len(step.changes) - 6} more"
        if outcome != "continue":
            actual += f"; {outcome.replace('_', ' ').upper()}"
        for note in self.store.observe(step, outcome):
            self._event("RULE FAILED: " + note)
            self.need_update = True
            self._bump("rule_failures")
        for note in self.fb.notes.get(step.t, []):
            self._event(f"t{step.t} {step.action}: {note}")
            self.need_update = True
        novel = getattr(step, "novel", False)
        noop = self.fb.is_noop(step)
        if noop:
            self._bump("noop_actions")
        self._check_goals(step, outcome)
        if self._pending is None:
            self._line(f"t{step.t} {step.action} | (explorer, no prediction) | {actual}")
            if step.changes:
                self._event(f"t{step.t} {step.action} (explorer step): {actual}")
            return
        exp, tests, tag = self._pending
        self._pending = None
        tk = frozenset(self.fb.tickers())
        v = check(exp, step, outcome, ignore=tk)
        real = "" if noop else "_real"   # steps where more than the tickers changed
        if v.ok and exp.informative:
            self._bump("predictions_ok")
            if real:
                self._bump("predictions_ok_real")
            self._recent = (self._recent + [True])[-TRUSTED_RUN:]
            # Credit each named assumption once per decision, not once per step:
            # a six-step sequence is one test of the beliefs behind it.
            fresh = [i for i in tests if i not in self._credited]
            self._credited |= set(fresh)
            self.store.supported_by_prediction(fresh, f"t{step.t}")
            self._line(f"t{step.t} {step.action} | {exp.describe()} | {actual} | OK")
            if noop:
                self._event(f"t{step.t} {step.action} was a NO-OP (only the every-action "
                            "changers changed), as you predicted.")
            if tag == "arrive" and self.plan:
                done = self.plan.pop(0)
                self._bump("subgoals_done")
                self._event(f"SUBGOAL DONE at t{step.t}: {done.describe()} - as predicted.")
                if not self.plan:
                    self.need_update = True
            if novel:
                self._drop_plan()
            return
        if not exp.informative:
            self._bump("predictions_none")
            self._line(f"t{step.t} {step.action} | (no prediction) | {actual}")
            self._event(f"t{step.t} {step.action}: you made no prediction; it did: {actual}")
        else:
            self._bump("predictions_wrong")
            if real:
                self._bump("predictions_wrong_real")
            self._recent = (self._recent + [False])[-TRUSTED_RUN:]
            self._line(f"t{step.t} {step.action} | {exp.describe()} | {actual} | WRONG")
            self._event(f"SURPRISE at t{step.t} {step.action}: {v.describe()}. "
                        f"You had predicted: {exp.describe()}. Assumptions this relied on: "
                        f"{', '.join(tests) or 'none named'}.")
            if exp.outcome == "level_complete" and outcome != "level_complete":
                hit = self.store.goals_failed(tests, f"t{step.t}",
                                              "predicted the level would complete; it did not")
                if hit:
                    self._event(f"Goal(s) {', '.join(hit)} lost confidence: the level did not "
                                "complete when they said it would.")
            if self.queue:
                self._event(f"The remaining {len(self.queue)} queued steps were dropped.")
        if noop:
            self._event(f"t{step.t} {step.action} was a NO-OP: nothing changed except the "
                        "objects that change on every action.")
        self._drop_plan()
        self.need_update = True

    def _check_goals(self, step: Step, outcome: str) -> None:
        """A goal's attain condition holding without the level completing is
        evidence against it (soft), once per arrival."""
        for g in self.store.goals():
            if not g.attain:
                continue
            sg = Subgoal.parse(g.attain)
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
            self._event(f"GOAL TEST at t{step.t}: {g.id}'s condition ({g.attain['object']} "
                        f"{g.attain['relation']} {g.attain['to']}) holds and the level did NOT "
                        f"complete - {g.id} is wrong or needs something first.")
            self.need_update = True

    def _line(self, s: str) -> None:
        self.history = [*self.history, s][-HISTORY:]

    # -- deciding -------------------------------------------------------------------------

    def _model_available(self) -> bool:
        if self.llm is None or len(self.calls) >= self.max_calls:
            return False
        return not (self.deadline and time.time() > self.deadline)

    def _trusted(self) -> bool:
        return len(self._recent) >= TRUSTED_RUN and all(self._recent)

    def _decide(self, obs: Observation) -> None:
        if not self._model_available():
            return
        if self.need_update:
            self._update(obs)
        elif self.since_update >= self.update_every:
            if self._trusted():
                self._bump("updates_skipped")
                self.since_update = 0
            else:
                self._update(obs)
        if self._model_available():
            self._act(obs)

    def _available_names(self, obs: Observation) -> list[str]:
        out = [f"ACTION{a}" for a in sorted(obs.available_actions) if a not in (RESET, CLICK)]
        if CLICK in obs.available_actions:
            out.append("CLICK(x,y)")
        return out

    def _baseline(self) -> Optional[int]:
        if not self.baselines or self.level >= len(self.baselines):
            return None
        return int(self.baselines[self.level])

    def _context(self, obs: Observation, with_events: bool, with_reasoning: bool) -> Context:
        c = Context(
            level=self.level, total_levels=self.total_levels,
            actions_on_level=self.level_actions, actions_total=self.total_actions,
            human_baseline=self._baseline(), available=self._available_names(obs),
            store=self.store.describe(), facts=self.fb.describe(),
            grid=make_scene(self.fb.grid).grid_text(), history=list(self.history),
            level_notes=list(self.level_notes),
            event="\n".join(self.events) if with_events else "",
            last_reasoning=self.last_reasoning[-REASONING_TAIL:] if with_reasoning else "",
            plan="\n".join(f"{k + 1}. {sg.describe()}" for k, sg in enumerate(self.plan)))
        if self.image and self.fb.grid is not None:
            with_image(c, self.fb.grid)
        return c

    def _call(self, kind: str, messages: list, schema: dict, tokens: int) -> Optional[dict]:
        """One model call. A truncated answer is retried once with twice the room;
        any other failure is counted and the step falls back to the explorer."""
        t0 = time.time()
        text, reasoning = "", ""
        for attempt in range(2):
            try:
                text = self.llm.complete(messages, max_tokens=tokens * (attempt + 1),
                                         temperature=self.temperature,
                                         seed=self.seed + len(self.calls), schema=schema)
                reasoning = (getattr(self.llm, "last", None) or {}).get("reasoning", "")
                break
            except Truncated:
                self._bump("truncated")
                text = ""
            except Exception as exc:  # a dead server must not end the game
                self._bump("llm_errors")
                self._event(f"(the {kind} call failed: {type(exc).__name__})")
                self.calls.append({"kind": kind, "level": self.level, "at": self.level_actions,
                                   "ok": False, "error": f"{type(exc).__name__}: {exc}"[:200],
                                   "seconds": round(time.time() - t0, 2), "prompt": PROMPT_VERSION})
                return None
        data = parse_json(text)
        self.calls.append({"kind": kind, "level": self.level, "at": self.level_actions,
                           "ok": data is not None, "seconds": round(time.time() - t0, 2),
                           "reasoning_chars": len(reasoning), "prompt": PROMPT_VERSION})
        self._bump(f"calls_{kind}")
        if kind == "act":
            self.last_reasoning = reasoning
        return data

    def _update(self, obs: Observation) -> None:
        live = len(self.store.live())
        if live > MAX_LIVE:
            self._event(f"The store holds {live} live assumptions, more than {MAX_LIVE}: "
                        "consolidate it - retire duplicates, superseded claims and any "
                        "descriptions of the current state.")
        data = self._call("update", update_messages(self._context(obs, True, True)),
                          UPDATE_SCHEMA, self.update_tokens)
        self.since_update = 0
        if data is None:
            return  # keep the events and the reasoning for the next attempt
        where = f"L{self.level + 1} t{self.level_actions}"
        done = self.store.apply(data.get("ops", []), where)
        self.calls[-1]["edits"] = len(done)
        self.events, self.last_reasoning, self.need_update = [], "", False

    def _act(self, obs: Observation) -> None:
        data = self._call("act", act_messages(self._context(obs, False, False)),
                          ACT_SCHEMA, self.act_tokens)
        if data is None:
            return
        tests = [str(t) for t in data.get("tests", []) if t][:8]
        q, bad = [], []
        for s in (data.get("steps") or [])[:MAX_STEPS]:
            if not isinstance(s, dict):
                continue
            a = self._to_action(str(s.get("action", "")), obs)
            if a is None:
                bad.append(str(s.get("action", ""))[:20])
                break  # later steps assumed this one happened
            q.append((a, Expectation.parse(s.get("expect")), tests, None))
        if bad:
            self._event(f"Your action {bad[0]!r} is not available or not understood; "
                        f"available: {', '.join(self._available_names(obs))}.")
            self.need_update = self.need_update or not q
        self.queue = q
        self.plan = [sg for sg in (Subgoal.parse(d) for d in (data.get("plan") or [])) if sg]
        self.plan_tests = tests
        self._credited = set()
        self.calls[-1]["steps"] = len(q)
        self.calls[-1]["subgoals"] = len(self.plan)
        self.calls[-1]["purpose"] = str(data.get("purpose", ""))[:200]
        self._bump("llm_steps", len(q))

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

    def _snapshot(self, obs: Observation) -> dict:
        objs = list(self.fb.objects.values())
        return {"n": len(objs), "colours": sorted({o.colour for o in objs}),
                "shapes": {o.shape for o in objs}, "actions": tuple(sorted(obs.available_actions)),
                "background": self.fb.background}

    def _finish_level(self, obs: Observation) -> None:
        won_with = action_name(self._last_action) if self._last_action else "?"
        if self._pending is not None:
            exp, tests, _ = self._pending
            if exp.outcome == "level_complete":
                self._bump("predictions_ok")
                self._bump("predictions_ok_real")
                self.store.supported_by_prediction(
                    [i for i in tests if i not in self._credited], f"L{self.level + 1} win")
                self.store.goals_won(tests, f"L{self.level + 1} win")
                verdict = "you predicted it"
            else:
                verdict = "you did NOT predict it (you expected: " + exp.describe() + ")"
            self._pending = None
        else:
            verdict = "an explorer step, no prediction"
        top = sorted(self.store.live(), key=lambda a: -a.confidence)[:6]
        base = self._baseline()
        self.level_notes.append(
            f"Level {self.level + 1}: won in {self.level_actions} actions"
            + (f" (human ~{base})" if base else "") + f"; the winning action was {won_with} - "
            f"{verdict}. Most confident assumptions then: "
            + "; ".join(f"{a.id} {a.statement}" for a in top))
        self._prev_level_start = self._level_start
        self._bump("levels")
        self.level = obs.levels_completed
        self.level_actions = 0
        pcs = self.fb.player_candidates()
        self.fb = FactBook4(prior_moves=dict(pcs[0]["moves"]) if pcs else {})
        self._last_action, self.queue, self.plan, self.history = None, [], [], []
        self._recent, self._attained = [], set()
        self.store.new_level(self.level)
        self.need_update = True

    def _describe_new_level(self, obs: Observation) -> None:
        now = self._snapshot(obs)
        prev = self._prev_level_start
        self._level_start = now
        lines = [f"NEW LEVEL {self.level + 1} started. All assumptions are now CARRIED: confirm "
                 "or revise them against this level before relying on them."]
        if prev:
            new_c = sorted(set(now["colours"]) - set(prev["colours"]))
            gone_c = sorted(set(prev["colours"]) - set(now["colours"]))
            new_s = len(now["shapes"] - prev["shapes"])
            lines.append(f"Compared with the start of the previous level: {now['n']} objects "
                         f"(was {prev['n']}); colours new {new_c or 'none'}, gone {gone_c or 'none'}; "
                         f"{new_s} object shapes not seen at the previous level's start"
                         + ("; background changed" if now["background"] != prev["background"] else "")
                         + ("; the available actions CHANGED" if now["actions"] != prev["actions"] else "")
                         + ".")
        self._event(" ".join(lines))

    def accuracy(self) -> dict:
        """Prediction accuracy, overall and on steps where more than the
        every-action changers changed (D-044: on ls20, 38 of 51 'correct'
        predictions were walking into a wall while a bar ticked)."""
        s = self.stats
        ok, bad = s.get("predictions_ok", 0), s.get("predictions_wrong", 0)
        okr, badr = s.get("predictions_ok_real", 0), s.get("predictions_wrong_real", 0)
        return {"all": round(ok / (ok + bad), 3) if ok + bad else None,
                "real": round(okr / (okr + badr), 3) if okr + badr else None,
                "n_real": okr + badr}

    def describe(self) -> str:
        clean = "CLEAN" if not self.stats.get("explore_steps") else \
            f"NOT CLEAN ({self.stats['explore_steps']} explorer steps)"
        return (f"level {self.level}, calls {len(self.calls)}, assumptions {len(self.store.live())}, "
                f"{clean}, acc {self.accuracy()}, "
                + ", ".join(f"{k}={v}" for k, v in sorted(self.stats.items())))
