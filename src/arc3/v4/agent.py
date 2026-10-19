"""The v4 controller: predict -> act -> check -> update assumptions (D-042).

Per step, deterministically:

1. The fact layer observes the new frame (v3's FactBook: tracked objects and
   exact typed changes).
2. Every assumption with a machine-checkable rule for the action just taken is
   checked (store.observe); the expectation the model attached to that action is
   checked (expect.check). A correct prediction supports the assumptions the
   model said it tested. A wrong one is a **surprise**: the rest of the queued
   sequence is dropped and an update is due.
3. If actions are queued, the next one is taken.
4. Otherwise the model is asked:
   - **update** — if anything surprising happened, at the start of a level, after
     a game over, or every ``update_every`` actions in any case (so beliefs
     formed while acting are written down);
   - **act** — always: one action, or up to six when the model is confident,
     each with an expectation.

So the model decides every action while it is learning (each surprise brings
it straight back), and runs short checked sequences once its predictions hold.
Without a model (none configured, call budget spent, or a call that fails) the
step goes to v1's graph explorer, which is kept in sync on every step.

Across levels the store is carried: every assumption is marked *carried* and
level-scoped ones lose confidence; code describes how the new level differs
from the last (colours, shapes, object count, actions) for the update call.

**What this does not do.**

- *No world-model code and no planner.* v3 asked the model for a simulator and
  planned against it; here the model plans in its head and code only checks.
  Long, exact plans (a 40-step maze route) are the obvious casualty; the
  verified simulator of v3 is the candidate to bring back as a tool for them.
- *No search over rival action choices.* One act call, one choice.
- *Expectations are about changes, not positions* (expect.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ..agent import GraphAgent
from ..llm import LLM, Truncated
from ..scene import make_scene
from ..types import CLICK, RESET, Action, GameState, Observation
from ..v3.facts import FactBook, Step, action_name
from ..v3.prompts import parse_json
from ..v3.world import parse_action
from .expect import Expectation, check
from .prompts import (ACT_SCHEMA, MAX_STEPS, PROMPT_VERSION, UPDATE_SCHEMA, Context,
                      act_messages, update_messages, with_image)
from .store import AssumptionStore

HISTORY = 14              # recent steps shown to the model
REASONING_TAIL = 4000     # characters of the last act call's thinking passed to update
EVENT_LINES = 30


@dataclass
class V4Agent:
    llm: Optional[LLM] = None
    seed: int = 0
    baselines: Optional[tuple] = None
    total_levels: Optional[int] = None
    temperature: float = 0.6
    image: bool = True
    max_calls: int = 800              # per game
    update_every: int = 10            # actions without an update before one is forced
    update_tokens: int = 16000        # thinking included
    act_tokens: int = 16000
    deadline: float = 0.0             # epoch seconds after which the model is not called
    graph: GraphAgent = field(default=None)  # type: ignore[assignment]
    fb: FactBook = field(default_factory=FactBook)
    store: AssumptionStore = field(default_factory=AssumptionStore)
    level: int = 0
    level_actions: int = 0
    total_actions: int = 0
    queue: list = field(default_factory=list)       # [(Action, Expectation, tests)]
    history: list = field(default_factory=list)
    level_notes: list = field(default_factory=list)
    events: list = field(default_factory=list)      # since the last update
    need_update: bool = True
    since_update: int = 0
    last_reasoning: str = ""
    calls: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    _last_action: Optional[Action] = None
    _pending: Optional[tuple] = None                # (Expectation, tests) for the last action
    _level_start: Optional[dict] = None
    _prev_level_start: Optional[dict] = None
    _credited: set = field(default_factory=set)     # ids credited for the current decision
    _available: tuple = ()

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
            self.queue = []
            self._event(f"GAME OVER after {self.level_actions} actions on this level; "
                        "RESET restarts the level.")
            self.need_update = True
            return self._emit(Action.simple(RESET), None)
        if obs.state == GameState.WIN or frame is None:
            return self._emit(Action.simple(RESET), None)

        if not self.queue:
            self._decide(obs)
        if self.queue:
            a, exp, tests = self.queue.pop(0)
            return self._emit(a, (exp, tests))
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
        if self._pending is None:
            self._line(f"t{step.t} {step.action} | (explorer, no prediction) | {actual}")
            if step.changes:
                self._event(f"t{step.t} {step.action} (explorer step): {actual}")
            return
        exp, tests = self._pending
        self._pending = None
        v = check(exp, step, outcome)
        if v.ok and exp.informative:
            self._bump("predictions_ok")
            # Credit each named assumption once per decision, not once per step:
            # a six-step sequence is one test of the beliefs behind it.
            fresh = [i for i in tests if i not in self._credited]
            self._credited |= set(fresh)
            self.store.supported_by_prediction(fresh, f"t{step.t}")
            self._line(f"t{step.t} {step.action} | {exp.describe()} | {actual} | OK")
            return
        if not exp.informative:
            self._bump("predictions_none")
            self._line(f"t{step.t} {step.action} | (no prediction) | {actual}")
            self._event(f"t{step.t} {step.action}: you made no prediction; it did: {actual}")
        elif not v.failures:
            # Everything named happened, plus changes the model did not mention
            # (seen on ls20: a bar that shrinks on every move, left out of every
            # expectation). Worth recording, not worth abandoning the sequence:
            # the next update sees it.
            self._bump("predictions_partial")
            self._credited |= set(tests)
            self._line(f"t{step.t} {step.action} | {exp.describe()} | {actual} | "
                       f"OK BUT ALSO: {'; '.join(v.unpredicted)}")
            self._event(f"t{step.t} {step.action}: as predicted, but also unpredicted: "
                        f"{'; '.join(v.unpredicted)}.")
            self.need_update = True
            return
        else:
            self._bump("predictions_wrong")
            self._line(f"t{step.t} {step.action} | {exp.describe()} | {actual} | WRONG")
            self._event(f"SURPRISE at t{step.t} {step.action}: {v.describe()}. "
                        f"You had predicted: {exp.describe()}. Assumptions this relied on: "
                        f"{', '.join(tests) or 'none named'}.")
            if self.queue:
                self._event(f"The remaining {len(self.queue)} queued steps were dropped.")
        self.queue = []
        self.need_update = True

    def _line(self, s: str) -> None:
        self.history = [*self.history, s][-HISTORY:]

    # -- deciding -------------------------------------------------------------------------

    def _model_available(self) -> bool:
        if self.llm is None or len(self.calls) >= self.max_calls:
            return False
        return not (self.deadline and time.time() > self.deadline)

    def _decide(self, obs: Observation) -> None:
        if not self._model_available():
            return
        if self.need_update or self.since_update >= self.update_every:
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
            last_reasoning=self.last_reasoning[-REASONING_TAIL:] if with_reasoning else "")
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
            q.append((a, Expectation.parse(s.get("expect")), tests))
        if bad:
            self._event(f"Your action {bad[0]!r} is not available or not understood; "
                        f"available: {', '.join(self._available_names(obs))}.")
            self.need_update = self.need_update or not q
        self.queue = q
        self._credited = set()
        self.calls[-1]["steps"] = len(q)
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
            exp, tests = self._pending
            if exp.outcome == "level_complete":
                self._bump("predictions_ok")
                self.store.supported_by_prediction(
                    [i for i in tests if i not in self._credited], f"L{self.level + 1} win")
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
        self.fb = FactBook()
        self._last_action, self.queue, self.history = None, [], []
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

    def describe(self) -> str:
        return (f"level {self.level}, calls {len(self.calls)}, assumptions {len(self.store.live())}, "
                + ", ".join(f"{k}={v}" for k, v in sorted(self.stats.items())))
