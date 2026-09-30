"""The v3 controller: assess -> plan or experiment -> act -> verify -> revise (D-039).

Per step, deterministically:

1. The fact layer observes the new frame (objects, changes, relations).
2. If the last action came with a prediction (it was part of a plan or an
   experiment run under the current world model), the prediction is checked
   against what happened. A mismatch is a **surprise**: the queue is dropped
   and the exact counterexample goes to the model.
3. If actions are queued, the next one is taken.
4. Otherwise a decision is made:
   - **Probe** at the start of a level: each simple action once (and clicks on
     a few differently coloured objects), because dynamics is cheap to test and
     goals are not (Guardrail 4).
   - **Assess** (LLM) when there is no model yet, after a surprise, when a plan
     ended without a win, after a game over, or every ``ASSESS_EVERY`` actions.
     The model's `predict` is verified against the recorded steps at once; while
     it mispredicts, **repair** (LLM) is called with the counterexamples, up to
     ``MAX_REPAIRS`` times — thinking is free, actions are not.
   - Then the model's chosen **next**: *plan* — search its world model for the
     shortest path to its goal and queue it with a predicted state per step;
     *experiment* — queue its test actions, predicted where the model can;
     *explore* — hand a few steps to v1's graph explorer.
5. A plan that completes with the goal predicted reached and every step
   verified, yet no win, **refutes the goal**; that goal is kept in a refuted
   list the model sees from then on.

Across levels, the last world model, goal and hypotheses are carried forward
as the starting point, and the facts of the winning move are described.

**What this does not do.**

- *No population of rival world models.* One `predict`/`goal` pair is live at a
  time; alternatives exist only as the model's stated hypotheses and the
  refuted-goal list. Rival models are the next extension if assessments churn.
- *Verification is object-level* (world.compare): boxes, colours, existence.
- *Clicks in plans are object centres only* (world.action_space).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..agent import GraphAgent
from ..llm import LLM
from ..scene import make_scene
from ..types import CLICK, RESET, Action, GameState, Observation
from .facts import FactBook
from .prompts import (
    ASSESS_SCHEMA,
    PROMPT_VERSION,
    REPAIR_SCHEMA,
    Context,
    assess_messages,
    parse_json,
    repair_messages,
)
from .world import State, WorldModel, compare, parse_action, plan, verify

ASSESS_EVERY = 30        # actions without an assessment before one is forced
MAX_REPAIRS = 2          # repair calls per assessment
TRUST = 0.9              # verification accuracy at which a model may be planned with
VERIFY_WINDOW = 80       # recent steps a model is verified on
EXPLORE_STEPS = 12       # steps handed to the explorer per 'explore'
MAX_EXPERIMENT = 8
PROBE_CLICKS = 4
MAX_CALLS = 200          # per game
MAX_LEVEL_CALLS = 40
ASSESS_TOKENS = 3000
REPAIR_TOKENS = 2000


@dataclass
class V3Agent:
    llm: Optional[LLM] = None
    seed: int = 0
    baselines: Optional[tuple] = None
    temperature: float = 0.0
    graph: GraphAgent = field(default=None)  # type: ignore[assignment]
    fb: FactBook = field(default_factory=FactBook)
    level: int = 0
    level_actions: int = 0
    queue: list = field(default_factory=list)          # [(Action, expected State | None, source)]
    wm: Optional[WorldModel] = None
    predict_code: str = ""
    goal_code: str = ""
    hypotheses: list = field(default_factory=list)
    ignore_ids: list = field(default_factory=list)
    check_text: str = ""
    counterexamples: list = field(default_factory=list)
    refuted_goals: list = field(default_factory=list)
    last_outcome: str = ""
    previous_level: str = ""
    trigger: str = "start"                           # why the next assessment is due
    since_assess: int = 0
    explore_left: int = 0
    probed: set = field(default_factory=set)
    calls: list = field(default_factory=list)        # diagnosis log, one dict per call
    stats: dict = field(default_factory=dict)
    _last_action: Optional[Action] = None
    _expect: Optional[tuple] = None                  # (State, source) for the last action
    _level_calls: int = 0
    _plan_goal_claim: str = ""
    _plan_last: bool = False
    _accuracy: float = 0.0
    _next: Optional[dict] = None

    def __post_init__(self) -> None:
        if self.graph is None:
            self.graph = GraphAgent(seed=self.seed, edge_band=1.0)

    # -- bookkeeping -----------------------------------------------------------

    def _bump(self, k: str, n: int = 1) -> None:
        self.stats[k] = self.stats.get(k, 0) + n

    def _available(self, obs: Observation) -> list[str]:
        out = [f"ACTION{a}" for a in sorted(obs.available_actions) if a not in (RESET, CLICK)]
        if CLICK in obs.available_actions:
            out.append("CLICK(x,y)")
        return out

    def _simple(self, obs: Observation) -> list[int]:
        return sorted(a for a in obs.available_actions if a not in (RESET, CLICK))

    def _state(self) -> State:
        return State.from_objects(self.fb.objects, self.fb.grid, self.fb.masks, self.fb.background)

    def _baseline(self) -> Optional[int]:
        if not self.baselines or self.level >= len(self.baselines):
            return None
        return int(self.baselines[self.level])

    # -- the loop ----------------------------------------------------------------

    def act(self, obs: Observation) -> Action:
        """The next action. **Returning it is free; the caller spends it.**"""
        graph_choice = self.graph.act(obs)
        frame = obs.last if obs.last is not None else self.fb.grid

        if obs.levels_completed > self.level:
            self._new_level(obs)
        if frame is not None:
            self.fb.observe(frame, self._last_action)

        if obs.state == GameState.GAME_OVER:
            self.queue, self._expect = [], None
            self.last_outcome = (f"GAME OVER after {self.level_actions} actions on this level "
                                 "(the level restarts on RESET).")
            self.trigger = "game over"
            return self._emit(Action.simple(RESET), None, "reset")
        if obs.state == GameState.WIN or frame is None:
            return self._emit(Action.simple(RESET), None, "reset")

        self._check_expectation()

        if not self.queue:
            self._decide(obs, graph_choice)
        if self.queue:
            a, exp, src = self.queue.pop(0)
            if src == "plan" and not self.queue:
                self._plan_last = True
            return self._emit(a, exp, src)
        self._bump("explore_steps")
        return self._emit(graph_choice, None, "explore")

    def _emit(self, a: Action, expected: Optional[State], source: str) -> Action:
        self.graph.took(a)
        self._last_action = a
        self._expect = (expected, source) if expected is not None else None
        self.level_actions += 1
        self.since_assess += 1
        if self.explore_left and source == "explore":
            self.explore_left -= 1
        self._bump(f"src_{source}")
        return a

    def _check_expectation(self) -> None:
        if self._expect is None:
            return
        expected, source = self._expect
        self._expect = None
        miss = compare(expected, self.fb.objects, self.ignore_ids)
        if miss:
            step = self.fb.steps[-1] if self.fb.steps else None
            ce = f"t{step.t} {step.action}: " + "; ".join(miss[:4]) if step else "; ".join(miss[:4])
            self.counterexamples = [ce, *self.counterexamples][:6]
            self.queue = []
            self.last_outcome = (f"SURPRISE during {source}: your predict() was wrong at {ce}. "
                                 "The rest of the plan was dropped.")
            self.trigger = "surprise"
            self._bump("surprises")
        elif source == "plan" and getattr(self, "_plan_last", False):
            # Every step of the plan was predicted correctly and it ended where
            # the model said the goal is met - yet the level did not end.
            self._plan_last = False
            if self.wm is not None and self.wm.goal(self._state()) >= 0.99:
                claim = self._plan_goal_claim or "the current goal()"
                self.refuted_goals.append(claim)
                self.last_outcome = ("Your plan executed exactly as predicted and reached a state "
                                     "your goal() scores 1.0, but the level did NOT end. That goal "
                                     "is refuted.")
                self.trigger = "goal refuted"
                self._bump("goals_refuted")

    # -- deciding ------------------------------------------------------------------

    def _decide(self, obs: Observation, graph_choice: Action) -> None:
        if self.trigger == "new level" and self.wm is not None and self.wm.goal_fn is not None:
            # Levels of a game usually share their rules. Plan with the last
            # level's model before probing (probing cost 4 actions per level on
            # the synthetic room): every planned step is still checked, so a
            # changed rule shows up as a surprise, and probing resumes then.
            steps = self.fb.play_steps()
            if steps:
                self._verify()
            if (not steps or self._accuracy >= 1.0) and self._queue_plan(obs):
                self.trigger = ""
                self._bump("carried_plans")
                return
        probe = self._next_probe(obs)
        if probe is not None:
            self.queue.append((probe, None, "probe"))
            return
        if self.explore_left > 0:
            return  # the explorer's choice is used
        if self.llm is None or len(self.calls) >= MAX_CALLS or self._level_calls >= MAX_LEVEL_CALLS:
            return
        if self.trigger or self.since_assess >= ASSESS_EVERY or self.wm is None:
            self._assess(obs)
        self._act_on_next(obs)

    def _next_probe(self, obs: Observation) -> Optional[Action]:
        for a in self._simple(obs):
            if a not in self.probed:
                self.probed.add(a)
                return Action.simple(a)
        if CLICK in obs.available_actions:
            clicked_colours = {k for k in self.probed if isinstance(k, tuple)}
            if len(clicked_colours) < PROBE_CLICKS:
                for o in sorted(self.fb.objects.values(), key=lambda o: -o.size):
                    key = ("click", o.colour)
                    if key not in self.probed:
                        self.probed.add(key)
                        return Action.click(min(63, o.x + o.w // 2), min(63, o.y + o.h // 2))
        return None

    def _context(self, obs: Observation) -> Context:
        return Context(
            level=self.level, actions_on_level=self.level_actions,
            human_baseline=self._baseline(), available=self._available(obs),
            facts=self.fb.describe(), grid=make_scene(self.fb.grid).grid_text(),
            hypotheses=self.hypotheses, predict_code=self.predict_code,
            goal_code=self.goal_code, ignore_ids=self.ignore_ids, check=self.check_text,
            counterexamples=self.counterexamples, refuted_goals=self.refuted_goals,
            last_outcome=self.last_outcome, previous_level=self.previous_level)

    def _call(self, kind: str, messages: list, schema: dict, tokens: int) -> Optional[dict]:
        t0 = time.time()
        try:
            text = self.llm.complete(messages, max_tokens=tokens, temperature=self.temperature,
                                     seed=self.seed + len(self.calls), schema=schema)
        except Exception as exc:  # a dead server must not end the game
            text = ""
            self._bump("llm_errors")
            self.last_outcome = f"(model call failed: {type(exc).__name__})"
        data = parse_json(text)
        self.calls.append({"kind": kind, "level": self.level, "at": self.level_actions,
                           "ok": data is not None, "seconds": round(time.time() - t0, 2),
                           "prompt": PROMPT_VERSION})
        self._level_calls += 1
        self._bump(f"calls_{kind}")
        return data

    def _assess(self, obs: Observation) -> None:
        data = self._call("assess", assess_messages(self._context(obs)), ASSESS_SCHEMA, ASSESS_TOKENS)
        self.since_assess = 0
        self.trigger = ""
        if data is None:
            self._next = {"kind": "explore", "purpose": "the model gave no usable answer"}
            return
        self.hypotheses = [h for h in data.get("hypotheses", []) if isinstance(h, dict)]
        self.ignore_ids = [int(i) for i in data.get("ignore_ids", []) if isinstance(i, (int, float))]
        self._install(str(data.get("predict_code", "")), str(data.get("goal_code", "")))
        goals = [h.get("claim", "") for h in self.hypotheses if h.get("kind") == "goal"]
        self._plan_goal_claim = goals[0] if goals else ""
        self._next = data.get("next") if isinstance(data.get("next"), dict) else {"kind": "explore"}
        repairs = 0
        while self._needs_repair() and repairs < MAX_REPAIRS and len(self.calls) < MAX_CALLS:
            repairs += 1
            fix = self._call("repair", repair_messages(self._context(obs)), REPAIR_SCHEMA, REPAIR_TOKENS)
            if fix is None:
                break
            if isinstance(fix.get("ignore_ids"), list):
                self.ignore_ids = [int(i) for i in fix["ignore_ids"] if isinstance(i, (int, float))]
            self._install(str(fix.get("predict_code", "")), self.goal_code)

    def _install(self, predict_code: str, goal_code: str) -> None:
        wm = WorldModel.compile(predict_code, goal_code)
        self.predict_code, self.goal_code, self.wm = predict_code, goal_code, wm
        self._verify()

    def _verify(self) -> None:
        if self.wm is None:
            return
        if self.wm.error and self.wm.predict_fn is None:
            self.check_text = f"does not compile: {self.wm.error}"
            self.counterexamples = []
            self._accuracy = 0.0
            return
        steps = self.fb.play_steps()[-VERIFY_WINDOW:]
        chk = verify(self.wm, steps, self.ignore_ids, self.fb.background)
        self.check_text = chk.describe() + (f"; runtime error: {self.wm.error}" if self.wm.error else "")
        self.counterexamples = chk.counterexamples
        self._accuracy = chk.accuracy if chk.n else 0.0
        self.calls and self.calls[-1].setdefault("accuracy", round(self._accuracy, 3))

    def _needs_repair(self) -> bool:
        return self.wm is not None and getattr(self, "_accuracy", 0.0) < 1.0 and bool(self.fb.play_steps())

    def _act_on_next(self, obs: Observation) -> None:
        nxt = getattr(self, "_next", None) or {"kind": "explore"}
        kind = nxt.get("kind", "explore")
        if kind == "plan" and self.wm is not None and getattr(self, "_accuracy", 0.0) >= TRUST:
            if self._queue_plan(obs):
                return
            kind = "experiment" if nxt.get("actions") else "explore"
        if kind == "experiment" and nxt.get("actions"):
            if self._queue_actions(nxt["actions"], obs, "experiment"):
                self.last_outcome = f"Ran your experiment: {nxt.get('purpose', '')}"
                return
        self.explore_left = EXPLORE_STEPS
        self.last_outcome = f"Explored {EXPLORE_STEPS} steps ({nxt.get('purpose', 'no plan available')})."
        self._bump("explore_handoffs")

    def _queue_plan(self, obs: Observation) -> bool:
        start = self._state()
        p = plan(self.wm, start, self._simple(obs), CLICK in obs.available_actions)
        self._bump("plans")
        if not p.actions:
            self.last_outcome = (f"The planner searched {p.nodes} states of your model and found no "
                                 f"path that raises goal() above {p.start_progress:.2f}.")
            self.trigger = "no plan"
            return False
        if not p.reached:
            self.last_outcome = (f"No path to goal()=1 in {p.nodes} states; following the best path "
                                 f"found (goal {p.start_progress:.2f} -> {p.progress:.2f}).")
        self._plan_last = False
        return self._queue_actions(p.actions, obs, "plan")

    def _queue_actions(self, actions: list, obs: Observation, source: str) -> bool:
        s = self._state()
        q = []
        for a in actions[: (200 if source == "plan" else MAX_EXPERIMENT)]:
            parsed = parse_action(str(a))
            if parsed is None or (parsed[0] not in obs.available_actions and parsed[0] != RESET):
                continue
            aid, x, y = parsed
            act = Action.click(min(63, x), min(63, y)) if aid == CLICK else Action.simple(aid)
            name = f"CLICK({act.x},{act.y})" if act.is_click else ("RESET" if aid == RESET else f"ACTION{aid}")
            expected = None
            if self.wm is not None and self.wm.predict_fn is not None and aid != RESET:
                s = self.wm.predict(s, name)
                expected = s
            q.append((act, expected, source))
        self.queue = q
        return bool(q)

    # -- levels ----------------------------------------------------------------------

    def _new_level(self, obs: Observation) -> None:
        won_with = self._last_action
        self.previous_level = (
            f"Level {self.level + 1} was won in {self.level_actions} actions; the winning action "
            f"was {won_with}. The hypotheses at that point were: "
            + "; ".join(f"[{h.get('kind')}] {h.get('claim')}" for h in self.hypotheses[:8])
            + ". Its predict() and goal() are carried over as your starting point.")
        self._bump("levels")
        self.level = obs.levels_completed
        self.level_actions = 0
        self._level_calls = 0
        self.fb = FactBook()
        self._last_action = None
        self.queue, self._expect = [], None
        self.refuted_goals, self.counterexamples = [], []
        self.probed = set()
        self.explore_left = 0
        self.trigger = "new level"

    def describe(self) -> str:
        return (f"level {self.level}, calls {len(self.calls)}, "
                + ", ".join(f"{k}={v}" for k, v in sorted(self.stats.items())))
