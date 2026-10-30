"""End-to-end agent evaluation: levels cleared and offline RHAE, cold start.

The first script in the repository that reports the competition metric for a
policy rather than a proxy (D-034). Every game is played once from a cold start
with a fixed action cap, and the score is computed by `client/budget.py` against
the human baselines shipped in each game's metadata.

Also runs the `explore` driver under the same cap, with a RESET on GAME_OVER, as
the baseline: before this agent it was the only policy that cleared any level.

Usage:
    python scripts/eval_agent.py [--actions 500] [--agents "graph;graph_nomodel;explore"]
                                 [--games ls20 vc33] [--out docs/experiments/agent.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arc3.agent import GraphAgent
from arc3.client.budget import ActionBudget
from arc3.client.env import EnvClient, available_environments, make_arcade
from arc3.agent_v2 import HypothesisAgent
from arc3.v3.agent import V3Agent
from arc3.v4.agent import V4Agent
from arc3.v5.agent import V5Agent
from arc3.explore import ExploringDriver
from arc3.llm import OpenAIChatLLM, RecordingLLM, ReplayLLM
from arc3.types import RESET, Action, GameState

#: Set from the command line in main(); module-level so play() stays picklable.
LLM_URL = ""
CALLS_DIR = "docs/experiments/llm_calls"
REPLAY = False
REPLAY_ONLY = False
LLM_EXTRA: dict = {}
LLM_MODEL = "local"
#: Development against a hosted API only (Guardrail 5): the key is read from the
#: environment variable named by --llm-key-env, never passed on the command line.
LLM_KEY = ""
LLM_TIMEOUT = 600.0


def base_name(agent_name: str) -> str:
    return agent_name.partition(":")[0]


def play(gid: str, agent_name: str, n_actions: int, seed: int, arcade, env_dir: str) -> dict:
    client = EnvClient(gid, seed=seed, record=False, arcade=arcade, environments_dir=env_dir)
    meta = client.metadata
    budget = ActionBudget(gid, meta.win_levels, meta.baseline_actions)
    if agent_name == "explore":
        driver = ExploringDriver(seed=seed)

        def choose(obs):
            if obs.state in (GameState.GAME_OVER, GameState.WIN):
                return Action.simple(RESET)
            a = driver.next_action(obs)
            return a if a is not None else Action.simple(RESET)
        agent = None
    else:
        # "graph_nomodel:local_min=0.2" -> GraphAgent(use_model=False, local_min=0.2)
        base, _, opts = agent_name.partition(":")
        kw = {k: float(v) for k, v in (o.split("=") for o in opts.split(",") if o)}
        if base in ("v2", "v3", "v4", "v5"):
            # The LLM at LLM_URL; every call recorded under CALLS_DIR, or
            # replayed from there when REPLAY is set (then no model is needed).
            llm = None
            if LLM_URL:
                calls = Path(CALLS_DIR) / f"{agent_name.replace(':', '_')}_s{seed}_{gid}.jsonl"
                live = OpenAIChatLLM(base_url=LLM_URL, model=LLM_MODEL, extra=LLM_EXTRA,
                                     api_key=LLM_KEY, allow_remote=bool(LLM_KEY),
                                     timeout=LLM_TIMEOUT)
                # A replay miss falls through to the live model and is recorded
                # in the same file, so an interrupted run resumes where it stopped.
                llm = (ReplayLLM(calls, fallback=RecordingLLM(live, calls) if not REPLAY_ONLY
                                 else None) if REPLAY
                       else RecordingLLM(live, calls))
            if base == "v5":
                agent = V5Agent(seed=seed, llm=llm, baselines=meta.baseline_actions,
                                total_levels=meta.win_levels,
                                temperature=float(kw.pop("temperature", 0.6)))
            elif base == "v4":
                agent = V4Agent(seed=seed, llm=llm, baselines=meta.baseline_actions,
                                total_levels=meta.win_levels,
                                temperature=float(kw.pop("temperature", 0.6)),
                                image=bool(kw.pop("image", 0)))
            elif base == "v3":
                agent = V3Agent(seed=seed, llm=llm, baselines=meta.baseline_actions,
                                temperature=float(kw.pop("temperature", 0.0)))
            else:
                agent = HypothesisAgent(seed=seed, llm=llm, **kw)
        else:
            agent = GraphAgent(seed=seed, use_model=(base == "graph"), **kw)
        choose = agent.act

    t0 = time.time()
    level_actions: list[int] = []
    for _ in range(n_actions):
        obs = client.current
        if obs.state == GameState.WIN:
            break
        before = obs.levels_completed
        obs = client.step(choose(obs))
        budget.observe(obs)
        if obs.levels_completed > before:
            level_actions.append(client.actions_used)
    res = budget.result()
    return {
        "game": gid,
        "agent": agent_name,
        "levels": res.levels_completed,
        "total_levels": res.total_levels,
        "score": res.score(),
        "actions": client.actions_used,
        "cleared_at": level_actions,
        "per_level": [
            {"i": lv.index, "actions": lv.actions, "baseline": lv.baseline,
             "score": lv.score()} for lv in res.levels if lv.completed
        ],
        "seconds": round(time.time() - t0, 1),
        "detail": agent.describe() if agent else "",
        # v2 only: (level, actions into level, hypotheses proposed, compiled) per call,
        # and the population as it stood at the end.
        "llm_calls": getattr(agent, "calls", None),
        "hypotheses": agent.pop.describe() if hasattr(agent, "pop") else None,
        "level_report": agent.level_report() if hasattr(agent, "level_report") else None,
        "stats": dict(agent.stats) if agent else None,
        # v4 only: the assumption store at the end, and every edit made to it.
        "assumptions": agent.store.describe() if hasattr(agent, "store") else None,
        "store_log": agent.store.log if hasattr(agent, "store") else None,
        "history": getattr(agent, "history", None) if base_name(agent_name) in ("v4", "v5") else None,
        # v4 (D-044): prediction accuracy on steps where more than the every-action
        # changers changed, and whether the explorer ever acted (not a clean run).
        "accuracy": agent.accuracy() if hasattr(agent, "accuracy") else None,
        "explorer_steps": (agent.stats.get("explore_steps", 0)
                           if agent is not None and hasattr(agent, "store") else None),
    }


def main() -> None:
    global LLM_URL, CALLS_DIR, REPLAY, REPLAY_ONLY, LLM_EXTRA, LLM_MODEL, LLM_KEY, LLM_TIMEOUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", type=int, default=500)
    ap.add_argument("--agents", default="graph;graph_nomodel;explore")
    ap.add_argument("--games", nargs="*")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env-dir", default="environment_files")
    ap.add_argument("--out")
    ap.add_argument("--llm-url", default="", help="OpenAI-compatible server for v2, e.g. http://127.0.0.1:8080")
    ap.add_argument("--calls-dir", default=CALLS_DIR)
    ap.add_argument("--replay", action="store_true", help="serve v2's LLM calls from --calls-dir")
    ap.add_argument("--replay-only", action="store_true", help="with --replay: never call the live model")
    ap.add_argument("--llm-extra", default="{}", help="JSON merged into every LLM request body")
    ap.add_argument("--llm-model", default="local", help="model name sent to the server")
    ap.add_argument("--llm-key-env", default="",
                    help="env var holding an API key for a hosted endpoint (development only; "
                         "setting it allows a non-local --llm-url)")
    ap.add_argument("--llm-timeout", type=float, default=600.0, help="seconds per LLM request")
    ap.add_argument("--jobs", type=int, default=1,
                    help="games played at once (threads; each builds its own arcade). "
                         "An LLM server batches concurrent requests, so this is the throughput knob")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    LLM_URL, CALLS_DIR = args.llm_url, args.calls_dir
    REPLAY, REPLAY_ONLY = args.replay, args.replay_only
    LLM_EXTRA = json.loads(args.llm_extra)
    LLM_MODEL = args.llm_model
    LLM_TIMEOUT = args.llm_timeout
    if args.llm_key_env:
        import os
        LLM_KEY = os.environ.get(args.llm_key_env, "")
        if not LLM_KEY:
            sys.exit(f"--llm-key-env {args.llm_key_env}: variable is empty or unset")

    arcade = make_arcade(environments_dir=args.env_dir)
    games = args.games or available_environments(args.env_dir)
    agents = args.agents.split(";")
    rows = []
    jobs = [(gid, name) for gid in games for name in agents]

    partial = Path(args.out).with_suffix(".partial.jsonl") if args.out else None
    lock = __import__("threading").Lock()

    def one(job):
        gid, name = job
        arc = arcade if args.jobs <= 1 else make_arcade(environments_dir=args.env_dir)
        try:
            r = play(gid, name, args.actions, args.seed, arc, args.env_dir)
        except Exception as exc:  # one game's crash must not lose the others' results
            import traceback
            traceback.print_exc()
            r = {"game": gid, "agent": name, "levels": 0, "total_levels": 0, "score": 0.0,
                 "actions": 0, "cleared_at": [], "seconds": 0, "error": repr(exc)[:500],
                 "detail": f"CRASHED: {exc!r}"[:300]}
        if partial is not None:
            with lock, partial.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, default=str) + "
")
        print(f"{gid:6s} {name:14s} levels {r['levels']}/{r['total_levels']} "
              f"score {100 * (r['score'] or 0):5.1f}%  cleared_at {r['cleared_at']}  "
              f"{r['seconds']}s  {r['detail']}", flush=True)
        return r
    if args.jobs <= 1:
        rows = [one(j) for j in jobs]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            rows = list(ex.map(one, jobs))

    print("\n== summary ==")
    for name in agents:
        rs = [r for r in rows if r["agent"] == name]
        lv = sum(r["levels"] for r in rs)
        tot = sum(r["total_levels"] or 0 for r in rs)
        sc = sum((r["score"] or 0) for r in rs) / len(rs)
        g = sum(1 for r in rs if r["levels"])
        print(f"{name:14s} levels {lv}/{tot}  games with a level {g}/{len(rs)}  "
              f"mean RHAE {100 * sc:.2f}%")
    if args.out:
        Path(args.out).write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1, default=str))


if __name__ == "__main__":
    main()
