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
from arc3.explore import ExploringDriver
from arc3.llm import OpenAIChatLLM, RecordingLLM, ReplayLLM
from arc3.types import RESET, Action, GameState

#: Set from the command line in main(); module-level so play() stays picklable.
LLM_URL = ""
CALLS_DIR = "docs/experiments/llm_calls"
REPLAY = False
REPLAY_ONLY = False


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
        if base == "v2":
            # v2 with the LLM at LLM_URL; every call recorded under CALLS_DIR,
            # or replayed from there when REPLAY is set (then no model is needed).
            llm = None
            if LLM_URL:
                calls = Path(CALLS_DIR) / f"{agent_name.replace(':', '_')}_s{seed}_{gid}.jsonl"
                live = OpenAIChatLLM(base_url=LLM_URL)
                llm = (ReplayLLM(calls, fallback=live if not REPLAY_ONLY else None) if REPLAY
                       else RecordingLLM(live, calls))
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
    }


def main() -> None:
    global LLM_URL, CALLS_DIR, REPLAY, REPLAY_ONLY
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
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    LLM_URL, CALLS_DIR = args.llm_url, args.calls_dir
    REPLAY, REPLAY_ONLY = args.replay, args.replay_only

    arcade = make_arcade(environments_dir=args.env_dir)
    games = args.games or available_environments(args.env_dir)
    agents = args.agents.split(";")
    rows = []
    for gid in games:
        for name in agents:
            r = play(gid, name, args.actions, args.seed, arcade, args.env_dir)
            rows.append(r)
            print(f"{gid:6s} {name:14s} levels {r['levels']}/{r['total_levels']} "
                  f"score {100 * (r['score'] or 0):5.1f}%  cleared_at {r['cleared_at']}  "
                  f"{r['seconds']}s  {r['detail']}", flush=True)

    print("\n== summary ==")
    for name in agents:
        rs = [r for r in rows if r["agent"] == name]
        lv = sum(r["levels"] for r in rs)
        tot = sum(r["total_levels"] for r in rs)
        sc = sum((r["score"] or 0) for r in rs) / len(rs)
        g = sum(1 for r in rs if r["levels"])
        print(f"{name:14s} levels {lv}/{tot}  games with a level {g}/{len(rs)}  "
              f"mean RHAE {100 * sc:.2f}%")
    if args.out:
        Path(args.out).write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
