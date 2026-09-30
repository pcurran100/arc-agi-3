"""Where is v2 weak? A five-link diagnosis of a benchmark run (D-035).

v2 can fail at five distinct places, and each has a different fix, so a bad
score is not one finding but one of five:

1. **Format** - does the model answer with code we can parse and compile?
   Fix: the prompt, or the parser.
2. **Informative** - are the compiled hypotheses anything but constant,
   erroring, or "goal reached" everywhere? Fix: the prompt, the scene.
3. **Steering** - do they change what the agent does (goal detours)?
   Fix: the value-to-action coupling (`goal_weight`, best-first rule).
4. **Correct** - when a level was won, did any hypothesis score the winning
   state highly (`wins_confirmed`)? This is the decisive link: it separates "the
   model cannot guess goals" from "guessing works but steering by it does not".
   Fix: the model, thinking mode, the evidence given to it.
5. **Net** - v2 against v1 game by game on the same seeds. Needs a v1 run.

Reads the `eval_s*.json` files a benchmark leaves (Kaggle `results/` or a local
`eval_agent.py --out`), and optionally the `llm_calls/*.jsonl` beside them.

Usage:
    python scripts/diagnose_v2.py RESULTS_DIR [--v1 V1_RESULTS.json ...]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        seed = d.get("seed", d.get("args", {}).get("seed"))
        for r in d["rows"]:
            r = dict(r)
            r["_seed"] = seed
            rows.append(r)
    return rows


def pct(a: float, b: float) -> str:
    return f"{100 * a / b:.0f}%" if b else "n/a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, help="directory with eval_s*.json (and llm_calls/)")
    ap.add_argument("--v1", type=Path, nargs="*", default=[],
                    help="v1 result files on the same seeds, for link 5")
    args = ap.parse_args()

    files = sorted(p for p in args.results.glob("eval_s*.json") if "_k" not in p.stem)
    rows = [r for r in load_rows(files) if str(r["agent"]).startswith("v2")]
    if not rows:
        raise SystemExit(f"no v2 rows in {args.results}")
    print(f"{len(rows)} v2 game runs from {len(files)} file(s)\n")

    # Link 1: format.
    calls = [c for r in rows for c in (r.get("llm_calls") or [])]
    proposed = sum(c[2] for c in calls)
    compiled = sum(c[3] for c in calls)
    empty = sum(1 for c in calls if c[2] == 0)
    print("1. FORMAT")
    print(f"   {len(calls)} LLM calls; {empty} returned no parseable hypothesis ({pct(empty, len(calls))})")
    print(f"   {proposed} hypotheses proposed, {compiled} compiled ({pct(compiled, proposed)})")
    errors = Counter()
    hyps = [h for r in rows for lv in (r.get("level_report") or []) for h in lv["hypotheses"]]
    for h in hyps:
        if h["compile_error"]:
            errors[h["compile_error"].split(":")[0]] += 1
    if errors:
        print("   compile errors:", ", ".join(f"{k} x{v}" for k, v in errors.most_common(5)))

    # Link 2: informative.
    ok = [h for h in hyps if h["compiled"]]
    flat = sum(1 for h in ok if not h["informative"])
    runtime = sum(1 for h in ok if h["errors"])
    falsey = sum(1 for h in ok if h["false_goals"] >= 3)
    print("\n2. INFORMATIVE")
    print(f"   of {len(ok)} compiled: {flat} flat ({pct(flat, len(ok))}), "
          f"{runtime} raised at runtime ({pct(runtime, len(ok))}), "
          f"{falsey} called >=3 non-winning states 'reached' ({pct(falsey, len(ok))})")

    # Link 3: steering.
    # A followed hypothesis steers through frontier cost on every navigation,
    # not only through explicit detours, so "had a focus" is the steering test;
    # detours are reported beside it as the stronger, rarer signal.
    stats = [(r.get("stats") or {}) for r in rows]
    detours = [s.get("goal_detours", 0) for s in stats]
    steered = sum(1 for s in stats if s.get("focus_switches", 0) or s.get("goal_detours", 0))
    print("\n3. STEERING")
    print(f"   games where a hypothesis was followed: {steered}/{len(rows)}; "
          f"games with explicit detours: {sum(1 for d in detours if d)}; "
          f"median detours {sorted(detours)[len(detours) // 2]}")

    # Link 4: correct.
    won = [lv for r in rows for lv in (r.get("level_report") or []) if lv["won"]]
    with_hyps = [lv for lv in won if any(h["compiled"] for h in lv["hypotheses"])]
    right = [lv for lv in with_hyps if any(h["wins_confirmed"] for h in lv["hypotheses"])]
    print("\n4. CORRECT (the decisive link)")
    print(f"   {len(won)} levels won; {len(with_hyps)} had live hypotheses at the time")
    print(f"   levels where some hypothesis scored the winning state >= 0.5: "
          f"{len(right)}/{len(with_hyps)} ({pct(len(right), len(with_hyps))})")
    goals = Counter(h["goal"] for lv in right for h in lv["hypotheses"] if h["wins_confirmed"])
    for g, n in goals.most_common(5):
        print(f"     confirmed: {g[:90]}  (x{n})")

    # Link 5: net, against v1 on the same seeds.
    print("\n5. NET vs v1")
    if not args.v1:
        print("   (no --v1 files given)")
    else:
        v1 = {(r["game"].split("-")[0], r["_seed"]): r for r in load_rows(args.v1)
              if not str(r["agent"]).startswith("v2")}
        better = worse = same = 0
        for r in rows:
            b = v1.get((r["game"].split("-")[0], r["_seed"]))
            if b is None:
                continue
            d = (r["score"] or 0) - (b["score"] or 0)
            better += d > 1e-9
            worse += d < -1e-9
            same += abs(d) <= 1e-9
            if abs(d) > 1e-9:
                print(f"   {r['game'][:6]} s{r['_seed']}: v2 {100 * (r['score'] or 0):.1f}% "
                      f"vs v1 {100 * (b['score'] or 0):.1f}%")
        print(f"   games better {better}, worse {worse}, same {same}")

    # The verdict: the first broken link is where to work.
    print("\nFIRST WEAK LINK:", end=" ")
    if calls and compiled < 0.5 * max(1, proposed) or empty > 0.5 * max(1, len(calls)):
        print("1 format - fix the prompt or parser before anything else")
    elif ok and (flat + runtime + falsey) > 0.7 * len(ok):
        print("2 informative - the model's hypotheses carry no signal")
    elif rows and steered < 0.3 * len(rows):
        print("3 steering - hypotheses exist but rarely change behaviour")
    elif with_hyps and len(right) < 0.3 * len(with_hyps):
        print("4 correct - the model does not recognise winning states")
    else:
        print("none obvious - compare against v1 (link 5) on more seeds")


if __name__ == "__main__":
    main()
