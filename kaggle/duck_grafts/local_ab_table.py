# SPDX-License-Identifier: MIT-0
# ARC-AGI-3 project (MIT-0). Tabulate local_run.py summaries: one row per (round, game), arms
# side by side, then per-arm means. Usage:
#   python kaggle/duck_grafts/local_ab_table.py docs/experiments/local/duck_ab_2026-11-02
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1])
    runs = {}
    for f in sorted(root.glob("r*_[AD]/summary.json")):
        rnd, arm = f.parent.name.rsplit("_", 1)
        runs[(rnd, arm)] = json.loads(f.read_text(encoding="utf-8"))
    rounds = sorted({r for r, _ in runs})
    print("| round | game | A levels | A score | A actions | A model RESETs | D levels | D score | D actions "
          "| D model RESETs | D hint turns (budget/life/effects of prompts) | gen. tokens A / D (k) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    totals = {"A": [], "D": []}
    for rnd in rounds:
        a, d = runs.get((rnd, "A")), runs.get((rnd, "D"))
        if not a or not d:
            continue
        ga = {g["game_id"]: g for g in a["games"]}
        for gd in d["games"]:
            g = ga[gd["game_id"]]
            h = gd["hint_turns"]
            totals["A"].append(g["final_score"] or 0.0)
            totals["D"].append(gd["final_score"] or 0.0)
            print(f"| {rnd} | {gd['game_id'][:4]} | {g['levels_completed']}/{g['number_of_levels']} | "
                  f"{g['final_score']:.2f} | {g['actions']} | {g['resets_model']} | "
                  f"{gd['levels_completed']}/{gd['number_of_levels']} | {gd['final_score']:.2f} | "
                  f"{gd['actions']} | {gd['resets_model']} | {h.get('budget', 0)}/{h.get('life', 0)}/"
                  f"{h.get('effects', 0)} of {gd['turn_prompts']} | "
                  f"{g['generated_tokens'] / 1000:.0f} / {gd['generated_tokens'] / 1000:.0f} |")
        print(f"| {rnd} | **mean** | | **{a['mean_score']:.2f}** | | | | **{d['mean_score']:.2f}** | | | | |")
    for arm, xs in totals.items():
        if xs:
            print(f"\n{arm}: mean over all game-runs {sum(xs) / len(xs):.2f} (n={len(xs)})")


if __name__ == "__main__":
    main()
