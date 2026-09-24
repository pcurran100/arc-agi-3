"""What, if anything, moves the playfield on the games random play cannot budge?

D-028 measured that **8 of 25 games change the playfield on fewer than 15% of
actions** under random play — vc33 on 0 of 99 — and that all eight score zero.
They are half the zeros and none of them is a modelling failure: there is
nothing in their corpus to model.

This asks the question that decides what to do about it, and the two answers
have very different consequences:

- **Something at the root moves it, and random play missed.** A *targeting*
  problem. Fixable by aiming better, and cheap.
- **Nothing at the root moves it.** A *depth* problem: the game needs a specific
  sequence before anything responds, and no single-action policy will ever find
  it. Much harder, and it is the condition D-028 named as putting these games
  beyond reach in the remaining time.

It also discharges an action item D-028 left open. D-021's click map hashed the
**whole frame**, so a click that merely ticked a move counter counted as a live
pixel producing a distinct outcome; vc33 was recorded as 100% live with 3
outcomes and has zero playfield activity. Every figure here is computed
**outside the bookkeeping band**, so "live" means *the game responded*, not *the
screen changed*.

Like the click map this is an **offline instrument only** (D-010 footing). It
brute-forces the action space by RESET-ing between probes, which is free locally
and impossible at evaluation where every probe is permanent (D-009).

Usage:
    python scripts/eval_playfield_map.py [--games a,b,c] [--depths 0,5,20]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from arc3.client.env import EnvClient, available_environments
from arc3.client.replay import read_episode
from arc3.drive import RandomObjectDriver, foreground_object_targets, object_targets
from arc3.dynamics.predict import bookkeeping_mask
from arc3.types import CLICK, GRID_H, GRID_W, Action

RESET = 0

#: D-028's eight. Fewer than 15% of actions move the playfield under random play.
STARVED = ("ft09", "lf52", "lp85", "s5i5", "sb26", "su15", "tn36", "vc33")


def band_for(game: str, corpus: str = "corpus") -> np.ndarray:
    """The bookkeeping band, learned from the recorded episode for this game.

    Uses the corpus rather than recomputing from scratch so that the band is the
    same one every other measurement in the repository used.
    """
    path = Path(corpus) / f"{game}.jsonl.gz"
    if not path.exists():
        return np.zeros((GRID_H, GRID_W), dtype=bool)
    ep = read_episode(path)
    trans = [t for t in ep.transitions() if t.action.id != 0]
    return bookkeeping_mask(trans[: max(1, int(len(trans) * 0.7))])


def _probe(client: EnvClient, action: Action, base: np.ndarray, band: np.ndarray) -> dict:
    """Take one action, classify what it did, and RESET back."""
    obs = client.step(action)
    frame = obs.last
    out = {"any": False, "playfield": False, "px": 0}
    if frame is not None:
        diff = frame != base
        out["any"] = bool(diff.any())
        out["playfield"] = bool((diff & ~band).any())
        out["px"] = int((diff & ~band).sum())
    return out


def scan_state(client: EnvClient, base: np.ndarray, band: np.ndarray, legal, stride: int) -> dict:
    """Try every non-click action, and click a grid of pixels, from one state."""
    acts_any = acts_pf = 0
    tried = 0
    best_px = 0
    live_targets: list[tuple[int, int]] = []

    def run(a: Action, tag=None):
        nonlocal acts_any, acts_pf, tried, best_px
        r = _probe(client, a, base, band)
        tried += 1
        acts_any += r["any"]
        acts_pf += r["playfield"]
        best_px = max(best_px, r["px"])
        if r["playfield"] and tag is not None:
            live_targets.append(tag)
        client.step(Action.simple(RESET))

    for a in legal:
        if a != CLICK:
            run(Action.simple(a))
    if CLICK in legal:
        for y in range(0, GRID_H, stride):
            for x in range(0, GRID_W, stride):
                run(Action.click(x, y), tag=(x, y))
    return {
        "tried": tried,
        "any": acts_any,
        "playfield": acts_pf,
        "max_playfield_px": best_px,
        "live_targets": live_targets[:20],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=",".join(STARVED))
    ap.add_argument("--depths", default="0,5,20,50")
    ap.add_argument("--stride", type=int, default=2, help="click grid stride (1 = all 4096)")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    logging.disable(logging.INFO)

    wanted = args.games.split(",")
    depths = [int(d) for d in args.depths.split(",")]
    ids = {g[:4]: g for g in available_environments()}

    print(f"probing OUTSIDE the bookkeeping band; click stride {args.stride}, seed {args.seed}\n")
    hdr = f"{'game':6s} {'depth':>6s} {'probes':>7s} {'any':>6s} {'PLAYFIELD':>10s} {'max px':>7s}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for short in wanted:
        gid = ids.get(short)
        if gid is None:
            continue
        band = band_for(short, args.corpus)
        client = EnvClient(gid, seed=args.seed, record=False)
        legal = sorted(a for a in client.current.available_actions if a != RESET)
        driver = RandomObjectDriver(seed=args.seed)

        # Fixed action sequence to reach each depth, replayed exactly.
        seq: list[Action] = []
        probe_client = EnvClient(gid, seed=args.seed, record=False)
        for _ in range(max(depths)):
            a = driver.next_action(probe_client.current)
            if a is None:
                break
            seq.append(a)
            probe_client.step(a)

        for d in depths:
            if d > len(seq):
                continue
            client.step(Action.simple(RESET))
            for a in seq[:d]:
                client.step(a)
            base = client.current.last
            if base is None:
                continue
            base = base.copy()
            legal_here = sorted(a for a in client.current.available_actions if a != RESET)
            t0 = time.time()
            r = scan_state(client, base, band, legal_here, args.stride)
            r.update({"game": short, "depth": d, "seconds": time.time() - t0,
                      "band_px": int(band.sum())})
            rows.append(r)
            print(
                f"{short:6s} {d:6d} {r['tried']:7d} {r['any']:6d} {r['playfield']:10d} "
                f"{r['max_playfield_px']:7d}"
            )
        print()

    if rows:
        print("=" * 60)
        never = sorted({r["game"] for r in rows} - {r["game"] for r in rows if r["playfield"] > 0})
        sometimes = sorted({r["game"] for r in rows if r["playfield"] > 0})
        print(f"games where SOME probe moved the playfield : {sometimes}")
        print(f"games where NOTHING did, at any depth tried: {never}")
        print()
        print("  -> 'sometimes' is a TARGETING problem: the move exists and random play missed it.")
        print("  -> 'never' is a DEPTH problem: no single action responds and a sequence is needed.")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
